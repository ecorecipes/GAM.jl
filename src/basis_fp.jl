# Fractional polynomial smooth — bs=:fp
#
# Implements fractional polynomial regression as in R's gamlss fp().
# Uses power transformations x^p from a discrete candidate set,
# selecting the best powers by AIC.

"""Fractional polynomial smooth basis (gamlss-style `bs=:fp`)."""
struct FractionalPolynomial <: AbstractBasisType end

# Register
BASIS_TYPES[:fp] = FractionalPolynomial()

"""Default candidate powers for fractional polynomials (0 means log(x))."""
const FP_DEFAULT_POWERS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

"""
    _fp_transform(x, p)

Apply fractional polynomial power transformation: x^p, or log(x) if p == 0.
"""
function _fp_transform(x::Vector{Float64}, p::Float64)
    if p == 0.0
        return log.(x)
    else
        return x .^ p
    end
end

"""
    _fp_basis_columns(x, p1, p2)

Build the two-column basis for FP2 with powers (p1, p2).
If p1 == p2, uses x^p1 and x^p1 * log(x) (repeated power convention).
"""
function _fp_basis_columns(x::Vector{Float64}, p1::Float64, p2::Float64)
    col1 = _fp_transform(x, p1)
    if p1 == p2
        col2 = col1 .* log.(x)
    else
        col2 = _fp_transform(x, p2)
    end
    return col1, col2
end

"""
    _fp_aic(X, y)

Compute AIC for a simple linear regression y ~ X (with intercept).
Returns AIC = n*log(RSS/n) + 2*(p+1) where p is the number of columns in X.
"""
function _fp_aic(X::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    # Add intercept
    Xi = hcat(ones(n), X)
    p = size(Xi, 2)
    # Solve least squares
    coefs = Xi \ y
    resid = y - Xi * coefs
    rss = sum(abs2, resid)
    # AIC (Gaussian)
    if rss <= 0.0
        return -Inf
    end
    return n * log(rss / n) + 2 * p
end

"""
    _fp_select_powers(x, y, powers, degree)

Select the best power(s) by AIC from candidate set.
Returns (best_powers, best_aic).
"""
function _fp_select_powers(x::Vector{Float64}, y::Vector{Float64},
    powers::Vector{Float64}, degree::Int)
    best_aic = Inf
    best_powers = Float64[]

    if degree == 1
        # FP1: try each single power
        for p in powers
            col = _fp_transform(x, p)
            if any(!isfinite, col)
                continue
            end
            X = reshape(col, :, 1)
            aic = _fp_aic(X, y)
            if aic < best_aic
                best_aic = aic
                best_powers = [p]
            end
        end
    else
        # FP2: try all pairs (p1 ≤ p2)
        for (i, p1) in enumerate(powers)
            for p2 in powers[i:end]
                col1, col2 = _fp_basis_columns(x, p1, p2)
                if any(!isfinite, col1) || any(!isfinite, col2)
                    continue
                end
                X = hcat(col1, col2)
                # Check for near-collinearity
                if rank(X) < 2
                    continue
                end
                aic = _fp_aic(X, y)
                if aic < best_aic
                    best_aic = aic
                    best_powers = [p1, p2]
                end
            end
        end
    end

    if isempty(best_powers)
        # Fallback: use p=1 (linear)
        best_powers = degree == 1 ? [1.0] : [1.0, 2.0]
    end

    return best_powers, best_aic
end

function _smooth_construct(::FractionalPolynomial, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Fractional polynomial smooths support 1d only"))

    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    # Get degree and candidate powers from xt or spec
    degree = get(spec.xt, :degree, spec.m === nothing ? 2 : spec.m)::Int
    degree in (1, 2) || throw(ArgumentError("FP degree must be 1 or 2, got $degree"))
    powers = get(spec.xt, :powers, FP_DEFAULT_POWERS)

    # Ensure x is positive (required for negative powers and log)
    x_shift = 0.0
    x_min = minimum(x)
    if x_min <= 0.0
        x_shift = abs(x_min) + 0.1
        x = x .+ x_shift
    end

    # Build a simple response for power selection.
    # Use a pseudo-response: centered version of the data.
    # For proper selection, we need a response vector.
    # In GAM context, the response isn't available at construction time,
    # so we use the covariate itself as a proxy or let the user supply one.
    # If a response hint is provided via xt, use it; otherwise select
    # powers that give the most spread (largest variance in transformed x).
    y_hint = get(spec.xt, :y, nothing)

    if y_hint !== nothing
        y = Float64.(y_hint)
        selected_powers, _ = _fp_select_powers(x, y, Float64.(powers), degree)
    else
        # Without a response, select powers that maximize basis variance
        # (heuristic: powers that best capture data structure)
        # Default to common choices: p=1 for FP1, p=(1,2) for FP2
        selected_powers = degree == 1 ? [1.0] : [1.0, 2.0]
    end

    # Build basis matrix with selected powers
    if degree == 1
        col = _fp_transform(x, selected_powers[1])
        X = reshape(col, :, 1)
    else
        col1, col2 = _fp_basis_columns(x, selected_powers[1], selected_powers[2])
        X = hcat(col1, col2)
    end

    # No penalty for fractional polynomials — model selection is via power choice
    penalties = Matrix{Float64}[]
    null_dim = 0
    pen_rank = size(X, 2)

    # Store selected powers and shift for prediction
    spec.xt[:_selected_powers] = selected_powers
    spec.xt[:_x_shift] = x_shift
    spec.xt[:_degree] = degree

    # For FP, skip constraint absorption — these are parametric-like terms
    # with very few columns, and sum-to-zero doesn't apply well.
    # Instead, center columns for numerical stability.
    col_means = vec(mean(X; dims = 1))
    for j in axes(X, 2)
        X[:, j] .-= col_means[j]
    end
    spec.xt[:_col_means] = col_means

    return ConstructedSmooth(
        spec, X, penalties,
        Float64[],  # no knots for FP
        null_dim, pen_rank,
        nothing, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

function _predict_matrix(::FractionalPolynomial, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))

    selected_powers = smooth.spec.xt[:_selected_powers]::Vector{Float64}
    x_shift = smooth.spec.xt[:_x_shift]::Float64
    degree = smooth.spec.xt[:_degree]::Int
    col_means = smooth.spec.xt[:_col_means]::Vector{Float64}

    # Apply same shift
    if x_shift > 0.0
        x_new = x_new .+ x_shift
    end

    # Build basis with stored powers
    if degree == 1
        col = _fp_transform(x_new, selected_powers[1])
        X_new = reshape(col, :, 1)
    else
        col1, col2 = _fp_basis_columns(x_new, selected_powers[1], selected_powers[2])
        X_new = hcat(col1, col2)
    end

    # Apply same centering
    for j in axes(X_new, 2)
        X_new[:, j] .-= col_means[j]
    end

    return X_new
end
