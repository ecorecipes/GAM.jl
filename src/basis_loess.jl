# Loess / local polynomial smooth — bs=:lo
#
# Implements a kernel basis smooth inspired by R's gamlss lo().
# Uses tricube kernel basis functions centered at knot locations with
# a ridge (identity) penalty for regularization.

"""Local polynomial (loess) smooth basis (gamlss-style `bs=:lo`)."""
struct LoessSmooth <: AbstractBasisType end

# Register
BASIS_TYPES[:lo] = LoessSmooth()

"""
    _tricube_kernel(u)

Tricube kernel: K(u) = (1 - |u|³)³ for |u| < 1, else 0.
"""
function _tricube_kernel(u::Real)
    au = abs(u)
    return au < 1.0 ? (1.0 - au^3)^3 : 0.0
end

"""
    _gaussian_kernel(u)

Gaussian kernel: K(u) = exp(-u²/2).
"""
function _gaussian_kernel(u::Real)
    return exp(-u^2 / 2)
end

"""
    _loess_kernel_matrix(x, knots, bandwidth; kernel=:tricube)

Build kernel basis matrix X where X[i,j] = K((x[i] - knots[j]) / h).
"""
function _loess_kernel_matrix(x::Vector{Float64}, knots::Vector{Float64},
    bandwidth::Float64; kernel::Symbol = :tricube)
    n = length(x)
    nk = length(knots)
    X = zeros(n, nk)
    kfun = kernel == :gaussian ? _gaussian_kernel : _tricube_kernel
    for j in 1:nk
        for i in 1:n
            u = (x[i] - knots[j]) / bandwidth
            X[i, j] = kfun(u)
        end
    end
    return X
end

function _smooth_construct(::LoessSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Loess smooths currently support 1d only"))

    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)

    # Extract loess-specific options from xt
    degree = get(spec.xt, :degree, 1)::Int
    degree in (1, 2) || throw(ArgumentError("Loess degree must be 1 or 2, got $degree"))
    span = get(spec.xt, :span, 0.75)::Float64
    kernel = get(spec.xt, :kernel, :tricube)::Symbol

    # Knot locations
    if user_knots !== nothing
        knots = Float64.(user_knots)
    else
        knots = knot_quantiles(x, k)
    end
    nk = length(knots)

    # Bandwidth: span * data range, scaled by knot density
    x_range = maximum(x) - minimum(x)
    bandwidth = span * x_range / max(nk - 1, 1)

    # Ensure bandwidth is positive
    if bandwidth < eps()
        bandwidth = 1.0
    end

    # Build kernel basis matrix
    X = _loess_kernel_matrix(x, knots, bandwidth; kernel = kernel)

    # For degree=2, augment with x*K columns for local quadratic effect
    if degree == 2
        X_lin = zeros(n, nk)
        for j in 1:nk
            for i in 1:n
                X_lin[i, j] = X[i, j] * (x[i] - knots[j])
            end
        end
        X = hcat(X, X_lin)
    end

    ncol = size(X, 2)

    # Ensure no zero columns (can happen with compact-support tricube kernels)
    col_norms = vec(sum(abs2, X; dims = 1))
    active = col_norms .> 1e-10
    if !all(active)
        X = X[:, active]
        ncol = size(X, 2)
    end

    # Penalty: ridge (identity) — the kernel localization provides the smoothing
    S = Matrix{Float64}(I, ncol, ncol)
    penalties = Matrix{Float64}[S]

    null_dim = 1  # constant function is approximately in the span
    pen_rank = ncol - null_dim

    # Store bandwidth and kernel info in xt for prediction
    spec.xt[:_bandwidth] = bandwidth
    spec.xt[:_kernel] = kernel
    spec.xt[:_degree] = degree
    spec.xt[:_active] = active

    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knots,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

function _predict_matrix(::LoessSmooth, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    knots = smooth.knots
    nk = length(knots)

    bandwidth = smooth.spec.xt[:_bandwidth]::Float64
    kernel = smooth.spec.xt[:_kernel]::Symbol
    degree = smooth.spec.xt[:_degree]::Int

    # Build kernel basis at new points
    X_new = _loess_kernel_matrix(x_new, knots, bandwidth; kernel = kernel)

    # Augment for degree=2
    if degree == 2
        n_new = length(x_new)
        X_lin = zeros(n_new, nk)
        for j in 1:nk
            for i in 1:n_new
                X_lin[i, j] = X_new[i, j] * (x_new[i] - knots[j])
            end
        end
        X_new = hcat(X_new, X_lin)
    end

    # Apply same column filter as construction
    active = smooth.spec.xt[:_active]::BitVector
    if !all(active)
        X_new = X_new[:, active]
    end

    # Apply constraint
    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = nullspace(C)
        return X_new * Z
    end
    return X_new
end
