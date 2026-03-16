# Shape-Constrained Additive Models (SCAM) — basis construction
#
# Implements SCOP-splines (shape constrained P-splines) following
# Pya & Wood (2015). Uses B-spline basis with Σ matrix reparameterization
# to enforce monotonicity, convexity/concavity constraints.
#
# Key idea: transform B-spline basis X₁ via constraint matrix Σ so that
# X = X₁ * Σ, then exponentiate certain coefficients to enforce positivity.

using LinearAlgebra: I as Eye

# ============================================================================
# Σ (Sigma) constraint matrix construction
# ============================================================================

"""
    _sigma_matrix(constraint_type, q) -> Matrix{Float64}

Construct the Σ constraint matrix of size q×q for the given shape constraint.
The constraint matrix transforms unconstrained B-spline coefficients into
shape-constrained ones: β_constrained = Σ * exp(β_unconstrained).

Returns the FULL q×q matrix (before identifiability column drops).
"""
function _sigma_matrix(::MonoIncBasis, q::Int)
    # Monotone increasing: lower triangular of 1's (cumulative sum)
    # β*_j = β₁ + ν₂ + ... + νⱼ where νᵢ ≥ 0
    Sig = ones(q, q)
    for i in 1:q, j in (i + 1):q
        Sig[i, j] = 0.0
    end
    return Sig
end

function _sigma_matrix(::MonoDecBasis, q::Int)
    # Monotone decreasing: negative lower triangular, positive first column
    # β*_j = β₁ - ν₂ - ... - νⱼ where νᵢ ≥ 0
    Sig = fill(-1.0, q, q)
    for i in 1:q, j in (i + 1):q
        Sig[i, j] = 0.0
    end
    Sig[:, 1] .= 1.0  # first column positive
    return Sig
end

function _sigma_matrix(::ConcaveBasis, q::Int)
    # Concave (f'' ≤ 0): differences of coefficients are decreasing
    # Uses (q-1)×(q-1) matrix on columns 2:q of the B-spline basis
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    Sig[1:qm1, 1] .= collect(1:qm1)
    for j in 2:qm1
        for i in j:qm1
            Sig[i, j] = -Float64(i - j + 1)
        end
    end
    return Sig
end

function _sigma_matrix(::ConvexBasis, q::Int)
    # Convex (f'' ≥ 0): differences of coefficients are increasing
    # Negation of concave Σ for off-diagonal, same first column
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    Sig[1:qm1, 1] .= collect(1:qm1)
    for j in 2:qm1
        for i in j:qm1
            Sig[i, j] = Float64(i - j + 1)
        end
    end
    return Sig
end

function _sigma_matrix(::MonoIncConvexBasis, q::Int)
    # Monotone increasing + convex: cumsum of cumsums
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    for j in 1:qm1
        for i in j:qm1
            Sig[i, j] = Float64(i - j + 1)
        end
    end
    return Sig
end

function _sigma_matrix(::MonoIncConcaveBasis, q::Int)
    # Monotone increasing + concave — matches R scam's smooth.construct.micv
    # Sig[i,j] = min(i, qm1 - j + 1) for i,j in 1:qm1
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    for j in 1:qm1
        for i in 1:qm1
            Sig[i, j] = Float64(min(i, qm1 - j + 1))
        end
    end
    return Sig
end

function _sigma_matrix(::MonoDecConvexBasis, q::Int)
    # Monotone decreasing + convex — negation of MICV Sigma
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    for j in 1:qm1
        for i in 1:qm1
            Sig[i, j] = -Float64(min(i, qm1 - j + 1))
        end
    end
    return Sig
end

function _sigma_matrix(::MonoDecConcaveBasis, q::Int)
    # Monotone decreasing + concave
    qm1 = q - 1
    Sig = zeros(qm1, qm1)
    for j in 1:qm1
        for i in j:qm1
            Sig[i, j] = -Float64(i - j + 1)
        end
    end
    return Sig
end

# ============================================================================
# Whether to drop the first B-spline column before applying Σ
# ============================================================================

# Monotone types: Σ is q×q on the full basis, then drop column 1
_drops_first_bspline_col(::MonoIncBasis) = false
_drops_first_bspline_col(::MonoDecBasis) = false
# Concave/convex/combined types: Σ is (q-1)×(q-1) on columns 2:q
_drops_first_bspline_col(::ConcaveBasis) = true
_drops_first_bspline_col(::ConvexBasis) = true
_drops_first_bspline_col(::MonoIncConvexBasis) = true
_drops_first_bspline_col(::MonoIncConcaveBasis) = true
_drops_first_bspline_col(::MonoDecConvexBasis) = true
_drops_first_bspline_col(::MonoDecConcaveBasis) = true

# ============================================================================
# Knot vector construction (matching scam's approach)
# ============================================================================

"""
    _scam_knots(x, q, m) -> Vector{Float64}

Construct a knot vector for SCOP-splines. `q` is the basis dimension,
`m` is the penalty order (default 2 for cubic). The spline order is m+2.

Interior knots are evenly spaced from min(x) to max(x), with boundary
knots extended outward uniformly.
"""
function _scam_knots(x::AbstractVector{<:Real}, q::Int, m::Int)
    nk = q + m + 2  # total number of knots
    lo, hi = minimum(x), maximum(x)
    n_interior = q - m
    interior = range(lo, hi; length = n_interior)
    dx = interior[2] - interior[1]

    xk = zeros(nk)
    xk[(m + 2):(q + 1)] .= interior
    for i in 1:(m + 1)
        xk[i] = xk[m + 2] - (m + 2 - i) * dx
    end
    for i in (q + 2):(q + m + 2)
        xk[i] = xk[q + 1] + (i - q - 1) * dx
    end
    return xk
end

# ============================================================================
# Shape-constrained smooth construction (unified for all constraint types)
# ============================================================================

function _smooth_construct(basis::AbstractConstrainedBasis, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Shape-constrained smooths only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    q = min(spec.k, n)
    q >= 4 || throw(ArgumentError("k=$q too small for shape-constrained smooth (need k ≥ 4)"))
    m = spec.m === nothing ? 2 : spec.m  # penalty order, default cubic
    spline_order = m + 2

    # Knot vector
    if user_knots !== nothing
        xk = Float64.(user_knots)
        length(xk) == q + m + 2 || throw(ArgumentError(
            "Expected $(q + m + 2) knots for k=$q, m=$m; got $(length(xk))"))
    else
        xk = _scam_knots(x, q, m)
    end

    # B-spline basis
    X1 = _bspline_basis(x, xk, spline_order)

    # Apply Σ constraint matrix
    drops_first = _drops_first_bspline_col(basis)
    if drops_first
        # Concave/convex types: work on columns 2:q
        Sig = _sigma_matrix(basis, q)  # (q-1) × (q-1)
        X = X1[:, 2:q] * Sig
    else
        # Monotone types: full q×q Σ, then drop first column
        Sig_full = _sigma_matrix(basis, q)  # q × q
        X_full = X1 * Sig_full
        X = X_full[:, 2:end]
        Sig = Sig_full[2:end, 2:end]
    end
    # X is n × (q-1), Sig is (q-1) × (q-1)

    ncol_X = size(X, 2)  # q-1

    # Sum-to-zero centering constraint
    cmX = vec(mean(X; dims = 1))
    X .-= cmX'

    # Penalty matrix: first-order differences of the constrained coefficients
    if drops_first
        # For concave/convex: penalty on coefficients 2:(q-1)
        P = _diff_matrix(ncol_X - 1, 1)  # (ncol_X-2) × (ncol_X-1)
        S_inner = P' * P  # (ncol_X-1) × (ncol_X-1)
        S = zeros(ncol_X, ncol_X)
        S[2:end, 2:end] .= S_inner
    else
        # For monotone: penalty on all q-1 constrained coefficients
        P = _diff_matrix(ncol_X, 1)  # (ncol_X-1) × ncol_X
        S = P' * P  # ncol_X × ncol_X
    end

    # Rescale penalty to match mgcv's smoothCon normalization:
    # S_new = S / maS where maS = ||S||_1 / ||X||_∞²
    # This makes smoothing parameters comparable across different smooth types.
    maXX = opnorm(X, Inf)^2
    maS_norm = opnorm(S, 1)
    if maS_norm > 0 && maXX > 0
        maS = maS_norm / maXX
        S ./= maS
    end

    penalties = Matrix{Float64}[S]

    # All coefficients must be positive (exponentiated during fitting)
    p_ident = trues(ncol_X)

    pen_rank = ncol_X - 1
    null_dim = 2  # unpenalized space: straight line (2 DoF)

    return ConstructedSmooth(
        spec, X, penalties,
        xk,
        null_dim, pen_rank,
        nothing, nothing, 0, 0,
        Sig, cmX, p_ident,
    )
end

"""
    _diff_matrix(n, d) -> Matrix{Float64}

Construct the d-th order differencing matrix of size (n-d) × n.
"""
function _diff_matrix(n::Int, d::Int)
    D = Matrix{Float64}(Eye, n, n)
    for _ in 1:d
        m = size(D, 1)
        D_new = zeros(m - 1, size(D, 2))
        for i in 1:(m - 1)
            D_new[i, :] .= D[i + 1, :] .- D[i, :]
        end
        D = D_new
    end
    return D
end

# ============================================================================
# Prediction matrix for shape-constrained smooths
# ============================================================================

function _predict_matrix(basis::AbstractConstrainedBasis, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    n = length(x_new)

    q = smooth.spec.k
    m = smooth.spec.m === nothing ? 2 : smooth.spec.m
    spline_order = m + 2
    xk = smooth.knots
    Sig = smooth.Sigma
    cmX = smooth.cmX

    # Inner knot range for extrapolation handling
    ll = xk[m + 2]
    ul = xk[end - m - 1]

    ind_in = ll .<= x_new .<= ul

    drops_first = _drops_first_bspline_col(basis)

    if all(ind_in)
        X1 = _bspline_basis(x_new, xk, spline_order)
        if drops_first
            X = X1[:, 2:q] * Sig
        else
            Sig_full = _sigma_matrix(basis, q)
            X = (X1 * Sig_full)[:, 2:end]
        end
    else
        # Linear extrapolation outside knot range
        eps_pts = [ll, ll, ul, ul]
        D_endpts = _bspline_deriv_design(xk, eps_pts, spline_order)

        X1_full = zeros(n, q)
        for i in 1:n
            if ind_in[i]
                X1_full[i:i, :] .= _bspline_basis([x_new[i]], xk, spline_order)
            elseif x_new[i] < ll
                # Linear extrapolation from lower bound
                X1_full[i, :] .= D_endpts[1, :] .+ (x_new[i] - ll) .* D_endpts[2, :]
            else
                # Linear extrapolation from upper bound
                X1_full[i, :] .= D_endpts[3, :] .+ (x_new[i] - ul) .* D_endpts[4, :]
            end
        end

        if drops_first
            X = X1_full[:, 2:q] * Sig
        else
            Sig_full = _sigma_matrix(basis, q)
            X = (X1_full * Sig_full)[:, 2:end]
        end
    end

    # Apply centering
    X .-= cmX'
    return X
end

"""
    _bspline_deriv_design(knots, pts, order) -> Matrix{Float64}

Evaluate B-spline basis and first derivative at points `pts`.
Returns a matrix where rows alternate: [value_at_pt1, deriv_at_pt1, value_at_pt2, ...].
Used for linear extrapolation at boundary points.
"""
function _bspline_deriv_design(knots::Vector{Float64}, pts::Vector{Float64}, order::Int)
    n_pts = length(pts) ÷ 2
    n_basis = length(knots) - order

    result = zeros(length(pts), n_basis)
    h = 1e-7  # finite difference step

    for idx in 1:n_pts
        pt = pts[2 * idx - 1]
        # Value
        result[2 * idx - 1, :] .= vec(_bspline_basis([pt], knots, order))
        # Derivative via central finite difference
        B_plus = vec(_bspline_basis([pt + h], knots, order))
        B_minus = vec(_bspline_basis([pt - h], knots, order))
        result[2 * idx, :] .= (B_plus .- B_minus) ./ (2h)
    end
    return result
end

# ============================================================================
# Helper: check if any smooth in a model is shape-constrained
# ============================================================================

"""
    has_shape_constraints(smooths) -> Bool

Return true if any smooth term has shape constraints (p_ident is not nothing).
"""
function has_shape_constraints(smooths::Vector{<:ConstructedSmooth})
    return any(sm -> sm.p_ident !== nothing, smooths)
end

"""
    build_p_ident(smooths, n_parametric, total_p) -> BitVector

Construct the global p_ident vector indicating which coefficients in the
full model matrix must be exponentiated during shape-constrained fitting.
"""
function build_p_ident(smooths::Vector{<:ConstructedSmooth}, n_parametric::Int, total_p::Int)
    p_ident = falses(total_p)
    for sm in smooths
        if sm.p_ident !== nothing
            p_ident[sm.first_para:sm.last_para] .= sm.p_ident
        end
    end
    return p_ident
end
