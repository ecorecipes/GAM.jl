# Adaptive smooths: P-spline basis with multiple local penalties — bs="ad"
#
# Adaptive smooths allow the smoothing parameter to vary locally across the
# covariate range. The basis matrix is identical to a standard P-spline, but
# the single difference penalty is split into n_penalties local penalties using
# a smooth partition-of-unity weighting scheme. Each local penalty receives its
# own smoothing parameter, estimated via REML/GCV in the outer iteration.
#
# Reference: Wood (2011) "Fast stable restricted maximum likelihood and marginal
# likelihood estimation of semiparametric generalized linear models", JRSS-B.

"""Adaptive smooth basis — P-spline with locally varying penalty (mgcv `bs="ad"`)."""
struct AdaptiveSmooth <: AbstractBasisType
    n_penalties::Int
end

AdaptiveSmooth() = AdaptiveSmooth(5)

BASIS_TYPES[:ad] = AdaptiveSmooth()

"""
    _ad_diff_matrix(k, d) -> Matrix{Float64}

Construct the d-th order finite difference matrix D of size (k-d) × k.
Used internally by adaptive smooth construction.
"""
function _ad_diff_matrix(k::Int, d::Int)
    d >= 0 || throw(ArgumentError("penalty order d must be ≥ 0"))
    d < k || throw(ArgumentError("penalty order d=$d must be < k=$k"))

    D = Matrix{Float64}(I, k, k)
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

"""
    _partition_of_unity_weights(n_rows, n_penalties) -> Vector{Vector{Float64}}

Build smooth partition-of-unity weights for `n_rows` difference rows split
into `n_penalties` regions, using a B-spline basis over the row index
(mgcv's `bs="ad"` weights the difference penalty by the columns of a small
B-spline basis; dimension 5 by default). B-splines sum to one on the
interior, giving an exact partition of unity there; a final normalization
guards the (extended-knot) edges.
"""
function _partition_of_unity_weights(n_rows::Int, n_penalties::Int)
    n_penalties >= 1 || throw(ArgumentError("n_penalties must be ≥ 1"))
    if n_penalties == 1 || n_rows == 1
        return [ones(n_rows) for _ in 1:1]
    end

    # Quadratic B-spline basis (order 3) of dimension n_penalties over the
    # row-index range [1, n_rows]; fall back to lower order if n_penalties
    # is very small.
    order = min(3, n_penalties)
    idx = collect(1.0:n_rows)
    knots = _bspline_knot_vector(idx, n_penalties, order - 1)
    W = _bspline_basis(idx, knots, order)   # n_rows × n_penalties

    weights = [W[:, j] for j in 1:n_penalties]

    # Normalize (exact partition of unity, incl. edges)
    for i in 1:n_rows
        total = sum(weights[j][i] for j in 1:n_penalties)
        if total > 0
            for j in 1:n_penalties
                weights[j][i] /= total
            end
        end
    end

    return weights
end

function _smooth_construct(basis::AdaptiveSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Adaptive smooths only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    spline_order = m_order + 2

    # Get n_penalties from xt or from the basis type default
    n_penalties = get(spec.xt, :n_penalties, basis.n_penalties)::Int
    n_penalties >= 1 || throw(ArgumentError("n_penalties must be ≥ 1"))

    # Build P-spline knot vector (same shared builder as PSpline)
    m2 = spline_order - 1
    knot_vec = _bspline_knot_vector(x, k, m2; user_knots = user_knots)

    # B-spline basis (identical to P-spline)
    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)

    # Build the raw difference matrix D: (actual_k - m_order) × actual_k
    D = _ad_diff_matrix(actual_k, m_order)
    n_rows = size(D, 1)

    # Clamp n_penalties to available rows
    n_pen = min(n_penalties, n_rows)

    # Build local penalties via partition-of-unity weighting of D
    pou_weights = _partition_of_unity_weights(n_rows, n_pen)
    penalties = Matrix{Float64}[]
    for j in 1:n_pen
        W_j = Diagonal(pou_weights[j])
        S_j = D' * W_j * D
        # Symmetrize for numerical safety
        S_j = (S_j + S_j') / 2
        push!(penalties, S_j)
    end

    null_dim = m_order
    pen_rank = actual_k - null_dim

    # Absorb identifiability constraints (same as P-spline)
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knot_vec,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

# Prediction matrix is identical to P-spline — the basis doesn't change,
# only the penalties differ.
function _predict_matrix(::AdaptiveSmooth, smooth::ConstructedSmooth, newdata)
    return _predict_matrix(PSpline(), smooth, newdata)
end
