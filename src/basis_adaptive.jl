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
    _adaptive_weight_basis(n_rows, n_penalties) -> Vector{Vector{Float64}}

Build the adaptive penalty weight basis `V`: the columns weight the rows of the
second-difference matrix, so `S_j = D' diag(V[:, j]) D`.

Follows mgcv's `smooth.construct.ad.smooth.spec` exactly
(`R/smooth.r:2467-2477`), which evaluates a **fixed** P-spline basis of
dimension `n_penalties` at `x = 1:(nk-2)/nk`, where `nk = n_rows + 2` is the
smoothing-basis dimension:

  * `n_penalties == 2` → `V = [1  x]` (no spline basis);
  * `n_penalties == 3` → order-3 basis (mgcv overrides to `m = 1`);
  * `n_penalties >= 4` → order-4 basis (`m = 2`, mgcv's `bs="ps"` default).

Note mgcv applies **no** normalization: for `n_penalties >= 3` the B-spline
columns already form an exact partition of unity over the evaluation points,
so `sum_j S_j == D'D`. That identity deliberately does *not* hold for
`n_penalties == 2`, where mgcv's `[1  x]` has row sums in `[1, 2]`.
"""
function _adaptive_weight_basis(n_rows::Int, n_penalties::Int)
    n_penalties >= 1 || throw(ArgumentError("n_penalties must be ≥ 1"))
    if n_penalties == 1 || n_rows == 1
        return [ones(n_rows)]
    end

    # mgcv evaluates the weight basis at 1:(nk-2)/nk, with nk the smoothing
    # basis dimension. The B-spline branches are invariant to an affine change
    # of this grid, but the `n_penalties == 2` branch below is not, so use
    # mgcv's actual values.
    nk = n_rows + 2
    xw = collect(1.0:n_rows) ./ nk

    if n_penalties == 2
        return [ones(n_rows), xw]
    end

    # mgcv: m = 2 (order 4) in general, overridden to m = 1 (order 3) at k == 3.
    order = n_penalties == 3 ? 3 : 4
    knots = _bspline_knot_vector(xw, n_penalties, order - 1)
    W = _bspline_basis(xw, knots, order)   # n_rows × n_penalties

    return [W[:, j] for j in 1:n_penalties]
end

"""
    _adaptive_n_penalties(spec, default) -> Int

Resolve the number of adaptive sub-penalties, following mgcv's convention in
which `m` is the **penalty basis size** (`p.order[1]`, default 5) rather than a
spline order. `xt[:n_penalties]` is retained as an explicit alias and wins if
both are supplied.
"""
function _adaptive_n_penalties(spec::SmoothSpec, default::Int)
    xt_val = get(spec.xt, :n_penalties, nothing)
    if spec.m !== nothing
        m_val = Int(spec.m)
        m_val >= 1 || throw(ArgumentError(
            "For an adaptive smooth, `m` is the number of sub-penalties and must be ≥ 1, got $m_val"))
        if xt_val === nothing
            @warn "For `bs=:ad`, `m` follows mgcv and sets the NUMBER of adaptive " *
                  "sub-penalties (mgcv's `p.order`, default 5) — it is not a spline " *
                  "order. The smoothing basis is always a cubic P-spline with a " *
                  "second-order difference penalty. Use `xt=Dict(:n_penalties => n)` " *
                  "to be explicit." maxlog = 1
            return m_val
        elseif Int(xt_val) != m_val
            throw(ArgumentError(
                "Conflicting adaptive penalty basis size: m=$m_val and " *
                "xt[:n_penalties]=$(Int(xt_val)). Supply only one."))
        end
    end
    return xt_val === nothing ? default : Int(xt_val)
end

function _smooth_construct(basis::AdaptiveSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Adaptive smooths only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)

    # mgcv fixes the smoothing basis for `bs="ad"` at a cubic P-spline with a
    # second-order difference penalty (`pobject$p.order <- c(2,2)`,
    # R/smooth.r:2454) regardless of `m`; `m` selects the penalty basis size.
    m_order = 2
    spline_order = 4

    n_penalties = _adaptive_n_penalties(spec, basis.n_penalties)

    # Build P-spline knot vector (same shared builder as PSpline)
    m2 = spline_order - 1
    knot_vec = _bspline_knot_vector(x, k, m2; user_knots = user_knots)

    # B-spline basis (identical to P-spline)
    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)

    # Build the raw difference matrix D: (actual_k - m_order) × actual_k
    D = _ad_diff_matrix(actual_k, m_order)
    n_rows = size(D, 1)

    # mgcv errors rather than silently shrinking the request (R/smooth.r:2463).
    n_penalties < actual_k - 2 || throw(ArgumentError(
        "penalty basis too large for smoothing basis: requested $n_penalties " *
        "sub-penalties for a basis of dimension $actual_k (need < $(actual_k - 2)). " *
        "Increase `k` or reduce `m`/`xt[:n_penalties]`."))
    n_pen = min(n_penalties, n_rows)

    # Build local penalties by weighting the rows of D (mgcv's V basis)
    pou_weights = _adaptive_weight_basis(n_rows, n_pen)
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
