# Gaussian process smooth — bs="gp"
#
# Implements GP smooths with several correlation functions.
# The model matrix is the correlation matrix evaluated at data/knot locations,
# and the penalty is the precision (inverse correlation) matrix.

"""Gaussian process smooth basis (mgcv `bs="gp"`)."""
struct GPSmooth <: AbstractBasisType end

# Register
BASIS_TYPES[:gp] = GPSmooth()

"""
Prediction cache for GP smooths: stores the length-scale used at fit time
so prediction uses the identical correlation function. (At fit the scale is
derived from the data range; the knot range differs because quantile knots
exclude the extremes.)
"""
struct GPPredictCache <: AbstractSmoothPredictCache
    scale::Float64
end

"""
    _gp_correlation(d, corfun, params)

Compute GP correlation for distance `d` given correlation function type.
"""
function _gp_correlation(d::Float64, corfun::Symbol, params::Vector{Float64})
    if corfun == :exponential
        return exp(-d)
    elseif corfun == :gaussian || corfun == :sqexp
        return exp(-d^2)
    elseif corfun == :matern32
        s = sqrt(3) * d
        return (1 + s) * exp(-s)
    elseif corfun == :mgcv_m32
        # mgcv's default `gp` correlation (gpE type 3): (1 + E)·exp(-E) with
        # E = distance/rho. This is Matérn 3/2 with length-scale rho/√3 — mgcv
        # omits the √3 that the standard parameterization (:matern32 above)
        # carries, so at the same nominal range its correlation decays √3 times
        # more slowly. Kept as a distinct name rather than folded into
        # :matern32 so both parameterizations stay available and explicit.
        return (1 + d) * exp(-d)
    elseif corfun == :matern52
        s = sqrt(5) * d
        return (1 + s + s^2 / 3) * exp(-s)
    elseif corfun == :power_exp
        p = isempty(params) ? 1.5 : params[1]
        return exp(-d^p)
    else
        throw(ArgumentError(
            "Unknown GP correlation function :$corfun; valid options are " *
            ":matern32 (default), :matern52, :exponential, :gaussian/:sqexp, " *
            ":power_exp"))
    end
end

"""
    _smooth_construct(::GPSmooth, spec, data, user_knots)

Low-rank kriging GP smooth (Kammann & Wand 2003 style; mgcv `bs="gp"` uses
the same low-rank construction). The basis is the cross-correlation matrix
`X = R_xk` between data and knots and the penalty is the knot correlation
matrix `S = R_kk`, so the implied prior covariance of the fitted function
is `R_xk R_kk⁻¹ R_kx ≈ R_xx` (Nystrom) — the proper low-rank GP model.

The default correlation function is Matérn 3/2, `(1 + √3 d)·exp(-√3 d)` with
`d = distance/range`, and the range is the span of the data.

Override both via `xt`: `xt = Dict(:corfun => :mgcv_m32, :scale => 2.0)`.
Valid `:corfun` values are `:matern32` (default), `:mgcv_m32`, `:matern52`,
`:exponential`, `:gaussian`/`:sqexp` and `:power_exp`.

!!! note "This is not mgcv's `bs="gp"`"
    mgcv's default `gp` correlation (`gpE` type 3) is `(1 + E)·exp(-E)` with
    `E = distance/rho` and `rho` the largest pairwise distance — Matérn 3/2
    with length-scale `rho/√3`, i.e. mgcv omits the `√3` this basis carries.
    `:corfun => :mgcv_m32` reproduces that correlation function exactly, but
    it does **not** make the fits agree, because the low-rank construction
    differs more fundamentally: mgcv eigen-reduces the correlation matrix by
    Lanczos (`slanczos`, as it does for `bs="tp"`), whereas this basis is the
    Kammann & Wand Nyström cross-correlation. Measured on `s(x, k=10)` over
    200 points of `sin(2πx) + N(0, 0.3²)`, edf is 7.98 here against mgcv's
    7.53; switching to `:mgcv_m32` moves it only to 7.92. Treat `bs=:gp` as a
    GP smoother in its own right, not as a port of mgcv's.
"""
function _smooth_construct(::GPSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("GP smooths currently support 1d only"))

    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    # A knot-based basis cannot carry more basis functions than there are
    # distinct covariate values; mgcv raises "A term has fewer unique
    # covariate combinations than specified maximum degrees of freedom"
    # rather than silently shrinking the basis.
    if user_knots === nothing
        n_unique = length(unique(x))
        n_unique >= spec.k || throw(ArgumentError(
            "s($var) has fewer unique covariate combinations ($n_unique) " *
            "than the basis dimension k=$(spec.k); reduce k (mgcv raises the " *
            "same error)"))
    end

    k = min(spec.k, n)

    # Knot locations
    if user_knots !== nothing
        knots = Float64.(user_knots)
    else
        knots = knot_quantiles(x, k)
    end
    nk = length(knots)

    # Correlation function. The default stays :matern32 — see the docstring for
    # why matching mgcv's default correlation alone does NOT buy parity here.
    corfun = Symbol(get(spec.xt, :corfun, :matern32))::Symbol
    params = Vector{Float64}(get(spec.xt, :params, Float64[]))

    # Range parameter: the data range by default (documented above)
    x_range = maximum(x) - minimum(x)
    # The ::Float64 assertion is load-bearing: spec.xt is Dict{Symbol,Any},
    # so get(...) infers Any and Float64(::Any) is not statically resolvable.
    # Without it `scale` is Any and the divisions in the n×nk loops below
    # become runtime dispatches, boxing every element.
    scale = Float64(get(spec.xt, :scale, x_range > 0 ? x_range : 1.0))::Float64

    # Correlation matrix at knot locations = the penalty. The small nugget
    # keeps it positive definite.
    R_kk = zeros(nk, nk)
    for i in 1:nk, j in 1:nk
        d = abs(knots[i] - knots[j]) / scale
        R_kk[i, j] = _gp_correlation(d, corfun, params)
    end
    R_kk += 1e-8 * I

    # Model matrix: cross-correlation between data and knots
    X = zeros(n, nk)
    for i in 1:n, j in 1:nk
        d = abs(x[i] - knots[j]) / scale
        X[i, j] = _gp_correlation(d, corfun, params)
    end

    # Penalty: S = R_kk (positive definite → no null space)
    S = Matrix(Symmetric(R_kk))

    penalties = Matrix{Float64}[S]
    null_dim = 0   # S is strictly positive definite
    pen_rank = nk

    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knots,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = GPPredictCache(scale),
    )
end

function _predict_matrix(::GPSmooth, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    knots = smooth.knots
    nk = length(knots)

    # Use the same length-scale as at fit time (stored in the predict cache).
    cache = smooth.predict_cache
    scale = if cache isa GPPredictCache
        cache.scale
    else
        # Fallback for smooths constructed without a cache
        maximum(knots) - minimum(knots)
    end

    # Must match the construction-time choice, or prediction silently uses a
    # different correlation function from the one the basis was built with.
    corfun = Symbol(get(smooth.spec.xt, :corfun, :matern32))::Symbol
    params = Vector{Float64}(get(smooth.spec.xt, :params, Float64[]))

    # Cross-correlation IS the basis — no factorization needed at predict
    X_new = zeros(length(x_new), nk)
    for i in eachindex(x_new), j in 1:nk
        d = abs(x_new[i] - knots[j]) / scale
        X_new[i, j] = _gp_correlation(d, corfun, params)
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X_new, 2))
        return X_new * Z
    end
    return X_new
end
