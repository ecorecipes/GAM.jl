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
    elseif corfun == :matern52
        s = sqrt(5) * d
        return (1 + s + s^2 / 3) * exp(-s)
    elseif corfun == :power_exp
        p = isempty(params) ? 1.5 : params[1]
        return exp(-d^p)
    else
        error("Unknown GP correlation function: $corfun")
    end
end

"""
    _smooth_construct(::GPSmooth, spec, data, user_knots)

Low-rank kriging GP smooth (Kammann & Wand 2003 style; mgcv `bs="gp"` uses
the same low-rank construction). The basis is the cross-correlation matrix
`X = R_xk` between data and knots and the penalty is the knot correlation
matrix `S = R_kk`, so the implied prior covariance of the fitted function
is `R_xk R_kk⁻¹ R_kx ≈ R_xx` (Nystrom) — the proper low-rank GP model.

The correlation function is Matérn 3/2 with the range parameter set to the
data range (so correlation decays over the span of the data), which is the
same order as mgcv's default range choice for `gp`. Override via
`xt = Dict(:scale => ...)`.
"""
function _smooth_construct(::GPSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("GP smooths currently support 1d only"))

    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)

    # Knot locations
    if user_knots !== nothing
        knots = Float64.(user_knots)
    else
        knots = knot_quantiles(x, k)
    end
    nk = length(knots)

    # Correlation function (default Matérn 3/2)
    corfun = :matern32
    params = Float64[]

    # Range parameter: the data range by default (documented above)
    x_range = maximum(x) - minimum(x)
    scale = Float64(get(spec.xt, :scale, x_range > 0 ? x_range : 1.0))

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

    corfun = :matern32
    params = Float64[]

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
