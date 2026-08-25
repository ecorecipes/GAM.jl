# Gaussian process smooth — bs="gp"
#
# A direct port of mgcv's Kammann & Wand (2003) Matérn spline and the other
# GP correlation families it supports (`R/smooth.r:3404-3595`).
#
# The construction, following `smooth.construct.gp.smooth.spec`:
#
#   1. knots = the unique covariate values, capped at `max.knots` (2000)
#   2. covariates and knots are centred by `shift = colMeans(x)`
#   3. `E = gpE(knt, knt, defn)` fixes the range `rho` (the largest pairwise
#      knot distance, unless supplied) and the correlation type
#   4. the leading `k - M` eigenpairs of `E` give `UZ`, and the penalty is
#      `diag([eigenvalues; zeros(M)])`
#   5. the model matrix is `[gpE(x, knt) * UZ | gpT(x)]`, where `gpT` is the
#      UNPENALIZED null space: `[1, x]`, or `[1]` for the stationary variant
#
# Step 5's null space is the substantive modelling point: mgcv shrinks a `gp`
# smooth toward a straight line, not toward zero.

"""Gaussian process smooth basis (mgcv `bs="gp"`)."""
struct GPSmooth <: AbstractBasisType end

# Register
BASIS_TYPES[:gp] = GPSmooth()

"""
Prediction cache for GP smooths.

Everything the correlation function depends on is resolved ONCE here at
construction and re-read at prediction: the centring `shift`, the centred
`knots`, the eigenvector block `UZ`, and the fully resolved correlation
definition (`gptype`, `rho`, `kappa`, `stationary`). Prediction must never
re-derive any of these from `spec.xt`, because a basis built with one
correlation function and predicted with another disagrees silently.
"""
struct GPPredictCache <: AbstractSmoothPredictCache
    shift::Float64
    knots::Vector{Float64}      # centred knot locations
    UZ::Matrix{Float64}         # nk × (k - M) eigenvectors of E
    gptype::Int                 # mgcv `gpE` type, 1..5
    rho::Float64                # range parameter
    kappa::Float64              # κ, used by the power-exponential type
    stationary::Bool            # true ⇒ null space is the intercept alone
    corfun::Symbol              # :mgcv, or a legacy named correlation
    params::Vector{Float64}     # legacy `xt[:params]`
end

"""
    _gp_cor_mgcv(e, gptype, kappa)

mgcv's `gpE` correlation functions (`R/smooth.r:3429-3436`), evaluated at the
already-scaled distance `e = d / rho`:

| type | correlation | |
|------|-------------|--|
| 1 | `(1 - 1.5e + 0.5e³)·1(e ≤ 1)` | spherical |
| 2 | `exp(-e^κ)` | power exponential |
| 3 | `(1 + e)·exp(-e)` | Matérn κ = 1.5 (mgcv's default) |
| 4 | `exp(-e) + e·exp(-e)·(1 + e/3)` | Matérn κ = 2.5 |
| 5 | `exp(-e) + e·exp(-e)·(1 + 0.4e + e²/15)` | Matérn κ = 3.5 |

Type 3 is *not* the textbook Matérn 3/2: mgcv omits the `√3` that the standard
parameterization carries, so at a given nominal range its correlation decays
`√3` times more slowly. That is mgcv's convention and this is a port of it.
"""
function _gp_cor_mgcv(e::Float64, gptype::Int, kappa::Float64)
    if gptype == 1
        return e <= 1.0 ? (1.0 - 1.5e + 0.5 * e^3) : 0.0
    elseif gptype == 2
        return exp(-e^kappa)
    elseif gptype == 3
        return (1.0 + e) * exp(-e)
    elseif gptype == 4
        ee = exp(-e)
        return ee + (e * ee) * (1.0 + e / 3.0)
    elseif gptype == 5
        ee = exp(-e)
        return ee + (e * ee) * (1.0 + 0.4e + e^2 / 15.0)
    else
        throw(ArgumentError(
            "GP correlation type $gptype is not one of mgcv's 1..5 " *
            "(1 spherical, 2 power exponential, 3/4/5 Matérn κ = 1.5/2.5/3.5)"))
    end
end

"""
    _gp_correlation(d, corfun, params)

Correlation for the *legacy* named functions, kept so that
`xt = Dict(:corfun => :matern32)` and friends keep working. `d` is the
distance already divided by the range. mgcv's own families are reached
through [`_gp_cor_mgcv`](@ref) and the `m` argument instead.
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
            ":matern32, :mgcv_m32, :matern52, :exponential, :gaussian/:sqexp, " *
            ":power_exp. mgcv's own families are selected with `m` instead " *
            "(see the `bs=:gp` docstring)."))
    end
end

# Dispatch a single scaled distance through whichever correlation the cache
# resolved at construction time.
@inline function _gp_eval(e::Float64, gptype::Int, kappa::Float64,
                          corfun::Symbol, params::Vector{Float64})
    return corfun === :mgcv ? _gp_cor_mgcv(e, gptype, kappa) :
           _gp_correlation(e, corfun, params)
end

"""
    _gp_E(xs, knots, rho, gptype, kappa, corfun, params) -> Matrix

mgcv's `gpE` (`R/smooth.r:3410-3439`): the `length(xs) × length(knots)`
matrix of correlations between evaluation points and knots, with distances
divided by `rho`.
"""
function _gp_E(xs::AbstractVector{Float64}, knots::Vector{Float64},
               rho::Float64, gptype::Int, kappa::Float64,
               corfun::Symbol, params::Vector{Float64})
    nk = length(knots)
    E = Matrix{Float64}(undef, length(xs), nk)
    @inbounds for j in 1:nk
        kj = knots[j]
        for i in eachindex(xs)
            E[i, j] = _gp_eval(abs(xs[i] - kj) / rho, gptype, kappa, corfun, params)
        end
    end
    return E
end

"""
    _gp_T(xs, stationary) -> Matrix

mgcv's `gpT` (`R/smooth.r:3405-3408`): the UNPENALIZED null-space block,
`[1, x]` in general and `[1]` for the stationary variant (`m < 0`). `xs` must
already be centred.
"""
function _gp_T(xs::AbstractVector{Float64}, stationary::Bool)
    n = length(xs)
    stationary && return ones(n, 1)
    T = Matrix{Float64}(undef, n, 2)
    @inbounds for i in 1:n
        T[i, 1] = 1.0
        T[i, 2] = xs[i]
    end
    return T
end

# Build [E(x, knt) * UZ | T(x)] in chunks of `nk` rows, as mgcv's
# `Predict.matrix.gp.smooth` does (R/smooth.r:3565-3590), so peak memory is
# O(nk²) rather than O(n·nk) for large n.
function _gp_model_matrix(x_raw::Vector{Float64}, c::GPPredictCache)
    # mgcv centres the covariates by the fit-time `shift` before evaluating
    # gpE/gpT, at both construction and prediction (R/smooth.r:3512, 3563).
    # Doing it here rather than at the call sites means prediction cannot
    # forget it.
    xs = x_raw .- c.shift
    n = length(xs)
    nk = length(c.knots)
    kE = size(c.UZ, 2)
    ncol = kE + (c.stationary ? 1 : 2)
    X = Matrix{Float64}(undef, n, ncol)
    step = max(nk, 1)
    lo = 1
    while lo <= n
        hi = min(lo + step - 1, n)
        rows = lo:hi
        xv = @view xs[rows]
        E = _gp_E(xv, c.knots, c.rho, c.gptype, c.kappa, c.corfun, c.params)
        @views mul!(X[rows, 1:kE], E, c.UZ)
        @views X[rows, (kE + 1):ncol] .= _gp_T(xv, c.stationary)
        lo = hi + 1
    end
    return X
end

"""
    _smooth_construct(::GPSmooth, spec, data, user_knots)

Gaussian process / Kammann & Wand (2003) Matérn smooth — a port of mgcv's
`bs="gp"` (`R/smooth.r:3441-3552`), matching it in correlation function,
range parameter, knot selection and null space.

`m` selects mgcv's correlation type, defaulting to **3** as mgcv does:

| `m` | correlation |
|-----|-------------|
| 1 | spherical |
| 2 | power exponential, `exp(-e^κ)` |
| 3 | Matérn κ = 1.5 — `(1 + e)exp(-e)` (default) |
| 4 | Matérn κ = 2.5 |
| 5 | Matérn κ = 3.5 |

A **negative** `m` selects mgcv's *stationary* variant, whose null space is
the intercept alone rather than `[1, x]`.

`xt` options: `:rho` (range; default the largest pairwise knot distance, as
mgcv does), `:kappa` (κ for `m = 2`; default 1), and `:max_knots` (default
2000, mirroring `xt = list(max.knots = )`).

The unpenalized null space `[1, x]` means a `gp` smooth shrinks toward a
straight line, not toward zero.

Legacy correlation functions remain available through
`xt = Dict(:corfun => :matern32)` — `:matern32`, `:matern52`, `:exponential`,
`:gaussian`/`:sqexp`, `:power_exp` and `:mgcv_m32` — with `:scale` as an alias
for `:rho`. These are **not** mgcv-compatible and are kept only so existing
code keeps working; `:corfun` also forces `null_dim` to match `m`'s sign as
above.

!!! note "Behaviour change"
    Before this became a port, `bs=:gp` defaulted to a textbook Matérn 3/2
    over quantile knots with a Nyström cross-correlation basis, a range taken
    from the data span, and no null space. Fits therefore differ from earlier
    versions of GAM.jl; they now agree with mgcv.
"""
function _smooth_construct(::GPSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("GP smooths currently support 1d only"))

    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    # mgcv: "A term has fewer unique covariate combinations than specified
    # maximum degrees of freedom" (R/smooth.r:3487-3488).
    xu = sort(unique(x))
    nu = length(xu)
    if user_knots === nothing
        nu >= spec.k || throw(ArgumentError(
            "s($var) has fewer unique covariate combinations ($nu) " *
            "than the basis dimension k=$(spec.k); reduce k (mgcv raises the " *
            "same error)"))
    end

    # --- correlation definition, resolved once ------------------------------
    corfun = Symbol(get(spec.xt, :corfun, :mgcv))::Symbol
    params = Vector{Float64}(get(spec.xt, :params, Float64[]))
    m_raw = spec.m === nothing ? 3 : spec.m
    stationary = m_raw < 0
    gptype = abs(m_raw)
    kappa = Float64(get(spec.xt, :kappa, 1.0))::Float64
    if corfun === :mgcv
        (1 <= gptype <= 5) || throw(ArgumentError(
            "s($var, bs=:gp, m=$m_raw): |m| must be 1..5 (mgcv's gpE types); " *
            "a negative m selects the stationary variant"))
        (0 < kappa <= 2) || throw(ArgumentError(
            "s($var, bs=:gp): xt[:kappa] must satisfy 0 < kappa <= 2 " *
            "(mgcv: \"incorrect arguments to GP smoother\")"))
    end

    # --- knots: unique covariate values, capped at max.knots ---------------
    max_knots = Int(get(spec.xt, :max_knots, 2000))::Int
    if user_knots !== nothing
        knots = sort(Float64.(user_knots))
    elseif n > max_knots && nu > max_knots
        # mgcv samples `max.knots` rows of the unique values under a fixed
        # seed. Reproducing R's RNG stream is impractical, so we take a
        # deterministic evenly spaced subsample of the sorted unique values
        # instead; this is the one place `bs=:gp` does not reproduce mgcv
        # exactly, and it only engages above `max_knots` unique values.
        idx = round.(Int, range(1, nu; length = max_knots))
        knots = xu[unique(idx)]
    else
        knots = xu
    end
    nk = length(knots)

    # --- centring (mgcv R/smooth.r:3511-3513) -------------------------------
    shift = sum(x) / n
    kc = knots .- shift   # `_gp_model_matrix` centres the covariates itself

    # --- range parameter ----------------------------------------------------
    # mgcv: `if (rho <= 0) rho <- max(E)` where E holds the knot-to-knot
    # distances (R/smooth.r:3425-3426, 3516) — so the range is the largest
    # pairwise distance among the KNOTS, not the span of the data.
    rho_opt = get(spec.xt, :rho, get(spec.xt, :scale, nothing))
    rho = if rho_opt === nothing
        r = nk > 1 ? (maximum(kc) - minimum(kc)) : 0.0
        r > 0 ? r : 1.0
    else
        Float64(rho_opt)::Float64
    end
    rho > 0 || throw(ArgumentError("s($var, bs=:gp): the range xt[:rho] must be positive"))

    # --- basis dimension and null space ------------------------------------
    null_dim = stationary ? 1 : 2          # gpT gives [1] or [1, x]
    bs_dim = spec.k
    if bs_dim < null_dim + 1
        throw(ArgumentError(
            "s($var, bs=:gp): k=$bs_dim is below the minimum $(null_dim + 1) " *
            "for this null space (mgcv resets to ncol(knt)+2 with a warning)"))
    end
    kE = bs_dim - null_dim                 # penalized (eigen) columns

    # --- eigen-reduction of the knot correlation matrix --------------------
    E_kk = _gp_E(kc, kc, rho, gptype, kappa, corfun, params)
    Esym = Symmetric((E_kk .+ E_kk') ./ 2)

    # Penalty: diag([eigenvalues; zeros(null_dim)]) (R/smooth.r:3532-3539).
    S = zeros(bs_dim, bs_dim)
    local UZ::Matrix{Float64}
    if kE < nk
        # mgcv: `slanczos(E, k, -1)` — the k eigenpairs of largest ABSOLUTE
        # eigenvalue, emitted in descending signed order. `_tprs_top_eigen`
        # implements exactly that convention.
        evals, UZ = _tprs_top_eigen(Esym, kE)
        @inbounds for i in 1:kE
            S[i, i] = evals[i]
        end
    else
        # mgcv's `else` branch: no point eigen-reducing, so U is the identity
        # and the penalty is E itself. That is only dimensionally consistent
        # when the penalized block is exactly the knot count.
        kE == nk || throw(ArgumentError(
            "s($var, bs=:gp): k=$bs_dim needs $kE penalized columns but only " *
            "$nk knots are available; reduce k or supply more knots"))
        UZ = Matrix{Float64}(I, nk, nk)
        S[1:nk, 1:nk] .= Esym
    end

    cache = GPPredictCache(shift, kc, UZ, gptype, rho, kappa,
                           stationary, corfun, params)
    X = _gp_model_matrix(x, cache)

    penalties = Matrix{Float64}[Matrix(Symmetric(S))]
    pen_rank = kE

    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knots,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = cache,
    )
end

function _predict_matrix(::GPSmooth, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))

    cache = smooth.predict_cache
    cache isa GPPredictCache || throw(ArgumentError(
        "GP smooth $(smooth.spec.label) has no prediction cache; the basis " *
        "cannot be rebuilt without the correlation definition fixed at fit time"))

    X_new = _gp_model_matrix(x_new, cache)

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X_new, 2))
        return X_new * Z
    end
    return X_new
end
