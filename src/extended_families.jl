# Extended families for GAM fitting
#
# Wraps distributions with extra parameter estimation support
# (e.g., NB theta, Tweedie power, Beta precision).
# Kept separate from Distributions.UnivariateDistribution to avoid type confusion.

using SpecialFunctions: digamma, trigamma, logabsgamma, loggamma
using ForwardDiff: ForwardDiff

"""
    ExtendedFamily

Abstract supertype for extended GAM families that either require estimation of
additional parameters beyond the mean (e.g., NB shape, Tweedie power,
Beta precision) or use quasi-likelihood variance functions outside the
standard distribution types.
"""
abstract type ExtendedFamily end

# ============================================================================
# Negative Binomial family
# ============================================================================

"""
    NegBinFamily(; theta=1.0, estimate_theta=true)

Negative Binomial family with estimated shape parameter θ.
Variance function: V(μ) = μ + μ²/θ.
Default link: `LogLink()`.
"""
mutable struct NegBinFamily <: ExtendedFamily
    theta::Float64
    estimate_theta::Bool
end

NegBinFamily(; theta::Real=1.0, estimate_theta::Bool=true) =
    NegBinFamily(Float64(theta), estimate_theta)

# ============================================================================
# Quasi-likelihood families
# ============================================================================

"""
    QuasiPoissonFamily()

Quasi-Poisson family for overdispersed count data.
Variance function: `V(μ) = μ`, with dispersion estimated separately and stored
on the fitted model as `scale`.
Default link: `LogLink()`.
"""
struct QuasiPoissonFamily <: ExtendedFamily end

"""
    QuasiBinomialFamily()

Quasi-binomial family for overdispersed binary or proportion data.
Variance function: `V(μ) = μ(1-μ)`, with dispersion estimated separately and
stored on the fitted model as `scale`.
Default link: `LogitLink()`.
"""
struct QuasiBinomialFamily <: ExtendedFamily end

# ============================================================================
# Tweedie family
# ============================================================================

"""
    TweedieFamily(; p=1.5, estimate_p=false)

Tweedie family with power parameter p ∈ (1, 2).
Variance function: V(μ) = μ^p.
Default link: `LogLink()`.
If `estimate_p=true`, GAM.jl updates `p` with a bounded profile-likelihood step
based on a Tweedie log-density series for `1 < p < 2`, bringing it closer to
mgcv's `tw()` / `ldTweedie` path.
"""
mutable struct TweedieFamily <: ExtendedFamily
    p::Float64
    estimate_p::Bool
end

TweedieFamily(; p::Real=1.5, estimate_p::Bool=false) =
    TweedieFamily(Float64(p), estimate_p)

# ============================================================================
# Beta regression family
# ============================================================================

"""
    BetaFamily(; phi=1.0, estimate_phi=true)

Beta regression family with precision parameter φ > 0.
Response must be in (0,1). Variance: μ(1-μ)/(1+φ).
Default link: `LogitLink()`.
"""
mutable struct BetaFamily <: ExtendedFamily
    phi::Float64
    estimate_phi::Bool
end

BetaFamily(; phi::Real=1.0, estimate_phi::Bool=true) =
    BetaFamily(Float64(phi), estimate_phi)

# ============================================================================
# Scaled t family (robust regression)
# ============================================================================

"""
    ScatFamily(; nu=nothing, sigma=nothing, estimate_theta=true, min_df=3.0)

Scaled t family for robust regression, matching mgcv's `scat()`. The response
is modelled as `y = μ + σ·t_ν`, so that

    V(μ) = σ²ν/(ν - 2)     (constant in μ, for ν > 2)

Both the degrees of freedom `ν` and the scale `σ` are estimated by default.
Heavy tails make the influence of a residual *redescend*: the curvature of the
deviance turns negative once `|y - μ| > σ√ν`, so gross outliers are downweighted
towards zero rather than dominating the fit as they do under `Normal()`.

Following mgcv, `ν` is parameterised as `ν = exp(θ₁) + min_df` (default
`min_df = 3`, which keeps the variance and the Fisher information finite) and
`σ = exp(θ₂)`. Supplying `nu`/`sigma` sets the starting values; passing
`estimate_theta=false` in addition holds them fixed, mirroring mgcv's
`scat(theta = c(nu, sig))`. If `nu ≤ min_df`, `min_df` is reset to `0.9ν` with a
warning, exactly as mgcv does.

Default link: `IdentityLink()` (`LogLink()` and `InverseLink()` also permitted,
as in mgcv).

# Examples
```julia
m = gam(@formula(y ~ s(x)), df; family = ScatFamily())
m.family.nu, m.family.sigma
```

# References
- Wood, S. N., Pya, N. and Säfken, B. (2016). Smoothing parameter and model
  selection for general smooth models. *JASA* 111, 1548–1575.
"""
mutable struct ScatFamily <: ExtendedFamily
    nu::Float64
    sigma::Float64
    estimate_theta::Bool
    min_df::Float64
    # Whether (nu, sigma) have been given data-dependent starting values.
    # mgcv does this in `preinitialize`; see `_preinitialize!`.
    initialized::Bool
end

function ScatFamily(; nu::Union{Real, Nothing} = nothing,
    sigma::Union{Real, Nothing} = nothing,
    estimate_theta::Bool = true,
    min_df::Real = 3.0)

    min_df = Float64(min_df)
    min_df >= 0.0 || throw(ArgumentError("ScatFamily: min_df must be ≥ 0, got $min_df"))

    if nu === nothing && sigma === nothing
        # mgcv's ini.theta = c(-2, -1) placeholder; overwritten by
        # `_preinitialize!` once the response is available.
        return ScatFamily(exp(-2.0) + min_df, exp(-1.0), estimate_theta, min_df, false)
    end
    (nu === nothing || sigma === nothing) &&
        throw(ArgumentError("ScatFamily: supply both `nu` and `sigma`, or neither"))

    ν = Float64(nu)
    σ = Float64(sigma)
    ν > 0.0 || throw(ArgumentError("ScatFamily: nu must be > 0, got $ν"))
    σ > 0.0 || throw(ArgumentError("ScatFamily: sigma must be > 0, got $σ"))
    if ν <= min_df
        min_df = 0.9 * ν
        @warn "ScatFamily: supplied nu is below min_df; min_df reset to $(min_df)"
    end
    return ScatFamily(ν, σ, estimate_theta, min_df, true)
end

# ============================================================================
# Common interface
# ============================================================================

"""Return the default link function for an extended family."""
_default_link(::NegBinFamily) = LogLink()
_default_link(::QuasiPoissonFamily) = LogLink()
_default_link(::QuasiBinomialFamily) = LogitLink()
_default_link(::TweedieFamily) = LogLink()
_default_link(::BetaFamily) = LogitLink()
_default_link(::ScatFamily) = IdentityLink()

"""Whether the family has an extra parameter to estimate."""
_has_extra_param(f::NegBinFamily) = f.estimate_theta
_has_extra_param(::QuasiPoissonFamily) = false
_has_extra_param(::QuasiBinomialFamily) = false
_has_extra_param(f::TweedieFamily) = f.estimate_p
_has_extra_param(f::BetaFamily) = f.estimate_phi
_has_extra_param(f::ScatFamily) = f.estimate_theta

"""Current value of the family's estimated extra parameter (NB θ, Tweedie p,
Beta φ, ELF log σ), for convergence checks of the alternating estimation."""
_extra_param_value(f::NegBinFamily) = f.theta
_extra_param_value(f::TweedieFamily) = f.p
_extra_param_value(f::BetaFamily) = f.phi
# ν is the slower-moving of the two scat parameters (σ is pinned by the
# residual scale within a single `estimate_theta!` call), so it governs the
# alternation convergence test in `outer.jl`.
_extra_param_value(f::ScatFamily) = f.nu
_extra_param_value(f::ExtendedFamily) = hasproperty(f, :theta) ? f.theta : 0.0

"""Whether the family provides Dd derivatives for proper PIRLS working weights."""
_has_Dd(::ExtendedFamily) = false
_has_Dd(::TweedieFamily) = true
# Required, not optional, for scat: V(μ) is constant, so the Fisher fallback
# in `pirls_extended` would reduce the fit to unweighted least squares and
# discard the family's entire robustness.
_has_Dd(::ScatFamily) = true

"""Dd derivatives (override for families that provide them)."""
_family_Dd(f::ExtendedFamily, y, mu, wt; level=0) = error("Dd not implemented for $(typeof(f))")
_family_Dd(f::TweedieFamily, y, mu, wt; level=0) = tweedie_Dd(f, y, mu, wt; level=level)
_family_Dd(f::ScatFamily, y, mu, wt; level=0) = scat_Dd(f, y, mu, wt; level=level)

"""
    tweedie_Dd(f::TweedieFamily, y, mu, wt; level=0)

Level-0 Tweedie deviance derivatives with respect to `μ`. This supplies the
gradient, observed curvature, and expected curvature needed for PIRLS working
responses/weights.
"""
function tweedie_Dd(f::TweedieFamily, y, mu, wt; level::Int=0)
    Dmu = Vector{Float64}(undef, length(y))
    Dmu2 = Vector{Float64}(undef, length(y))
    EDmu2 = Vector{Float64}(undef, length(y))
    p = f.p

    @inbounds for i in eachindex(y, mu, wt)
        yi = Float64(y[i])
        mui = max(Float64(mu[i]), eps())
        wi = Float64(wt[i])

        invmup = inv(mui^p)
        scale = 2.0 * wi * invmup

        Dmu[i] = scale * (mui - yi)
        Dmu2[i] = scale * ((1.0 - p) + p * yi / mui)
        EDmu2[i] = scale
    end

    return Dict{Symbol, Any}(
        :Dmu => Dmu,
        :Dmu2 => Dmu2,
        :EDmu2 => EDmu2,
    )
end

"""Name string for display."""
_family_name(::NegBinFamily) = "NegativeBinomial"
_family_name(::QuasiPoissonFamily) = "QuasiPoisson"
_family_name(::QuasiBinomialFamily) = "QuasiBinomial"
_family_name(::TweedieFamily) = "Tweedie"
_family_name(::BetaFamily) = "Beta"
_family_name(::ScatFamily) = "Scaled t"

# ============================================================================
# Variance functions
# ============================================================================

function _variance(f::NegBinFamily, mu)
    θ = f.theta
    return mu .+ mu .^ 2 ./ θ
end

function _variance(::QuasiPoissonFamily, mu)
    return mu
end

function _variance(::QuasiBinomialFamily, mu)
    return mu .* (1.0 .- mu)
end

function _variance(f::TweedieFamily, mu)
    return mu .^ f.p
end

function _variance(f::BetaFamily, mu)
    return mu .* (1.0 .- mu) ./ (1.0 + f.phi)
end

# Constant in μ: Var(μ + σ·t_ν) = σ²ν/(ν-2). The `zero(mu) +` keeps the method
# usable for both a scalar (via `_variance_scalar`) and a vector of means.
function _variance(f::ScatFamily, mu)
    v = f.sigma^2 * f.nu / max(f.nu - 2.0, eps())
    return @. zero(mu) + v
end

# Scalar variance and dV/dμ for the Fletcher scale estimator
# (mirrors _variance_scalar/_dvariance_scalar_mu for standard families)
_variance_scalar(f::ExtendedFamily, mu::Float64) = float(_variance(f, mu))
_dvariance_scalar_mu(f::NegBinFamily, mu::Float64) = 1.0 + 2.0 * mu / f.theta
_dvariance_scalar_mu(::QuasiPoissonFamily, mu::Float64) = 1.0
_dvariance_scalar_mu(::QuasiBinomialFamily, mu::Float64) = 1.0 - 2.0 * mu
_dvariance_scalar_mu(f::TweedieFamily, mu::Float64) = f.p * mu^(f.p - 1.0)
_dvariance_scalar_mu(f::BetaFamily, mu::Float64) = (1.0 - 2.0 * mu) / (1.0 + f.phi)
# V(μ) = σ²ν/(ν-2) does not depend on μ.
_dvariance_scalar_mu(::ScatFamily, ::Float64) = 0.0

# ============================================================================
# Deviance functions
# ============================================================================

@inline function _poisson_unit_deviance(yi, mui)
    if yi > 0
        return 2.0 * (yi * log(yi / mui) - (yi - mui))
    end
    return 2.0 * mui
end

@inline function _binomial_unit_deviance(yi, mui)
    di = 0.0
    if yi > 0
        di += yi * log(yi / mui)
    end
    if yi < 1
        di += (1 - yi) * log((1 - yi) / (1 - mui))
    end
    return 2.0 * di
end

function _deviance(f::NegBinFamily, y, mu, wt)
    θ = f.theta
    dev = 0.0
    for i in eachindex(y, mu, wt)
        yi = y[i]
        mui = max(mu[i], eps())
        d = 0.0
        if yi > 0
            d += yi * log(yi / mui)
        end
        d -= (yi + θ) * log((yi + θ) / (mui + θ))
        dev += wt[i] * d
    end
    return 2.0 * dev
end

function _deviance(::QuasiPoissonFamily, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        dev += wt[i] * _poisson_unit_deviance(y[i], mui)
    end
    return dev
end

function _deviance(::QuasiBinomialFamily, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        mui = clamp(mu[i], eps(), 1.0 - eps())
        dev += wt[i] * _binomial_unit_deviance(y[i], mui)
    end
    return dev
end

"""
Tweedie unit deviance (without the leading factor 2):
    y^(2-p)/((1-p)(2-p)) - y·μ^(1-p)/(1-p) + μ^(2-p)/(2-p)
with the Poisson (p→1) and Gamma (p→2) limits handled explicitly.
"""
function _tweedie_unit_deviance(p::Real, yi::Real, mui::Real)
    if abs(p - 1.0) < 1e-10
        # Poisson limit
        if yi > 0
            return yi * log(yi / mui) - (yi - mui)
        else
            return mui
        end
    elseif abs(p - 2.0) < 1e-10
        # Gamma limit
        if yi > 0
            return -log(yi / mui) + (yi - mui) / mui
        else
            return -log(eps() / mui) + (eps() - mui) / mui
        end
    else
        t3 = yi > 0 ? yi^(2 - p) / ((1 - p) * (2 - p)) : 0.0
        t2 = yi * mui^(1 - p) / (1 - p)
        t1 = mui^(2 - p) / (2 - p)
        return t3 - t2 + t1
    end
end

function _deviance(f::TweedieFamily, y, mu, wt)
    p = f.p
    dev = 0.0
    for i in eachindex(y, mu, wt)
        dev += wt[i] * _tweedie_unit_deviance(p, y[i], max(mu[i], eps()))
    end
    return 2.0 * dev
end

function _deviance(f::BetaFamily, y, mu, wt)
    φ = f.phi
    dev = 0.0
    for i in eachindex(y, mu, wt)
        yi = clamp(y[i], eps(), 1.0 - eps())
        mui = clamp(mu[i], eps(), 1.0 - eps())
        # Beta deviance based on log-likelihood ratio
        # -2 * [ll(y; y, phi) - ll(y; mu, phi)]
        ll_sat = _beta_loglik_single(yi, yi, φ)
        ll_mod = _beta_loglik_single(yi, mui, φ)
        dev += wt[i] * (-2.0 * (ll_mod - ll_sat))
    end
    return dev
end

# mgcv scat(): dev.resids = wt (ν+1) log1p((1/ν)((y-μ)/σ)²), which is
# 2(ℓ_saturated - ℓ) since the saturated fit μ = y zeroes the log1p term.
function _deviance(f::ScatFamily, y, mu, wt)
    ν = f.nu
    σ = max(f.sigma, eps())
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        r = (Float64(y[i]) - Float64(mu[i])) / σ
        dev += Float64(wt[i]) * (ν + 1.0) * log1p(r^2 / ν)
    end
    return dev
end

function _beta_loglik_single(y, mu, φ)
    a = mu * φ
    b = (1.0 - mu) * φ
    a = max(a, eps())
    b = max(b, eps())
    y = clamp(y, eps(), 1.0 - eps())
    return (a - 1.0) * log(y) + (b - 1.0) * log(1.0 - y) +
           logabsgamma(a + b)[1] - logabsgamma(a)[1] - logabsgamma(b)[1]
end

# ============================================================================
# Deviance residuals
# ============================================================================

function _deviance_residuals(f::NegBinFamily, y, mu, wt)
    θ = f.theta
    r = similar(y)
    for i in eachindex(y, mu, wt)
        yi = y[i]
        mui = max(mu[i], eps())
        d = 0.0
        if yi > 0
            d += yi * log(yi / mui)
        end
        d -= (yi + θ) * log((yi + θ) / (mui + θ))
        r[i] = sign(yi - mui) * sqrt(max(2.0 * wt[i] * d, 0.0))
    end
    return r
end

function _deviance_residuals(::QuasiPoissonFamily, y, mu, wt)
    r = similar(y)
    @inbounds for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        di = wt[i] * _poisson_unit_deviance(y[i], mui)
        r[i] = sign(y[i] - mui) * sqrt(max(di, 0.0))
    end
    return r
end

function _deviance_residuals(::QuasiBinomialFamily, y, mu, wt)
    r = similar(y)
    @inbounds for i in eachindex(y, mu, wt)
        mui = clamp(mu[i], eps(), 1.0 - eps())
        di = wt[i] * _binomial_unit_deviance(y[i], mui)
        r[i] = sign(y[i] - mui) * sqrt(max(di, 0.0))
    end
    return r
end

function _deviance_residuals(f::TweedieFamily, y, mu, wt)
    r = similar(y)
    @inbounds for i in eachindex(y, mu, wt)
        di = 2.0 * wt[i] * _tweedie_unit_deviance(f.p, y[i], max(mu[i], eps()))
        r[i] = sign(y[i] - mu[i]) * sqrt(max(di, 0.0))
    end
    return r
end

function _deviance_residuals(f::BetaFamily, y, mu, wt)
    r = similar(y)
    for i in eachindex(y, mu, wt)
        yi = clamp(y[i], eps(), 1.0 - eps())
        mui = clamp(mu[i], eps(), 1.0 - eps())
        ll_sat = _beta_loglik_single(yi, yi, f.phi)
        ll_mod = _beta_loglik_single(yi, mui, f.phi)
        di = -2.0 * (ll_mod - ll_sat)
        r[i] = sign(yi - mui) * sqrt(max(wt[i] * di, 0.0))
    end
    return r
end

function _deviance_residuals(f::ScatFamily, y, mu, wt)
    ν = f.nu
    σ = max(f.sigma, eps())
    r = similar(y)
    @inbounds for i in eachindex(y, mu, wt)
        resid = Float64(y[i]) - Float64(mu[i])
        di = Float64(wt[i]) * (ν + 1.0) * log1p((resid / σ)^2 / ν)
        r[i] = sign(resid) * sqrt(max(di, 0.0))
    end
    return r
end

# ============================================================================
# Mu clamping
# ============================================================================

function _clamp_mu(::NegBinFamily, mu)
    return max.(mu, eps())
end

function _clamp_mu(::QuasiPoissonFamily, mu)
    return max.(mu, eps())
end

function _clamp_mu(::QuasiBinomialFamily, mu)
    return clamp.(mu, eps(), 1.0 - eps())
end

function _clamp_mu(::TweedieFamily, mu)
    return max.(mu, eps())
end

function _clamp_mu(::BetaFamily, mu)
    return clamp.(mu, eps(), 1.0 - eps())
end

# μ is unrestricted on the real line (mgcv's validmu only checks finiteness).
_clamp_mu(::ScatFamily, mu) = mu

# ============================================================================
# Null deviance
# ============================================================================

function _null_deviance(f::NegBinFamily, y, wt)
    mu = max(mean(y), eps())
    return _deviance(f, y, fill(mu, length(y)), wt)
end

function _null_deviance(f::QuasiPoissonFamily, y, wt)
    mu = max(mean(y), eps())
    return _deviance(f, y, fill(mu, length(y)), wt)
end

function _null_deviance(f::QuasiBinomialFamily, y, wt)
    mu = clamp(mean(y), eps(), 1.0 - eps())
    return _deviance(f, y, fill(mu, length(y)), wt)
end

function _null_deviance(f::TweedieFamily, y, wt)
    mu = max(mean(y), eps())
    return _deviance(f, y, fill(mu, length(y)), wt)
end

function _null_deviance(f::BetaFamily, y, wt)
    mu = clamp(mean(y), eps(), 1.0 - eps())
    return _deviance(f, y, fill(mu, length(y)), wt)
end

# The intercept-only scat fit is an M-estimate of location, not the mean, so the
# null deviance needs an actual 1-D minimisation (mgcv does the same via
# `find.null.dev`, which runs IRLS on the intercept-only model). The deviance is
# not convex in μ for a heavy-tailed t, so a Brent search over the data range is
# used rather than a local Newton step from the mean.
function _null_deviance(f::ScatFamily, y, wt)
    n = length(y)
    lo, hi = extrema(y)
    lo == hi && return _deviance(f, y, fill(float(lo), n), wt)
    mu_buf = Vector{Float64}(undef, n)
    objective(m) = begin
        fill!(mu_buf, Float64(m))
        _deviance(f, y, mu_buf, wt)
    end
    mu_opt, dev_opt = _brent_minimize(objective, float(lo), float(hi);
        tol = 1e-10 * max(hi - lo, 1.0), maxiter = 200)
    # Guard against Brent settling in a local basin worse than the mean.
    dev_mean = objective(mean(y))
    return min(dev_opt, dev_mean)
end

# ============================================================================
# Initialization
# ============================================================================

function _initialize_mu(::NegBinFamily, y)
    return max.(y, eps()) .+ 0.1
end

function _initialize_mu(::QuasiPoissonFamily, y)
    return max.(y, eps()) .+ 0.1
end

function _initialize_mu(::QuasiBinomialFamily, y)
    return clamp.(y, 0.01, 0.99)
end

function _initialize_mu(::TweedieFamily, y)
    return max.(y, eps()) .+ 0.1
end

function _initialize_mu(::BetaFamily, y)
    return clamp.(y, 0.01, 0.99)
end

# mgcv scat(): mustart <- y + (y == 0) * 0.1
_initialize_mu(::ScatFamily, y) = float.(y) .+ (y .== 0) .* 0.1

# ============================================================================
# Theta / extra parameter estimation
# ============================================================================

"""
    estimate_theta!(family::NegBinFamily, y, mu, wt, scale)

Estimate NB shape parameter θ by Newton iteration on the profile
log-likelihood given μ (MASS `theta.ml`-style alternation, not mgcv's
REML-embedded θ estimation). Uses digamma/trigamma.

Called from two places by design: periodically inside the extended-family
P-IRLS loop (every 3 iterations, so θ tracks μ as the fit evolves) and once
per outer smoothing-parameter iteration (see `outer.jl`), so θ settles at the
converged fit. Both call sites update the same `family.theta`.
"""
function estimate_theta!(family::NegBinFamily, y, mu, wt, scale)
    !family.estimate_theta && return

    θ = family.theta
    n = length(y)

    for iter in 1:50
        # Log saturated likelihood derivatives w.r.t. θ (on log(θ) scale)
        # lsth1 = d/dθ [log-likelihood terms involving θ only]
        g1 = 0.0  # gradient
        g2 = 0.0  # Hessian

        for i in eachindex(y, mu, wt)
            yi = y[i]
            mui = max(mu[i], eps())
            wi = wt[i]

            # First derivative of log-likelihood w.r.t. θ
            # ∂ℓ/∂θ = Σ wᵢ [log(θ) - log(yᵢ+θ) + digamma(yᵢ+θ) - digamma(θ) + 1 - θ/(yᵢ+θ) + θ*log(θ/(μᵢ+θ)) + (μᵢ-yᵢ)/(μᵢ+θ)]
            # Simplified gradient on θ scale:
            g1 += wi * (digamma(yi + θ) - digamma(θ) + log(θ) - log(mui + θ) +
                        (mui - yi) / (mui + θ))

            # Second derivative for Hessian: dg1/dθ
            # d/dθ [digamma(y+θ) - digamma(θ) + log(θ) - log(μ+θ) + (μ-y)/(μ+θ)]
            # = trigamma(y+θ) - trigamma(θ) + 1/θ - 2/(μ+θ) + (θ+y)/(μ+θ)²
            # (matches MASS::theta.ml's information matrix)
            g2 += wi * (trigamma(yi + θ) - trigamma(θ) + 1.0 / θ -
                        2.0 / (mui + θ) + (θ + yi) / (mui + θ)^2)
        end

        # (No scale adjustment: the Newton step -g1/g2 is invariant to a
        # common factor, and NB fixes the dispersion at 1 anyway.)

        # Operate on log(θ) for positivity: chain rule
        g1_log = g1 * θ
        g2_log = g2 * θ^2 + g1 * θ

        # Newton step on log(θ)
        if abs(g2_log) < eps()
            break
        end
        step = -g1_log / g2_log

        # Clamp step size
        step = clamp(step, -2.0, 2.0)

        log_θ_new = log(θ) + step
        log_θ_new = clamp(log_θ_new, log(1e-4), log(1e6))
        θ_new = exp(log_θ_new)

        # Convergence check
        if abs(θ_new - θ) / (abs(θ) + 1e-8) < 1e-6
            θ = θ_new
            break
        end
        θ = θ_new
    end

    family.theta = max(θ, 1e-4)
    return nothing
end

"""
    estimate_theta!(family::BetaFamily, y, mu, wt, scale)

Estimate Beta precision φ by Newton iteration on the
log-likelihood. Uses digamma/trigamma.
"""
function estimate_theta!(family::BetaFamily, y, mu, wt, scale)
    !family.estimate_phi && return

    φ = family.phi
    n = length(y)

    for iter in 1:50
        g1 = 0.0
        g2 = 0.0

        for i in eachindex(y, mu, wt)
            yi = clamp(y[i], eps(), 1.0 - eps())
            mui = clamp(mu[i], eps(), 1.0 - eps())
            wi = wt[i]

            a = mui * φ
            b = (1.0 - mui) * φ

            # ∂ℓ/∂φ = Σ wᵢ [μᵢ(log yᵢ - digamma(a)) + (1-μᵢ)(log(1-yᵢ) - digamma(b)) + digamma(φ)]
            # Simplified:
            g1 += wi * (mui * (log(yi) - digamma(a)) +
                        (1.0 - mui) * (log(1.0 - yi) - digamma(b)) +
                        digamma(φ) - log(1.0))  # ∂ lgamma(φ)/∂φ = digamma(φ)

            g2 += wi * (-mui^2 * trigamma(a) -
                        (1.0 - mui)^2 * trigamma(b) +
                        trigamma(φ))
        end

        # Newton step on log(φ)
        g1_log = g1 * φ
        g2_log = g2 * φ^2 + g1 * φ

        if abs(g2_log) < eps()
            break
        end

        # For maximum likelihood, we want to maximize, so step = -g1/g2 for finding zero of gradient
        # But g2 should be negative at maximum, so -g1/g2 > 0 when g1 > 0
        step = -g1_log / g2_log
        step = clamp(step, -2.0, 2.0)

        log_φ_new = log(φ) + step
        log_φ_new = clamp(log_φ_new, log(1e-4), log(1e6))
        φ_new = exp(log_φ_new)

        if abs(φ_new - φ) / (abs(φ) + 1e-8) < 1e-6
            φ = φ_new
            break
        end
        φ = φ_new
    end

    family.phi = max(φ, 1e-4)
    return nothing
end

"""
    estimate_theta!(family::TweedieFamily, y, mu, wt, scale)

Estimate Tweedie power parameter `p` in `(1, 2)` with a bounded
profile-likelihood update based on the Dunn-Smyth Tweedie series used by mgcv.
"""
const _TWEEDIE_P_LOWER = 1.01
const _TWEEDIE_P_UPPER = 1.99
const _TWEEDIE_LOGDENSITY_TOL = eps(Float64)^2
const _TWEEDIE_LOGDENSITY_MAX_TERMS = 1_000_000
const _TWEEDIE_LOGPHI_LOWER = log(1e-8)
const _TWEEDIE_LOGPHI_UPPER = log(1e8)
const _TWEEDIE_LOGPHI_WINDOW = 4.0

function _tweedie_profile_scale(y, mu, wt, p::Float64, fallback_scale::Float64)
    numer = 0.0
    denom = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        v_i = max(mu[i]^p, eps())
        numer += wt[i] * (y[i] - mu[i])^2 / v_i
        denom += wt[i]
    end
    if !(denom > 0.0) || !isfinite(numer)
        return clamp(fallback_scale, 1e-8, 1e8)
    end
    φ = numer / denom
    return clamp(isfinite(φ) ? φ : fallback_scale, 1e-8, 1e8)
end

function _tweedie_logdensity_positive(y::Float64, mu::Float64, p::Float64, phi::Float64;
        tol::Float64=_TWEEDIE_LOGDENSITY_TOL,
        max_terms::Int=_TWEEDIE_LOGDENSITY_MAX_TERMS)
    twop = 2.0 - p
    onep = 1.0 - p
    alpha_pos = twop / (p - 1.0)
    logy = log(y)
    w_base = -alpha_pos * log(p - 1.0) - log(phi) / (p - 1.0) - log(twop)

    j_mode = y^twop / (phi * twop)
    if !isfinite(j_mode) || j_mode > max_terms
        return -Inf
    end

    j_max = floor(Int, j_mode)
    if j_mode - j_max > 0.5 || j_max < 1
        j_max += 1
    end
    if abs(Float64(j_max) - j_mode) > 1.0
        return -Inf
    end

    logw(j::Int, loggamma_j1::Float64) =
        j * w_base - loggamma_j1 - logabsgamma(j * alpha_pos)[1] + j * alpha_pos * logy
    logw(j::Int) = logw(j, logabsgamma(j + 1.0)[1])

    wmax = logw(j_max)
    improved = true
    while improved
        improved = false
        if j_max < max_terms
            w_up = logw(j_max + 1)
            if w_up > wmax
                j_max += 1
                wmax = w_up
                improved = true
                continue
            end
        end
        if j_max > 1
            w_down = logw(j_max - 1)
            if w_down > wmax
                j_max -= 1
                wmax = w_down
                improved = true
            end
        end
    end

    cutoff = wmax + log(tol)
    sum_scaled = 1.0

    loggamma_j1 = logabsgamma(j_max + 1.0)[1]
    converged_up = false
    for j in (j_max + 1):max_terms
        loggamma_j1 += log(Float64(j))
        wj = logw(j, loggamma_j1)
        if wj < cutoff
            converged_up = true
            break
        end
        sum_scaled += exp(wj - wmax)
    end
    if !converged_up && j_max < max_terms
        return -Inf
    end

    loggamma_j1 = logabsgamma(j_max + 1.0)[1]
    for j in (j_max - 1):-1:1
        loggamma_j1 -= log(Float64(j + 1))
        wj = logw(j, loggamma_j1)
        if wj < cutoff
            break
        end
        sum_scaled += exp(wj - wmax)
    end

    if !(sum_scaled > 0.0) || !isfinite(sum_scaled)
        return -Inf
    end

    mu1p = mu^onep
    l_base = mu1p * (y / onep - mu / twop) / phi
    return l_base - logy + wmax + log(sum_scaled)
end

function _tweedie_logdensity(y::Real, mu::Real, p::Float64, phi::Float64;
        tol::Float64=_TWEEDIE_LOGDENSITY_TOL,
        max_terms::Int=_TWEEDIE_LOGDENSITY_MAX_TERMS)
    yi = Float64(y)
    mui = Float64(mu)
    if !(isfinite(yi) && isfinite(mui) && isfinite(p) && isfinite(phi))
        return -Inf
    end
    if yi < 0.0 || !(mui > 0.0) || !(phi > 0.0) || !(1.0 < p < 2.0)
        return -Inf
    end
    if yi == 0.0
        return -mui^(2.0 - p) / (phi * (2.0 - p))
    end
    return _tweedie_logdensity_positive(yi, mui, p, phi; tol=tol, max_terms=max_terms)
end

function _tweedie_total_loglik(y, mu, wt, p::Float64, phi::Float64)
    ll = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        wi = wt[i]
        wi == 0.0 && continue
        lli = _tweedie_logdensity(y[i], mu[i], p, phi)
        if !isfinite(lli)
            return -Inf
        end
        ll += wi * lli
    end
    return ll
end

function _tweedie_profile_loglik(y, mu, wt, p::Float64, fallback_scale::Float64)
    φ_guess = _tweedie_profile_scale(y, mu, wt, p, fallback_scale)
    ll_guess = _tweedie_total_loglik(y, mu, wt, p, φ_guess)
    best_ll = ll_guess
    best_phi = φ_guess

    logphi_guess = log(φ_guess)
    lower = max(_TWEEDIE_LOGPHI_LOWER, logphi_guess - _TWEEDIE_LOGPHI_WINDOW)
    upper = min(_TWEEDIE_LOGPHI_UPPER, logphi_guess + _TWEEDIE_LOGPHI_WINDOW)
    if upper > lower
        objective(logphi) = begin
            ll = _tweedie_total_loglik(y, mu, wt, p, exp(Float64(logphi)))
            return isfinite(ll) ? -ll : Inf
        end
        logphi_opt, obj_opt = _brent_minimize(objective, lower, upper; tol=1e-3, maxiter=25)
        ll_opt = -obj_opt
        if isfinite(ll_opt) && ll_opt > best_ll
            best_ll = ll_opt
            best_phi = exp(logphi_opt)
        end
    end
    return best_ll, best_phi
end

function estimate_theta!(family::TweedieFamily, y, mu, wt, scale)
    !family.estimate_p && return

    μmin, μmax = extrema(mu)
    if !(μmin > 0.0) || !isfinite(μmin) || !isfinite(μmax)
        return
    end
    if log(μmax) - log(μmin) < 1e-6
        return
    end

    p_old = clamp(family.p, _TWEEDIE_P_LOWER, _TWEEDIE_P_UPPER)
    fallback_scale = clamp(isfinite(scale) ? scale : 1.0, 1e-8, 1e8)
    cache = Dict{Float64, Tuple{Float64, Float64}}()
    function profiled_loglik(p::Float64)
        get!(cache, p) do
            _tweedie_profile_loglik(y, mu, wt, p, fallback_scale)
        end
    end
    objective(p) = begin
        ll, _ = profiled_loglik(clamp(Float64(p), _TWEEDIE_P_LOWER, _TWEEDIE_P_UPPER))
        return isfinite(ll) ? -ll : Inf
    end

    ll_old, _ = profiled_loglik(p_old)
    p_opt, obj_opt = _brent_minimize(objective, _TWEEDIE_P_LOWER, _TWEEDIE_P_UPPER;
        tol = 1e-3, maxiter = 30)
    ll_opt = -obj_opt
    if !isfinite(ll_opt)
        return
    end

    p_target = clamp(Float64(p_opt), _TWEEDIE_P_LOWER, _TWEEDIE_P_UPPER)
    if isfinite(ll_old) && ll_opt < ll_old - 1e-6 * max(abs(ll_old), 1.0)
        p_target = p_old
    end

    step = clamp(p_target - p_old, -0.25, 0.25)
    family.p = clamp(p_old + step, _TWEEDIE_P_LOWER, _TWEEDIE_P_UPPER)
    return nothing
end

# ============================================================================
# Scale estimation for extended families
# ============================================================================

"""Whether the family estimates scale (like Gaussian) or has fixed scale=1."""
_estimates_scale(::NegBinFamily) = false
_estimates_scale(::QuasiPoissonFamily) = true
_estimates_scale(::QuasiBinomialFamily) = true
_estimates_scale(::TweedieFamily) = true
_estimates_scale(::BetaFamily) = false
# σ is a family parameter estimated by `estimate_theta!`, so the GLM
# dispersion is fixed at 1 (mgcv reports `Scale est. = 1` for scat).
_estimates_scale(::ScatFamily) = false

# ============================================================================
# Scaled t: deviance derivatives, likelihood and parameter estimation
# ============================================================================

"""
    scat_Dd(f::ScatFamily, y, mu, wt; level=0)

Deviance derivatives for the scaled t family, transcribed from mgcv's
`scat()\$Dd` (mgcv 1.9-4). With `ν = exp(θ₁) + min_df` and `σ = exp(θ₂)`, and
writing `r = y - μ`, `a = 1 + r²/(νσ²)`:

- `level = 0` returns `Dmu`, `Dmu2` (observed curvature) and `EDmu2`
  (expected curvature, `2w(ν+1)/(σ²(ν+3))`);
- `level = 1` additionally returns the mixed derivatives with respect to the
  two internal parameters `θ = (θ₁, θ₂)`: `Dth`, `Dmuth`, `Dmu2th`, `Dmu3`,
  `EDmu3` and `EDmu2th`.

`Dmu2` is *not* positive: it turns negative once `|r| > σ√ν`, which is the
redescending influence that makes the family robust. `pirls_extended` floors
the resulting working weight at `eps()`, so such observations contribute
essentially nothing to the fit — the intended behaviour here, and the reason
the Dd path (rather than the constant-variance Fisher fallback) is mandatory
for this family.

!!! note "Divergence from mgcv"
    mgcv's `scat()` omits the prior weight from `EDmu2` and from the first
    column of `EDmu2th`, while including it everywhere else (and while its own
    `nb()` includes it in `EDmu2`). That looks like an oversight in mgcv; the
    weight is included here, so results agree with mgcv for unit prior weights
    and differ — in GAM.jl's favour — otherwise.
"""
function scat_Dd(f::ScatFamily, y, mu, wt; level::Int = 0)
    level <= 1 || throw(ArgumentError(
        "scat_Dd: only levels 0 and 1 are implemented (got level = $level)"))

    ν = f.nu
    σ = max(f.sigma, eps())
    ν1 = ν + 1.0
    ν2 = ν - f.min_df        # = exp(θ₁), i.e. dν/dθ₁
    ν2ν = ν2 / ν
    σ2 = σ^2
    n = length(y)

    Dmu = Vector{Float64}(undef, n)
    Dmu2 = Vector{Float64}(undef, n)
    EDmu2 = Vector{Float64}(undef, n)
    edmu2_unit = 2.0 * ν1 / σ2 / (ν + 3.0)

    if level == 0
        @inbounds for i in eachindex(y, mu, wt)
            wi = Float64(wt[i])
            ym = Float64(y[i]) - Float64(mu[i])
            a = 1.0 + (ym / σ)^2 / ν
            nusig2a = ν * σ2 * a
            fi = ν1 * ym / nusig2a
            f1 = ym / nusig2a
            Dmu[i] = -2.0 * wi * fi
            Dmu2[i] = 2.0 * wi * ν1 * (1.0 / nusig2a - 2.0 * f1^2)
            EDmu2[i] = wi * edmu2_unit
        end
        return Dict{Symbol, Any}(:Dmu => Dmu, :Dmu2 => Dmu2, :EDmu2 => EDmu2)
    end

    Dmu3 = Vector{Float64}(undef, n)
    EDmu3 = zeros(n)
    Dth = Matrix{Float64}(undef, n, 2)
    Dmuth = Matrix{Float64}(undef, n, 2)
    Dmu2th = Matrix{Float64}(undef, n, 2)
    EDmu2th = Matrix{Float64}(undef, n, 2)
    edmu2th1_unit = 4.0 / (σ2 * (ν + 3.0)^2) * ν2

    @inbounds for i in eachindex(y, mu, wt)
        wi = Float64(wt[i])
        ym = Float64(y[i]) - Float64(mu[i])
        a = 1.0 + (ym / σ)^2 / ν
        sig2a = σ2 * a
        nusig2a = ν * sig2a
        fi = ν1 * ym / nusig2a
        f1 = ym / nusig2a
        fym = fi * ym
        ff1 = fi * f1
        f1ym = f1 * ym
        fymf1 = fym * f1
        ymsig2a = ym / sig2a
        ν1nusig2a = ν1 / nusig2a

        Dmu[i] = -2.0 * wi * fi
        Dmu2[i] = 2.0 * wi * ν1 * (1.0 / nusig2a - 2.0 * f1^2)
        EDmu2[i] = wi * edmu2_unit
        Dmu3[i] = 4.0 * wi * fi * (3.0 / nusig2a - 4.0 * f1^2)

        Dth[i, 1] = wi * ν2 * (log(a) - fym / ν)
        Dth[i, 2] = -2.0 * wi * fym
        Dmuth[i, 1] = 2.0 * wi * (fi - ymsig2a - fymf1) * ν2ν
        Dmuth[i, 2] = 4.0 * wi * fi * (1.0 - f1ym)
        Dmu2th[i, 1] = 2.0 * wi * (-ν1nusig2a + 1.0 / sig2a + 5.0 * ff1 -
                                   2.0 * f1ym / sig2a - 4.0 * fymf1 * f1) * ν2ν
        Dmu2th[i, 2] = 4.0 * wi * (-ν1nusig2a + 5.0 * ff1 - 4.0 * ff1 * f1ym)
        EDmu2th[i, 1] = wi * edmu2th1_unit
        EDmu2th[i, 2] = -2.0 * EDmu2[i]
    end

    return Dict{Symbol, Any}(
        :Dmu => Dmu, :Dmu2 => Dmu2, :EDmu2 => EDmu2, :Dmu3 => Dmu3,
        :EDmu3 => EDmu3, :Dth => Dth, :Dmuth => Dmuth, :Dmu2th => Dmu2th,
        :EDmu2th => EDmu2th,
    )
end

"""
    _scat_loglik(ν, σ, y, mu, wt)

Log-likelihood of the scaled t family. Kept generic in `ν` and `σ` so that
`ForwardDiff` can differentiate it in `estimate_theta!`. Equals `-aic/2` from
mgcv's `scat()\$aic`.
"""
function _scat_loglik(ν, σ, y, mu, wt)
    const_term = loggamma((ν + 1) / 2) - loggamma(ν / 2) - log(σ) - log(π * ν) / 2
    ll = zero(const_term)
    half_ν1 = (ν + 1) / 2
    @inbounds for i in eachindex(y, mu, wt)
        r = (Float64(y[i]) - Float64(mu[i])) / σ
        ll += Float64(wt[i]) * (const_term - half_ν1 * log1p(r^2 / ν))
    end
    return ll
end

"""Log-likelihood used for AIC; see [`_scat_loglik`](@ref)."""
_family_loglik(f::ScatFamily, y, mu, wt) = _scat_loglik(f.nu, f.sigma, y, mu, wt)

# Bounds on the internal parameters θ = (log(ν - min_df), log σ). The upper ν
# bound corresponds to ν ≈ 1e6, which is numerically indistinguishable from the
# Gaussian limit; mgcv likewise reports ν as Inf beyond exp(999).
const _SCAT_LOGNU_LOWER = log(1e-6)
const _SCAT_LOGNU_UPPER = log(1e6)
const _SCAT_LOGSIG_LOWER = log(1e-10)
const _SCAT_LOGSIG_UPPER = log(1e10)

"""
    estimate_theta!(family::ScatFamily, y, mu, wt, scale)

Estimate the scaled t degrees of freedom `ν` and scale `σ` by maximising the
conditional log-likelihood given the current `mu`, using safeguarded Newton
steps on `θ = (log(ν - min_df), log σ)`.

This follows the same alternating profile-likelihood scheme as
[`estimate_theta!(::NegBinFamily, ...)`](@ref) rather than mgcv's
REML-embedded θ estimation: `pirls_extended` calls it periodically as the fit
evolves and once more at convergence, and `outer.jl` alternates it with the
smoothing-parameter update. The two-dimensional gradient and Hessian are taken
by `ForwardDiff` on [`_scat_loglik`](@ref); at two parameters this costs
essentially nothing per call and removes the largest transcription risk in the
family.

`scale` is unused — the scaled t carries its own dispersion in `σ`, so the GLM
scale is fixed at 1 (`_estimates_scale(::ScatFamily) = false`).
"""
function estimate_theta!(family::ScatFamily, y, mu, wt, scale)
    family.estimate_theta || return nothing

    min_df = family.min_df
    nll(θ) = -_scat_loglik(exp(θ[1]) + min_df, exp(θ[2]), y, mu, wt)

    θ = [clamp(log(max(family.nu - min_df, 1e-8)), _SCAT_LOGNU_LOWER, _SCAT_LOGNU_UPPER),
         clamp(log(max(family.sigma, 1e-12)), _SCAT_LOGSIG_LOWER, _SCAT_LOGSIG_UPPER)]
    fval = nll(θ)
    isfinite(fval) || return nothing

    θ_trial = similar(θ)
    for _ in 1:50
        g = ForwardDiff.gradient(nll, θ)
        all(isfinite, g) || break
        maximum(abs, g) < 1e-10 && break

        H = ForwardDiff.hessian(nll, θ)
        # Ridge the Hessian until it is positive definite, so the step is a
        # descent direction even where the profile likelihood is flat in ν
        # (which it is whenever the data are effectively Gaussian).
        δ = nothing
        base = max(1e-8 * maximum(abs, H), 1e-10)
        for attempt in 0:10
            Ht = attempt == 0 ? H : H + (base * 10.0^(attempt - 1)) * I
            F = try
                cholesky(Symmetric(Ht))
            catch
                nothing
            end
            if F !== nothing
                δ = F \ g
                break
            end
        end
        δ === nothing && break

        step = 1.0
        improved = false
        for _halve in 1:30
            θ_trial[1] = clamp(θ[1] - step * δ[1], _SCAT_LOGNU_LOWER, _SCAT_LOGNU_UPPER)
            θ_trial[2] = clamp(θ[2] - step * δ[2], _SCAT_LOGSIG_LOWER, _SCAT_LOGSIG_UPPER)
            f_trial = nll(θ_trial)
            if isfinite(f_trial) && f_trial < fval
                improved = true
                break
            end
            step *= 0.5
        end
        improved || break

        moved = max(abs(θ_trial[1] - θ[1]), abs(θ_trial[2] - θ[2]))
        copyto!(θ, θ_trial)
        fval = nll(θ)
        moved < 1e-10 && break
    end

    family.nu = exp(θ[1]) + min_df
    family.sigma = exp(θ[2])
    return nothing
end

"""
    _preinitialize!(family, y)

Give a family's extra parameters data-dependent starting values, mirroring
mgcv's `preinitialize` hook. A no-op for every family except the scaled t,
whose scale must start near the spread of the response for P-IRLS to make
progress. Runs once per family object: warm-started refits keep the values
carried over from the previous outer iteration.
"""
_preinitialize!(::ExtendedFamily, y) = nothing

function _preinitialize!(f::ScatFamily, y)
    f.initialized && return nothing
    # mgcv scat(): Theta <- c(1.5, log(0.8 * sd(y)))
    sd_y = length(y) > 1 ? std(y) : 1.0
    isfinite(sd_y) && sd_y > 0 || (sd_y = 1.0)
    f.nu = exp(1.5) + f.min_df
    f.sigma = 0.8 * sd_y
    f.initialized = true
    return nothing
end

"""
    _use_expected_information(family) -> Bool

Whether `pirls_extended` should form the *final* working weights (those that
determine the EDF, hat values and covariance matrix) from the expected
curvature `EDmu2` rather than the observed curvature `Dmu2`.

`true` by default. It is `false` for the scaled t, where `EDmu2` is constant in
`μ`: using it would give the fit the effective degrees of freedom and standard
errors of an ordinary least-squares fit, discarding precisely the outlier
downweighting the family exists to provide. mgcv likewise reports scat EDFs
from the Newton weights.
"""
_use_expected_information(::ExtendedFamily) = true
_use_expected_information(::ScatFamily) = false
