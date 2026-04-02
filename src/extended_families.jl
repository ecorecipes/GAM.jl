# Extended families for GAM fitting
#
# Wraps distributions with extra parameter estimation support
# (e.g., NB theta, Tweedie power, Beta precision).
# Kept separate from Distributions.UnivariateDistribution to avoid type confusion.

using SpecialFunctions: digamma, trigamma, logabsgamma

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
# Common interface
# ============================================================================

"""Return the default link function for an extended family."""
_default_link(::NegBinFamily) = LogLink()
_default_link(::QuasiPoissonFamily) = LogLink()
_default_link(::QuasiBinomialFamily) = LogitLink()
_default_link(::TweedieFamily) = LogLink()
_default_link(::BetaFamily) = LogitLink()

"""Whether the family has an extra parameter to estimate."""
_has_extra_param(f::NegBinFamily) = f.estimate_theta
_has_extra_param(::QuasiPoissonFamily) = false
_has_extra_param(::QuasiBinomialFamily) = false
_has_extra_param(f::TweedieFamily) = f.estimate_p
_has_extra_param(f::BetaFamily) = f.estimate_phi

"""Whether the family provides Dd derivatives for proper PIRLS working weights."""
_has_Dd(::ExtendedFamily) = false
_has_Dd(::TweedieFamily) = true

"""Dd derivatives (override for families that provide them)."""
_family_Dd(f::ExtendedFamily, y, mu, wt; level=0) = error("Dd not implemented for $(typeof(f))")
_family_Dd(f::TweedieFamily, y, mu, wt; level=0) = tweedie_Dd(f, y, mu, wt; level=level)

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

function _deviance(f::TweedieFamily, y, mu, wt)
    p = f.p
    dev = 0.0
    for i in eachindex(y, mu, wt)
        yi = y[i]
        mui = max(mu[i], eps())
        # Tweedie unit deviance: 2 * [y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p)]
        if abs(p - 1.0) < 1e-10
            # Poisson limit
            if yi > 0
                d = yi * log(yi / mui) - (yi - mui)
            else
                d = mui
            end
        elseif abs(p - 2.0) < 1e-10
            # Gamma limit
            if yi > 0
                d = -log(yi / mui) + (yi - mui) / mui
            else
                d = -log(eps() / mui) + (eps() - mui) / mui
            end
        else
            if yi > 0
                t3 = yi^(2 - p) / ((1 - p) * (2 - p))
            else
                t3 = 0.0
            end
            t2 = yi * mui^(1 - p) / (1 - p)
            t1 = mui^(2 - p) / (2 - p)
            d = t3 - t2 + t1
        end
        dev += wt[i] * d
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
    dev_total = _deviance(f, y, mu, wt)
    # Per-observation deviance
    for i in eachindex(y, mu, wt)
        f_single = TweedieFamily(p=f.p, estimate_p=false)
        di = _deviance(f_single, [y[i]], [mu[i]], [wt[i]])
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

# ============================================================================
# Theta / extra parameter estimation
# ============================================================================

"""
    estimate_theta!(family::NegBinFamily, y, mu, wt, scale)

Estimate NB shape parameter θ by Newton iteration on the
log saturated likelihood. Uses digamma/trigamma.
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

            # Second derivative for Hessian
            g2 += wi * (trigamma(yi + θ) - trigamma(θ) + 1.0 / θ -
                        2.0 / (mui + θ) + (mui - yi) / (mui + θ)^2)
        end

        # Adjust for scale
        g1 /= (2.0 * scale)
        g2 /= (2.0 * scale)

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
