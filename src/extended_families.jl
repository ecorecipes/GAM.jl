# Extended families for GAM fitting
#
# Wraps distributions with extra parameter estimation support
# (e.g., NB theta, Tweedie power, Beta precision).
# Kept separate from Distributions.UnivariateDistribution to avoid type confusion.

using SpecialFunctions: digamma, trigamma, logabsgamma

"""
    ExtendedFamily

Abstract supertype for extended GAM families that require estimation of
additional parameters beyond the mean (e.g., NB shape, Tweedie power,
Beta precision).
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
# Tweedie family
# ============================================================================

"""
    TweedieFamily(; p=1.5, estimate_p=false)

Tweedie family with power parameter p ∈ (1, 2).
Variance function: V(μ) = μ^p.
Default link: `LogLink()`.
Currently supports fixed p only.
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
_default_link(::TweedieFamily) = LogLink()
_default_link(::BetaFamily) = LogitLink()

"""Whether the family has an extra parameter to estimate."""
_has_extra_param(f::NegBinFamily) = f.estimate_theta
_has_extra_param(f::TweedieFamily) = f.estimate_p
_has_extra_param(f::BetaFamily) = f.estimate_phi

"""Whether the family provides Dd derivatives for proper PIRLS working weights."""
_has_Dd(::ExtendedFamily) = false

"""Dd derivatives (override for families that provide them)."""
_family_Dd(f::ExtendedFamily, y, mu, wt; level=0) = error("Dd not implemented for $(typeof(f))")

"""Name string for display."""
_family_name(::NegBinFamily) = "NegativeBinomial"
_family_name(::TweedieFamily) = "Tweedie"
_family_name(::BetaFamily) = "Beta"

# ============================================================================
# Variance functions
# ============================================================================

function _variance(f::NegBinFamily, mu)
    θ = f.theta
    return mu .+ mu .^ 2 ./ θ
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

Placeholder for Tweedie power parameter estimation (not yet implemented).
"""
function estimate_theta!(family::TweedieFamily, y, mu, wt, scale)
    # Power parameter estimation requires ldTweedie; skip for now
    return nothing
end

# ============================================================================
# Scale estimation for extended families
# ============================================================================

"""Whether the family estimates scale (like Gaussian) or has fixed scale=1."""
_estimates_scale(::NegBinFamily) = false
_estimates_scale(::TweedieFamily) = true
_estimates_scale(::BetaFamily) = false
