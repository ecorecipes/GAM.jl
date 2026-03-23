# GAMLSS — Generalized Additive Models for Location, Scale, and Shape
#
# Uses Distributions.jl for density evaluation and GLM.jl Link types
# for each parameter, following the same conventions as GLM.jl.
#
# Design:
#   - For distributions where GAMLSS params match Distributions.jl params
#     (e.g., Normal(μ,σ)), pass the distribution directly to gamlss().
#   - For distributions needing reparameterization (e.g., Gamma as μ,CV
#     instead of shape,scale), use named types (GammaLocationScale, etc.)
#   - Links are GLM.Link objects, one per parameter, specified separately.
#   - nll_obs uses Distributions.logpdf — ForwardDiff handles derivatives.

# ═══════════════════════════════════════════════════════════════════════
# Link helpers
# ═══════════════════════════════════════════════════════════════════════

_link_symbol(::IdentityLink) = :identity
_link_symbol(::LogLink) = :log
_link_symbol(::LogitLink) = :logit
_link_symbol(::InverseLink) = :inverse
_link_symbol(::GLM.Link) = :unknown

_apply_link_inv(::IdentityLink, η) = η
_apply_link_inv(::LogLink, η) = exp(η)
_apply_link_inv(::LogitLink, η) = 1 / (1 + exp(-η))
_apply_link_inv(::InverseLink, η) = 1 / η
_apply_link_inv(link::GLM.Link, η) = GLM.linkinv(link, η)

# ═══════════════════════════════════════════════════════════════════════
# DistFamily{D}: for distributions whose params match Distributions.jl
# ═══════════════════════════════════════════════════════════════════════

"""
    DistFamily{D<:UnivariateDistribution} <: MultiParameterFamily

Wraps a `Distributions.jl` distribution for use with `gamlss()`.
Only used for distributions whose GAMLSS parameters match the
Distributions.jl constructor directly (e.g., `Normal(μ, σ)`).

Constructed automatically by `gamlss()` when passed a distribution instance.
"""
struct DistFamily{D<:UnivariateDistribution} <: MultiParameterFamily
    dist::D
    links::Vector{<:GLM.Link}
    pnames::Vector{String}
end

nparams(f::DistFamily) = length(f.links)
param_names(f::DistFamily) = f.pnames
param_links(f::DistFamily) = [_link_symbol(l) for l in f.links]

# ── Normal: params (μ, σ) match Distributions.jl directly ──

_gamlss_nparams(::Normal) = 2
_gamlss_default_links(::Normal) = [IdentityLink(), LogLink()]
_gamlss_param_names(::Normal) = ["mu", "sigma"]

_gamlss_construct(::Normal, params) = Normal(params[1], max(params[2], 1e-10))

function _gamlss_initial_eta(::Normal, links, y)
    n = length(y)
    [fill(GLM.linkfun(links[1], mean(y)), n),
     fill(GLM.linkfun(links[2], max(std(y), 0.01)), n)]
end

function nll_obs(f::DistFamily, y_i, η_vec)
    params = ntuple(k -> _apply_link_inv(f.links[k], η_vec[k]), length(f.links))
    d = _gamlss_construct(f.dist, params)
    return -logpdf(d, y_i)
end

function initial_eta(f::DistFamily, y::AbstractVector)
    _gamlss_initial_eta(f.dist, f.links, y)
end

# ═══════════════════════════════════════════════════════════════════════
# GammaLocationScale — Gamma reparameterized as (μ, σ)
#   μ = mean, σ = coefficient of variation
#   Maps to Gamma(α=1/σ², θ=μσ²)
# ═══════════════════════════════════════════════════════════════════════

"""
    GammaLocationScale(; links=[LogLink(), LogLink()])

Gamma distribution reparameterized for GAMLSS:
- Parameter 1: μ (mean), default link: `LogLink()`
- Parameter 2: σ (coefficient of variation), default link: `LogLink()`

Maps to `Gamma(α=1/σ², θ=μσ²)` so E[Y]=μ, Var[Y]=μ²σ².
"""
struct GammaLocationScale <: MultiParameterFamily
    links::Vector{GLM.Link}
end
GammaLocationScale(; links = [LogLink(), LogLink()]) = GammaLocationScale(links)

nparams(::GammaLocationScale) = 2
param_names(::GammaLocationScale) = ["mu", "sigma"]
param_links(f::GammaLocationScale) = [_link_symbol(l) for l in f.links]

function nll_obs(f::GammaLocationScale, y_i, η_vec)
    μ = _apply_link_inv(f.links[1], η_vec[1])
    σ = _apply_link_inv(f.links[2], η_vec[2])
    α = 1 / max(σ^2, 1e-10)
    θ = max(μ, 1e-10) * max(σ^2, 1e-10)
    return -logpdf(Gamma(α, θ), y_i)
end

function initial_eta(f::GammaLocationScale, y::AbstractVector)
    μ = max(mean(y), 0.01)
    σ = max(std(y) / μ, 0.01)
    n = length(y)
    [fill(GLM.linkfun(f.links[1], μ), n),
     fill(GLM.linkfun(f.links[2], σ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# BetaRegression — Beta reparameterized as (μ, φ)
#   μ = mean, φ = precision
#   Maps to Beta(α=μφ, β=(1-μ)φ)
# ═══════════════════════════════════════════════════════════════════════

"""
    BetaRegression(; links=[LogitLink(), LogLink()])

Beta distribution reparameterized for GAMLSS / beta regression:
- Parameter 1: μ (mean), default link: `LogitLink()`
- Parameter 2: φ (precision), default link: `LogLink()`

Maps to `Beta(α=μφ, β=(1-μ)φ)` so E[Y]=μ, Var[Y]=μ(1-μ)/(1+φ).
"""
struct BetaRegression <: MultiParameterFamily
    links::Vector{GLM.Link}
end
BetaRegression(; links = [LogitLink(), LogLink()]) = BetaRegression(links)

nparams(::BetaRegression) = 2
param_names(::BetaRegression) = ["mu", "phi"]
param_links(f::BetaRegression) = [_link_symbol(l) for l in f.links]

function nll_obs(f::BetaRegression, y_i, η_vec)
    μ = _apply_link_inv(f.links[1], η_vec[1])
    φ = _apply_link_inv(f.links[2], η_vec[2])
    μc = clamp(μ, 1e-6, 1 - 1e-6)
    φc = max(φ, 1e-6)
    y_c = clamp(y_i, 1e-10, 1 - 1e-10)
    return -logpdf(Beta(μc * φc, (1 - μc) * φc), y_c)
end

function initial_eta(f::BetaRegression, y::AbstractVector)
    μ = clamp(mean(y), 0.01, 0.99)
    v = max(var(y), 1e-6)
    φ = max(μ * (1 - μ) / v - 1, 1.0)
    n = length(y)
    [fill(GLM.linkfun(f.links[1], μ), n),
     fill(GLM.linkfun(f.links[2], φ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# NegativeBinomialLocationScale — NB reparameterized as (μ, σ)
#   μ = mean, σ = overdispersion parameter
#   Maps to NegativeBinomial(r=1/σ², p=r/(r+μ))
# ═══════════════════════════════════════════════════════════════════════

"""
    NegativeBinomialLocationScale(; links=[LogLink(), LogLink()])

Negative Binomial reparameterized for GAMLSS (NBI parameterization):
- Parameter 1: μ (mean), default link: `LogLink()`
- Parameter 2: σ (overdispersion), default link: `LogLink()`

Maps to `NegativeBinomial(r=1/σ², p=r/(r+μ))` so E[Y]=μ, Var[Y]=μ+μ²σ².
"""
struct NegativeBinomialLocationScale <: MultiParameterFamily
    links::Vector{GLM.Link}
end
NegativeBinomialLocationScale(; links = [LogLink(), LogLink()]) =
    NegativeBinomialLocationScale(links)

nparams(::NegativeBinomialLocationScale) = 2
param_names(::NegativeBinomialLocationScale) = ["mu", "sigma"]
param_links(f::NegativeBinomialLocationScale) = [_link_symbol(l) for l in f.links]

function nll_obs(f::NegativeBinomialLocationScale, y_i, η_vec)
    μ = _apply_link_inv(f.links[1], η_vec[1])
    σ = _apply_link_inv(f.links[2], η_vec[2])
    r = 1 / max(σ^2, 1e-10)
    p = r / (r + max(μ, 1e-10))
    return -logpdf(NegativeBinomial(r, p), round(Int, max(y_i, 0)))
end

function initial_eta(f::NegativeBinomialLocationScale, y::AbstractVector)
    μ = max(mean(y), 0.1)
    v = max(var(y), μ + 0.01)
    σ = sqrt(max((v - μ) / (μ^2), 0.01))
    n = length(y)
    [fill(GLM.linkfun(f.links[1], μ), n),
     fill(GLM.linkfun(f.links[2], σ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# InverseGaussianLocationScale — IG reparameterized as (μ, σ)
#   μ = mean, σ = CV (coefficient of variation)
#   Maps to InverseGaussian(μ, λ=μ/σ²)
# ═══════════════════════════════════════════════════════════════════════

"""
    InverseGaussianLocationScale(; links=[LogLink(), LogLink()])

Inverse Gaussian reparameterized for GAMLSS:
- Parameter 1: μ (mean), default link: `LogLink()`
- Parameter 2: σ (coefficient of variation), default link: `LogLink()`

Maps to `InverseGaussian(μ, λ=μ/σ²)` so E[Y]=μ, Var[Y]=μ³/λ=μ²σ².
"""
struct InverseGaussianLocationScale <: MultiParameterFamily
    links::Vector{GLM.Link}
end
InverseGaussianLocationScale(; links = [LogLink(), LogLink()]) =
    InverseGaussianLocationScale(links)

nparams(::InverseGaussianLocationScale) = 2
param_names(::InverseGaussianLocationScale) = ["mu", "sigma"]
param_links(f::InverseGaussianLocationScale) = [_link_symbol(l) for l in f.links]

function nll_obs(f::InverseGaussianLocationScale, y_i, η_vec)
    μ = _apply_link_inv(f.links[1], η_vec[1])
    σ = _apply_link_inv(f.links[2], η_vec[2])
    μc = max(μ, 1e-10)
    λ = μc / max(σ^2, 1e-10)
    return -logpdf(InverseGaussian(μc, λ), y_i)
end

function initial_eta(f::InverseGaussianLocationScale, y::AbstractVector)
    μ = max(mean(y), 0.01)
    σ = max(std(y) / μ, 0.01)
    n = length(y)
    [fill(GLM.linkfun(f.links[1], μ), n),
     fill(GLM.linkfun(f.links[2], σ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# Hand-coded derivatives for common families (bypass ForwardDiff)
# ═══════════════════════════════════════════════════════════════════════

"""
Hand-coded NLL derivatives for Normal(μ, σ) with IdentityLink, LogLink.
NLL_i = 0.5*log(2π) + η₂ + 0.5*(y_i - η₁)² * exp(-2η₂)
Avoids per-observation ForwardDiff, giving ~5-10× speedup.
"""
function nll_derivs!(family::DistFamily{<:Normal}, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    n = length(y)
    η₁ = η_list[1]  # μ (identity link)
    η₂ = η_list[2]  # log(σ) (log link)
    @inbounds for i in 1:n
        r = y[i] - η₁[i]
        s2inv = exp(-2.0 * η₂[i])
        r2s2inv = r * r * s2inv

        # Gradients: col 1 = ∂NLL/∂η₁, col 2 = ∂NLL/∂η₂
        out[i, 1] = -r * s2inv
        out[i, 2] = 1.0 - r2s2inv

        # Hessian: col 3 = ∂²NLL/∂η₁², col 4 = ∂²NLL/∂η₁∂η₂, col 5 = ∂²NLL/∂η₂²
        out[i, 3] = s2inv
        out[i, 4] = 2.0 * r * s2inv
        out[i, 5] = 2.0 * r2s2inv
    end
    return out
end

"""Vectorized NLL total for Normal — avoids per-obs function call overhead."""
function nll_total(family::DistFamily{<:Normal}, y::AbstractVector,
                   η_list::Vector{<:AbstractVector})
    n = length(y)
    η₁ = η_list[1]
    η₂ = η_list[2]
    total = 0.0
    c = 0.5 * log(2π)
    @inbounds for i in 1:n
        r = y[i] - η₁[i]
        total += c + η₂[i] + 0.5 * r * r * exp(-2.0 * η₂[i])
    end
    return total
end

"""
Hand-coded NLL derivatives for GammaLocationScale(μ, σ) with LogLink, LogLink.
η₁ = log(μ), η₂ = log(σ). α = exp(-2η₂) = 1/σ², θ = exp(η₁+2η₂) = μσ².
NLL = α·log(θ) + lgamma(α) − (α−1)·log(y) + y/θ.
Verified against ForwardDiff via Symbolics.jl derivation.
"""
function nll_derivs!(family::GammaLocationScale, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return _nll_derivs_ad!(family, out, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]  # log(μ)
    η₂ = η_list[2]  # log(σ)
    @inbounds for i in 1:n
        yi = y[i]
        # α = exp(-2η₂), θ = exp(η₁+2η₂), so α·θ = exp(η₁) = μ
        α = exp(-2.0 * η₂[i])
        θ = exp(η₁[i] + 2.0 * η₂[i])
        μ = exp(η₁[i])
        yoθ = yi / θ
        logθ = η₁[i] + 2.0 * η₂[i]
        logyi = log(max(yi, 1e-300))
        ψα = digamma(α)
        ψ1α = trigamma(α)

        # ∂NLL/∂η₁ = α − y/θ  (from Symbolics: (θα − y)/θ = (μ − y)/θ)
        g1 = α - yoθ

        # ∂NLL/∂η₂ = −2α(logθ + ψα − logyi) + 2(α − y/θ)
        # (from dα/dη₂ = −2α, dθ/dη₂ = 2θ, and lgamma term ψα·(−2α))
        dNdα_nolg = logθ - logyi  # ∂(NLL−lgamma)/∂α = log(θ) − log(y)
        g2 = -2.0 * α * (dNdα_nolg + ψα) + 2.0 * (α - yoθ)

        # ∂²NLL/∂η₁² = y/θ  (since ∂(α−y/θ)/∂η₁ = y/θ via dθ/dη₁=θ)
        h11 = yoθ

        # ∂²NLL/∂η₁∂η₂ = 2(y/θ − α)  (from ∂(α−y/θ)/∂η₂ = −2α+2y/θ)
        h12 = 2.0 * (yoθ - α)

        # ∂²NLL/∂η₂²: full product rule on g2 = dNdα·(−2α) + dNdθ·(2θ)
        # h22 = d(dNdα)/dη₂·(−2α) + dNdα·(4α) + d(dNdθ)/dη₂·(2θ) + dNdθ·(4θ)
        dNdα = logθ + ψα - logyi
        dNdθ = α / θ - yi / (θ * θ)
        ddNdα_dη₂ = 2.0 - 2.0 * α * ψ1α
        ddNdθ_dη₂ = -4.0 * α / θ + 4.0 * yi / (θ * θ)
        h22 = ddNdα_dη₂ * (-2.0 * α) + dNdα * (4.0 * α) +
              ddNdθ_dη₂ * (2.0 * θ) + dNdθ * (4.0 * θ)

        out[i, 1] = g1
        out[i, 2] = g2
        out[i, 3] = h11
        out[i, 4] = h12
        out[i, 5] = h22
    end
    return out
end

"""Vectorized NLL total for GammaLocationScale with LogLink, LogLink."""
function nll_total(family::GammaLocationScale, y::AbstractVector,
                   η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return invoke(nll_total, Tuple{MultiParameterFamily, AbstractVector,
                      Vector{<:AbstractVector}}, family, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]; η₂ = η_list[2]
    total = 0.0
    @inbounds for i in 1:n
        α = exp(-2.0 * η₂[i])
        θ = exp(η₁[i] + 2.0 * η₂[i])
        total += α * log(θ) + lgamma(α) - (α - 1.0) * log(max(y[i], 1e-300)) + y[i] / θ
    end
    return total
end

"""
Hand-coded NLL derivatives for BetaRegression(μ, φ) with LogitLink, LogLink.
η₁ = logit(μ), η₂ = log(φ). μ = sigmoid(η₁), φ = exp(η₂).
a = μφ, b = (1−μ)φ. NLL = lgamma(a)+lgamma(b)−lgamma(a+b)−(a−1)log(y)−(b−1)log(1−y).
Chain rule through (a,b) with digamma/trigamma.
Verified against ForwardDiff via Symbolics.jl derivation.
"""
function nll_derivs!(family::BetaRegression, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    if !(family.links[1] isa LogitLink && family.links[2] isa LogLink)
        return _nll_derivs_ad!(family, out, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]  # logit(μ)
    η₂ = η_list[2]  # log(φ)
    @inbounds for i in 1:n
        yi = clamp(y[i], 1e-10, 1.0 - 1e-10)
        # μ = sigmoid(η₁), φ = exp(η₂)
        eη₁ = exp(-η₁[i])
        μ = 1.0 / (1.0 + eη₁)
        μ = clamp(μ, 1e-6, 1.0 - 1e-6)
        μ1 = 1.0 - μ
        dμ = μ * μ1           # dμ/dη₁
        d2μ = dμ * (1.0 - 2.0 * μ)  # d²μ/dη₁²

        φ = exp(η₂[i])
        φ = max(φ, 1e-6)

        a = μ * φ;  b = μ1 * φ
        logyi = log(yi);  log1yi = log(1.0 - yi)
        ψa = digamma(a);  ψb = digamma(b);  ψab = digamma(a + b)
        ψ1a = trigamma(a); ψ1b = trigamma(b); ψ1ab = trigamma(a + b)

        # NLL partials w.r.t. (a, b)
        Na = ψa - ψab - logyi
        Nb = ψb - ψab - log1yi
        Naa = ψ1a - ψ1ab
        Nbb = ψ1b - ψ1ab
        Nab = -ψ1ab

        # da/dη₁ = φ·dμ, db/dη₁ = −φ·dμ
        dadη₁ = φ * dμ;  dbdη₁ = -φ * dμ

        # g1 = Na·dadη₁ + Nb·dbdη₁ = φ·dμ·(Na − Nb)
        g1 = Na * dadη₁ + Nb * dbdη₁

        # g2 = Na·a + Nb·b  (da/dη₂=a, db/dη₂=b)
        g2 = Na * a + Nb * b

        # h11: second-order chain rule with d²a/dη₁²=φ·d2μ, d²b/dη₁²=−φ·d2μ
        h11 = Naa * dadη₁^2 + 2.0 * Nab * dadη₁ * dbdη₁ + Nbb * dbdη₁^2 +
              Na * φ * d2μ + Nb * (-φ * d2μ)

        # h12: ∂²NLL/∂η₁∂η₂
        # d(dadη₁)/dη₂ = dμ·φ = dadη₁, d(dbdη₁)/dη₂ = dbdη₁
        h12 = (Naa * a + Nab * b) * dadη₁ + (Nab * a + Nbb * b) * dbdη₁ +
              Na * dadη₁ + Nb * dbdη₁

        # h22: ∂²NLL/∂η₂² (d²a/dη₂²=a, d²b/dη₂²=b)
        h22 = Naa * a^2 + 2.0 * Nab * a * b + Nbb * b^2 +
              Na * a + Nb * b

        out[i, 1] = g1
        out[i, 2] = g2
        out[i, 3] = h11
        out[i, 4] = h12
        out[i, 5] = h22
    end
    return out
end

"""
Hand-coded NLL derivatives for NegativeBinomialLocationScale(μ, σ) with LogLink, LogLink.
η₁ = log(μ), η₂ = log(σ). r = 1/σ² = exp(-2η₂), p = r/(r+μ).
NLL = −lgamma(y+r) + lgamma(y+1) + lgamma(r) − r·log(r/(r+μ)) − y·log(μ/(r+μ)).
Verified against ForwardDiff via Symbolics.jl derivation.
"""
function nll_derivs!(family::NegativeBinomialLocationScale, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return _nll_derivs_ad!(family, out, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]  # log(μ)
    η₂ = η_list[2]  # log(σ)
    @inbounds for i in 1:n
        yi = max(y[i], 0.0)
        μ = exp(η₁[i])
        r = exp(-2.0 * η₂[i])  # r = 1/σ²
        rmu = r + μ

        ψyr = digamma(yi + r);  ψr = digamma(r)
        ψ1yr = trigamma(yi + r); ψ1r = trigamma(r)
        logrmu = log(rmu); logr = log(r)

        # ∂NLL/∂μ = (r+y)/(r+μ) − y/μ
        dNdμ = (r + yi) / rmu - yi / μ

        # ∂NLL/∂r = −ψ(y+r) + ψ(r) + log(r+μ) − log(r) − 1 + (r+y)/(r+μ)
        dNdr = -ψyr + ψr + logrmu - logr - 1.0 + (r + yi) / rmu

        # dr/dη₂ = −2r, d²r/dη₂² = 4r
        drdη₂ = -2.0 * r

        # g1 = dNdμ·μ (chain rule: dμ/dη₁ = μ)
        g1 = dNdμ * μ

        # g2 = dNdr·(−2r)
        g2 = dNdr * drdη₂

        # ∂²NLL/∂μ² = −(r+y)/(r+μ)² + y/μ²
        d2Ndμ2 = -(r + yi) / (rmu * rmu) + yi / (μ * μ)

        # ∂²NLL/∂r² = −ψ₁(y+r) + ψ₁(r) − 1/r + 1/(r+μ) + (μ−y)/(r+μ)²
        d2Ndr2 = -ψ1yr + ψ1r - 1.0 / r + 1.0 / rmu + (μ - yi) / (rmu * rmu)

        # ∂²NLL/∂μ∂r = (μ−y)/(r+μ)²
        d2Ndμdr = (μ - yi) / (rmu * rmu)

        # h11 = d2Ndμ2·μ² + dNdμ·μ
        h11 = d2Ndμ2 * μ * μ + dNdμ * μ

        # h12 = d2Ndμdr·μ·(−2r)
        h12 = d2Ndμdr * μ * drdη₂

        # h22 = d2Ndr2·(−2r)² + dNdr·4r
        h22 = d2Ndr2 * drdη₂ * drdη₂ + dNdr * 4.0 * r

        out[i, 1] = g1
        out[i, 2] = g2
        out[i, 3] = h11
        out[i, 4] = h12
        out[i, 5] = h22
    end
    return out
end

"""Vectorized NLL total for NegativeBinomialLocationScale with LogLink, LogLink."""
function nll_total(family::NegativeBinomialLocationScale, y::AbstractVector,
                   η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return invoke(nll_total, Tuple{MultiParameterFamily, AbstractVector,
                      Vector{<:AbstractVector}}, family, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]; η₂ = η_list[2]
    total = 0.0
    @inbounds for i in 1:n
        yi = max(y[i], 0.0)
        μ = exp(η₁[i])
        r = exp(-2.0 * η₂[i])
        rmu = r + μ
        total += -lgamma(yi + r) + lgamma(yi + 1.0) + lgamma(r) -
                 r * log(r / rmu) - yi * log(μ / rmu)
    end
    return total
end

"""
Hand-coded NLL derivatives for InverseGaussianLocationScale(μ, σ) with LogLink, LogLink.
η₁ = log(μ), η₂ = log(σ). λ = μ/σ².
NLL = −0.5·log(μ) + log(σ) + (y−μ)²/(2μσ²y) + const(y).
Q = (y−μ)²/(2μσ²y) = y/(2μσ²) − 1/σ² + μ/(2σ²y).
"""
function nll_derivs!(family::InverseGaussianLocationScale, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return _nll_derivs_ad!(family, out, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]  # log(μ)
    η₂ = η_list[2]  # log(σ)
    @inbounds for i in 1:n
        yi = max(y[i], 1e-300)
        μ = exp(η₁[i])
        σ2 = exp(2.0 * η₂[i])     # σ²
        σ2inv = 1.0 / σ2           # 1/σ²

        # Q decomposition: Q = y/(2μσ²) − 1/σ² + μ/(2σ²y)
        t1 = yi / (2.0 * μ * σ2)   # y/(2μσ²)
        t3 = μ / (2.0 * σ2 * yi)   # μ/(2σ²y)

        # g1 = ∂NLL/∂η₁ = μ·∂NLL/∂μ = −0.5 − y/(2μσ²) + μ/(2σ²y)
        g1 = -0.5 - t1 + t3

        # g2 = ∂NLL/∂η₂ = 1 − (y−μ)²/(μσ²y)
        rv = yi - μ
        Q2 = rv * rv * σ2inv / (μ * yi)  # (y−μ)²/(μσ²y)
        g2 = 1.0 - Q2

        # h11 = y/(2μσ²) + μ/(2σ²y)
        h11 = t1 + t3

        # h12 = y/(μσ²) − μ/(σ²y) = 2(t1 − t3)
        h12 = 2.0 * (t1 - t3)

        # h22 = 2(y−μ)²/(μσ²y)
        h22 = 2.0 * Q2

        out[i, 1] = g1
        out[i, 2] = g2
        out[i, 3] = h11
        out[i, 4] = h12
        out[i, 5] = h22
    end
    return out
end

"""Vectorized NLL total for InverseGaussianLocationScale with LogLink, LogLink."""
function nll_total(family::InverseGaussianLocationScale, y::AbstractVector,
                   η_list::Vector{<:AbstractVector})
    if !all(l -> l isa LogLink, family.links)
        return invoke(nll_total, Tuple{MultiParameterFamily, AbstractVector,
                      Vector{<:AbstractVector}}, family, y, η_list)
    end
    n = length(y)
    η₁ = η_list[1]; η₂ = η_list[2]
    total = 0.0
    @inbounds for i in 1:n
        yi = max(y[i], 1e-300)
        μ = exp(η₁[i])
        σ2 = exp(2.0 * η₂[i])
        λ = μ / σ2
        total += -0.5 * log(λ / (2π * yi^3)) + λ * (yi - μ)^2 / (2.0 * μ^2 * yi)
    end
    return total
end

"""AD fallback for nll_derivs! (used when hand-coded not available)."""
function _nll_derivs_ad!(family::MultiParameterFamily, out::Matrix{Float64},
                         y::AbstractVector, η_list::Vector{<:AbstractVector})
    K = nparams(family)
    n = length(y)
    @inbounds for i in 1:n
        η_vec = [η_list[k][i] for k in 1:K]
        f_i = η -> nll_obs(family, y[i], η)
        val, grad, hess = DifferentiationInterface.value_gradient_and_hessian(f_i, _ad_backend, η_vec)
        for k in 1:K
            out[i, grad_col(k)] = grad[k]
        end
        for j in 1:K
            for ii in 1:j
                out[i, hess_col(K, ii, j)] = hess[ii, j]
            end
        end
    end
    return out
end

# ═══════════════════════════════════════════════════════════════════════
# GamlssControl — control parameters for RS/CG solvers
# ═══════════════════════════════════════════════════════════════════════

"""
    GamlssControl

Control parameters for GAMLSS fitting, matching R's gamlss.control().

# Fields
- `c_crit`: global deviance convergence criterion (default 0.001)
- `n_cyc`: max outer iterations (default 20)
- `i_cc`: inner convergence criterion for RS/CG (default 0.001)
- `i_cyc`: max inner iterations for CG (default 50)
- `mu_step`, `sigma_step`, `nu_step`, `tau_step`: step sizes per parameter (default 1.0)
- `autostep`: automatic step halving when deviance increases (default true)
- `gd_tol`: global deviance tolerance (default Inf)
- `trace`: print progress (default false)
"""
struct GamlssControl
    c_crit::Float64
    n_cyc::Int
    i_cc::Float64
    i_cyc::Int
    mu_step::Float64
    sigma_step::Float64
    nu_step::Float64
    tau_step::Float64
    autostep::Bool
    gd_tol::Float64
    trace::Bool
    sp_method::Symbol  # :efs, :local_ml, :local_gaic, :local_gcv
    gaic_k::Float64    # penalty multiplier for GAIC (k=2 → AIC, k=log(n) → BIC)
end

"""
    gamlss_control(; c_crit=0.001, n_cyc=20, i_cc=0.001, i_cyc=50,
                     mu_step=1.0, sigma_step=1.0, nu_step=1.0, tau_step=1.0,
                     autostep=true, gd_tol=Inf, trace=false,
                     sp_method=:efs, gaic_k=2.0)

Construct a [`GamlssControl`](@ref) with default or custom fitting parameters.

# Arguments
- `c_crit`: global deviance convergence criterion
- `n_cyc`: maximum outer iterations
- `i_cc`: inner convergence criterion for RS/CG steps
- `i_cyc`: maximum inner iterations for CG
- `mu_step`, `sigma_step`, `nu_step`, `tau_step`: step sizes per distribution parameter
- `autostep`: enable automatic step halving when deviance increases
- `gd_tol`: global deviance tolerance
- `trace`: print iteration progress
- `sp_method`: smoothing parameter method — `:efs`, `:local_ml`, `:local_gaic`, or `:local_gcv`
- `gaic_k`: penalty multiplier for GAIC (`2.0` = AIC, `log(n)` = BIC)

# Returns
- A `GamlssControl` instance.

# Examples
```julia
ctrl = gamlss_control(n_cyc=50, trace=true)
m = gamlss(formulas, data, family; gamlss_ctrl=ctrl)
```
"""
function gamlss_control(; c_crit::Real=0.001, n_cyc::Int=20,
                          i_cc::Real=0.001, i_cyc::Int=50,
                          mu_step::Real=1.0, sigma_step::Real=1.0,
                          nu_step::Real=1.0, tau_step::Real=1.0,
                          autostep::Bool=true, gd_tol::Real=Inf,
                          trace::Bool=false,
                          sp_method::Symbol=:efs, gaic_k::Real=2.0)
    sp_method in (:efs, :local_ml, :local_gaic, :local_gcv) ||
        throw(ArgumentError("sp_method must be :efs, :local_ml, :local_gaic, or :local_gcv"))
    GamlssControl(Float64(c_crit), n_cyc, Float64(i_cc), i_cyc,
                  Float64(mu_step), Float64(sigma_step), Float64(nu_step),
                  Float64(tau_step), autostep, Float64(gd_tol), trace,
                  sp_method, Float64(gaic_k))
end

"""Get step size for parameter k from GamlssControl."""
function _get_step(ctrl::GamlssControl, k::Int)
    k == 1 && return ctrl.mu_step
    k == 2 && return ctrl.sigma_step
    k == 3 && return ctrl.nu_step
    k == 4 && return ctrl.tau_step
    return 1.0
end

# ═══════════════════════════════════════════════════════════════════════
# Legacy aliases
# ═══════════════════════════════════════════════════════════════════════

"""Alias: `GaussianLS()` → `DistFamily(Normal(), ...)`"""
GaussianLS() = DistFamily(Normal(), [IdentityLink(), LogLink()], ["mu", "sigma"])
"""Alias: `GammaLS()` → `GammaLocationScale()`"""
GammaLS() = GammaLocationScale()
"""Alias: `BetaLS()` → `BetaRegression()`"""
BetaLS() = BetaRegression()
"""Alias: `NegBinLS()` → `NegativeBinomialLocationScale()`"""
NegBinLS() = NegativeBinomialLocationScale()

# ═══════════════════════════════════════════════════════════════════════
# gamlss() — main interface
# ═══════════════════════════════════════════════════════════════════════

"""
    gamlss(formulas, data, family; links, control, sp, trace) -> MultiParameterModel

Fit a Generalized Additive Model for Location, Scale, and Shape (GAMLSS).

# Arguments
- `formulas`: vector of `@gam_formula`, one per distribution parameter.
  A single formula is replicated for all parameters.
- `data`: a table (DataFrame, etc.)
- `family`: a distribution or family:
  - **Direct**: `Normal()` — params match Distributions.jl
  - **Reparameterized**: `GammaLocationScale()`, `BetaRegression()`,
    `NegativeBinomialLocationScale()`, `InverseGaussianLocationScale()`
  - **Extreme value**: `GEVFamily()`, `GPDFamily()`, `EGPD1Family()`, etc.

# Keyword Arguments
- `links`: vector of `GLM.Link` objects, one per parameter.
  Default: canonical links for the family. Only for distributions passed
  directly (e.g., `Normal()`). Reparameterized families accept links
  via their constructor: `GammaLocationScale(links=[...])`.
- `control`: fitting control (see [`mp_control`](@ref))
- `sp`: fixed log smoothing parameters (default: estimate via REML)
- `trace`: print iteration progress

# Examples
```julia
using GAM, DataFrames, Distributions, GLM

# Normal — pass Distributions.jl type directly, links specified separately
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))],
           df, Normal())

# Normal with custom links
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))],
           df, Normal(); links=[IdentityLink(), IdentityLink()])

# Gamma — reparameterized (μ, σ=CV), links in constructor
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))],
           df, GammaLocationScale())

# Beta regression
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ 1)],
           df, BetaRegression())

# GEV extreme value (MultiParameterFamily passed through)
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
           df, GEVFamily())
```
"""
function gamlss(formulas, data, family::UnivariateDistribution;
    links::Union{Nothing, Vector{<:GLM.Link}} = nothing,
    method::Symbol = :efs,
    control::MPFitControl = mp_control(),
    gamlss_ctrl::GamlssControl = gamlss_control(),
    sp = nothing, trace::Bool = false,
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    _validate_gamlss_family(family)
    K = _gamlss_nparams(family)
    actual_links = links === nothing ? _gamlss_default_links(family) : links
    length(actual_links) == K || throw(ArgumentError(
        "Expected $K links for $(typeof(family)), got $(length(actual_links))"))
    pnames = _gamlss_param_names(family)

    df = DistFamily(family, actual_links, pnames)

    if priors !== nothing
        return _fit_gamlss_bayes(formulas, data, df, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains)
    end
    return gamlss(formulas, data, df; method = method, control = control,
                  gamlss_ctrl = gamlss_ctrl, sp = sp, trace = trace)
end

function gamlss(formulas, data, family::MultiParameterFamily;
    links = nothing,
    method::Symbol = :efs,
    control::MPFitControl = mp_control(),
    gamlss_ctrl::GamlssControl = gamlss_control(),
    sp = nothing, trace::Bool = false,
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    if priors !== nothing
        return _fit_gamlss_bayes(formulas, data, family, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains)
    end

    ctrl = MPFitControl(control.inner_maxit, control.inner_tol,
        control.outer_maxit, control.outer_tol,
        control.step_max, trace)

    return _gamlss_fit(formulas, data, family, ctrl, sp, trace; method = method,
                       gamlss_ctrl = gamlss_ctrl)
end

"""Internal fitting function shared by all gamlss paths."""
function _gamlss_fit(formulas, data, family::MultiParameterFamily,
    ctrl::MPFitControl, sp, trace::Bool;
    method::Symbol = :efs, gamlss_ctrl::GamlssControl = gamlss_control())

    K = nparams(family)

    if formulas isa FormulaTerm || formulas isa GamFormula
        formulas = fill(formulas, K)
    end
    length(formulas) == K || throw(ArgumentError(
        "Expected $K formulas for $(typeof(family)), got $(length(formulas))"))

    cols = Tables.columntable(data)
    y = _extract_response(formulas[1], cols)
    n = length(y)

    X_list = Vector{Matrix{Float64}}(undef, K)
    smooths_list = Vector{Vector{ConstructedSmooth}}(undef, K)
    offset = 0

    for k in 1:K
        Xk, smoothsk = _build_design_matrix(formulas[k], cols, n, offset)
        X_list[k] = Xk
        smooths_list[k] = smoothsk
        offset += size(Xk, 2)
    end

    p = offset
    param_offsets = cumsum([0; [size(X, 2) for X in X_list]])

    Sl = build_penalty_matrices(smooths_list, param_offsets)
    nsp = length(Sl)
    Mp = sum(1 + sum(sm.null_dim for sm in smooths; init = 0) for smooths in smooths_list)

    η_init = initial_eta(family, y)
    β_init = zeros(p)
    for k in 1:K
        s = param_offsets[k] + 1
        β_init[s] = mean(η_init[k])
    end

    if sp !== nothing
        log_sp = Float64.(sp)
    else
        log_sp = _init_log_sp_hessian(family, y, X_list, Sl, β_init, param_offsets, nsp)
    end

    # Dispatch to solver
    if method == :rs || method == :cg
        return _gamlss_fit_rscg(method, family, y, X_list, smooths_list, Sl,
                                β_init, log_sp, param_offsets, ctrl, gamlss_ctrl,
                                nsp, Mp, p, n, sp)
    end

    # Default: EFS solver
    if sp !== nothing || nsp == 0
        S = zeros(p, p)
        for (j, Sj) in enumerate(Sl)
            S .+= exp(log_sp[j]) .* Sj
        end
        β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_init, S, ctrl)
        reml_val = nll_pen
    else
        log_sp, β_opt, reml_val = mp_efs_outer(family, y, X_list, Sl, β_init,
            log_sp, param_offsets, ctrl; Mp = Mp)
        conv = true
    end

    η_fit = _compute_eta(X_list, β_opt, param_offsets, K)

    S = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        if j <= length(log_sp)
            S .+= exp(log_sp[j]) .* Sj
        end
    end

    Vp, Vc, H0 = mp_covariance(family, y, X_list, β_opt, S, param_offsets)
    edf = diag(Vp * H0)
    nll_val = nll_total(family, y, η_fit)

    # Compute LAML for model comparison
    laml = mp_laml(family, y, X_list, β_opt, S, Sl, log_sp, param_offsets; Mp = Mp)

    idpars = Vector{Int}(undef, p)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        idpars[s:e] .= k
    end

    return MultiParameterModel(
        family, β_opt, η_fit, X_list, smooths_list, log_sp,
        edf, Vp, Vc, nll_val, reml_val, laml, y, n, conv, idpars, param_offsets)
end
