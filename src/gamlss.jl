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
    control::MPFitControl = mp_control(),
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
    return gamlss(formulas, data, df; control = control, sp = sp, trace = trace)
end

function gamlss(formulas, data, family::MultiParameterFamily;
    links = nothing,
    control::MPFitControl = mp_control(),
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

    return _gamlss_fit(formulas, data, family, ctrl, sp, trace)
end

"""Internal fitting function shared by all gamlss paths."""
function _gamlss_fit(formulas, data, family::MultiParameterFamily,
    ctrl::MPFitControl, sp, trace::Bool)

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

    idpars = Vector{Int}(undef, p)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        idpars[s:e] .= k
    end

    return MultiParameterModel(
        family, β_opt, η_fit, X_list, smooths_list, log_sp,
        edf, Vp, Vc, nll_val, reml_val, y, n, conv, idpars, param_offsets)
end
