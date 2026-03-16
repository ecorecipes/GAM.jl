# GAMLSS families for GAM.jl
#
# Location-Scale-Shape families following the R gamlss package convention.
# Built on the MultiParameterFamily framework, using Distributions.jl for
# density evaluation and ForwardDiff for automatic derivatives.
#
# Each family defines nll_obs(family, y_i, η_vec) using Distributions.logpdf,
# where η_vec contains the linear predictors for each distribution parameter.

using SpecialFunctions: logbeta, loggamma

# ═══════════════════════════════════════════════════════════════════════
# GaussianLS — Normal(μ, σ) with μ = η₁, log(σ) = η₂
# ═══════════════════════════════════════════════════════════════════════

"""
    GaussianLS()

Gaussian location-scale family: Y ~ Normal(μ, σ).

- η₁ = μ (identity link)
- η₂ = log(σ) (log link for positivity)

This is the simplest GAMLSS family and a useful test case.
"""
struct GaussianLS <: MultiParameterFamily end

nparams(::GaussianLS) = 2
param_names(::GaussianLS) = ["mu", "sigma"]
param_links(::GaussianLS) = [:identity, :log]

function nll_obs(::GaussianLS, y_i, η_vec)
    μ = η_vec[1]
    σ = exp(η_vec[2])
    return -logpdf(Normal(μ, σ), y_i)
end

function initial_eta(::GaussianLS, y::AbstractVector)
    μ = mean(y)
    σ = max(std(y), 0.01)
    n = length(y)
    return [fill(μ, n), fill(log(σ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# GammaLS — Gamma(shape=1/σ², scale=μσ²) with μ = exp(η₁), log(σ) = η₂
# ═══════════════════════════════════════════════════════════════════════

"""
    GammaLS()

Gamma location-scale family: Y ~ Gamma(shape, scale) parameterized as:

- η₁ = log(μ) (log link — mean)
- η₂ = log(σ) (log link — CV, so Var = μ²σ²)

Shape α = 1/σ², Scale θ = μσ² → E[Y] = μ, Var[Y] = μ²σ².
"""
struct GammaLS <: MultiParameterFamily end

nparams(::GammaLS) = 2
param_names(::GammaLS) = ["mu", "sigma"]
param_links(::GammaLS) = [:log, :log]

function nll_obs(::GammaLS, y_i, η_vec)
    μ = exp(η_vec[1])
    σ = exp(η_vec[2])
    α = 1.0 / (σ * σ)           # shape
    θ = μ * σ * σ                # scale = μ/α = μσ²
    return -logpdf(Gamma(α, θ), y_i)
end

function initial_eta(::GammaLS, y::AbstractVector)
    μ = max(mean(y), 0.01)
    σ = max(std(y) / μ, 0.01)  # coefficient of variation
    n = length(y)
    return [fill(log(μ), n), fill(log(σ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# BetaLS — Beta(α, β) with logit(μ) = η₁, log(φ) = η₂
# ═══════════════════════════════════════════════════════════════════════

"""
    BetaLS()

Beta location-scale family: Y ~ Beta(α, β) where:

- η₁ = logit(μ) (logit link — mean)
- η₂ = log(φ) (log link — precision)

α = μφ, β = (1-μ)φ → E[Y] = μ, Var[Y] = μ(1-μ)/(1+φ).
"""
struct BetaLS <: MultiParameterFamily end

nparams(::BetaLS) = 2
param_names(::BetaLS) = ["mu", "phi"]
param_links(::BetaLS) = [:logit, :log]

function nll_obs(::BetaLS, y_i, η_vec)
    μ = 1.0 / (1.0 + exp(-η_vec[1]))   # logit⁻¹
    φ = exp(η_vec[2])
    α = μ * φ
    β = (1.0 - μ) * φ
    y_c = clamp(y_i, 1e-10, 1.0 - 1e-10)
    return -logpdf(Beta(α, β), y_c)
end

function initial_eta(::BetaLS, y::AbstractVector)
    μ = clamp(mean(y), 0.01, 0.99)
    v = max(var(y), 1e-6)
    φ = max(μ * (1 - μ) / v - 1.0, 1.0)
    n = length(y)
    return [fill(log(μ / (1 - μ)), n), fill(log(φ), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# PoissonLS — Negative-binomial as Poisson + overdispersion
# NegBinLS: Y ~ NB(μ, σ) with log(μ) = η₁, log(σ) = η₂
# ═══════════════════════════════════════════════════════════════════════

"""
    NegBinLS()

Negative Binomial location-scale family: Y ~ NB(r, p) parameterized as:

- η₁ = log(μ) (log link — mean)
- η₂ = log(σ) (log link — overdispersion)

r = 1/σ², p = 1/(1 + μσ²) → E[Y] = μ, Var[Y] = μ + μ²σ².
This is the NBI parameterization used by R gamlss.
"""
struct NegBinLS <: MultiParameterFamily end

nparams(::NegBinLS) = 2
param_names(::NegBinLS) = ["mu", "sigma"]
param_links(::NegBinLS) = [:log, :log]

function nll_obs(::NegBinLS, y_i, η_vec)
    μ = exp(η_vec[1])
    σ² = exp(2.0 * η_vec[2])
    r = 1.0 / σ²                    # size parameter
    p = r / (r + μ)                  # success probability
    return -logpdf(NegativeBinomial(r, p), round(Int, max(y_i, 0)))
end

function initial_eta(::NegBinLS, y::AbstractVector)
    μ = max(mean(y), 0.1)
    v = max(var(y), μ + 0.01)
    σ² = max((v - μ) / (μ * μ), 0.01)
    n = length(y)
    return [fill(log(μ), n), fill(0.5 * log(σ²), n)]
end

# ═══════════════════════════════════════════════════════════════════════
# gamlss() — unified interface for all multi-parameter families
# ═══════════════════════════════════════════════════════════════════════

"""
    gamlss(formulas, data, family; control, sp, trace) -> MultiParameterModel

Fit a Generalized Additive Model for Location, Scale, and Shape (GAMLSS).

This is the unified interface for **all** multi-parameter distribution models,
including GAMLSS location-scale families, extreme value models (GEV, GPD),
and extended GPD models. The `evgam` function is an alias.

# Arguments
- `formulas`: a vector of `@gam_formula` or `FormulaTerm`, one per distribution
  parameter. A single formula is replicated for all parameters.
- `data`: a table (DataFrame, NamedTuple, etc.)
- `family`: any `MultiParameterFamily`:
  - **GAMLSS**: `GaussianLS()`, `GammaLS()`, `BetaLS()`, `NegBinLS()`
  - **Extreme value**: `GEVFamily()`, `GPDFamily()`
  - **Extended GPD**: `EGPD1Family()`, `EGPD2Family()`, etc.
- `control`: fitting control parameters (see [`mp_control`](@ref))
- `sp`: optional fixed smoothing parameters (log scale)
- `trace`: print iteration progress

# Examples
```julia
using GAM, DataFrames

# Gaussian location-scale
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))],
           df, GaussianLS())

# GEV extreme value (same interface)
m = gamlss([@gam_formula(y ~ s(x)), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
           df, GEVFamily())
```
"""
function gamlss(formulas, data, family::MultiParameterFamily;
    control::MPFitControl = mp_control(),
    sp = nothing, trace::Bool = false)

    ctrl = MPFitControl(control.inner_maxit, control.inner_tol,
        control.outer_maxit, control.outer_tol,
        control.step_max, trace)

    return evgam(formulas, data, family; control=ctrl, sp=sp, trace=trace)
end
