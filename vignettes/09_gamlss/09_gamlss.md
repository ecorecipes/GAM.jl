# GAMLSS: Location, Scale, and Shape Models
GAM.jl Contributors

- [Introduction](#introduction)
- [Setup](#setup)
- [Normal Location-Scale Model](#normal-location-scale-model)
  - [Data](#data)
  - [Fitting the model](#fitting-the-model)
  - [Extracting fitted parameters](#extracting-fitted-parameters)
  - [Comparison with standard GAM](#comparison-with-standard-gam)
  - [Constant-variance special case](#constant-variance-special-case)
- [Gamma Location-Scale Model](#gamma-location-scale-model)
  - [Data](#data-1)
  - [Fitting](#fitting)
  - [Results](#results)
  - [Why not just `Gamma()` in `gam()`?](#why-not-just-gamma-in-gam)
- [Beta Regression](#beta-regression)
  - [Data](#data-2)
  - [Fitting](#fitting-1)
  - [Results](#results-1)
- [Custom Links](#custom-links)
- [Available Families](#available-families)
- [Effective Degrees of Freedom](#effective-degrees-of-freedom)
- [R Comparison](#r-comparison)
- [Summary](#summary)

## Introduction

A standard GAM models the **mean** of a response distribution as a
smooth function of covariates, keeping other distribution parameters
(variance, shape) constant. **GAMLSS** (Generalized Additive Models for
Location, Scale, and Shape) extends this by allowing *every* parameter
of the response distribution to depend on covariates through smooth
functions.

For a response $y_i$ with distribution
$\mathcal{D}(\theta_1, \theta_2, \ldots, \theta_K)$, GAMLSS models each
parameter via its own additive predictor:

$$g_k(\theta_{ki}) = \beta_{0k} + f_{k1}(x_{1i}) + \cdots + f_{kJ_k}(x_{J_k i}), \quad k = 1, \ldots, K$$

where $g_k$ is the link function for parameter $k$. This is useful when:

- **Variance changes** with covariates (heteroscedasticity)
- **Skewness or tail behavior** varies across the covariate space
- You need a **distributional regression** rather than a mean regression

GAM.jl’s `gamlss()` function provides this capability using
Distributions.jl types and GLM.jl link functions.

## Setup

``` julia
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
```

``` julia
using GAM
using CSV
using DataFrames
using Distributions
using GLM: LogLink, LogitLink, IdentityLink
using Statistics: mean, std, var, cor
using StatsAPI: fitted, deviance
using LinearAlgebra: diag
using Random
```

## Normal Location-Scale Model

The simplest GAMLSS: a Gaussian response where both the mean $\mu(x)$
and standard deviation $\sigma(x)$ vary smoothly with a covariate.

### Data

We use data with:

- Location: $\mu(x) = 2 + 1.5\sin(x)$
- Scale: $\sigma(x) = 0.3 + 0.2\cos(x)$

``` julia
dat = CSV.read("data_normal_ls.csv", DataFrame)
println("n = $(nrow(dat)), y range: [$(round(minimum(dat.y); digits=2)), $(round(maximum(dat.y); digits=2))]")
```

    n = 500, y range: [-0.33, 4.21]

### Fitting the model

With `gamlss()`, pass `Normal()` directly — its parameters $(μ, σ)$
match the Distributions.jl parameterization. Each parameter gets its own
formula:

``` julia
m = gamlss(
    [@gam_formula(y ~ s(x, k=15, bs=:cr)),   # μ formula
     @gam_formula(y ~ s(x, k=10, bs=:cr))],   # log(σ) formula
    dat, Normal()
)
println("Converged: $(m.converged)")
println("Negative log-likelihood: $(round(m.nll; digits=2))")
println("Total EDF: $(round(sum(m.edf); digits=1))")
```

    Converged: true
    Negative log-likelihood: 25.99
    Total EDF: 18.6

The default links are `IdentityLink()` for $\mu$ and `LogLink()` for
$\sigma$, matching the standard GAMLSS parameterization.

### Extracting fitted parameters

The fitted linear predictors are stored in `m.fitted_eta`. Apply the
inverse link to recover the distribution parameters:

``` julia
μ_fit = m.fitted_eta[1]                    # identity link → μ directly
σ_fit = exp.(m.fitted_eta[2])              # log link → exp to get σ

println("μ: cor with truth = $(round(cor(μ_fit, dat.mu_true); digits=5))")
println("σ: cor with truth = $(round(cor(σ_fit, dat.sigma_true); digits=5))")
```

    μ: cor with truth = 0.99861
    σ: cor with truth = 0.98639

### Comparison with standard GAM

A standard `gam()` assumes constant variance. Let’s compare:

``` julia
m_gam = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), dat)
μ_gam = m_gam.fitted_values
println("Standard GAM μ: cor = $(round(cor(μ_gam, dat.mu_true); digits=5))")
println("GAMLSS μ:       cor = $(round(cor(μ_fit, dat.mu_true); digits=5))")
println("Max |μ_gam - μ_gamlss|: $(round(maximum(abs.(μ_gam - μ_fit)); digits=4))")
```

    Standard GAM μ: cor = 0.99873
    GAMLSS μ:       cor = 0.99861
    Max |μ_gam - μ_gamlss|: 0.0357

The location estimates are similar, but GAMLSS additionally captures how
the spread changes.

### Constant-variance special case

When $\sigma$ is modeled as intercept-only (`y ~ 1`), GAMLSS reduces to
a standard GAM:

``` julia
m_const = gamlss(
    [@gam_formula(y ~ s(x, k=15, bs=:cr)),
     @gam_formula(y ~ 1)],
    dat, Normal()
)
σ_const = exp.(m_const.fitted_eta[2])
println("σ range: [$(round(minimum(σ_const); digits=4)), $(round(maximum(σ_const); digits=4))]")
println("σ CV: $(round(std(σ_const)/mean(σ_const)*100; digits=2))%")
```

    σ range: [0.3294, 0.3294]
    σ CV: 0.0%

## Gamma Location-Scale Model

For positive continuous data, `GammaLocationScale()` reparameterizes the
Gamma distribution as:

- $\mu$ = mean (link: `LogLink()`)
- $\sigma$ = coefficient of variation (link: `LogLink()`)

This maps to `Gamma(α = 1/σ², θ = μσ²)` so that $E[Y] = \mu$ and
$\text{Var}[Y] = \mu^2 \sigma^2$.

### Data

``` julia
dat_g = CSV.read("data_gamma_ls.csv", DataFrame)
println("n = $(nrow(dat_g)), y range: [$(round(minimum(dat_g.y); digits=3)), $(round(maximum(dat_g.y); digits=2))]")
println("All positive: $(all(dat_g.y .> 0))")
```

    n = 500, y range: [0.563, 7.16]
    All positive: true

### Fitting

``` julia
m_g = gamlss(
    [@gam_formula(y ~ s(x, k=15, bs=:cr)),   # log(μ)
     @gam_formula(y ~ s(x, k=10, bs=:cr))],   # log(σ)
    dat_g, GammaLocationScale()
)
println("Converged: $(m_g.converged)")
```

    Converged: true

### Results

``` julia
μ_fit = exp.(m_g.fitted_eta[1])   # log link for μ
σ_fit = exp.(m_g.fitted_eta[2])   # log link for σ (CV)

println("μ: cor with truth = $(round(cor(μ_fit, dat_g.mu_true); digits=5))")
println("σ: cor with truth = $(round(cor(σ_fit, dat_g.sigma_true); digits=5))")
println("Mean fitted CV: $(round(mean(σ_fit); digits=3))")
```

    μ: cor with truth = 0.83194
    σ: cor with truth = 0.99327
    Mean fitted CV: 0.368

### Why not just `Gamma()` in `gam()`?

A standard `gam(...; family=Gamma())` only models the mean with a single
scale parameter. The GAMLSS version lets the CV *vary* with $x$,
capturing heterogeneity in the dispersion.

## Beta Regression

For responses bounded in $(0, 1)$ — proportions, rates, or compositional
data — `BetaRegression()` reparameterizes the Beta distribution as:

- $\mu$ = mean (link: `LogitLink()`)
- $\phi$ = precision (link: `LogLink()`)

This maps to `Beta(α = \mu\phi, β = (1-\mu)\phi)` so that $E[Y] = \mu$
and $\text{Var}[Y] = \mu(1-\mu)/(1+\phi)$.

### Data

``` julia
dat_b = CSV.read("data_beta_reg.csv", DataFrame)
println("n = $(nrow(dat_b)), y range: [$(round(minimum(dat_b.y); digits=4)), $(round(maximum(dat_b.y); digits=4))]")
```

    n = 400, y range: [0.0071, 0.7526]

### Fitting

Here $\phi$ is constant (intercept-only), so we model only the mean as
smooth:

``` julia
m_b = gamlss(
    [@gam_formula(y ~ s(x, k=15, bs=:cr)),   # logit(μ)
     @gam_formula(y ~ 1)],                     # log(φ)
    dat_b, BetaRegression()
)
println("Converged: $(m_b.converged)")
```

    Converged: true

### Results

``` julia
logit_inv(x) = 1 / (1 + exp(-x))
μ_fit = logit_inv.(m_b.fitted_eta[1])
φ_fit = exp.(m_b.fitted_eta[2])

println("μ: cor with truth = $(round(cor(μ_fit, dat_b.mu_true); digits=5))")
println("Fitted precision φ ≈ $(round(mean(φ_fit); digits=1)) (true: 15.0)")
```

    μ: cor with truth = 0.99727
    Fitted precision φ ≈ 16.5 (true: 15.0)

## Custom Links

Links are specified separately from the family, just like in GLM.jl. For
`Normal()`, pass them via the `links` keyword:

``` julia
# Use log link for both μ and σ (e.g., when μ must be positive)
Random.seed!(123)
x_pos = collect(range(0.1, 3.0; length=200))
μ_pos = exp.(0.5 .+ 0.3 .* sin.(2 .* x_pos))
y_pos = μ_pos .+ 0.2 .* randn(200)
y_pos = max.(y_pos, 0.01)
dat_pos = DataFrame(x=x_pos, y=y_pos)

m_custom = gamlss(
    [@gam_formula(y ~ s(x, k=10, bs=:cr)),
     @gam_formula(y ~ 1)],
    dat_pos, Normal();
    links=[LogLink(), LogLink()]  # both parameters on log scale
)
println("Converged: $(m_custom.converged)")
println("Fitted μ range: [$(round(minimum(exp.(m_custom.fitted_eta[1])); digits=2)), $(round(maximum(exp.(m_custom.fitted_eta[1])); digits=2))]")
```

    Converged: true
    Fitted μ range: [1.26, 2.17]

For reparameterized families, pass links to the constructor:

``` julia
m_g2 = gamlss(
    [@gam_formula(y ~ s(x, k=10, bs=:cr)),
     @gam_formula(y ~ 1)],
    dat_g, GammaLocationScale(links=[LogLink(), LogLink()])
)
println("Converged: $(m_g2.converged)")
```

    Converged: true

## Available Families

GAM.jl’s `gamlss()` supports several distribution families:

| Family | Parameters | Default links | Type |
|----|----|----|----|
| `Normal()` | μ (mean), σ (SD) | Identity, Log | Direct Distributions.jl |
| `GammaLocationScale()` | μ (mean), σ (CV) | Log, Log | Reparameterized |
| `BetaRegression()` | μ (mean), φ (precision) | Logit, Log | Reparameterized |
| `NegativeBinomialLocationScale()` | μ (mean), σ (overdispersion) | Log, Log | Reparameterized |
| `InverseGaussianLocationScale()` | μ (mean), σ (CV) | Log, Log | Reparameterized |
| `GEVFamily()` | μ, σ, ξ | Identity, Log, Identity | Extreme value |
| `GPDFamily()` | σ, ξ | Log, Identity | Extreme value |

Legacy aliases (`GaussianLS()`, `GammaLS()`, `BetaLS()`, `NegBinLS()`)
are also available for backward compatibility.

## Effective Degrees of Freedom

Each smooth in each parameter equation has its own EDF, estimated via
REML:

``` julia
println("Normal location-scale model:")
K = GAM.nparams(m.family)
offsets = m.param_offsets
for k in 1:K
    s = offsets[k] + 1
    e = offsets[k + 1]
    edf_k = sum(m.edf[s:e])
    pname = GAM.param_names(m.family)[k]
    println("  $(pname): total EDF = $(round(edf_k; digits=2)) ($(e-s+1) coefficients)")
end
```

    Normal location-scale model:
      mu: total EDF = 11.51 (15 coefficients)
      sigma: total EDF = 7.07 (10 coefficients)

## R Comparison

The R `gamlss` package (with `gamlss.add` for smooth terms) fits similar
models. Here we verify GAM.jl matches R:

``` julia
using RCall

# Re-fit the Normal location-scale model for comparison
m_norm = gamlss(
    [@gam_formula(y ~ s(x, k=15, bs=:cr)),
     @gam_formula(y ~ s(x, k=10, bs=:cr))],
    dat, Normal())
μ_fit_norm = m_norm.fitted_eta[1]
σ_fit_norm = exp.(m_norm.fitted_eta[2])

R"""
suppressPackageStartupMessages({
    library(mgcv)
    library(gamlss)
    library(gamlss.add)
})
"""

# Pass data to R and fit Normal location-scale
@rput dat
R"""
m_r <- gamlss(y ~ ga(~s(x, k=15, bs="cr")),
              sigma.formula = ~ ga(~s(x, k=10, bs="cr")),
              family = NO(), data = dat, trace = FALSE,
              control = gamlss.control(n.cyc = 100, trace = FALSE))
mu_r <- fitted(m_r, "mu")
sigma_r <- fitted(m_r, "sigma")
"""
@rget mu_r sigma_r

n_julia = length(μ_fit_norm)
n_r = length(mu_r)
if n_julia == n_r
    println("μ: Julia–R correlation = $(round(cor(μ_fit_norm, mu_r); digits=6))")
    println("σ: Julia–R correlation = $(round(cor(σ_fit_norm, sigma_r); digits=6))")
    println("μ: max |diff| = $(round(maximum(abs.(μ_fit_norm - mu_r)); digits=4))")
    println("σ: max |diff| = $(round(maximum(abs.(σ_fit_norm - sigma_r)); digits=4))")
else
    println("Length mismatch: Julia=$n_julia, R=$n_r — comparing first $n_r")
    μ_j = μ_fit_norm[1:n_r]
    σ_j = σ_fit_norm[1:n_r]
    println("μ: Julia–R correlation = $(round(cor(μ_j, mu_r); digits=6))")
    println("σ: Julia–R correlation = $(round(cor(σ_j, sigma_r); digits=6))")
end
```

    μ: Julia–R correlation = 0.99984
    σ: Julia–R correlation = 0.999633
    μ: max |diff| = 0.0938
    σ: max |diff| = 0.0219

## Summary

- **`gamlss()`** fits distributional regression models where each
  parameter gets its own smooth predictor
- Pass **Distributions.jl types** directly (`Normal()`) when parameters
  match, or use **reparameterized types** (`GammaLocationScale()`,
  `BetaRegression()`) when they don’t
- **Links** are always separate — via `links=` keyword for direct types,
  or in the family constructor for reparameterized types
- An intercept-only scale formula (`y ~ 1`) recovers the standard GAM as
  a special case
- Smoothing parameters are estimated jointly via **REML**
- Results can be compared against R’s `gamlss` package with `gamlss.add`
  smooth terms
