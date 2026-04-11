# GAM.jl

*Generalized Additive Models for Julia*

GAM.jl is a comprehensive Julia port of R's [mgcv](https://cran.r-project.org/package=mgcv) ecosystem, providing
penalized regression spline GAMs with automatic smoothness estimation. It also implements
functionality from gamlss, scam, qgam, and evgam — all in pure Julia with a **9.81x**
geometric-mean speedup over R in the latest checked-in benchmark snapshot.

## Features

- **28 smooth basis types**: thin plate (`:tp`, `:ts`), cubic (`:cr`, `:cs`, `:cc`),
  P-splines (`:ps`), cyclic P-splines (`:cps`), B-splines (`:bs`), Gaussian process (`:gp`),
  loess (`:lo`), fractional polynomial (`:fp`), Duchon splines (`:ds`), adaptive (`:ad`),
  spherical splines (`:sos`), SPDE Matérn (`:spde`), Markov random fields (`:mrf`),
  soap film (`:so`), factor-smooth interactions (`:fs`), constrained factor smooth (`:sz`),
  random effects (`:re`), tensor products (`te`/`ti`/`t2`),
  and 8 SCAM shape-constrained bases (`:mpi`, `:mpd`, `:cx`, `:cv`, `:micx`, `:micv`, `:mdcx`, `:mdcv`)
- **Automatic smoothing**: REML, ML, and GCV smoothing parameter estimation via
  Extended Fellner-Schall (EFS) method
- **Multiple families**: Gaussian, Poisson, Binomial, Gamma, Inverse Gaussian,
  plus extended families (Negative Binomial, quasi-Poisson, quasi-Binomial, Tweedie, Beta)
- **GAMLSS**: distributional regression for location, scale, and shape parameters
  with RS and CG solvers (GaussianLS, GammaLocationScale, BetaRegression, and more)
- **SCAM**: shape-constrained additive models (monotonicity, convexity)
- **QGAM**: quantile regression GAMs via extended log-F likelihood
- **evgam**: extreme value GAMs (GEV, GPD, EGPD families)
- **BAM**: `bam()` for memory-efficient fitting of large datasets with discretization
- **GAMM**: `gamm()` for mixed-effects GAMs with `@gamm_formula` syntax, including
  PQL estimation for non-Gaussian families (Poisson, Binomial, Gamma)
- **ANOVA for GAMs**: `anova_gam()` for smooth significance testing and nested
  model comparison (F-test and χ² test)
- **GINLA**: integrated nested Laplace approximation for posterior inference
- **Bayesian inference**: Turing.jl integration via `smooth2random` conversion
- **Side constraints**: `gam.side` identifiability constraints for overlapping smooths
- **Gratia-style diagnostics**: `smooth_estimates`, `derivatives`, `partial_residuals`,
  `appraise`, `rootogram`, `posterior_samples`, `fitted_samples`, `data_slice`
- **JuliaStats integration**: follows StatsModels.jl/GLM.jl conventions with
  `@formulak` syntax and full StatsBase interface
- **Tested against R**: comprehensive integration tests comparing results against mgcv

## Quick Start

```julia
using GAM, DataFrames

# Generate data
n = 200
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# Fit a GAM
m = gam(@formulak(y ~ s(x, k=15, bs=:cr)), df)

# Inspect the fit
m                    # pretty-printed summary
coef(m)              # coefficients
deviance(m)          # model deviance
m.edf                # effective degrees of freedom per smooth
m.scale              # estimated scale parameter

# Diagnostics
gam_check(m)         # residual diagnostics
smooth_estimates(m)  # evaluate smooths on a grid
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/GAM.jl")
```

## Contents

```@contents
Pages = [
    "tutorial.md",
    "smooths.md",
    "formulas.md",
    "families.md",
    "gamlss.md",
    "scam.md",
    "qgam.md",
    "evgam.md",
    "bam.md",
    "gamm.md",
    "bayesian.md",
    "diagnostics.md",
    "mgcv.md",
    "api.md",
]
Depth = 2
```
