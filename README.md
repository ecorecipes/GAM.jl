# GAM.jl

[![Build Status](https://github.com/ecorecipes/GAM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/GAM.jl/actions/workflows/CI.yml)

A comprehensive Julia implementation of Generalized Additive Models, inspired by R's [mgcv](https://cran.r-project.org/package=mgcv) package by Simon N. Wood. GAM.jl follows the conventions of [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) and [GLM.jl](https://github.com/JuliaStats/GLM.jl) and implements most of the [StatsAPI](https://github.com/JuliaStats/StatsAPI.jl) model interface.

It covers a large fraction of mgcv's day-to-day functionality (smooths, families, REML/GCV smoothness selection, `by` variables, offsets, prediction with standard errors and per-term contributions) plus several companion packages (gamlss, scam, qgam, evgam). Some mgcv features are not yet implemented — see [Scope and limitations](#scope-and-limitations).

## Features

- **Smooth term specification** — `s()`, `te()`, `ti()`, `t2()` with 30 basis types including thin-plate regression splines, cubic regression splines, P-splines, tensor products, random effects, soap films, Markov random fields, and Gaussian processes
- **Automatic smoothness estimation** — REML/ML via Extended Fellner-Schall (EFS, default) or Newton optimization; GCV/UBRE via direct criterion optimization
- **GLM families** — Gaussian, Poisson, Binomial, Gamma, InverseGaussian, NegativeBinomial, Tweedie, Beta
- **Multi-parameter models (GAMLSS)** — location-scale-shape regression with RS and CG solvers, local ML/GAIC/GCV smoothing parameter selection
- **Shape-constrained smooths (SCAM)** — monotone increasing/decreasing, convex/concave constraints and combinations
- **Quantile regression (QGAM)** — Extended Log-F families with automatic calibration
- **Extreme value models** — GEV, GPD, and extended GPD families
- **Large-scale fitting (BAM)** — chunked accumulation of the normal equations for large datasets
- **Mixed models (GAMM)** — random intercepts/slopes via `gamm()` with `GAM.@formula(...)`
- **Bayesian inference** — Turing.jl extension for posterior sampling with smooth-aware priors
- **Diagnostics** — gratia-style smooth estimates, derivatives, posterior samples, concurvity, rootograms
- **Side constraints** — automatic identifiability constraints when smooths share covariates
- **`by` variables** — varying-coefficient smooths (numeric `by`) and factor-`by` smooths (one penalized smooth per level), including factor-`by` for shape-constrained (SCAM) smooths
- **Offsets** — `gam(...; offset=...)` for known additive terms on the link scale (e.g. log-exposure in rate models), supported for ordinary, extended-family, and shape-constrained fits
- **Term selection** — `gam(...; select=true)` adds a null-space penalty to every smooth (Marra & Wood 2011) so whole terms can be shrunk out of the model

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/GAM.jl")
```

Requires Julia ≥ 1.9.

## Quick Start

```julia
using GAM, DataFrames, Distributions, Random

# Generate data
Random.seed!(42)
n = 500
x1 = rand(n) .* 2π
x2 = rand(n)
y = sin.(x1) .+ 3 .* x2.^2 .+ randn(n) .* 0.3
df = DataFrame(; y, x1, x2)

# Fit a GAM with two smooth terms
m = gam(@formula(y ~ s(x1, k=15, bs=:cr) + s(x2, k=10)), df)

# Standard StatsAPI interface
using StatsAPI
coef(m)              # coefficients
fitted(m)            # fitted values
deviance(m)          # deviance
predict(m, df)       # predictions (with new data)
coeftable(m)         # coefficient table with p-values
r2(m)                # R-squared
```

## Smooth Term Types

`@formula` is the public formula interface. It covers ordinary linear terms,
smooths, and GAMM random effects. If another package also exports `@formula`,
use `GAM.@formula(...)` or `using GAM: @formula`.

| Syntax | Basis | Description |
|--------|-------|-------------|
| `s(x, bs=:tp)` | Thin-plate regression spline | Default. Isotropic, optimal for 1–3 dimensions |
| `s(x, bs=:ts)` | Thin-plate with shrinkage | Extra penalty on null space for selection |
| `s(x, bs=:cr)` | Cubic regression spline | Knot-based, fast for large data |
| `s(x, bs=:cs)` | Cubic spline with shrinkage | Adds shrinkage penalty to CR spline |
| `s(x, bs=:cc)` | Cyclic cubic spline | For periodic data (e.g., time of day) |
| `s(x, bs=:ps)` | P-spline | B-spline basis with difference penalty |
| `s(x, bs=:cps)` | Cyclic P-spline | Periodic P-spline |
| `s(x, bs=:bs)` | B-spline | Unpenalized B-spline basis |
| `s(x, bs=:gp)` | Gaussian process | GP covariance as smooth |
| `s(x, bs=:ds)` | Duchon spline | Generalized thin-plate spline |
| `s(x, bs=:re)` | Random effect | i.i.d. Gaussian random effects |
| `s(x, bs=:mrf)` | Markov random field | Spatial smoothing on discrete regions |
| `s(x, y, bs=:so)` | Soap film | Smoothing over complex domains with boundaries |
| `s(x, y, bs=:fs)` | Factor-smooth interaction | Smooth varying by factor level |
| `s(x, bs=:sos)` | Spherical spline | Smoothing on the sphere (lat/lon) |
| `s(x, bs=:spde)` | SPDE Matérn | Stochastic PDE Matérn field |
| `s(x, bs=:lo)` | Loess | Local regression basis |
| `s(x, bs=:ad)` | Adaptive | Spatially adaptive smoothness |
| `s(x, bs=:fp)` | Fractional polynomial | Fractional-polynomial basis |
| `s(x, bs=:sz)` | Constrained factor smooth | Sum-to-zero factor smooth |
| `te(x, y)` | Tensor product | Full interaction (main effects + interaction) |
| `ti(x, y)` | Tensor interaction | Interaction only (marginals excluded) |
| `t2(x, y)` | Alternative tensor product | ANOVA-style tensor with single penalty per margin |

Smooths also accept `by=` for varying-coefficient and factor-smooth models:

```julia
gam(@formula(y ~ z + s(x, by=z)), df)        # numeric by: z * f(x)
gam(@formula(y ~ g + s(x, by=g)), df)        # factor by: one smooth per level of g
```

## Family and Link Support

The six core GLM families — `Normal`, `Poisson`, `Binomial`, `Bernoulli`, `Gamma`, `InverseGaussian` — are supported with their standard links, plus package-specific extended families:

```julia
# Gaussian (default)
gam(@formula(y ~ s(x)), df)

# Poisson with log link
gam(@formula(y ~ s(x)), df, Poisson(), LogLink())

# Negative binomial
gam(@formula(y ~ s(x)), df, NegBinFamily(1.0))

# Tweedie
gam(@formula(y ~ s(x)), df, TweedieFamily(1.5))

# Beta regression
gam(@formula(y ~ s(x)), df, BetaFamily())
```

## Multi-Parameter Models (GAMLSS)

Model all distribution parameters (location, scale, shape) as smooth functions:

```julia
using GAM, DataFrames, Random

Random.seed!(1)
n = 1000
x = randn(n)
μ = sin.(x)
σ = exp.(0.5 .* x)
y = μ .+ σ .* randn(n)

df = DataFrame(; y, x)

# Gaussian location-scale model
m = gam(
    [
        @formula(y ~ s(x, k=15)),   # mean model
        @formula(y ~ s(x, k=10)),   # log-sd model
    ],
    df,
    GaussianLS(),
)
```

Supported GAMLSS families: `GaussianLS`, `GammaLocationScale`, `BetaRegression`, `NegativeBinomialLocationScale`, `InverseGaussianLocationScale`.

Solver options via `gamlss_control(sp_method=...)`: `:efs` (default, fastest), `:local_ml`, `:local_gaic`, `:local_gcv`.

## Shape-Constrained Models (SCAM)

Enforce monotonicity, convexity, or concavity constraints on smooth terms:

```julia
using GAM

# Monotone increasing smooth
m = gam(@formula(y ~ s(x, bs=:mpi, k=15)), df)

# Convex smooth
m = gam(@formula(y ~ s(x, bs=:cx, k=15)), df)

# Combined: monotone increasing and concave
m = gam(@formula(y ~ s(x, bs=:micv, k=15)), df)
```

Constraint types: `:mpi` (monotone increasing), `:mpd` (monotone decreasing), `:cx` (convex), `:cv` (concave), `:micx` (increasing + convex), `:micv` (increasing + concave), `:mdcx` (decreasing + convex), `:mdcv` (decreasing + concave).

## Quantile Regression (QGAM)

Fit quantile regression GAMs with automatic calibration:

```julia
# Single quantile
m = qgam(@formula(y ~ s(x, k=15)), df, 0.5)  # median

# Multiple quantiles
fits = mqgam(@formula(y ~ s(x, k=15)), df, [0.1, 0.25, 0.5, 0.75, 0.9])

# Extract a single fit
m50 = qdo(fits, 0.5)
```

## Extreme Value Models

Model block maxima (GEV) or threshold exceedances (GPD):

```julia
# GEV model for annual maxima
m = evgam(
    [
        @formula(y ~ s(x, k=10)),   # location
        @formula(y ~ s(x, k=8)),    # log-scale
        @formula(y ~ 1),            # shape (constant)
    ],
    df,
    GEVFamily(),
)
```

## Large-Scale Fitting (BAM)

For large datasets, `bam` fits via chunked accumulation of the normal equations
(keeping memory bounded regardless of row count):

```julia
m = bam(@formula(y ~ s(x1, k=20) + s(x2, k=20)), df)
```

## Mixed Models (GAMM)

GAMs with random effects via MixedModels.jl:

```julia
m = gamm(
    GAM.@formula(y ~ s(x, k=10) + (1 | group)),
    df
)
```

## Bayesian Inference

Posterior sampling via Turing.jl extension:

```julia
using GAM, Turing, Distributions

m_bayes = gam(@formula(y ~ s(x, k=10)), df;
    priors = PriorSpec(sds = Exponential(1.0)),
    nsamples = 1000,
    nchains = 2)

# Posterior summaries
coef(m_bayes)
coeftable(m_bayes)

# Bayesian model scoring
l = loo(m_bayes)
l.looic

w = waic(m_bayes)
w.waic
```

## Diagnostics

```julia
# Model diagnostics
gam_check(m)          # residual plots, basis adequacy
k_check(m)            # basis dimension check
concurvity(m)         # concurvity indices

# Smooth estimates (gratia-style)
se = smooth_estimates(m)
dr = derivatives(m; select=1)        # derivatives of first smooth
pr = partial_residuals(m; select=1)  # partial residuals

# Posterior uncertainty
ps = posterior_samples(m; n=1000)

# Model overview
overview(m)
```

## Prediction

```julia
predict(m, newdata)                    # link scale (η)
predict(m, newdata; type=:response)    # response scale (μ)
predict(m, newdata; se=true)           # (predictions, standard errors)
predict(m, newdata; type=:terms)       # per-term contributions (NamedTuple)
predict(m, newdata; offset=off)        # supply the offset used at fitting

# Linear-predictor (design) matrix Xp such that Xp * coef(m) == η,
# for building custom predictions/intervals (mgcv's type="lpmatrix")
Xp = lpmatrix(m, newdata)
```

## Performance

The latest checked-in benchmark snapshot (`benchmark/results.txt`, 2026-04-01) shows an overall geometric mean speedup of **9.81x** over R on Julia 1.12.5 / R 4.5.2 / macOS ARM64. Both sides use the same data, knot count `k`, and `method="REML"`; Julia timings exclude JIT compilation (warm-up runs) and R timings exclude interpreter startup. The harness measures *fitting time*, not fit equivalence — it does not assert that the two implementations return identical coefficients (correctness is covered by the R-comparison tests instead). The BAM row compares Julia's chunked QR against mgcv's `bam(method="fREML")` without `discrete=TRUE`, i.e. different algorithms; the BAM and SCAM "families" are each a single benchmark.

| Benchmark family | Speedup |
|-----------|---------|
| GAM fitting | 9.84x |
| BAM | 4.66x |
| Prediction | 20.17x |
| Basis construction | 6.72x |
| SCAM | 6.81x |
| QGAM | 4.74x |
| GAMLSS | 14.20x |

Regenerate the checked-in benchmark snapshot with:

```bash
julia --project=. benchmark/refresh_results.jl
```

For the full per-benchmark table, see `benchmark/results.txt`.

## Scope and limitations

GAM.jl is not a line-for-line port of mgcv. Notable mgcv features that are **not** yet implemented:

- Specialized families such as ordered-categorical (`ocat`), zero-inflated Poisson (`ziP`), Cox proportional hazards (`cox.ph`), and multinomial (`multinom`) — though location-scale models are covered by the GAMLSS and evgam interfaces
- Linear functional terms / the summation convention (matrix arguments to `s()`)
- `na.action`-style missing-data handling (rows with missing/non-finite values must be removed before fitting)
- AR1 residual correlation in `bam`

`offset` and `by` variables work for ordinary, extended-family, and shape-constrained (SCAM) fits; factor-`by` is not supported for the linear-constraint (SCASM) solver, and `select=true` applies to ordinary and extended-family GAMs (not the constrained solvers).

## Term Selection

```julia
# Add a null-space penalty to every smooth so entire terms can be removed
m = gam(@formula(y ~ s(x1) + s(x2) + s(x3)), df; select=true)
```

With `select=true`, a smooth whose effect is negligible is shrunk to zero
effective degrees of freedom (mgcv's `select=TRUE`), giving automatic variable
selection alongside smoothness estimation.

## How It Works

GAM.jl follows the same mathematical framework as mgcv:

1. **Basis construction** — Covariates are expanded into smooth basis matrices via `smooth_construct()`
2. **Penalized fitting** — Penalized Iteratively Reweighted Least Squares (P-IRLS) optimizes the penalized log-likelihood
3. **Smoothness estimation** — The Extended Fellner-Schall (EFS) method (Wood & Fasiolo, 2017) iteratively updates smoothing parameters to optimize REML/GCV/ML
4. **Side constraints** — Automatic identifiability constraints are applied when smooths share covariates (mgcv's `gam.side`)
5. **Inference** — Bayesian covariance matrices (Vp) provide approximate confidence intervals and p-values

The key difference from mgcv: GAM.jl is written in pure Julia (no C code), leveraging Julia's BLAS/LAPACK bindings, multiple dispatch, and JIT compilation for performance.

## Testing

GAM.jl has roughly 1,900 test assertions across 48 test files, including:

- Unit tests for all basis types, families, and link functions
- End-to-end tests for GAM, BAM, SCAM, QGAM, GAMLSS, GAMM, evgam, GINLA
- R comparison tests validating fitted values, EDF, deviance, and smoothing parameters against mgcv, scam, qgam, gamlss, and evgam reference output
- Bayesian inference tests with Turing.jl
- Side constraint tests validated against mgcv's `gam.side`

R-comparison tests that require a live R installation (via RCall) are skipped automatically when R or the relevant R package is unavailable, or when `GAM_SKIP_RCALL=true`. The GAMLSS, side-constraint, and SPDE comparisons run against checked-in reference output and do not need R. Run the suite with `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Dependencies

**Core:** StatsModels.jl, GLM.jl, Distributions.jl, StatsBase.jl, StatsAPI.jl, LinearAlgebra, SparseArrays

**Extensions (loaded on demand):**
- [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) — GAMM support
- [Turing.jl](https://github.com/TuringLang/Turing.jl) — Bayesian inference
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl) — Visualization

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071–1081.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society Series B*, 73(1), 3–36.
- Rigby, R.A. & Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society Series C*, 54(3), 507–554.
- Fasiolo, M., Wood, S.N., Zaffran, M., Nedellec, R., & Goude, Y. (2021). Fast calibrated additive quantile regression. *Journal of the American Statistical Association*, 116(535), 1402–1413.
- Pya, N. & Wood, S.N. (2015). Shape constrained additive models. *Statistics and Computing*, 25(3), 543–559.

## Author

[Simon Frost](https://github.com/sdwfrost) ([@sdwfrost](https://github.com/sdwfrost))

## License

MIT
