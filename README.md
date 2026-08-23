# GAM.jl

[![Build Status](https://github.com/ecorecipes/GAM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/GAM.jl/actions/workflows/CI.yml)

A comprehensive Julia implementation of Generalized Additive Models, inspired by R's [mgcv](https://cran.r-project.org/package=mgcv) package by Simon N. Wood. GAM.jl follows the conventions of [StatsModels.jl](https://github.com/JuliaStats/StatsModels.jl) and [GLM.jl](https://github.com/JuliaStats/GLM.jl) and implements most of the [StatsAPI](https://github.com/JuliaStats/StatsAPI.jl) model interface.

It covers a large fraction of mgcv's day-to-day functionality (smooths, families, REML/GCV smoothness selection, `by` variables, offsets, prediction with standard errors and per-term contributions) plus several companion packages (gamlss, scam, qgam, evgam). Some mgcv features are not yet implemented — see [Scope and limitations](#scope-and-limitations).

## Features

- **Smooth term specification** — `s()`, `te()`, `ti()`, `t2()` with 30 registered basis types including thin-plate regression splines, cubic regression splines, P-splines, tensor products, random effects, soap films, Markov random fields, and Gaussian processes (a few are documented approximations of their mgcv namesakes — see the table below)
- **Automatic smoothness estimation** — REML/ML via Extended Fellner-Schall (EFS, default) or Newton optimization; GCV/UBRE via direct criterion optimization
- **GLM families** — Gaussian, Poisson, Binomial, Gamma, InverseGaussian, NegativeBinomial, Tweedie, Beta
- **Multi-parameter models (GAMLSS)** — location-scale-shape regression with RS and CG solvers, local ML/GAIC/GCV smoothing parameter selection
- **Shape-constrained smooths (SCAM)** — monotone increasing/decreasing, convex/concave constraints and combinations
- **Quantile regression (QGAM)** — Extended Log-F families with automatic calibration
- **Extreme value models** — GEV, GPD, and extended GPD families
- **Large-scale fitting (BAM)** — chunked accumulation of the normal equations for large datasets
- **Mixed models (GAMM)** — random intercepts/slopes via `gamm()` with `GAM.@formula(...)`, fitted by a pure-Julia penalized-smooth backend (PQL for non-Gaussian families)
- **Prior weights** — `gam(...; weights=...)` for observation weights, as in mgcv
- **Bayesian inference** — Turing.jl extension for posterior sampling with smooth-aware priors
- **Diagnostics** — gratia-style smooth estimates, derivatives, posterior samples, concurvity, rootograms
- **Side constraints** — automatic identifiability constraints when smooths share covariates
- **`by` variables** — varying-coefficient smooths (numeric `by`) and factor-`by` smooths (one penalized smooth per level), including factor-`by` for shape-constrained (SCAM) smooths
- **Offsets** — `gam(...; offset=...)` for known additive terms on the link scale (e.g. log-exposure in rate models), supported for ordinary, extended-family, shape-constrained, nested-effect, mixed (`gamm`), and multi-parameter (`gamlss`/`evgam`/`qgam`) fits — the last accepting per-parameter offsets
- **Term selection** — `gam(...; select=true)` adds a null-space penalty to every smooth (Marra & Wood 2011) so whole terms can be shrunk out of the model
- **Nested effects** — gamFactory-style `s_nest()` smooths of estimated covariate transformations: single-index/distributed-lag (`trans_linear`), adaptive exponential smoothing (`trans_nexpsm`), and kernel smoothing (`trans_mgks`), fitted by joint penalized Newton with EFS smoothing selection via `gam_nl()` (or `gam()`, which routes automatically)

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
| `s(x, bs=:bs)` | B-spline | Penalized B-spline (integrated squared derivative penalty, as in mgcv) |
| `s(x, bs=:gp)` | Gaussian process | Matérn 3/2 covariance as smooth |
| `s(x, bs=:ds)` | Duchon spline | Currently an alias for the thin-plate spline (`:tp`) |
| `s(x, bs=:re)` | Random effect | i.i.d. Gaussian random effects |
| `s(x, bs=:mrf)` | Markov random field | Spatial smoothing on discrete regions (requires an `xt` neighborhood structure) |
| `s(x, y, bs=:so)` | Soap film | Smoothing over complex domains with boundaries (approximation of mgcv's construction) |
| `s(x, y, bs=:fs)` | Factor-smooth interaction | Smooth varying by factor level |
| `s(lat, lon, bs=:sos)` | Spherical spline | Smoothing on the sphere; approximation of mgcv's spline-on-the-sphere kernels |
| `s(x, bs=:spde)` | SPDE Matérn | Stochastic PDE Matérn field |
| `s(x, bs=:lo)` | Loess | Local regression basis |
| `s(x, bs=:ad)` | Adaptive | Spatially adaptive smoothness (mgcv-style B-spline penalty weights) |
| `s(x, bs=:sc)` | Shape-constrained B-spline | Linear-constraint (SCASM) B-spline basis |
| `s(x, bs=:scad)` | Shape-constrained adaptive | Linear-constraint (SCASM) adaptive basis |
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
gam(@formula(y ~ s(x)), df, NegBinFamily(theta=1.0))

# Tweedie
gam(@formula(y ~ s(x)), df, TweedieFamily(p=1.5))

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

## Nested Effects

Smooths of *estimated* covariate transformations (Fasiolo et al. 2025; R's
[gamFactory](https://github.com/mfasiolo/gamFactory)): the inner parameters
are fitted jointly with the outer spline.

```julia
# Single-index effect over lagged covariates: s(a'x), a estimated
m = gam_nl(@formula(y ~ s(x0) + s_nest(l1, l2, l3, trans=trans_linear(), k=10)), df)
inner_coef(m)                      # estimated index direction (unit norm)

# Adaptive exponential smoothing of a time-ordered series
m = gam_nl(@formula(y ~ s_nest(x, trans=trans_nexpsm())), df)

# Gaussian-kernel smoothing of z over coordinates
m = gam_nl(@formula(y ~ s_nest(z, cx, cy, trans=trans_mgks())), df)
```

The inner output is standardized and the outer cubic spline uses a fixed
symmetric knot range with an `s(0) = 0` constraint and linear extrapolation,
following the paper. `predict(m, newdata; se=true)` returns delta-method
standard errors that propagate the joint uncertainty of all coefficients —
inner transformation parameters included — through the composition.
`trans_mgks(nn=50)` uses fixed nearest-neighbor sets (as in the paper) for
O(n·nn) evaluation. `gam_nl` accepts `offset=` and `weights=` like `gam()`
(supply the same offset again at `predict`); unsupported options error
rather than being ignored. Supported families: `Normal`, `Poisson`,
`Bernoulli`/`Binomial`, `Gamma`. `gam()` routes formulas containing `s_nest`
to `gam_nl` automatically, and the test suite compares fits against R's
gamFactory live when it is installed.

## Large-Scale Fitting (BAM)

For large datasets, `bam` fits via chunked accumulation of the normal equations
(keeping memory bounded regardless of row count):

```julia
m = bam(@formula(y ~ s(x1, k=20) + s(x2, k=20)), df)
```

## Mixed Models (GAMM)

GAMs with random effects, fitted by a pure-Julia backend that represents random
effects as identity-penalized smooths (LAMS), with PQL for non-Gaussian families:

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
gam_check(m)          # text diagnostics: convergence, k-index with p-values
                      # (plots via appraise(m) with Plots.jl loaded)
k_check(m)            # basis dimension check (mgcv-style k-index + p-value)
concurvity(m)         # concurvity indices (worst/observed/estimate)
appraise(m)           # residual QQ with simulated reference bands
                      # (default method=:simulate; :normal for normal theory)
leverage(m)           # hat diagonals (sums to the model EDF)
cooksdistance(m)      # Cook's distances (influence)

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

## Vignettes

Twelve Quarto vignettes walk through the package, each with an R companion in
its `R/` subdirectory running the equivalent analysis (mgcv, scam, qgam,
gamlss, evgam, gamFactory) on the same checked-in data:

1. [Introduction](vignettes/01_introduction/01_introduction.qmd)
2. [Basis types](vignettes/02_basis_types/02_basis_types.qmd)
3. [Multiple smooths and concurvity](vignettes/03_multiple_smooths/03_multiple_smooths.qmd)
4. [Families and links](vignettes/04_families/04_families.qmd)
5. [Diagnostics](vignettes/05_diagnostics/05_diagnostics.qmd)
6. [Extreme values (GEV/GPD)](vignettes/06_extreme_values/06_extreme_values.qmd)
7. [Shape constraints (SCAM)](vignettes/07_shape_constraints/07_shape_constraints.qmd)
8. [Quantile regression (QGAM)](vignettes/08_quantile_regression/08_quantile_regression.qmd)
9. [GAMLSS](vignettes/09_gamlss/09_gamlss.qmd)
10. [Mixed models (GAMM)](vignettes/10_gamm/10_gamm.qmd)
11. [Bayesian GAMs](vignettes/11_bayesian_gam/11_bayesian_gam.qmd)
12. [Nested effects](vignettes/12_nested_effects/12_nested_effects.qmd)

Rendered GFM versions (`.md`) are checked in alongside the sources; see
[vignettes/README.md](vignettes/README.md) for rendering and data-generation
details.

## Performance

<!-- BENCH-REFRESH -->
The latest checked-in benchmark snapshot (`benchmark/results.txt`, 2026-08-23) shows an overall geometric mean speedup of **11.16x** over R on Julia 1.12.5 / R 4.6.1 / macOS ARM64. Both sides use the same data, knot count `k`, and `method="REML"`; Julia timings exclude JIT compilation (warm-up runs) and R timings exclude interpreter startup. The harness measures *fitting time*, not fit equivalence — it does not assert that the two implementations return identical coefficients (correctness is covered by the elementwise R-comparison tests instead). The BAM row compares Julia's chunked accumulation against mgcv's `bam(method="fREML")` without `discrete=TRUE`, i.e. different algorithms; the BAM and SCAM "families" are each a single benchmark, and the SCAM figure reflects that `scam()` now performs full GCV criterion optimization (matching R scam's method) rather than the faster REML-flavored EFS updates.

| Benchmark family | Speedup |
|-----------|---------|
| GAM fitting | 19.99x |
| BAM | 4.62x |
| Prediction | 7.51x |
| Basis construction | 9.81x |
| SCAM | 1.98x |
| QGAM | 4.77x |
| GAMLSS | 12.82x |

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
- AR1 residual correlation in `bam`; `bam`'s covariate discretization (`discrete=TRUE`) — `bam` uses chunked accumulation of the normal equations only
- Smoothing-parameter-uncertainty corrections (mgcv's `Vc`); `unconditional=true` in `smooth_estimates`/`posterior_samples` warns and uses the conditional covariance
- Smooth-term test statistics use a documented simplification of mgcv's `testStat` (EDFs and p-value conclusions match mgcv; the statistics themselves can differ for heavily penalized smooths)

Some behaviors differ from R by design: quasi families report `NaN`
log-likelihood/AIC (R's `NA` convention), and nested effects (`s_nest`) support
the `Normal`, `Poisson`, `Bernoulli`/`Binomial`, and `Gamma` families with
identity/log/logit links. The optional MixedModels.jl GAMM backend is disabled
(the pure-Julia backend is the supported path).

Some basis types are documented approximations rather than exact ports of their
mgcv namesakes: `:sos` (planar kernel on great-circle distances), `:so` (grid-PDE
soap film), `:ds` (alias of `:tp`), and `:t2` (an alternative tensor
construction, not Wood–Scheipl–Faraway). Fits with these bases will differ from
mgcv's.

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

The key difference from mgcv: GAM.jl's model-fitting code is written in Julia rather than C, leveraging Julia's BLAS/LAPACK bindings, multiple dispatch, and JIT compilation for performance. (Two compiled libraries are used indirectly: BLAS/LAPACK for linear algebra, and the OSQP C solver for linear-constraint SCASM fits.)

## Testing

GAM.jl has roughly 2,300 test assertion macros across 53 test files, including:

- Unit tests for all basis types, families, and link functions
- End-to-end tests for GAM, BAM, SCAM, QGAM, GAMLSS, GAMM, evgam, GINLA
- R comparison tests validating fitted values, EDF, deviance, and smoothing parameters against mgcv, scam, qgam, gamlss, and evgam reference output
- Bayesian inference tests with Turing.jl
- Side constraint tests validated against mgcv's `gam.side`
- Live nested-effects comparisons against gamFactory, and elementwise parity checks (smoothing parameters, coefficients, prediction SEs, AIC) against mgcv. On numerically flat REML ridges the smoothing parameter is only weakly identified, so the suite compares fitted values/EDF/criterion there rather than raw smoothing parameters
- Aqua.jl static quality checks (ambiguities, piracy, stale deps, exports)

R-comparison tests that require a live R installation (via RCall) are skipped automatically when R or the relevant R package is unavailable, or when `GAM_SKIP_RCALL=true`. The GAMLSS, side-constraint, and SPDE comparisons run against checked-in reference output and do not need R. Run the suite with `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Dependencies

**Core:** StatsModels.jl, GLM.jl, Distributions.jl, StatsBase.jl, StatsAPI.jl, OSQP.jl, ForwardDiff.jl, DifferentiationInterface.jl, PSIS.jl, SpecialFunctions.jl, Tables.jl, Reexport.jl, LinearAlgebra, SparseArrays

**Extensions (loaded on demand):**
- [Turing.jl](https://github.com/TuringLang/Turing.jl) — Bayesian inference
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl) — Visualization

GAMM fitting uses the built-in pure-Julia backend; no MixedModels.jl dependency is required.

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071–1081.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society Series B*, 73(1), 3–36.
- Rigby, R.A. & Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society Series C*, 54(3), 507–554.
- Fasiolo, M., Wood, S.N., Zaffran, M., Nedellec, R., & Goude, Y. (2021). Fast calibrated additive quantile regression. *Journal of the American Statistical Association*, 116(535), 1402–1413.
- Pya, N. & Wood, S.N. (2015). Shape constrained additive models. *Statistics and Computing*, 25(3), 543–559.
- Fasiolo, M., et al. (2025). Scalable smoothing with nested models. *arXiv:2511.19234*.

## Author

[Simon Frost](https://github.com/sdwfrost) ([@sdwfrost](https://github.com/sdwfrost))

## License

MIT
