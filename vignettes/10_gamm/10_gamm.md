# GAMM: Generalized Additive Mixed Models
Simon Frost

- [Introduction](#introduction)
  - [Model specification](#model-specification)
- [Setup](#setup)
- [Example 1: Gaussian GAMM with Random
  Intercepts](#example-1-gaussian-gamm-with-random-intercepts)
  - [Data](#data)
  - [Fitting with `gamm()`](#fitting-with-gamm)
  - [Random effects](#random-effects)
  - [Variance components](#variance-components)
  - [Comparison with true values](#comparison-with-true-values)
  - [Visualizing the Gaussian GAMM](#visualizing-the-gaussian-gamm)
  - [Equivalence with
    `s(subject, bs=:re)`](#equivalence-with-ssubject-bsre)
- [Example 2: Poisson GAMM for Count
  Data](#example-2-poisson-gamm-for-count-data)
  - [Fitting](#fitting)
  - [Random effects](#random-effects-1)
  - [Visualizing the Poisson GAMM](#visualizing-the-poisson-gamm)
- [Example 3: Equivalent Formula
  Interfaces](#example-3-equivalent-formula-interfaces)
  - [Using `@formula` with `(1|group)`](#using-formula-with-1group)
  - [Using `@formula` with `re(group)`](#using-formula-with-regroup)
  - [Consistency with `re(group)` and
    `s(group, bs=:re)`](#consistency-with-regroup-and-sgroup-bsre)
- [Prediction](#prediction)
- [Summary](#summary)

## Introduction

A **Generalized Additive Mixed Model** (GAMM) combines the smooth,
nonlinear covariate effects of a GAM with **grouped random effects** —
random intercepts and slopes that vary across levels of a grouping
factor (e.g., subjects, sites, or time series). GAMMs are essential
when:

- Observations are **clustered** (repeated measures on subjects, spatial
  replicates at sites)
- Between-cluster variability needs to be **explicitly modeled** (not
  just absorbed by smooth terms)
- You want to separate **population-level smooth trends** from
  **cluster-specific deviations**

In penalized regression terms, a random intercept
$b_j \sim N(0, \sigma^2_b)$ is equivalent to adding a ridge penalty on
the group indicator columns — which is precisely what `s(group, bs=:re)`
already does in a standard GAM. GAM.jl’s `gamm()` provides the familiar
mixed-model syntax `(1|group)` as a convenient front-end to this same
machinery.

### Model specification

For a response $y_{ij}$ (observation $i$ in group $j$):

$$g(\mu_{ij}) = \mathbf{x}_i^\top \boldsymbol{\beta} + \sum_k f_k(x_{ik}) + b_j, \qquad b_j \sim N(0, \sigma^2_b)$$

where $g$ is the link function, $f_k$ are smooth functions estimated via
penalized regression splines, and $b_j$ are random intercepts.

## Setup

``` julia
using GAM
using CSV
using DataFrames
using Distributions
using GLM: LogLink, IdentityLink
using Statistics: mean, std, var, cor
using StatsAPI: fitted, deviance, nobs, coef, predict
using Plots
using Printf
```

## Example 1: Gaussian GAMM with Random Intercepts

### Data

Simulated data with 12 subjects, each observed 40 times. The
population-level signal is $\mu(x) = 1.5\sin(1.5x)$, with
subject-specific random intercepts ($\sigma_b = 0.6$) and residual noise
($\sigma_\varepsilon = 0.4$). The dataset is produced by
`vignettes/generate_data.jl` from exactly this process with a fixed
seed. (With only 12 subjects, the *sample* standard deviation of the
drawn intercepts differs noticeably from $\sigma_b$ — estimates should
be compared to the drawn effects in `re_true`, not to the population
value.)

``` julia
dat = CSV.read("data_gaussian_gamm.csv", DataFrame)
println("n = $(nrow(dat)), subjects = $(length(unique(dat.subject)))")
println("y range: [$(round(minimum(dat.y); digits=2)), $(round(maximum(dat.y); digits=2))]")
```

    n = 480, subjects = 12
    y range: [-2.47, 3.06]

### Fitting with `gamm()`

The `(1 | subject)` syntax specifies a random intercept for each subject
level:

``` julia
m = gamm(@formula(y ~ s(x, k=15) + (1 | subject)), dat)
println(m)
```

    ┌ Warning: Random effect grouping variable :subject is numeric (Float64). This will be treated as a categorical grouping variable. If this is intentional, convert to CategoricalArray or String first.
    └ @ GAM ~/Projects/gam/GAM.jl/src/validation.jl:283
    Generalized Additive Mixed Model

    Family: Normal
    Link:   IdentityLink

    Fixed Effects Coefficients:
      β[1] =   0.184662

    Smooth Terms:
      s(x,bs=tp)            edf =  11.61

    Variance Components:
     Group                 Term                      Variance      Std.Dev.    Levels
     ──────────────────────────────────────────────────────────────────────────────
     subject               Intercept                 0.216080      0.464844        12
     Residual                                        0.147337      0.383845          

    Deviance:          67.2705
    REML:             269.2223
    Scale est.:       0.147337
    n = 480

### Random effects

Extract per-subject random intercept estimates (BLUPs) using `ranef()`:

``` julia
re = ranef(m)
est = vec(re.subject.effects)
levels = re.subject.levels
for (lev, eff) in zip(levels, est)
    @printf("  Subject %2s: b̂ = %+.3f\n", lev, eff)
end
```

      Subject 1.0: b̂ = +0.364
      Subject 2.0: b̂ = -0.504
      Subject 3.0: b̂ = -0.609
      Subject 4.0: b̂ = +0.404
      Subject 5.0: b̂ = +0.419
      Subject 6.0: b̂ = -0.363
      Subject 7.0: b̂ = +0.338
      Subject 8.0: b̂ = -0.178
      Subject 9.0: b̂ = -0.199
      Subject 10.0: b̂ = -0.442
      Subject 11.0: b̂ = -0.081
      Subject 12.0: b̂ = +0.849

### Variance components

`VarCorr()` returns the estimated random effect standard deviation:

``` julia
vc = VarCorr(m)
for v in vc
    @printf("  %s: σ = %.4f  (n_levels = %d)\n", v.label, v.std, v.n_levels)
end
residual_scale = m isa GAM.GammModel ? m.gam_model.scale : m.scale
@printf("  Residual: σ = %.4f\n", sqrt(residual_scale))
```

      Intercept: σ = 0.4648  (n_levels = 12)
      Residual: σ = 0.3838  (n_levels = 480)
      Residual: σ = 0.3838

### Comparison with true values

``` julia
true_re = [dat.re_true[findfirst(dat.subject .== s)] for s in sort(unique(dat.subject))]
@printf("Correlation of estimated vs true RE: %.4f\n", cor(est, true_re))
```

    Correlation of estimated vs true RE: 0.9875

### Visualizing the Gaussian GAMM

``` julia
# Population smooth on a prediction grid (unknown subject → zero RE)
x_grid = collect(range(minimum(dat.x), maximum(dat.x); length=200))
pop_pred = predict(m, DataFrame(x=x_grid, subject=fill(999, length(x_grid))))
mu_true_grid = 1.5 .* sin.(1.5 .* x_grid)
subject_levels = sort(unique(dat.subject))

p1 = scatter(dat.x, dat.y; group=dat.subject, markersize=2, alpha=0.45,
    xlabel="x", ylabel="y", title="Gaussian GAMM: data by subject",
    label="")
for (i, subj) in enumerate(subject_levels)
    subj_pred = predict(m, DataFrame(x=x_grid, subject=fill(subj, length(x_grid))))
    plot!(p1, x_grid, subj_pred; color=i, alpha=0.35, linewidth=1.25,
        label=i == 1 ? "subject-specific fits" : "")
end
plot!(p1, x_grid, pop_pred; color=:black, linewidth=3, linestyle=:dash, label="population mean")
plot!(p1, x_grid, mu_true_grid; color=:red, linewidth=2, linestyle=:dot, label="true population mean")

n_groups = length(levels)
p2 = bar(1:n_groups, [est true_re]; label=["Estimated" "True"], legend=:topright,
    xlabel="Subject", ylabel="Random intercept",
    title="Random intercepts: estimated vs true",
    xticks=(1:n_groups, string.(levels)))

plot(p1, p2; layout=(1, 2), size=(900, 400))
```

![](10_gamm_files/figure-commonmark/cell-8-output-1.svg)

### Equivalence with `s(subject, bs=:re)`

In GAM.jl, `gamm(@formula(y ~ s(x) + (1|subject)), ...)` is
mathematically equivalent to
`gam(@formula(y ~ s(x) + s(subject, bs=:re)), ...)`. Both treat the
random intercept as a smooth with identity penalty:

``` julia
m_gam = gam(@formula(y ~ s(x, k=15) + s(subject, bs=:re)), dat)
@printf("Fitted values correlation: %.6f\n", cor(fitted(m), fitted(m_gam)))
@printf("Scale (gamm): %.6f\n", m.gam_model.scale)
@printf("Scale (gam):  %.6f\n", m_gam.scale)
```

    Fitted values correlation: 1.000000
    Scale (gamm): 0.147337
    Scale (gam):  0.147337

## Example 2: Poisson GAMM for Count Data

Count data with 8 sites, each observed 60 times. The true log-rate has a
smooth trend plus site-specific random intercepts ($\sigma_b = 0.4$;
again generated by `vignettes/generate_data.jl`, and with only 8 sites
the sample spread of the drawn $b_j$ can differ substantially from
$\sigma_b$):

$$\log(\lambda_{ij}) = 1 + 0.8\sin(x_i) + b_j, \qquad b_j \sim N(0, 0.16)$$

``` julia
dat2 = CSV.read("data_poisson_gamm.csv", DataFrame)
println("n = $(nrow(dat2)), sites = $(length(unique(dat2.site)))")
println("y range: [$(minimum(dat2.y)), $(maximum(dat2.y))]")
```

    n = 480, sites = 8
    y range: [0.0, 16.0]

### Fitting

Pass `Poisson()` as the family — the canonical `LogLink` is used
automatically:

``` julia
m2 = gamm(@formula(y ~ s(x, k=15) + (1 | site)), dat2, Poisson())
println(m2)
```

    ┌ Warning: Random effect grouping variable :site is numeric (Float64). This will be treated as a categorical grouping variable. If this is intentional, convert to CategoricalArray or String first.
    └ @ GAM ~/Projects/gam/GAM.jl/src/validation.jl:283
    Generalized Additive Mixed Model

    Family: Poisson
    Link:   LogLink

    Fixed Effects Coefficients:
      β[1] =   0.930386

    Smooth Terms:
      s(x,bs=tp)            edf =   6.82

    Variance Components:
     Group                 Term                      Variance      Std.Dev.    Levels
     ──────────────────────────────────────────────────────────────────────────────
     site                  Intercept                 0.104307      0.322967         8
     Residual                                        0.980400      0.990151          

    Deviance:         494.4476
    REML:             476.0931
    Scale est.:       0.980400
    n = 480

### Random effects

``` julia
re2 = ranef(m2)
est2 = vec(re2.site.effects)
true_re2 = [dat2.re_true[findfirst(dat2.site .== s)] for s in sort(unique(dat2.site))]
@printf("RE correlation with truth: %.4f\n", cor(est2, true_re2))

vc2 = VarCorr(m2)
sd_drawn = std([dat2.re_true[findfirst(dat2.site .== s)] for s in sort(unique(dat2.site))])
@printf("Estimated σ_RE: %.4f (population σ_b: 0.4; sd of the 8 drawn effects: %.4f)\n",
    vc2[1].std, sd_drawn)
```

    RE correlation with truth: 0.9201
    Estimated σ_RE: 0.3230 (population σ_b: 0.4; sd of the 8 drawn effects: 0.3575)

### Visualizing the Poisson GAMM

``` julia
# Population smooth on prediction grid
x_grid2 = collect(range(minimum(dat2.x), maximum(dat2.x); length=200))
pop_pred2 = predict(m2, DataFrame(x=x_grid2, site=fill(999, length(x_grid2))))
lambda_true_grid = exp.(1 .+ 0.8 .* sin.(x_grid2))

site_levels = sort(unique(dat2.site))
p1 = scatter(dat2.x, dat2.y; group=dat2.site, markersize=2, alpha=0.45,
    xlabel="x", ylabel="y (count)", title="Poisson GAMM: data by site",
    label="")
for (i, site) in enumerate(site_levels)
    site_pred = predict(m2, DataFrame(x=x_grid2, site=fill(site, length(x_grid2))))
    plot!(p1, x_grid2, site_pred; color=i, alpha=0.35, linewidth=1.25,
        label=i == 1 ? "site-specific fits" : "")
end
plot!(p1, x_grid2, pop_pred2; color=:black, linewidth=3, linestyle=:dash, label="population mean")
plot!(p1, x_grid2, lambda_true_grid; color=:red, linewidth=2, linestyle=:dot, label="true population mean")

n_sites = length(site_levels)
p2 = bar(1:n_sites, [est2 true_re2]; label=["Estimated" "True"], legend=:topright,
    xlabel="Site", ylabel="Random intercept (log-scale)",
    title="Site random effects: estimated vs true",
    xticks=(1:n_sites, string.(site_levels)))

plot(p1, p2; layout=(1, 2), size=(900, 400))
```

![](10_gamm_files/figure-commonmark/cell-13-output-1.svg)

## Example 3: Equivalent Formula Interfaces

GAM.jl supports multiple ways to specify random effects:

### Using `@formula` with `(1|group)`

StatsModels.jl parses `(1|group)` as a `FunctionTerm{typeof(|)}`, which
`gamm()` detects automatically:

``` julia
_scale(m) = m isa GAM.GammModel ? m.gam_model.scale : m.scale
m3a = gamm(@formula(y ~ cr(x, 15) + (1|subject)), dat)
@printf("@formula path: scale = %.6f\n", _scale(m3a))
```

    ┌ Warning: Random effect grouping variable :subject is numeric (Float64). This will be treated as a categorical grouping variable. If this is intentional, convert to CategoricalArray or String first.
    └ @ GAM ~/Projects/gam/GAM.jl/src/validation.jl:283
    @formula path: scale = 0.147337

### Using `@formula` with `re(group)`

The `re()` convenience function provides an alternative syntax:

``` julia
m3b = gamm(@formula(y ~ cr(x, 15) + re(subject)), dat)
@printf("re() path: scale = %.6f\n", _scale(m3b))
```

    ┌ Warning: Random effect grouping variable :subject is numeric (Float64). This will be treated as a categorical grouping variable. If this is intentional, convert to CategoricalArray or String first.
    └ @ GAM ~/Projects/gam/GAM.jl/src/validation.jl:283
    re() path: scale = 0.147337

### Consistency with `re(group)` and `s(group, bs=:re)`

These formula forms produce equivalent fits (agreeing up to convergence
tolerance, so fitted-value correlations are ≈ 1 and scales agree
closely):

``` julia
m3c = gam(@formula(y ~ s(x, k=15) + s(subject, bs=:re)), dat)
@printf("cor((1|subject), re(subject)): %.6f\n", cor(fitted(m3a), fitted(m3b)))
@printf("cor((1|subject), s(subject, bs=:re)): %.6f\n", cor(fitted(m3a), fitted(m3c)))
```

    cor((1|subject), re(subject)): 1.000000
    cor((1|subject), s(subject, bs=:re)): 1.000000

## Prediction

Predict on new data — unknown group levels get zero random effect
contribution:

``` julia
# Known subjects
df_known = DataFrame(x=[0.0, 0.5, 1.0], subject=[1, 2, 3])
pred_known = predict(m, df_known)
@printf("Predictions (known subjects): [%.3f, %.3f, %.3f]\n", pred_known...)

# New subject (never seen)
df_new = DataFrame(x=[0.0, 0.5, 1.0], subject=[999, 999, 999])
pred_new = predict(m, df_new)
@printf("Predictions (new subject):    [%.3f, %.3f, %.3f]\n", pred_new...)
```

    Predictions (known subjects): [0.511, 0.706, 1.070]
    Predictions (new subject):    [0.147, 1.210, 1.679]

## Summary

| Feature             | Syntax                                            |
|---------------------|---------------------------------------------------|
| Random intercept    | `(1 \| group)` or `re(group)`                     |
| Gaussian family     | `gamm(formula, data)`                             |
| Non-Gaussian        | `gamm(formula, data, Poisson())`                  |
| Equivalent GAM      | `gam(@formula(y ~ s(x) + s(group, bs=:re)), ...)` |
| Random effects      | `ranef(m)`                                        |
| Variance components | `VarCorr(m)`                                      |
| Prediction          | `predict(m, newdata)`                             |

GAM.jl’s `gamm()` delegates to the same proven PIRLS+REML machinery used
by `gam()`, treating random effects as smooth terms with identity
penalty matrices. This means GAMM results are numerically equivalent to
the corresponding `s(group, bs=:re)` formulation, but with the
convenience of mixed-model notation.
