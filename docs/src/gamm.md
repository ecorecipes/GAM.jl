# [Mixed Models (GAMM)](@id gamm)

Generalized Additive Mixed Models (GAMM) combine smooth terms with random
effects for hierarchical/grouped data. GAM.jl's `gamm()` function provides
mixed-model GAMs with a formula syntax that integrates both smooth and
random effects.

## When to Use GAMM

- Longitudinal / panel data with subject-level random effects
- Multilevel data with group-level intercepts or slopes
- When you need explicit random effect estimates (BLUPs)
- Non-Gaussian responses with grouped data (Poisson GAMM, Binomial GAMM)

For simpler cases, `s(:group, bs=:re)` or `s(:x, :group, bs=:fs)` in a
standard `gam()` call may suffice. Use `gamm()` when you need the full
mixed-model machinery (variance components, BLUPs, crossed random effects).

## Interface

```julia
gamm(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,
)
```

For non-Gaussian families, `gamm()` automatically uses Penalized
Quasi-Likelihood (PQL), matching R's `mgcv::gamm()` which calls
`MASS::glmmPQL` internally.

## The `@gamm_formula` Macro

`@gamm_formula` extends `@gam_formula` with lme4-style random effects syntax:

```julia
# Random intercept for subject
@gamm_formula(y ~ s(x, k=10) + (1 | subject))

# Random intercept and slope
@gamm_formula(y ~ s(x, k=10) + (1 + x | subject))

# Crossed random effects
@gamm_formula(y ~ s(x) + (1 | subject) + (1 | item))
```

## Examples

### Random Intercept (Gaussian)

The most common GAMM: a smooth trend plus subject-specific intercepts.
Equivalent to R's `gamm(y ~ s(x), random=list(subject=~1))`.

```julia
using GAM, DataFrames, Random, Distributions
Random.seed!(42)

n_subjects = 20
n_per = 50
n = n_subjects * n_per

subject = repeat(1:n_subjects; inner=n_per)
x = repeat(range(0, 2π; length=n_per); outer=n_subjects) |> collect
re = randn(n_subjects)
y = sin.(x) .+ re[subject] .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y, subject=string.(subject))

m = gamm(@gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)), df)
```

### Extracting Random Effects with `ranef()`

`ranef()` returns the Best Linear Unbiased Predictors (BLUPs) for each
random effect grouping factor:

```julia
# Random effect estimates (BLUPs) — returns a Dict of grouping factor => values
re_estimates = ranef(m)
```

Each key in the returned dictionary corresponds to a grouping factor (e.g.,
`"subject"`), and the values are the estimated random effects for each level.

### Variance Components with `VarCorr()`

`VarCorr()` extracts variance and correlation parameters for all random
effects, equivalent to R's `VarCorr()` from nlme/lme4:

```julia
vc = VarCorr(m)
```

This returns a `VarCorrResult` showing the estimated variance (and standard
deviation) for each random effect term, plus the residual variance.

### Random Slopes

Model subject-specific linear trends alongside a population-level smooth.
Equivalent to R's `gamm(y ~ s(x), random=list(subject=~1+x))`:

```julia
re_slope = 0.3 .* randn(n_subjects)
y_slope = sin.(x) .+ re[subject] .+ re_slope[subject] .* x .+ 0.3 .* randn(n)
df_slope = DataFrame(x=x, y=y_slope, subject=string.(subject))

m_slope = gamm(
    @gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 + x | subject)),
    df_slope,
)

# Inspect the variance-covariance of random effects
VarCorr(m_slope)
```

### Crossed Random Effects

When observations are grouped by two or more non-nested factors:

```julia
n_items = 10
item = repeat(string.(1:n_items); outer=n ÷ n_items)
re_item = 0.5 .* randn(n_items)
y_crossed = sin.(x) .+ re[subject] .+ re_item[parse.(Int, item)] .+ 0.3 .* randn(n)
df_crossed = DataFrame(x=x, y=y_crossed, subject=string.(subject), item=item)

m_crossed = gamm(
    @gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 | subject) + (1 | item)),
    df_crossed,
)

ranef(m_crossed)    # BLUPs for both subject and item
VarCorr(m_crossed)  # variance components for both grouping factors
```

### Poisson GAMM (PQL)

For count data with grouped structure. `gamm()` automatically switches to
Penalized Quasi-Likelihood (PQL) for non-Gaussian families. Equivalent to
R's `gamm(y ~ s(x), family=poisson, random=list(subject=~1))`:

```julia
mu_pois = exp.(0.5 .* sin.(x) .+ re[subject] .+ 1.0)
y_pois = Float64.([rand(Poisson(m)) for m in mu_pois])
df_pois = DataFrame(x=x, y=y_pois, subject=string.(subject))

m_pois = gamm(
    @gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)),
    df_pois;
    family=Poisson(),
    link=LogLink(),
)

coef(m_pois)       # fixed effects on log scale
ranef(m_pois)      # subject-level random intercepts
VarCorr(m_pois)    # random effect variance
```

!!! note "PQL Estimation"
    PQL iterates between GAM fitting on working responses and random effect
    estimation via generalized BLUP. It matches R's `MASS::glmmPQL` used
    internally by `mgcv::gamm()`. PQL is reliable for moderate random effect
    variances but can be biased when variance components are large.

### Binomial GAMM

For binary outcomes with grouped data:

```julia
p_bin = 1.0 ./ (1.0 .+ exp.(-1.0 .* sin.(x) .- re[subject]))
y_bin = Float64.([rand(Bernoulli(p)) for p in p_bin])
df_bin = DataFrame(x=x, y=y_bin, subject=string.(subject))

m_bin = gamm(
    @gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)),
    df_bin;
    family=Binomial(),
    link=LogitLink(),
)
```

## GammModel

`gamm()` returns a `GammModel` object containing:

- `.gam` — the underlying GAM model
- `.lme` — the mixed model component
- Standard StatsBase methods work on the `GammModel` directly

```julia
coef(m)       # fixed effect coefficients
vcov(m)       # variance-covariance of fixed effects
ranef(m)      # random effect estimates (BLUPs)
VarCorr(m)    # variance components
deviance(m)   # model deviance
nobs(m)       # number of observations
```

## GAMM vs `gam()` with `bs=:re`

For simple random intercepts, you can use `s(:group, bs=:re)` in a standard
`gam()` call. The approaches are mathematically equivalent but differ in
interface:

```julia
# Using gam() with random effect smooth — simpler, no BLUPs
m_re = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(subject, bs=:re)), df)

# Using gamm() — gives BLUPs, VarCorr, mixed model diagnostics
m_gamm = gamm(@gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)), df)
```

Use `gamm()` when you need random effect predictions, variance components,
or non-Gaussian PQL estimation.

## See Also

- [Getting Started](@ref getting-started) for a quick example
- [Smooth Terms](@ref smooth-terms) — `bs=:re` and `bs=:fs` for simpler random effect smooths
- [API Reference](@ref api-reference) for `gamm`, `GammModel`, `@gamm_formula`, `ranef`, `VarCorr`
