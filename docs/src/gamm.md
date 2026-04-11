# [Mixed Models (GAMM)](@id gamm)

Generalized Additive Mixed Models (GAMM) combine smooth terms with random
effects for hierarchical/grouped data. GAM.jl's `gamm()` function provides
mixed-model GAMs with a formula syntax that integrates both smooth and
random effects.

```@setup gamm
using GAM, DataFrames, Random, Distributions
using GLM: LogLink, LogitLink
Random.seed!(42)

n_subjects = 12
n_per = 12
n = n_subjects * n_per

subject = repeat(1:n_subjects; inner=n_per)
x = repeat(range(0, 2π; length=n_per); outer=n_subjects) |> collect
re = 0.6 .* randn(n_subjects)
y = sin.(x) .+ re[subject] .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y, subject=string.(subject))
```

## When to Use GAMM

- Longitudinal / panel data with subject-level random effects
- Multilevel data with group-level intercepts or slopes
- When you need explicit random effect estimates (BLUPs)
- Non-Gaussian responses with grouped data (Poisson GAMM, Binomial GAMM)

For simpler cases, `s(:group, bs=:re)` or `s(:x, :group, bs=:fs)` in a
standard `gam()` call may suffice. Use `gamm()` when you need the full
mixed-model machinery (variance components, BLUPs, crossed random effects).

## Interface

```text
gamm(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,
)
```

For non-Gaussian families, `gamm()` automatically uses Penalized
Quasi-Likelihood (PQL), matching R's `mgcv::gamm()` which calls
`MASS::glmmPQL` internally.

## Recommended Formula Syntax

Use `GAM.@formula(...)` for GAMM models. It supports the same keyword smooth
syntax as `@formulak`, plus lme4-style random effects:

```@example gamm
GAM.@formula(y ~ s(x, k=10) + (1 | subject));
GAM.@formula(y ~ s(x, k=10, bs=:cr) + (1 + x | subject));
GAM.@formula(y ~ s(x, k=10, bs=:cr) + (1 | subject) + (1 | item));
nothing
```

`@gamm_formula` remains a compatibility alias, but new code should prefer
`GAM.@formula(...)`.

## Examples

### Random Intercept (Gaussian)

The most common GAMM: a smooth trend plus subject-specific intercepts.
Equivalent to R's `gamm(y ~ s(x), random=list(subject=~1))`.

```@example gamm
m = gamm(GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)), df);
nothing
```

### Extracting Random Effects with `ranef()`

`ranef()` returns the Best Linear Unbiased Predictors (BLUPs) for each
random effect grouping factor:

```@example gamm
re_estimates = ranef(m);
nothing
```

Each key in the returned dictionary corresponds to a grouping factor (e.g.,
`"subject"`), and the values are the estimated random effects for each level.

### Variance Components with `VarCorr()`

`VarCorr()` extracts variance and correlation parameters for all random
effects, equivalent to R's `VarCorr()` from nlme/lme4:

```@example gamm
vc = VarCorr(m);
nothing
```

This returns a `VarCorrResult` showing the estimated variance (and standard
deviation) for each random effect term, plus the residual variance.

### Random Slopes

Model subject-specific linear trends alongside a population-level smooth.
Equivalent to R's `gamm(y ~ s(x), random=list(subject=~1+x))`:

```@example gamm
re_slope = 0.15 .* randn(n_subjects)
y_slope = sin.(x) .+ re[subject] .+ re_slope[subject] .* x .+ 0.3 .* randn(n)
df_slope = DataFrame(x=x, y=y_slope, subject=string.(subject))

m_slope = gamm(
    GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 + x | subject)),
    df_slope,
);

VarCorr(m_slope);
nothing
```

### Crossed Random Effects

When observations are grouped by two or more non-nested factors:

```@example gamm
n_items = 6
item = repeat(string.(1:n_items); inner=n ÷ n_items)
re_item = 0.4 .* randn(n_items)
y_crossed = sin.(x) .+ re[subject] .+ re_item[parse.(Int, item)] .+ 0.3 .* randn(n)
df_crossed = DataFrame(x=x, y=y_crossed, subject=string.(subject), item=item)

m_crossed = gamm(
    GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 | subject) + (1 | item)),
    df_crossed,
);

ranef(m_crossed);
VarCorr(m_crossed);
nothing
```

### Poisson GAMM (PQL)

For count data with grouped structure. `gamm()` automatically switches to
Penalized Quasi-Likelihood (PQL) for non-Gaussian families. Equivalent to
R's `gamm(y ~ s(x), family=poisson, random=list(subject=~1))`:

```@example gamm
mu_pois = exp.(0.3 .* sin.(x) .+ re[subject] .+ 1.0)
y_pois = Float64.(rand.(Poisson.(mu_pois)))
df_pois = DataFrame(x=x, y=y_pois, subject=string.(subject))

m_pois = gamm(
    GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)),
    df_pois;
    family=Poisson(),
    link=LogLink(),
);

GAM.coef(m_pois);
ranef(m_pois);
VarCorr(m_pois);
nothing
```

!!! note "PQL Estimation"
    PQL iterates between GAM fitting on working responses and random effect
    estimation via generalized BLUP. It matches R's `MASS::glmmPQL` used
    internally by `mgcv::gamm()`. PQL is reliable for moderate random effect
    variances but can be biased when variance components are large.

### Binomial GAMM

For binary outcomes with grouped data:

```@example gamm
p_bin = 1.0 ./ (1.0 .+ exp.(-1.0 .* sin.(x) .- re[subject]))
y_bin = Float64.(rand.(Bernoulli.(p_bin)))
df_bin = DataFrame(x=x, y=y_bin, subject=string.(subject))

m_bin = gamm(
    GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)),
    df_bin;
    family=Binomial(),
    link=LogitLink(),
);
nothing
```

## GammModel

`gamm()` returns a `GammModel` object containing:

- `.gam_model` — the underlying `GamModel` for the fixed and smooth terms
- `.random_effects` / `.random_coefs` / `.random_vars` — the grouped random-effect structure
- Standard StatsAPI methods work on the `GammModel` directly

```@example gamm
GAM.coef(m);
GAM.vcov(m);
ranef(m);
VarCorr(m);
GAM.deviance(m);
GAM.nobs(m);
nothing
```

## GAMM vs `gam()` with `bs=:re`

For simple random intercepts, you can use `s(:group, bs=:re)` in a standard
`gam()` call. The approaches are mathematically equivalent but differ in
interface:

```@example gamm
m_re = gam(GAM.@formula(y ~ s(x, k=15, bs=:cr) + s(subject, bs=:re)), df);
m_gamm = gamm(GAM.@formula(y ~ s(x, k=15, bs=:cr) + (1 | subject)), df);
nothing
```

Use `gamm()` when you need random effect predictions, variance components,
or non-Gaussian PQL estimation.

## See Also

- [Getting Started](@ref getting-started) for a quick example
- [Smooth Terms](@ref smooth-terms) — `bs=:re` and `bs=:fs` for simpler random effect smooths
- [API Reference](@ref api-reference) for `gamm`, `GammModel`, `GAM.@formula`, `ranef`, `VarCorr`
