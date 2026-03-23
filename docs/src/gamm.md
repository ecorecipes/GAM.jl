# Mixed Models (GAMM)

Generalized Additive Mixed Models (GAMM) combine smooth terms with random
effects for hierarchical/grouped data. GAM.jl's `gamm()` function provides
mixed-model GAMs with a formula syntax that integrates both smooth and
random effects.

## When to Use GAMM

- Longitudinal / panel data with subject-level random effects
- Multilevel data with group-level intercepts or slopes
- When you need explicit random effect estimates (BLUPs)

For simpler cases, `s(:group, bs=:re)` or `s(:x, :group, bs=:fs)` in a
standard `gam()` call may suffice.

## Interface

```julia
gamm(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,
)
```

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

### Random Intercept

```julia
using GAM, DataFrames

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

### Inspecting Random Effects

```julia
# Random effect estimates (BLUPs)
ranef(m)

# Variance components
VarCorr(m)
```

### Random Slopes

```julia
m = gamm(
    @gamm_formula(y ~ s(x, k=15, bs=:cr) + (1 + x | subject)),
    df,
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
ranef(m)      # random effect estimates
VarCorr(m)    # variance components
```

## See Also

- [Getting Started](@ref) for a quick example
- [Smooth Terms](@ref) — `bs=:re` and `bs=:fs` for simpler random effect smooths
- [API Reference](@ref) for `gamm`, `GammModel`, `@gamm_formula`, `ranef`, `VarCorr`
