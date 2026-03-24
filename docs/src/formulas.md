# [Formula Syntax](@id formula-syntax)

GAM.jl uses the `@gam_formula` macro to specify models. This extends
StatsModels.jl's formula syntax with smooth term support.

## The `@gam_formula` Macro

Unlike `@formula`, `@gam_formula` supports keyword arguments in `s()`, `te()`,
and `ti()` calls:

```julia
@gam_formula(y ~ 1 + s(x, k=15, bs=:cr))
@gam_formula(y ~ s(x1) + s(x2, k=20))
@gam_formula(y ~ x1 + s(x2, k=10, bs=:ps))
@gam_formula(y ~ s(x1, bs=:cr) + s(x2, bs=:cr) + ti(x1, x2, k=5))
```

## Components

### Parametric Terms

Standard linear terms work as in StatsModels:

```julia
@gam_formula(y ~ 1 + x1 + x2)    # intercept + two linear effects
```

An intercept is included by default.

### Smooth Terms

Smooth terms are specified with `s()`:

```julia
@gam_formula(y ~ s(x))                    # default TPRS smooth
@gam_formula(y ~ s(x, k=20, bs=:cr))      # CR spline, k=20
@gam_formula(y ~ s(x, by=:group))         # varying coefficient
```

### Tensor Products

For smooth interactions between variables:

```julia
@gam_formula(y ~ te(x1, x2))              # tensor product
@gam_formula(y ~ s(x1) + s(x2) + ti(x1, x2))  # ANOVA decomposition
```

## Comparison with R's mgcv

| R mgcv | GAM.jl |
|--------|--------|
| `y ~ s(x)` | `@gam_formula(y ~ s(x))` |
| `y ~ s(x, k=20, bs="cr")` | `@gam_formula(y ~ s(x, k=20, bs=:cr))` |
| `y ~ te(x1, x2)` | `@gam_formula(y ~ te(x1, x2))` |
| `y ~ s(x, by=group)` | `@gam_formula(y ~ s(x, by=:group))` |

Key differences:
- Basis types use Julia symbols (`:cr`) instead of R strings (`"cr"`)
- The `@gam_formula` macro replaces R's built-in formula support
- Variable names are symbols (`:x`) rather than bare names

## API Reference

See [API Reference](@ref api-reference) for full documentation of `GamFormula`
and `@gam_formula`.
