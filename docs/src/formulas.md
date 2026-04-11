# [Formula Syntax](@id formula-syntax)

GAM.jl uses `@formulak` for GAM-specific formula syntax. When you import
`@formula` from GAM, keyword smooth calls are automatically diverted to
`@formulak`, while ordinary StatsModels formulas continue to use the standard
`@formula` path.

If another package also exports `@formula` (for example `GLM`), qualify it as
`GAM.@formula(...)` or import it explicitly with `using GAM: @formula`.

## The `@formulak` Macro

`@formulak` supports keyword arguments in `s()`, `te()`, and `ti()` calls:

```julia
@formula(y ~ 1 + s(x, k=15, bs=:cr))   # auto-diverts to @formulak
@formulak(y ~ 1 + s(x, k=15, bs=:cr))
@formulak(y ~ s(x1) + s(x2, k=20))
@formulak(y ~ x1 + s(x2, k=10, bs=:ps))
@formulak(y ~ s(x1, bs=:cr) + s(x2, bs=:cr) + ti(x1, x2, k=5))
```

## Components

### Parametric Terms

Standard linear terms work as in StatsModels:

```julia
@formulak(y ~ 1 + x1 + x2)    # intercept + two linear effects
```

An intercept is included by default.

### Smooth Terms

Smooth terms are specified with `s()`:

```julia
@formulak(y ~ s(x))                    # default TPRS smooth
@formulak(y ~ s(x, k=20, bs=:cr))      # CR spline, k=20
@formulak(y ~ s(x, by=:group))         # varying coefficient
```

### Tensor Products

For smooth interactions between variables:

```julia
@formulak(y ~ te(x1, x2))              # tensor product
@formulak(y ~ s(x1) + s(x2) + ti(x1, x2))  # ANOVA decomposition
```

## Comparison with R's mgcv

| R mgcv | GAM.jl |
|--------|--------|
| `y ~ s(x)` | `@formulak(y ~ s(x))` |
| `y ~ s(x, k=20, bs="cr")` | `@formulak(y ~ s(x, k=20, bs=:cr))` |
| `y ~ te(x1, x2)` | `@formulak(y ~ te(x1, x2))` |
| `y ~ s(x, by=group)` | `@formulak(y ~ s(x, by=:group))` |

Key differences:
- Basis types use Julia symbols (`:cr`) instead of R strings (`"cr"`)
- GAM's `@formula` keeps ordinary StatsModels behavior but routes keyword smooths to `@formulak`
- Variable names are symbols (`:x`) rather than bare names

## API Reference

See [API Reference](@ref api-reference) for full documentation of `GamFormula`
and `@formulak`.
