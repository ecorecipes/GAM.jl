# [Formula Syntax](@id formula-syntax)

GAM.jl's public formula interface is `@formula`. It covers ordinary linear
terms, smooth constructors with keyword arguments, and GAMM random effects such
as `(1 | group)` and `re(group)`.

If another package also exports `@formula` (for example `GLM`), qualify it as
`GAM.@formula(...)` or import it explicitly with `using GAM: @formula`.

```@setup formulas
using GAM
```

## `@formula`

```@example formulas
f1 = @formula(y ~ 1 + s(x, k=15, bs=:cr))
f2 = @formula(y ~ s(x1) + s(x2, k=20))
f3 = @formula(y ~ x1 + s(x2, k=10, bs=:ps))
f4 = @formula(y ~ s(x1, bs=:cr) + s(x2, bs=:cr) + ti(x1, x2, k=5))
[f1, f2, f3, f4]
```

## Components

### Parametric Terms

Standard linear terms work as in StatsModels:

```@example formulas
@formula(y ~ 1 + x1 + x2)    # intercept + two linear effects
```

An intercept is included by default.

### Smooth Terms

Smooth terms are specified with `s()`:

```@example formulas
(
    @formula(y ~ s(x)),                    # default TPRS smooth
    @formula(y ~ s(x, k=20, bs=:cr)),      # CR spline, k=20
    @formula(y ~ s(x, by=:group)),         # varying coefficient
)
```

### Tensor Products

For smooth interactions between variables:

```@example formulas
(
    @formula(y ~ te(x1, x2)),                  # tensor product
    @formula(y ~ s(x1) + s(x2) + ti(x1, x2)), # ANOVA decomposition
)
```

## Comparison with R's mgcv

| R mgcv | GAM.jl |
|--------|--------|
| `y ~ s(x)` | `@formula(y ~ s(x))` |
| `y ~ s(x, k=20, bs="cr")` | `@formula(y ~ s(x, k=20, bs=:cr))` |
| `y ~ te(x1, x2)` | `@formula(y ~ te(x1, x2))` |
| `y ~ s(x, by=group)` | `@formula(y ~ s(x, by=:group))` |

Key differences:
- Basis types use Julia symbols (`:cr`) instead of R strings (`"cr"`)
- The same `@formula` surface covers linear terms, smooths, and GAMM random effects
- If another package exports `@formula`, qualify it as `GAM.@formula(...)`

## API Reference

See [API Reference](@ref api-reference) for full documentation of `GAM.@formula`
and the corresponding formula types.
