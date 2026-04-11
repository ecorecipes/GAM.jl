# [Formula Syntax](@id formula-syntax)

GAM.jl's public formula interface is `@formula`. Ordinary StatsModels formulas
keep the usual behavior, while GAM smooth terms with keyword arguments — and
GAMM random effects like `(1 | group)` — are automatically routed to GAM.jl's
extended parser. `@formulak` remains available when you want to opt into the
GAM-specific path explicitly.

If another package also exports `@formula` (for example `GLM`), qualify it as
`GAM.@formula(...)` or import it explicitly with `using GAM: @formula`.

```@setup formulas
using GAM
```

## `@formula` and `@formulak`

```@example formulas
@formula(y ~ 1 + s(x, k=15, bs=:cr));
@formulak(y ~ 1 + s(x, k=15, bs=:cr));
@formula(y ~ s(x1) + s(x2, k=20));
@formula(y ~ x1 + s(x2, k=10, bs=:ps));
@formula(y ~ s(x1, bs=:cr) + s(x2, bs=:cr) + ti(x1, x2, k=5));
nothing
```

## Components

### Parametric Terms

Standard linear terms work as in StatsModels:

```@example formulas
@formula(y ~ 1 + x1 + x2);    # intercept + two linear effects
nothing
```

An intercept is included by default.

### Smooth Terms

Smooth terms are specified with `s()`:

```@example formulas
@formula(y ~ s(x));                    # default TPRS smooth
@formula(y ~ s(x, k=20, bs=:cr));      # CR spline, k=20
@formula(y ~ s(x, by=:group));         # varying coefficient
nothing
```

### Tensor Products

For smooth interactions between variables:

```@example formulas
@formula(y ~ te(x1, x2));              # tensor product
@formula(y ~ s(x1) + s(x2) + ti(x1, x2));  # ANOVA decomposition
nothing
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
- GAM's `@formula` keeps ordinary StatsModels behavior but routes keyword smooths
  and GAMM random effects to GAM.jl's extended formula parser
- Variable names are symbols (`:x`) rather than bare names

## API Reference

See [API Reference](@ref api-reference) for full documentation of `GAM.@formula`,
`GAM.@formulak`, and the corresponding formula types.
