# [Extreme Values (evgam)](@id evgam)

GAM.jl includes extreme value GAM functionality following Youngman (2022).
This allows modeling the tails of distributions — useful in hydrology,
meteorology, finance, and other fields where extreme events matter.

## Families

### GEV (Generalized Extreme Value)

For block maxima data. Three parameters: location (μ), scale (σ), shape (ξ).

```julia
evgam(formula_mu, formula_sigma, formula_xi, data;
    family=GEVFamily())
```

### GPD (Generalized Pareto Distribution)

For threshold exceedances. Two parameters: scale (σ), shape (ξ).

```julia
evgam(formula_sigma, formula_xi, data;
    family=GPDFamily())
```

### Extended GPD (EGPD)

Extensions of the GPD with additional flexibility:

| Family | Description |
|--------|-------------|
| `EGPD1Family()` | Extended GPD model 1 |
| `EGPD2Family()` | Extended GPD model 2 |
| `EGPD3Family()` | Extended GPD model 3 |
| `EGPD4Family()` | Extended GPD model 4 |

## Multi-Parameter Model Specification

Pass a vector of formulas, one per distribution parameter. With
`@gam_formula`, repeat the response on each entry:

```julia
# GEV: location, scale, shape
m = evgam(
    [
        @gam_formula(y ~ s(x, k=15, bs=:cr)),   # location μ
        @gam_formula(y ~ s(x, k=10, bs=:cr)),   # log(scale σ)
        @gam_formula(y ~ 1),                    # shape ξ (constant)
    ],
    df,
    GEVFamily(),
)
```

## Examples

### GEV for Annual Maxima

```julia
using GAM, DataFrames

n = 200
year = range(1, n) |> collect
# Non-stationary GEV: location trend, constant scale and shape
mu = 10.0 .+ 0.05 .* year
sigma = 2.0
xi = 0.1
y = mu .+ sigma .* ((-log.(rand(n))).^(-xi) .- 1) ./ xi
df = DataFrame(year=year, y=y)

m = evgam(
    [
        @gam_formula(y ~ s(year, k=10, bs=:cr)),  # location
        @gam_formula(y ~ 1),                      # scale
        @gam_formula(y ~ 1),                      # shape
    ],
    df,
    GEVFamily(),
)
```

### GPD for Threshold Exceedances

```julia
# Exceedances above a threshold
threshold = 10.0
x = randn(300)
sigma = exp.(0.5 .+ 0.3 .* x)
xi = 0.1
y = sigma .* ((rand(300)).^(-xi) .- 1) ./ xi
df = DataFrame(x=x, y=y)

m = evgam(
    [
        @gam_formula(y ~ s(x, k=10, bs=:cr)),  # log(scale)
        @gam_formula(y ~ 1),                   # shape
    ],
    df,
    GPDFamily(),
)
```

### Non-Stationary GEV

All three GEV parameters as smooth functions:

```julia
m = evgam(
    [
        @gam_formula(y ~ s(year, k=15, bs=:cr)),
        @gam_formula(y ~ s(year, k=10, bs=:cr)),
        @gam_formula(y ~ s(year, k=5, bs=:cr)),
    ],
    df,
    GEVFamily(),
)
```

## See Also

- [Families & Models](@ref families) for evgam family types
- [API Reference](@ref api-reference) for `evgam`, `GEVFamily`, `GPDFamily`, `MultiParameterFamily`
