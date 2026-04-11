# [Extreme Values (evgam)](@id evgam)

GAM.jl includes extreme value GAM functionality following Youngman (2022).
This allows modeling the tails of distributions — useful in hydrology,
meteorology, finance, and other fields where extreme events matter.

```@setup evgam
using GAM, DataFrames, Random
Random.seed!(42)

n = 90
year = collect(1:n)
mu = 10.0 .+ 0.03 .* year
sigma = 1.5
xi = 0.1
y = mu .+ sigma .* ((-log.(rand(n))).^(-xi) .- 1) ./ xi
df = DataFrame(year=year, y=y)

x = randn(120)
sigma_gpd = exp.(0.4 .+ 0.2 .* x)
xi_gpd = 0.1
y_gpd = sigma_gpd .* (rand(length(x)).^(-xi_gpd) .- 1) ./ xi_gpd
df_gpd = DataFrame(x=x, y=y_gpd)
```

## Families

### GEV (Generalized Extreme Value)

For block maxima data. Three parameters: location (μ), scale (σ), shape (ξ).

```text
evgam(formula_mu, formula_sigma, formula_xi, data;
    family=GEVFamily())
```

### GPD (Generalized Pareto Distribution)

For threshold exceedances. Two parameters: scale (σ), shape (ξ).

```text
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
`@formula`, repeat the response on each entry:

```text
# GEV: location, scale, shape
evgam(
    [
        @formula(y ~ s(x, k=15, bs=:cr)),   # location μ
        @formula(y ~ s(x, k=10, bs=:cr)),   # log(scale σ)
        @formula(y ~ 1),                    # shape ξ (constant)
    ],
    data,
    GEVFamily(),
)
```

## Examples

### GEV for Annual Maxima

```@example evgam
m = evgam(
    [
        @formula(y ~ s(year, k=10, bs=:cr)),  # location
        @formula(y ~ 1),                      # scale
        @formula(y ~ 1),                      # shape
    ],
    df,
    GEVFamily(),
);
nothing
```

### GPD for Threshold Exceedances

```@example evgam
m_gpd = evgam(
    [
        @formula(y ~ s(x, k=10, bs=:cr)),  # log(scale)
        @formula(y ~ 1),                   # shape
    ],
    df_gpd,
    GPDFamily(),
);
nothing
```

### Non-Stationary GEV

All three GEV parameters as smooth functions:

```@example evgam
m_ns = evgam(
    [
        @formula(y ~ s(year, k=15, bs=:cr)),
        @formula(y ~ s(year, k=10, bs=:cr)),
        @formula(y ~ s(year, k=5, bs=:cr)),
    ],
    df,
    GEVFamily(),
);
nothing
```

## See Also

- [Families & Models](@ref families) for evgam family types
- [API Reference](@ref api-reference) for `evgam`, `GEVFamily`, `GPDFamily`, `MultiParameterFamily`
