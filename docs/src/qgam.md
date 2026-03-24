# [Quantile Regression (QGAM)](@id qgam)

Quantile GAMs estimate conditional quantile functions rather than conditional
means. This is useful when you want to understand how the entire conditional
distribution changes with covariates — not just the average.

GAM.jl's QGAM implementation follows Fasiolo et al. (2021), using the extended
log-F (ELF) likelihood for smooth, calibrated quantile estimation.

## Key Concepts

- **Quantile regression**: estimate the τ-th quantile of y|x instead of E[y|x]
- **ELF family**: a smooth loss function that approximates the pinball loss
- **Calibration**: automatic learning rate selection to control coverage

## Interface

### Single Quantile

```julia
qgam(formula, data; qu=0.5, control=gam_control())
```

Fits a GAM for a single quantile level `qu ∈ (0, 1)`.

### Multiple Quantiles

```julia
mqgam(formula, data; qu=[0.1, 0.25, 0.5, 0.75, 0.9])
```

Fits all specified quantiles efficiently, sharing basis construction.
Returns a `MqgamResult` from which individual models can be extracted.

### Extract Individual Quantile

```julia
qdo(mqgam_result, quantile_level)
```

Extracts a fitted model for a single quantile from an `mqgam` result.

## Families

| Family | Description |
|--------|-------------|
| `ELFFamily(qu)` | Extended log-F for quantile `qu` |
| `ELFLSSFamily(qu)` | ELF with location-scale-shape for flexible tails |

In most cases you do not need to construct these directly — `qgam()` and
`mqgam()` handle family construction internally.

## Examples

### Median Regression

```julia
using GAM, DataFrames

n = 500
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ (0.1 .+ 0.3 .* x ./ (2π)) .* randn(n)
df = DataFrame(x=x, y=y)

m = qgam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df; qu=0.5)
```

### Multiple Quantiles

```julia
fits = mqgam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df;
    qu=[0.1, 0.25, 0.5, 0.75, 0.9])

# Extract individual quantile models
m10 = qdo(fits, 0.1)
m50 = qdo(fits, 0.5)
m90 = qdo(fits, 0.9)
```

### Poisson-like Counts with Quantile Regression

```julia
using Distributions

mu = exp.(1.0 .+ 0.5 .* sin.(x))
y = Float64.([rand(Poisson(m)) for m in mu])
df = DataFrame(x=x, y=y)

# Quantile regression on count data
m = qgam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df;
    qu=0.75, family=Poisson(), link=LogLink())
```

## Calibration

QGAM includes an automatic calibration step that selects the learning rate
(smoothing of the ELF loss) to achieve correct coverage. This is performed
internally during fitting. The calibration ensures that the estimated quantile
has the correct nominal coverage probability.

## See Also

- [Families & Models](@ref families) for ELF family details
- [Getting Started](@ref getting-started) for a quick QGAM example
- [API Reference](@ref api-reference) for `qgam`, `mqgam`, `qdo`
