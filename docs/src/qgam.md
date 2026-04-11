# [Quantile Regression (QGAM)](@id qgam)

Quantile GAMs estimate conditional quantile functions rather than conditional
means. This is useful when you want to understand how the entire conditional
distribution changes with covariates — not just the average.

GAM.jl's QGAM implementation follows Fasiolo et al. (2021), using the extended
log-F (ELF) likelihood for smooth, calibrated quantile estimation.

```@setup qgam
using GAM, DataFrames, Random, Distributions
using GLM: LogLink
Random.seed!(42)

n = 180
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ (0.1 .+ 0.2 .* x ./ (2π)) .* randn(n)
df = DataFrame(x=x, y=y)

mu = exp.(0.8 .+ 0.4 .* sin.(x))
y_count = Float64.(rand.(Poisson.(mu)))
df_count = DataFrame(x=x, y=y_count)
```

## Key Concepts

- **Quantile regression**: estimate the τ-th quantile of y|x instead of E[y|x]
- **ELF family**: a smooth loss function that approximates the pinball loss
- **Calibration**: automatic learning rate selection to control coverage

## Interface

### Single Quantile

```text
qgam(formula, data, 0.5; control=gam_control())
```

Fits a GAM for a single quantile level `qu ∈ (0, 1)`.

### Location-Scale Quantile Model

```text
qgam([mu_formula, sigma_formula], data, 0.9)
```

Fits a two-formula `ELFLSSFamily` model where the first formula controls the
quantile location and the second controls the covariate-dependent scale.
Returns a `MultiParameterModel`.

### Multiple Quantiles

```text
mqgam(formula, data, [0.1, 0.25, 0.5, 0.75, 0.9])
```

Fits all specified quantiles efficiently, sharing basis construction.
Returns a `MqgamResult` from which individual models can be extracted.

### Extract Individual Quantile

```text
qdo(mqgam_result, quantile_level)
```

Extracts a fitted model for a single quantile from an `mqgam` result.

## Families

| Family | Description |
|--------|-------------|
| `ELFFamily(qu)` | Extended log-F for quantile `qu` |
| `ELFLSSFamily(qu)` | ELF with covariate-dependent scale for location-scale quantile fits |

In most cases you do not need to construct these directly — `qgam()` and
`mqgam()` handle family construction internally.

## Examples

### Median Regression

```@example qgam
m = qgam(@formula(y ~ s(x, k=20, bs=:cr)), df, 0.5);
nothing
```

### Multiple Quantiles

```@example qgam
fits = mqgam(@formula(y ~ s(x, k=20, bs=:cr)), df,
    [0.1, 0.25, 0.5, 0.75, 0.9]);

m10 = qdo(fits, 0.1);
m50 = qdo(fits, 0.5);
m90 = qdo(fits, 0.9);
nothing
```

### Poisson-like Counts with Quantile Regression

```@example qgam
m_count = qgam(@formula(y ~ s(x, k=15, bs=:cr)), df_count, 0.75;
    family=Poisson(), link=LogLink());
nothing
```

### Location-Scale Quantile Regression

```@example qgam
fit = qgam([
    @formula(y ~ s(x, k=20, bs=:cr)),
    @formula(y ~ 0 + s(x, k=10, bs=:cr))
], df, 0.9);

chk = check_qgam(fit);
nothing
```

This path uses `ELFLSSFamily` and lets the ELF scale vary with covariates.
`cqcheck`, `check_qgam`, and `quantile_residuals` all work on these fits too.

You can also predict both fitted parameters at new data:

```@example qgam
newdf = DataFrame(x=range(minimum(x), maximum(x); length=50) |> collect)

fit_mat, se_mat = GAM.predict(fit, newdf; type=:response, se=true);
mu_hat = fit_mat[:, 1];
sigma_hat = fit_mat[:, 2];
nothing
```

`type=:link` returns the linear predictors for each parameter; `type=:response`
applies the corresponding inverse links (for `ELFLSSFamily`, `μ` and `σ`).
When `se=true`, `predict` returns a tuple `(fit, se_fit)` of matrices with the
same shape.

## Calibration

QGAM includes an automatic calibration step that selects the learning rate
(smoothing of the ELF loss) to achieve correct coverage. This is performed
internally during fitting. The calibration ensures that the estimated quantile
has the correct nominal coverage probability.

## See Also

- [Families & Models](@ref families) for ELF family details
- [Getting Started](@ref getting-started) for a quick QGAM example
- [API Reference](@ref api-reference) for `qgam`, `mqgam`, `qdo`
