# [GAMLSS](@id gamlss)

Generalized Additive Models for Location, Scale, and Shape (GAMLSS) extend
standard GAMs by modeling multiple distribution parameters as smooth functions
of covariates. While a standard GAM models only the mean (location), GAMLSS
simultaneously models the variance (scale) and potentially higher parameters (shape).

```@setup gamlss
using GAM, DataFrames, Random, Distributions
Random.seed!(42)

n = 160
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ (0.1 .+ 0.2 .* abs.(cos.(x))) .* randn(n)
df = DataFrame(x=x, y=y)

x_gamma = range(0.1, 3.0; length=n) |> collect
mu = exp.(0.3 .+ 0.4 .* x_gamma)
sigma = 0.2 .+ 0.05 .* x_gamma
y_gamma = [rand(Gamma(mu[i]^2 / sigma[i]^2, sigma[i]^2 / mu[i])) for i in eachindex(x_gamma)]
df_gamma = DataFrame(x=x_gamma, y=y_gamma)
```

## When to Use GAMLSS

- **Heteroscedastic data**: variance changes with covariates
- **Non-standard distributions**: where shape parameters vary
- **Distributional regression**: full predictive distributions, not just means

## Supported Families

| Family | Parameters | Description |
|--------|-----------|-------------|
| `GaussianLS()` | μ, σ | Normal with varying mean and variance |
| `GammaLocationScale()` | μ, σ | Gamma with varying location and scale |
| `BetaRegression()` | μ, φ | Beta with varying mean and precision |
| `NegativeBinomialLocationScale()` | μ, θ | Negative binomial with varying mean and shape |
| `InverseGaussianLocationScale()` | μ, σ | Inverse Gaussian with varying location and scale |

Custom families can be created using `DistFamily`.

## Interface

```text
gam([
    @formula(y ~ s(x1)),
    @formula(y ~ 1 + s(x2)),
], data, GaussianLS();
    method = :efs,
    gamlss_ctrl = gamlss_control(),
)
```

Pass a vector of formulas, one per distribution parameter. With
`@formula`, repeat the response on each entry, e.g.
`[@formula(y ~ s(x)), @formula(y ~ 1)]`. The legacy `gamlss(...)` wrapper
still works, but `gam(..., family)` is now the preferred entry point.

When a single formula is provided, `gam` replicates it for all distribution
parameters:

```@example gamlss
m_same = gam(@formula(y ~ s(x, k=12, bs=:cr)), df, GaussianLS());
nothing
```

## Solvers

GAMLSS uses an outer iteration over distribution parameters with inner GAM fits:

- **RS** (Rigby-Stasinopoulos): default algorithm, updates each parameter in turn
- **CG** (Cole-Green): uses the full joint Hessian for faster convergence

Set via `method = :rs` or `method = :cg`.

## Smoothing Parameter Estimation

| Method | Description |
|--------|-------------|
| `:efs` | Extended Fellner-Schall (default) — fast, usually robust |
| `:local_ml` | Local ML estimation at each outer step |
| `:local_gaic` | Local GAIC-based estimation |
| `:local_gcv` | Local GCV-based estimation |

## Examples

### Gaussian Location-Scale

```@example gamlss
m = gam(
    [
        @formula(y ~ s(x, k=15, bs=:cr)),  # μ model
        @formula(y ~ s(x, k=10, bs=:cr)),  # log(σ) model
    ],
    df,
    GaussianLS(),
);
nothing
```

### Gamma Location-Scale

```@example gamlss
m_gamma = gam(
    [
        @formula(y ~ s(x, k=15, bs=:cr)),
        @formula(y ~ s(x, k=10, bs=:cr)),
    ],
    df_gamma,
    GammaLocationScale(),
);
nothing
```

### Controlling the Fit

```@example gamlss
ctrl = gamlss_control(
    n_cyc = 20,             # max outer iterations
    c_crit = 1e-7,          # convergence tolerance
    sp_method = :efs,       # smoothing parameter method
    trace = false,
)

m_ctrl = gam(
    [
        @formula(y ~ s(x, k=15)),
        @formula(y ~ s(x, k=10)),
    ],
    df,
    GaussianLS();
    method = :rs,
    gamlss_ctrl = ctrl,
);
nothing
```

## See Also

- [Families & Models](@ref families) for the full family list
- [Getting Started](@ref getting-started) for a quick GAMLSS example
- [API Reference](@ref api-reference) for `gam`, `gamlss`, `GamlssControl`, and family types
