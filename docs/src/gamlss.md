# [GAMLSS](@id gamlss)

Generalized Additive Models for Location, Scale, and Shape (GAMLSS) extend
standard GAMs by modeling multiple distribution parameters as smooth functions
of covariates. While a standard GAM models only the mean (location), GAMLSS
simultaneously models the variance (scale) and potentially higher parameters (shape).

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

```julia
gamlss([
    @gam_formula(y ~ s(x1)),
    @gam_formula(y ~ 1 + s(x2)),
], data, GaussianLS();
    method = :efs,
    gamlss_ctrl = gamlss_control(),
)
```

Pass a vector of formulas, one per distribution parameter. With
`@gam_formula`, repeat the response on each entry, e.g.
`[@gam_formula(y ~ s(x)), @gam_formula(y ~ 1)]`.

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

```julia
using GAM, DataFrames

n = 500
x = range(0, 2π; length=n) |> collect
# Mean varies as sin(x), variance increases with |cos(x)|
y = sin.(x) .+ (0.1 .+ 0.3 .* abs.(cos.(x))) .* randn(n)
df = DataFrame(x=x, y=y)

m = gamlss(
    [
        @gam_formula(y ~ s(x, k=15, bs=:cr)),  # μ model
        @gam_formula(y ~ s(x, k=10, bs=:cr)),  # log(σ) model
    ],
    df,
    family=GaussianLS(),
)
```

### Gamma Location-Scale

```julia
x = range(0.1, 3.0; length=500) |> collect
mu = exp.(0.5 .* x)
sigma = 0.2 .+ 0.1 .* x
y = [rand(Gamma(mu[i]^2 / sigma[i]^2, sigma[i]^2 / mu[i])) for i in 1:500]
df = DataFrame(x=x, y=y)

m = gamlss(
    [
        @gam_formula(y ~ s(x, k=15, bs=:cr)),
        @gam_formula(y ~ s(x, k=10, bs=:cr)),
    ],
    df,
    family=GammaLocationScale(),
)
```

### Controlling the Fit

```julia
ctrl = gamlss_control(
    n_cyc = 50,             # max outer iterations
    c_crit = 1e-7,          # convergence tolerance
    sp_method = :efs,       # smoothing parameter method
    trace = true,           # print progress
)

m = gamlss(
    [
        @gam_formula(y ~ s(x, k=15)),
        @gam_formula(y ~ s(x, k=10)),
    ],
    df,
    family=GaussianLS(),
    method = :rs,
    gamlss_ctrl = ctrl,
)
```

## See Also

- [Families & Models](@ref families) for the full family list
- [Getting Started](@ref getting-started) for a quick GAMLSS example
- [API Reference](@ref api-reference) for `gamlss`, `GamlssControl`, and family types
