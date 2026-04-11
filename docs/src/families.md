# [Families & Models](@id families)

GAM.jl supports a wide range of distribution families for different model types.

```@setup families
using GAM, DataFrames, Random, Distributions
Random.seed!(42)

n = 120
x = range(0.1, 3.0; length=n) |> collect
y = sin.(x) .+ 0.2 .* randn(n)
df = DataFrame(x=x, y=y)

counts = Float64.(rand.(Poisson.(exp.(0.4 .+ 0.2 .* x))))
df_count = DataFrame(x=x, y=counts)

p_bin = clamp.(0.2 .+ 0.6 .* (x .- minimum(x)) ./ (maximum(x) - minimum(x)), 0.05, 0.95)
y_bin = Float64.(rand.(Bernoulli.(p_bin)))
df_bin = DataFrame(x=x, y=y_bin)

props = clamp.(0.15 .+ 0.7 .* (x .- minimum(x)) ./ (maximum(x) - minimum(x)) .+ 0.05 .* randn(n), 0.01, 0.99)
df_prop = DataFrame(x=x, y=props)

y_pos = exp.(0.4 .* sin.(x) .+ 0.2 .* randn(n))
df_pos = DataFrame(x=x, y=y_pos)

y_zero = max.(0.0, exp.(0.4 .* sin.(x) .+ 0.2 .* randn(n)) .- 1.0)
df_tweedie = DataFrame(x=x, y=y_zero)

x_ls = range(0, 2π; length=n) |> collect
y_ls = sin.(x_ls) .+ (0.1 .+ 0.2 .* abs.(cos.(x_ls))) .* randn(n)
df_ls = DataFrame(x=x_ls, y=y_ls)

year = collect(1:n)
y_gev = 10.0 .+ 0.03 .* year .+ abs.(randn(n))
df_gev = DataFrame(x=year, y=y_gev)
```

## Standard Exponential Families

These are used with `gam()` and work like GLM.jl families:

| Family | Use case | Link |
|--------|----------|------|
| `Gaussian()` | Continuous data | `IdentityLink()` |
| `Poisson()` | Count data | `LogLink()` |
| `Binomial()` | Binary / proportion data | `LogitLink()` |
| `Gamma()` | Positive continuous data | `InverseLink()` |
| `InverseGaussian()` | Positive continuous data | `InverseSquaredLink()` |

## Extended Families

Extended families either estimate additional distribution parameters alongside
the regression coefficients or use quasi-likelihood variance functions with an
estimated dispersion stored in `m.scale`.

### Negative Binomial

For overdispersed count data where the variance exceeds the mean.

```@example families
m = gam(@formula(y ~ s(x)), df_count;
    family=NegBinFamily(theta=1.0));
nothing
```

The shape parameter θ is estimated automatically. Variance: μ + μ²/θ.

### Quasi-Poisson

For overdispersed count data when you want Poisson mean structure with an
estimated dispersion parameter rather than a fully specified count distribution.

```@example families
m = gam(@formula(y ~ s(x)), df_count;
    family=QuasiPoissonFamily());
nothing
```

The unit variance function is `V(μ) = μ`, and the fitted dispersion is reported
as `m.scale`.

### Quasi-Binomial

For overdispersed binary or proportion data when a binomial mean-variance
relationship is appropriate up to a multiplicative dispersion factor.

```@example families
m = gam(@formula(y ~ s(x)), df_bin;
    family=QuasiBinomialFamily());
nothing
```

The unit variance function is `V(μ) = μ(1-μ)`, and the fitted dispersion is
reported as `m.scale`.

### Tweedie

For non-negative data with exact zeros, common in insurance and ecology.

```@example families
m = gam(@formula(y ~ s(x)), df_tweedie;
    family=TweedieFamily(p=1.5));
nothing
```

Power parameter p ∈ (1, 2). Variance: μᵖ.

To update the power parameter during fitting, pass `estimate_p=true`:

```@example families
m = gam(@formula(y ~ s(x)), df_tweedie;
    family=TweedieFamily(p=1.3, estimate_p=true));
nothing
```

This uses a bounded profile-likelihood update based on the Tweedie log density
for `1 < p < 2`.

For fitted Tweedie GAMs, `loglikelihood(m)` and `StatsAPI.aic(m)` / `StatsAPI.bic(m)`
paths use that exact Tweedie series likelihood instead of the generic
`-deviance/2` approximation.

`TweedieFamily` also provides level-0 deviance derivatives for PIRLS. With the
default `LogLink()`, the inner iteration therefore uses Tweedie-specific
working responses and bounded Newton weights derived from the deviance itself.
Higher-order power-parameter derivatives are still handled separately by the
bounded profile-likelihood update.

### Beta Regression

For response data in (0, 1), such as proportions.

```@example families
m = gam(@formula(y ~ s(x)), df_prop;
    family=BetaFamily(phi=1.0));
nothing
```

Precision parameter φ is estimated. Variance: μ(1-μ)/(1+φ).

## GAMLSS Families

GAMLSS families model multiple distribution parameters (location, scale, shape)
simultaneously. Use them through `gam(..., family)`; the legacy
[`gamlss()`](@ref) wrapper remains available. See [GAMLSS](gamlss.md) for full details.

| Family | Parameters | Use case |
|--------|-----------|----------|
| `GaussianLS()` | μ (location), σ (scale) | Heteroscedastic normal data |
| `GammaLocationScale()` | μ (location), σ (scale) | Positive data with varying dispersion |
| `BetaRegression()` | μ (location), φ (precision) | Proportions with varying precision |
| `NegativeBinomialLocationScale()` | μ (location), θ (shape) | Overdispersed counts |
| `InverseGaussianLocationScale()` | μ (location), σ (scale) | Positive data |

```@example families
m = gam([
    @formula(y ~ s(x)),
    @formula(y ~ s(x)),
], df_ls, GaussianLS());
nothing
```

## QGAM Families

Quantile regression families based on the extended log-F (ELF) distribution.
Used with [`qgam()`](@ref). See [Quantile Regression (QGAM)](qgam.md) for details.

| Family | Description |
|--------|-------------|
| `ELFFamily(qu)` | Extended log-F for a single quantile `qu` |
| `ELFLSSFamily(qu)` | ELF with covariate-dependent scale for location-scale quantile fits |

```@example families
m = qgam(@formula(y ~ s(x)), df, 0.5);
nothing
```

## evgam Families

Extreme value families for modeling tails of distributions. Used with
[`evgam()`](@ref). See [Extreme Values (evgam)](evgam.md) for details.

| Family | Description |
|--------|-------------|
| `GEVFamily()` | Generalized extreme value (location, scale, shape) |
| `GPDFamily()` | Generalized Pareto distribution (scale, shape) |
| `EGPD1Family()` | Extended GPD, model 1 |
| `EGPD2Family()` | Extended GPD, model 2 |
| `EGPD3Family()` | Extended GPD, model 3 |
| `EGPD4Family()` | Extended GPD, model 4 |

```@example families
m = evgam([
    @formula(y ~ s(x)),
    @formula(y ~ s(x)),
    @formula(y ~ 1),
], df_gev, GEVFamily());
nothing
```

## API Reference

See [API Reference](@ref api-reference) for full documentation of all family types.
