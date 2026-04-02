# [Families & Models](@id families)

GAM.jl supports a wide range of distribution families for different model types.

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

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=NegBinFamily(theta=1.0))
```

The shape parameter θ is estimated automatically. Variance: μ + μ²/θ.

### Quasi-Poisson

For overdispersed count data when you want Poisson mean structure with an
estimated dispersion parameter rather than a fully specified count distribution.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=QuasiPoissonFamily())
```

The unit variance function is `V(μ) = μ`, and the fitted dispersion is reported
as `m.scale`.

### Quasi-Binomial

For overdispersed binary or proportion data when a binomial mean-variance
relationship is appropriate up to a multiplicative dispersion factor.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=QuasiBinomialFamily())
```

The unit variance function is `V(μ) = μ(1-μ)`, and the fitted dispersion is
reported as `m.scale`.

### Tweedie

For non-negative data with exact zeros, common in insurance and ecology.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=TweedieFamily(p=1.5))
```

Power parameter p ∈ (1, 2). Variance: μᵖ.

To update the power parameter during fitting, pass `estimate_p=true`:

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=TweedieFamily(p=1.3, estimate_p=true))
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

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=BetaFamily(phi=1.0))
```

Precision parameter φ is estimated. Variance: μ(1-μ)/(1+φ).

## GAMLSS Families

GAMLSS families model multiple distribution parameters (location, scale, shape)
simultaneously. Used with [`gamlss()`](@ref). See [GAMLSS](@ref) for full details.

| Family | Parameters | Use case |
|--------|-----------|----------|
| `GaussianLS()` | μ (location), σ (scale) | Heteroscedastic normal data |
| `GammaLocationScale()` | μ (location), σ (scale) | Positive data with varying dispersion |
| `BetaRegression()` | μ (location), φ (precision) | Proportions with varying precision |
| `NegativeBinomialLocationScale()` | μ (location), θ (shape) | Overdispersed counts |
| `InverseGaussianLocationScale()` | μ (location), σ (scale) | Positive data |

```julia
m = gamlss(@gam_formula(y ~ s(x)), @gam_formula(~ s(x)), df;
    family=GaussianLS())
```

## QGAM Families

Quantile regression families based on the extended log-F (ELF) distribution.
Used with [`qgam()`](@ref). See [Quantile Regression (QGAM)](@ref) for details.

| Family | Description |
|--------|-------------|
| `ELFFamily(qu)` | Extended log-F for a single quantile `qu` |
| `ELFLSSFamily(qu)` | ELF with covariate-dependent scale for location-scale quantile fits |

```julia
m = qgam(@gam_formula(y ~ s(x)), df, 0.5)
```

## evgam Families

Extreme value families for modeling tails of distributions. Used with
[`evgam()`](@ref). See [Extreme Values (evgam)](@ref) for details.

| Family | Description |
|--------|-------------|
| `GEVFamily()` | Generalized extreme value (location, scale, shape) |
| `GPDFamily()` | Generalized Pareto distribution (scale, shape) |
| `EGPD1Family()` | Extended GPD, model 1 |
| `EGPD2Family()` | Extended GPD, model 2 |
| `EGPD3Family()` | Extended GPD, model 3 |
| `EGPD4Family()` | Extended GPD, model 4 |

```julia
m = evgam(@gam_formula(y ~ s(x)), @gam_formula(~ s(x)), @gam_formula(~ 1), df;
    family=GEVFamily())
```

## API Reference

See [API Reference](@ref api-reference) for full documentation of all family types.
