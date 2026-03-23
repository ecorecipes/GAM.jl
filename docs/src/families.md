# Families & Models

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

Extended families estimate additional distribution parameters alongside the
regression coefficients.

### Negative Binomial

For overdispersed count data where the variance exceeds the mean.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=NegBinFamily(theta=1.0))
```

The shape parameter θ is estimated automatically. Variance: μ + μ²/θ.

### Tweedie

For non-negative data with exact zeros, common in insurance and ecology.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=TweedieFamily(p=1.5))
```

Power parameter p ∈ (1, 2). Variance: μᵖ.

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
| `ELFLSSFamily(qu)` | ELF with location-scale-shape for flexible tails |

```julia
m = qgam(@gam_formula(y ~ s(x)), df; qu=0.5)
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

```@docs
GAM.NegBinFamily
GAM.TweedieFamily
GAM.BetaFamily
GAM.GaussianLS
GAM.GammaLocationScale
GAM.BetaRegression
GAM.NegativeBinomialLocationScale
GAM.InverseGaussianLocationScale
GAM.ELFFamily
GAM.ELFLSSFamily
GAM.GEVFamily
GAM.GPDFamily
GAM.MultiParameterFamily
```
