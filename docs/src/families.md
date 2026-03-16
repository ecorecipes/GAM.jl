# Extended Families

Beyond the standard exponential families (Gaussian, Poisson, Binomial, Gamma),
GAM.jl supports extended families where additional distribution parameters are
estimated alongside the regression coefficients.

## Negative Binomial

For overdispersed count data where the variance exceeds the mean.

```julia
using GAM

m = gam(@gam_formula(y ~ s(x)), df;
    family=NegBinFamily(theta=1.0))
```

The shape parameter θ is estimated automatically. Variance: μ + μ²/θ.

## Tweedie

For non-negative data with exact zeros, common in insurance and ecology.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=TweedieFamily(p=1.5))
```

Power parameter p ∈ (1, 2). Variance: μᵖ.

## Beta Regression

For response data in (0, 1), such as proportions.

```julia
m = gam(@gam_formula(y ~ s(x)), df;
    family=BetaFamily(phi=1.0))
```

Precision parameter φ is estimated. Variance: μ(1-μ)/(1+φ).

## API Reference

```@docs
GAM.NegBinFamily
GAM.TweedieFamily
GAM.BetaFamily
```
