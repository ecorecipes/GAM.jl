# Comparison with R's mgcv

GAM.jl aims to reproduce the results of R's mgcv package while following
Julia and JuliaStats conventions.

## Key Differences

### Syntax

| Feature | R mgcv | GAM.jl |
|---------|--------|--------|
| Fit a GAM | `gam(y ~ s(x), data=df)` | `gam(@gam_formula(y ~ s(x)), df)` |
| Basis type | `bs="cr"` | `bs=:cr` |
| Family | `family=poisson()` | `family=Poisson()` |
| Link | implicit | `link=LogLink()` |
| Method | `method="REML"` | `method=:REML` |
| Summary | `summary(m)` | `m` (pretty-printed) |
| Coefficients | `coef(m)` | `coef(m)` |
| Deviance | `deviance(m)` | `deviance(m)` |
| Predict | `predict(m, newdata)` | manual via `predict_matrix` |

### Architecture

- **mgcv** uses S3 classes and C code for performance
- **GAM.jl** uses Julia's type dispatch and generic linear algebra (no C code)
- **mgcv** uses `gam.fit3` (standard) / `gam.fit4` (extended) / `gam.fit5` (GAMLSS)
- **GAM.jl** uses a single P-IRLS engine with dispatch on family type

### Smoothing Parameter Estimation

Both use the Extended Fellner-Schall (EFS) method as the default optimizer.
GAM.jl's EFS implementation follows Wood & Fasiolo (2017).

### Numerical Accuracy

For cubic regression spline (`bs=:cr`) bases, GAM.jl produces numerically
identical results to mgcv (fitted values correlation = 1.0, identical deviance
and EDF). For TPRS and P-spline bases, results are statistically equivalent
(correlation > 0.999) but may differ slightly due to implementation details
in basis construction.

## Available Features

| Feature | mgcv | GAM.jl |
|---------|------|--------|
| TPRS (`bs="tp"`) | ✅ | ✅ |
| Cubic splines (`bs="cr"`) | ✅ | ✅ |
| P-splines (`bs="ps"`) | ✅ | ✅ |
| B-splines (`bs="bs"`) | ✅ | ✅ |
| Random effects (`bs="re"`) | ✅ | ✅ |
| Tensor products (`te()`) | ✅ | ✅ |
| Tensor interactions (`ti()`) | ✅ | ✅ |
| REML / ML / GCV | ✅ | ✅ |
| Extended families (NB, Tweedie) | ✅ | ✅ |
| Gaussian family | ✅ | ✅ |
| Poisson family | ✅ | ✅ |
| Binomial family | ✅ | ✅ |
| Gamma family | ✅ | ✅ |
| `gam.check` diagnostics | ✅ | ✅ |
| Adaptive smooths | ✅ | ❌ |
| Soap film smooths | ✅ | ❌ |
| GAMLSS | ✅ | ❌ |
| BAM (large data) | ✅ | ❌ |
| GAMM (mixed models) | ✅ | ❌ |

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3-36.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071-1081.
