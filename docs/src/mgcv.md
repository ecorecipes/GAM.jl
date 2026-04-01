# [Comparison with R's mgcv](@id mgcv-comparison)

GAM.jl aims to reproduce the results of R's mgcv package (and the broader
mgcv ecosystem: gamlss, scam, qgam, evgam) while following Julia and JuliaStats
conventions. On typical benchmarks GAM.jl achieves a **5.5× speedup** over R.

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
| GAMLSS | `gaulss()` family | `gamlss(..., family=GaussianLS())` |
| SCAM | `scam(y ~ s(x, bs="mpi"))` | `scam(@gam_formula(y ~ s(x, bs=:mpi)), df)` |
| QGAM | `qgam(y ~ s(x), qu=0.5)` | `qgam(@gam_formula(y ~ s(x)), df; qu=0.5)` |
| BAM | `bam(y ~ s(x))` | `bam(@gam_formula(y ~ s(x)), df)` |
| GAMM | `gamm(y ~ s(x))` | `gamm(@gamm_formula(y ~ s(x) + (1\|group)), df)` |

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

## Feature Comparison

### Core GAM

| Feature | R (mgcv) | GAM.jl |
|---------|----------|--------|
| TPRS (`bs="tp"`, `:ts`) | ✅ | ✅ |
| Cubic splines (`:cr`, `:cs`, `:cc`) | ✅ | ✅ |
| P-splines (`:ps`) | ✅ | ✅ |
| Cyclic P-splines (`:cps`) | ✅ | ✅ |
| B-splines (`:bs`) | ✅ | ✅ |
| Gaussian process (`:gp`) | ✅ | ✅ |
| Duchon splines (`:ds`) | ✅ | ✅ |
| Markov random field (`:mrf`) | ✅ | ✅ |
| Soap film (`:so`) | ✅ | ✅ |
| Factor-smooth (`:fs`) | ✅ | ✅ |
| Random effects (`:re`) | ✅ | ✅ |
| Tensor products (`te`/`ti`) | ✅ | ✅ |
| REML / ML / GCV | ✅ | ✅ |
| Extended families (NB, quasi, Tweedie, Beta) | ✅ | ✅ |
| Side constraints (`gam.side`) | ✅ | ✅ |
| `gam.check` diagnostics | ✅ | ✅ |
| Adaptive smooths | ✅ | ❌ |

### Extended Models

| Feature | R package | GAM.jl |
|---------|-----------|--------|
| GAMLSS (distributional regression) | gamlss / mgcv | ✅ `gamlss()` |
| BAM (large data) | mgcv | ✅ `bam()` |
| GAMM (mixed models) | mgcv | ✅ `gamm()` |
| SCAM (shape constraints) | scam | ✅ `scam()` |
| QGAM (quantile regression) | qgam | ✅ `qgam()` |
| evgam (extreme values) | evgam | ✅ `evgam()` |
| GINLA (posterior inference) | mgcv | ✅ `ginla()` |
| Bayesian (MCMC) | — | ✅ Turing.jl integration |

### Diagnostics

| Feature | R (gratia / mgcv) | GAM.jl |
|---------|-------------------|--------|
| Residual checks (`gam.check`) | ✅ | ✅ `gam_check()` |
| Basis dimension check (`k.check`) | ✅ | ✅ `k_check()` |
| Concurvity | ✅ | ✅ `concurvity()` |
| Smooth estimates | ✅ gratia | ✅ `smooth_estimates()` |
| Derivatives of smooths | ✅ gratia | ✅ `derivatives()` |
| Partial residuals | ✅ gratia | ✅ `partial_residuals()` |
| Posterior samples | ✅ gratia | ✅ `posterior_samples()` |
| Fitted samples | ✅ gratia | ✅ `fitted_samples()` |
| Rootogram | ✅ gratia | ✅ `rootogram()` |
| Appraise (multi-panel) | ✅ gratia | ✅ `appraise()` |
| Data slicing | ✅ gratia | ✅ `data_slice()` |

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3-36.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071-1081.
- Rigby, R.A. & Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape. *JRSS-C*, 54(3), 507-554.
- Pya, N. & Wood, S.N. (2015). Shape constrained additive models. *Statistics and Computing*, 25(3), 543-559.
- Fasiolo, M. et al. (2021). Fast calibrated additive quantile regression. *JASA*, 116(535), 1402-1413.
- Youngman, B.D. (2022). evgam: An R package for generalized additive extreme value models. *JSS*, 103(3).
