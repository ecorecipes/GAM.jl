# [Comparison with R's mgcv](@id mgcv-comparison)

GAM.jl aims to reproduce the results of R's mgcv package (and the broader
ecosystem: gamlss, scam, qgam, evgam, gamFactory) while following Julia and
JuliaStats conventions. In the latest checked-in benchmark snapshot, GAM.jl
achieves an **11.16x geometric-mean speedup** over R (fitting time; fit
equivalence is verified separately — see below).

## Key Differences

### Syntax

| Feature | R mgcv | GAM.jl |
|---------|--------|--------|
| Fit a GAM | `gam(y ~ s(x), data=df)` | `gam(@formula(y ~ s(x)), df)` |
| Basis type | `bs="cr"` | `bs=:cr` |
| Family | `family=poisson()` | `family=Poisson()` |
| Link | implicit | `link=LogLink()` |
| Method | `method="REML"` | `method=:REML` |
| Summary | `summary(m)` | `m` (pretty-printed) |
| Coefficients | `coef(m)` | `coef(m)` |
| Deviance | `deviance(m)` | `deviance(m)` |
| Predict | `predict(m, newdata, se.fit=TRUE)` | `predict(m, newdata; se=true, type=:response)` |
| GAMLSS | `gaulss()` family | `gam(@formula(y ~ s(x)), df, GaussianLS())` |
| SCAM | `scam(y ~ s(x, bs="mpi"))` | `gam(@formula(y ~ s(x, bs=:mpi)), df)` |
| QGAM | `qgam(y ~ s(x), qu=0.5)` | `qgam(@formula(y ~ s(x)), df, 0.5)` |
| BAM | `bam(y ~ s(x))` | `bam(@formula(y ~ s(x)), df)` |
| GAMM | `gamm(y ~ s(x))` | `gamm(@formula(y ~ s(x) + (1 \| group)), df)` |
| Nested effects | gamFactory `gam_nl` / `s_nest` | `gam_nl(@formula(y ~ s_nest(l1, l2, l3, trans=trans_linear())), df)` |

### Architecture

- **mgcv** uses S3 classes and C code for performance
- **GAM.jl** is written in Julia (BLAS/LAPACK underneath; the SCASM
  linear-constraint solver uses the OSQP C library)
- **mgcv** uses `gam.fit3` (standard) / `gam.fit4` (extended) / `gam.fit5` (GAMLSS)
- **GAM.jl** uses family-specialized P-IRLS engines (standard, extended,
  SCAM, SCASM, BAM) sharing the same mathematical conventions

### Smoothing Parameter Estimation

Both use the Extended Fellner-Schall (EFS) method as the default optimizer.
GAM.jl's EFS implementation follows Wood & Fasiolo (2017).

### Numerical Accuracy (measured, asserted in the test suite)

On the reference Gaussian cubic-spline model, GAM.jl matches mgcv
**elementwise**: smoothing parameter to log-difference 0.0000, coefficients
to 9e-8, and prediction standard errors to 5e-7 (Poisson: sp within 0.012,
SEs within 0.2%). AIC agrees within ~0.5 (mgcv's corrected-edf convention).
Nested effects match gamFactory's index directions to |cosine| > 0.999.
Two caveats: smooth-test F statistics use a documented simplification of
mgcv's `testStat` (edf and p-value conclusions match; statistics can
differ), and on numerically flat REML ridges the smoothing parameter is
only weakly identified, so fitted values/EDF — not raw sp — are the
meaningful comparison there.

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
| Adaptive smooths (`:ad`) | ✅ | ✅ |
| Spherical splines (`:sos`) | ✅ | ⚠️ approximation |
| Soap film (`:so`) | ✅ | ⚠️ approximation |
| `t2()` tensor construction (Wood–Scheipl–Faraway) | ✅ | ✅ (verified against mgcv: matching columns, penalty count, ranks, supports) |
| Linear functional terms (matrix args) | ✅ | ❌ |
| `bam(discrete=TRUE)` covariate discretization | ✅ | ❌ (chunked accumulation instead) |
| Smoothing-parameter-uncertainty `Vc` / `unconditional=TRUE` | ✅ | ❌ |

### Extended Models

| Feature | R package | GAM.jl |
|---------|-----------|--------|
| GAMLSS (distributional regression) | gamlss / mgcv | ✅ `gam(..., family)` |
| BAM (large data) | mgcv | ✅ `bam()` |
| GAMM (mixed models) | mgcv | ✅ `gamm()` |
| SCAM (shape constraints) | scam | ✅ `gam(...)` |
| QGAM (quantile regression) | qgam | ✅ `qgam()` |
| evgam (extreme values) | evgam | ✅ `evgam()` |
| GINLA (posterior inference) | mgcv | ✅ `ginla()` |
| Nested effects (single-index etc.) | gamFactory | ✅ `gam_nl()` / `s_nest()` |
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
| Influence measures | ✅ (base R) | ✅ `leverage()`, `cooksdistance()` |

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *JRSS-B*, 73(1), 3-36.
- Wood, S.N. & Fasiolo, M. (2017). A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071-1081.
- Rigby, R.A. & Stasinopoulos, D.M. (2005). Generalized additive models for location, scale and shape. *JRSS-C*, 54(3), 507-554.
- Pya, N. & Wood, S.N. (2015). Shape constrained additive models. *Statistics and Computing*, 25(3), 543-559.
- Fasiolo, M. et al. (2021). Fast calibrated additive quantile regression. *JASA*, 116(535), 1402-1413.
- Youngman, B.D. (2022). evgam: An R package for generalized additive extreme value models. *JSS*, 103(3).
