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
| Method | `method="REML"` (default: `"GCV.Cp"`) | `method=:REML` (default: `:REML`) |
| Knots | `knots=list(x=c(0,52))` | `knots=Dict(:x => [0.0, 52.0])` |
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

### ⚠️ Two conventions that differ — read before comparing side by side

Both of these make *correct* code in each language fit a **different model**,
which is the most common source of "GAM.jl doesn't match mgcv" reports.

#### 1. Tensor `k` is per-marginal in both packages — but changed in GAM.jl 0.2

For tensor smooths (`te`, `ti`, `t2`) a scalar `k` is now the dimension of
*each* marginal basis, recycled across margins — **exactly mgcv's
convention** — so an mgcv model ports with its `k` unchanged:

```julia
# mgcv:  te(x, z, k = 5)          → 5 per margin, 24 columns (post-constraint)
te(:x, :z, k = 5)                 # identical model in GAM.jl

# mgcv:  te(x, y, z, k = 4)       → 4 per margin
te(:x, :y, :z, k = 4)             # identical
te(:x, :z, k = [4, 7])            # unequal margins, mgcv's k = c(4, 7)
```

**The convention changed in GAM.jl 0.2.** Before, a scalar `k` was a *total*
dimension hint split as `round(Int, k^(1/d))` per margin, so `te(x, z, k=25)`
meant 5×5; it now means 25×25. Code written against the old behaviour should
switch to the explicit vector form (`k = [5, 5]`) to keep its basis size —
old scalar values carried over unchanged now produce much larger bases.

Plain `s()` smooths are unaffected: `k` has always meant the same thing in
both packages there.

#### 2. The default smoothing-parameter method differs

mgcv's `gam()` defaults to `method = "GCV.Cp"`; GAM.jl's `gam()` defaults to
`method = :REML`. Comparing defaults therefore compares two different
criteria, and REML generally selects smoother fits than GCV.

**When the methods are matched, the engines agree to numerical precision.**
On a matched tensor model the REML score at the optimum is `95.289118`
(GAM.jl) versus `95.28912` (mgcv), the criterion surfaces correlate at
`0.99994` over a grid spanning `log λ ∈ [-12, 12]`, and the located optima
agree to about `1e-3` in log-sp under both REML and GCV. Pass
`method = "REML"` in R when comparing.

**`"GCV.Cp"` is not always `:GCV`.** mgcv picks the actual criterion from
whether the scale is known (`R/mgcv.r:1946-1965`): with an estimated scale
(Gaussian, Gamma) `method = "GCV.Cp"` optimizes **GCV**, but with a known
scale (Poisson, Binomial) it optimizes **UBRE**. GAM.jl takes the criterion
literally from `method`, so the Julia equivalent of `"GCV.Cp"` is `:GCV` for
Gaussian/Gamma and `:UBRE` for Poisson/Binomial. Matching them accordingly
reproduces mgcv closely — on a Poisson `s(x, bs="cr", k=10)` fit, `:UBRE`
gives score `0.158781` against mgcv's `0.158781`, sp `129.97` vs `129.96`
and EDF `5.9892` vs `5.9894` — whereas `:GCV` on the same data optimizes a
genuinely different criterion (score `1.1676`, EDF `5.7504`).

The achieved score is available as `sp_criterion(m)`, the analogue of mgcv's
`b$gcv.ubre`. Both are minimized, so REML/ML values are negative log marginal
likelihoods and lower is better.

`:NCV` (neighbourhood cross validation, mgcv's `ncv.c`) is also available.
Its criterion matches mgcv to every printed digit at fixed `sp` (e.g.
`6.310240995` against `6.310240995` at `sp = 0.001`), and for a Gaussian
identity model — where the underlying Newton step is exact — it reproduces a
brute-force leave-one-out refit to `7.8e-16`. One documented divergence:
mgcv computes analytic derivatives of the NCV score to drive a Newton
optimizer, whereas GAM.jl supplies the criterion and selects with the existing
derivative-free optimizer, reaching the same optimum in more iterations
(free-fit `sp` differs by ~0.03% on a flat optimum).

All four remaining criteria agree with mgcv: `:ML` matches to ~4e-16 on the Gaussian
reference (the range-space determinant `MLpenalty1` uses and the ML-profiled
scale are both ported — an earlier 1–8% gap is fixed), and `:REML`, `:GCV`
and `:UBRE` agree to between `4e-12` and `2e-4` depending on family and link
(non-canonical links use mgcv's full-Newton working weights in the score).

### Architecture

- **mgcv** uses S3 classes and C code for performance
- **GAM.jl** is written in Julia (BLAS/LAPACK underneath; the SCASM
  linear-constraint solver uses the OSQP C library)
- **mgcv** uses `gam.fit3` (standard) / `gam.fit4` (extended) / `gam.fit5` (GAMLSS)
- **GAM.jl** uses family-specialized P-IRLS engines (standard, extended,
  SCAM, SCASM, BAM) sharing the same mathematical conventions

### Smoothing Parameter Estimation

The **criteria** are the same; the **optimizers** are not.

mgcv defaults to `optimizer = c("outer", "newton")` — Newton's method on the
smoothness criterion using exact analytic first and second derivatives, with
the stable reparameterization of Wood (2011, Appendix B). GAM.jl defaults to
the Extended Fellner-Schall fixed point of Wood & Fasiolo (2017), which needs
no second derivatives (mgcv offers the same via `optimizer = "efs"`).

They locate the same optima: on a matched tensor model the REML score at the
optimum agrees to `95.289118` vs `95.28912` and the criterion surfaces
correlate at `0.99994`. GAM.jl typically needs more outer iterations to get
there. The omitted stable reparameterization was measured not to degrade the
surface: disagreement with mgcv does not grow with the smoothing-parameter
ratio out to `e^24`, and the criterion shows no kinking.

### Other algorithmic differences (measured)

| Component | mgcv / R package | GAM.jl | Measured consequence |
|-----------|------------------|--------|----------------------|
| SCAM smoothness selection | BFGS on analytic GCV derivatives | coarse global scan + bracketed golden section | GAM.jl's selected GCV is **equal or lower on all six datasets tested**; because the criterion is flat, the selected EDF can still differ |
| QGAM calibration | `tuneLearnFast` re-solves smoothing parameters inside the loss | smoothing parameters frozen at the preliminary fit | GAM.jl's calibration error is **smaller** (0.010/0.000/0.013 vs R's 0.020/0.013/0.023 at τ = 0.2/0.5/0.8); EDF is roughly 2× smaller, i.e. right levels with smoother curves |
| Nested effects | gamFactory: hand-coded derivative blocks, LAML/BFGS | Gauss–Newton with automatic differentiation, EFS | outputs verified equal (index directions to \|cosine\| > 0.999, fitted correlation > 0.999) |
| `bam` normal equations | block QR updating | chunked accumulation of `X'WX` | equivalent in practice: for realistic spline designs `cond(X)` stays in the tens, and at genuine rank deficiency **both** approaches fail alike |
| GAMLSS | R gamlss RS/CG | same RS/CG algorithms | outputs verified equivalent |

### Numerical Accuracy (measured, asserted in the test suite)

On the reference Gaussian cubic-spline model, GAM.jl matches mgcv
**elementwise**: smoothing parameter to log-difference 0.0000, coefficients
to 9e-8, and prediction standard errors to 5e-7 (Poisson: sp within 0.012,
SEs within 0.2%). `aic(m)` follows mgcv's `AIC()` convention (`edf2`-based)
and agrees to ~1e-4 on REML fits.
Nested effects match gamFactory's index directions to |cosine| > 0.999.
Four caveats: smooth-test F statistics use a simplification of mgcv's
`testStat`, so the printed statistic can differ from `summary.gam`'s; on
numerically flat REML ridges the smoothing parameter is only weakly
identified, so fitted values/EDF — not raw sp — are the meaningful
comparison there; the `tp`, `gp` and `sos` bases are **rotation-equivalent**
to mgcv's, not elementwise — fits, EDF and predictions match at fixed `sp`
(to ~1e-12 or better) but raw coefficient vectors differ, so never compare
those elementwise; `bs=:sz` uses a **single** penalty on the deviation term where
mgcv uses one per factor level, so its bases and coefficient count match
mgcv exactly but its deviation edf is larger when levels differ markedly in
how far they depart from the common curve (10.24 vs 14.63 on a three-region
seasonal model, deviance 158.46 vs 158.00) — compare deviance and fitted
curves for `:sz`, not per-term edf; and `fs` smoothing parameters do **not**
transfer from mgcv. The penalty structure matches, but mgcv's `nat.param(type=1)`
parameterisation puts the null-space components on a different footing, so
feeding mgcv's `sp` in makes agreement *worse* rather than better — on a
random-slope trajectory model, 0.52% of fitted range against 0.16% for the
same model fitted freely, and larger still on other data. The size is
dataset-dependent; the direction is not. Compare `fs` fits at freely
selected `sp`, where they agree closely (fitted correlation 0.9999971, edf
89.69 vs 89.64 on that model), and compare edf rather than `sp`. Smoothing parameters
*do* transfer for `cr`/`ps`/`tp`/`sos`/`ad`/`t2` and factor-`by` smooths.

!!! note "The `testStat` simplification does not cost test size"
    The differing statistic is a difference in the *statistic*, not in
    calibrated inference. On identical null replicates the empirical
    rejection rates are within Monte Carlo error of mgcv's at every level:

    | α | GAM.jl | mgcv |
    |---|---|---|
    | 0.01 | 0.003 | 0.003 |
    | 0.05 | 0.037 | 0.035 |
    | 0.10 | 0.077 | 0.070 |

    (Gaussian, 400 replicates; Poisson gives the same picture — 0.062 vs
    0.052 at α = 0.05.) Both engines are mildly non-uniform in the same
    direction, a known property of REML-fitted smooth p-values rather than a
    defect in either.

## Feature Comparison

### Core GAM

| Feature | R (mgcv) | GAM.jl |
|---------|----------|--------|
| TPRS (`bs="tp"`, `:ts`) | ✅ | ✅ |
| Cubic splines (`:cr`, `:cs`, `:cc`) | ✅ | ✅ |
| P-splines (`:ps`) | ✅ | ✅ |
| Cyclic P-splines (`:cps`) | ✅ | ✅ |
| B-splines (`:bs`) | ✅ | ✅ |
| Gaussian process (`:gp`) | ✅ | ✅ 1-D only; `m` selects mgcv's correlation type |
| Duchon splines (`:ds`) | ✅ | ⚠️ alias for `:tp` — warns; not Duchon's fractional-order basis |
| Markov random field (`:mrf`) | ✅ | ✅ |
| Factor-smooth (`:fs`) | ✅ | ✅ |
| Random effects (`:re`) | ✅ | ✅ |
| Tensor products (`te`/`ti`) | ✅ | ✅ |
| REML / ML / GCV | ✅ | ✅ |
| Extended families (NB, quasi, Tweedie, Beta) | ✅ | ✅ |
| Side constraints (`gam.side`) | ✅ | ✅ |
| `gam.check` diagnostics | ✅ | ✅ |
| Adaptive smooths (`:ad`) | ✅ | ✅ |
| Spherical splines (`:sos`) | ✅ | Port of mgcv's kernel (`m = -2…4`, default 0); degrees like mgcv (`xt=Dict(:units=>:radians)` to opt out). `sp` values are portable between the packages |
| Soap film (`:so`) | ✅ | ⚠️ approximation |
| `t2()` tensor construction (Wood–Scheipl–Faraway) | ✅ | ✅ (verified against mgcv: matching columns, penalty count, ranks, supports) |
| Linear functional terms (matrix args) | ✅ | ❌ |
| `bam(discrete=TRUE)` covariate discretization | ✅ | ✅ `bam(...; discrete=true)` — 1-D smooths (any basis), `te`, `bs=:re` and factor-`by` terms are binned; numeric-`by`, `ti` and `t2` stay dense. Within the binned smooths, `:tp`/`:ts`/`:cr`/`:cs`/`:cc`/`:ps`/`:cps`/`:bs` are also *constructed* at the unique values (the `n`-row basis is never formed); `:ad`/`:gp`/`:fp`/`:lo`/`:ds` are built densely then binned — same answer, higher one-off cost. Approximate by covariate rounding, as in mgcv |
| Smoothing-parameter-uncertainty `Vc` / `unconditional=TRUE` | ✅ | ✅ `vcov_corrected`, `edf2`, and `unconditional=true` in `predict`/`smooth_estimates`/`derivatives`/`posterior_samples` |

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
| Basis dimension check (`k.check`) | ✅ | ✅ `k_check()` — **reproducible by default**; mgcv's draws from R's global RNG (`sample()`, `R/plots.r:220`) so repeated calls differ unless you `set.seed()` first. Pass `seed = nothing` for mgcv's behaviour |
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
