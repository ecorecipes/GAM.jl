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

#### 1. `k` counts basis functions **per margin** in mgcv, **in total** in GAM.jl

For tensor smooths (`te`, `ti`, `t2`), mgcv's `k` is the dimension of *each*
marginal basis, so the tensor has `k^d` columns. GAM.jl's `k` is the *total*
target dimension, split as `round(Int, k^(1/d))` per margin (floored at 3).
Measured, for `te(x, z)` after the identifiability constraint:

| `k` | mgcv columns | GAM.jl columns |
|-----|--------------|----------------|
| 4   | 15           | 8              |
| 5   | 24           | 8              |
| 9   | 80           | 8              |
| 16  | 255          | 15             |
| 25  | 624          | 24             |

So `te(x, z, k=5)` is a 24-column smooth in mgcv and an 8-column smooth in
GAM.jl — a threefold difference in flexibility from identical source text.
(Because of the floor at 3 per margin, `k=4`, `5` and `9` all give 8 columns.)

**To match mgcv, raise `k` to the power of the number of covariates:**

```julia
# mgcv:  te(x, z, k = 5)          → 5 per margin, 24 columns
# GAM.jl equivalent:
te(:x, :z, k = 25)                # 25^(1/2) = 5 per margin, 24 columns

# mgcv:  te(x, y, z, k = 4)       → 4 per margin, 63 columns
te(:x, :y, :z, k = 64)            # 64^(1/3) = 4 per margin
```

In general `k_julia = k_mgcv^d`. Per-margin dimensions can also be given
directly as a vector, `te(:x, :z, k = [4, 7])`, which sidesteps the
conversion entirely.

Plain `s()` smooths are unaffected: `k` means the same thing in both.

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

One known gap: GAM.jl's **`:ML` score differs from mgcv's** by roughly 1–8%
(e.g. `50.5654` vs `46.6654` on a Gaussian reference fit). mgcv's ML changes
the determinant terms rather than only dropping the `Mp` correction
(`R/gam.fit3.r:545-546` sets `REML <- -1` for ML, feeding a different
determinant path). `:REML`, `:GCV` and `:UBRE` all agree with mgcv to
between `4e-12` and `2e-4`.

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
SEs within 0.2%). AIC agrees within ~0.5 (mgcv's corrected-edf convention).
Nested effects match gamFactory's index directions to |cosine| > 0.999.
Two caveats: smooth-test F statistics use a simplification of mgcv's
`testStat`, so the printed statistic can differ from `summary.gam`'s; and on
numerically flat REML ridges the smoothing parameter is only weakly
identified, so fitted values/EDF — not raw sp — are the meaningful
comparison there.

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
| Gaussian process (`:gp`) | ✅ | ✅ |
| Duchon splines (`:ds`) | ✅ | ⚠️ alias for `:tp` — warns; not Duchon's fractional-order basis |
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
| Spherical splines (`:sos`) | ✅ | Port of mgcv's kernel (`m = -2…4`, default 0); degrees like mgcv (`xt=Dict(:units=>:radians)` to opt out). `sp` values are portable between the packages |
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
