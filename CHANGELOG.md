# Changelog

## 0.2.0 (2026-08-23)

A full review-and-fix release: two deep code reviews, a package-wide fix pass,
live validation against R (mgcv 1.9.4, scam, qgam, evgam, gratia, gamFactory),
and a new nested-effects feature. Commits `ff81e2c..d6e72f9`.

### Breaking / behavior changes

- **SCAM standard errors** are now delta-method transformed (`Vp`/`Ve` match
  the stored exponentiated coefficients, as in R's scam `Vp.t`); all SCAM
  SEs, CIs, and `predict(se=true)` values change by design.
- **`scam()` defaults to `method=:GCV`** with genuine GCV/UBRE criterion
  optimization (matching R's scam); `gam()` auto-routing still passes its own
  method through.
- **Quasi families return `NaN`** from `loglikelihood`/`aic` (R's `NA`
  convention); other families now use full log-likelihoods including
  saturated terms, so absolute AIC values changed.
- **Default scale estimator is Fletcher (2012)** (`gam_control(scale_est=...)`
  selects `:fletcher`, `:pearson`, or `:deviance`); Gamma/Tweedie-type fits
  report slightly different scales than before.
- **`ts`/`cs` shrinkage smooths** are single-modified-penalty constructions
  (one smoothing parameter, `null_space_dim = 0`), matching mgcv, instead of
  the previous double-penalty form.
- **GAMM random effects are unconstrained** (mgcv/lme4 convention; previously
  sum-to-zero), random slopes get separate variance components, and PQL fits
  report family-scale deviance.
- **`qgam` default `err = 0.05`** (R qgam's default; previously an n-dependent
  heuristic reaching 0.5), and quantile residuals use the exact ELF CDF.
- **`partial_residuals` returns working residuals** (previously response
  residuals on the wrong scale for non-identity links).
- **`concurvity(full=true)` returns all three mgcv measures**
  (`worst`/`observed`/`estimate`) as a NamedTuple.
- **`gam_check`/`k_check` use a real mgcv-style k-index** (differenced
  residuals, permutation p-value; low = bad) instead of the previous
  `edf/k'` ratio.
- **`BamControl`** now has only `chunk_size`; `discrete`/`max_unique`/
  `nthreads` are deprecated no-ops (bam never discretized or threaded).
- **Silently ignored options now work or error**: `sp=`/`fx=` are honored on
  every path, `id=` and unsupported `method` values throw, `start=` is wired
  through, and the disabled MixedModels GAMM backend errors instead of
  producing invalid fits.

### Fixed

- Core engine: `predict(type=:terms)` and coefficient names with categorical
  parametric terms; per-penalty EFS log-determinant derivatives in all
  optimizer paths; offset-aware null deviance; saturation guards and
  protected Cholesky factorizations; score-based EFS convergence; honest
  convergence flags and iteration counts; plain-`@formula` categorical
  support; stabilized `log|S|₊` for multi-penalty blocks.
- Bases: tensor products with thin-plate margins predict correctly; linear
  extrapolation for `cr` and B-spline bases; corrected GP (Kammann–Wand)
  basis/penalty pair; TPRS default-order rule and full polynomial null bases;
  mgcv-faithful `fs` (fully penalized) and `sz` (sum-to-zero across levels);
  random slopes for `bs=:re`; corrected null-space dimensions; B-spline
  penalty weights for the adaptive basis.
- Families: GEV/GPD ξ→0 limit derivatives (machine-precision verified); ELF
  deviance residuals; Mp constants in REML/LAML; negative working-Hessian
  handling in GAMLSS; indefinite-Hessian recovery in the multiparameter EFS
  loop and the constrained (OSQP) inner Newton.
- BAM: `Ve`, REML score comparability with `gam()`, step halving,
  `offset`/`select` support.
- Bayes: `PriorSpec.b` wired to the samplers; canonical PSIS-LOO estimator;
  coefficient summaries with parametric covariates.
- gratia/diagnostics/plots: genuinely simulated simultaneous intervals;
  batched derivative computation; working contour and smooth-estimate plot
  recipes.

### Added

- **Nested effects** (Fasiolo et al. 2025 / gamFactory): `s_nest()` with
  `trans_linear`, `trans_nexpsm`, and `trans_mgks` inner transformations,
  fitted by `gam_nl()` (with `gam()` auto-routing); delta-method
  `predict(se=true)`; O(n·nn) kernel smoothing via fixed neighborhoods;
  `inner_coef`; live gamFactory comparison tests.
- Elementwise R-parity tests: smoothing parameters, coefficients, prediction
  SEs, and AIC vs mgcv; new `bs=:cc` and `bs=:fs` comparisons (Gaussian-CR
  agreement: sp log-diff 0.0000, coefficients 8.7e-8, SEs 5.1e-7).
- Vignette 12 (nested effects) with a gamFactory R companion on shared data;
  seeded data-generation script (`vignettes/generate_data.jl`) for all
  narrated datasets; re-rendered vignette suite.
- Prior weights documentation, offset support across fitters, term selection
  (`select=true`) interactions with fixed smoothing parameters.
- CI job running the live R comparison suite (non-blocking), including
  gamFactory installed from GitHub.

### Docs

- README corrected throughout (basis table, family constructors, counts,
  dependency list, scope and limitations) and extended with vignette index
  and nested-effects sections; vignette narratives now match their
  checked-in data; benchmark harness auto-stamps versions/platform;
  `CONTRIBUTING.md` documents the style policy; Manifests untracked.

## 0.1.0

Initial development version.
