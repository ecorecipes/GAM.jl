# Changelog

## 0.2.0 (2026-08-23)

A full review-and-fix release: four deep code reviews, three package-wide
fix passes, live validation against R (mgcv 1.9.4, scam, qgam, evgam,
gratia, egpd, gamFactory), and a new nested-effects feature. Commits
`ff81e2c..HEAD` on this branch.

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
  residuals on the wrong scale for non-identity links), and now returns a
  typed **`PartialResiduals` struct** (long format: `smooth`, `xname`, `x`,
  `residual`; Tables.jl-compatible, with a plot recipe) instead of a `Dict`.
- **`appraise` defaults to simulated reference bands** (`method = :simulate`,
  the gratia/`qq.gam` convention); `method = :normal` retains the previous
  normal-theory behavior.
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
- qgam/ELF smooth fits: the extended-family P-IRLS could declare convergence
  on a spurious low-curvature plateau, leaving fitted quantiles uncalibrated
  (the empirical fraction below the fitted quantile did not track τ);
  convergence is now stationarity-checked and quantile fits are calibrated,
  with fraction-below-quantile regression tests.
- Gamma/inverse-link (and other inverse-type link) fits: invalid means are
  now handled consistently — P-IRLS enforces valid μ and `predict` no longer
  returns unclamped out-of-domain values (previously a Gamma model could
  predict large negative means while reporting converged).
- `bs=:fp` (fractional polynomial) without `fx=true` no longer throws:
  fractional-polynomial smooths are now auto-unpenalized by construction
  (their few-column basis carries no wiggliness penalty), documented and
  tested.
- TPRS with more basis functions than unique covariate values now raises an
  informative error (previously a crash at one sample size and a silent
  edf≈0 degenerate fit at another), matching mgcv's behavior.
- The smoothing-scale floor is now relative to the response magnitude, so
  models of very small-magnitude responses are no longer oversmoothed.
- `gamm()` accepts `offset=`; gamlss's smoothing-method keyword is
  canonicalized (`method=` accepted alongside the legacy `sp_method=`).
- `scam(method=:GCV)` optimizer: warm-started PIRLS evaluations across large
  smoothing-parameter jumps could report inconsistent (deviance, edf) pairs,
  making the cyclic golden-section search converge to a criterion-worse point
  than R scam's optimum. The optimizer now cold-starts a coarse global scan
  per coordinate, refines within the bracket, and guards against unconverged
  warm evaluations; the selected criterion now matches or beats R scam's on
  the reference model, and the live-R comparison asserts it.
- `bam()`: the three remaining unprotected Cholesky factorizations (hat/edf
  helper, Gaussian fast path, final rebuild) now use the escalating-ridge
  recovery, so near-singular models fit through `bam()` wherever `gam()`
  succeeds; the bam outer loop also gained the core loop's score-based EFS
  convergence, eliminating flat-ridge walks to the smoothing-parameter clamp.
- `gam_nl`: the EFS update target now uses per-group log-determinant
  derivatives for overlapping penalties (te margins), fixing data-dependent
  slow convergence; a final gradient-stopped Newton polish (with the index
  scale direction projected out) runs after smoothing-parameter convergence,
  restoring exact constant-offset absorption; LogLink helpers are
  overflow-guarded.

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
  `offset`/`select` support; protected Cholesky in the inner P-IRLS loop and
  EFS traces computed by per-block solves instead of full inverses.
- `scam` records real outer-iteration counts and stores `NaN` (not a
  mislabeled GCV score) in the `reml` slot; extended families honor
  `gam_control(scale_est=...)` (analytic Fletcher) and reach a stationary
  NB θ on fixed-smoothing-parameter paths; Fletcher's correction uses prior
  weights.
- The plain-`@formula` (FormulaTerm) path applies side-identifiability
  constraints identically to `@formulak` (previously skipped, leaving
  overlapping tensor margins unconstrained).
- Bayes: `PriorSpec.b` wired to the samplers; canonical PSIS-LOO estimator;
  coefficient summaries with parametric covariates.
- gratia/diagnostics/plots: genuinely simulated simultaneous intervals;
  batched derivative computation; working contour and smooth-estimate plot
  recipes.

### Added
- `NestedControl`/`nested_control()` for `gam_nl`, matching the package's
  control-struct convention (iteration limits, tolerance, and a wired
  `trace` option); the loose `outer_maxit`/`newton_maxit`/`tol` keywords
  are deprecated aliases.
- Offsets for multi-parameter models: `gamlss`, `evgam`, and the
  vector-formula `gam`/`qgam` routes accept `offset=` — a single length-n
  vector (offset on the first linear predictor, e.g. log-exposure) or a
  length-K per-parameter vector of `nothing`/vectors. Offsets enter all
  solvers (Newton/EFS and RS/CG), are stored on the model
  (`MultiParameterModel.offsets`, a new field), and are used by
  `predict`/`fitted` on training data, with `predict(..., offset=)` for new
  data. Verified against mgcv `gaulss` with an offset (fitted-location
  max-abs 2.1e-5). Not supported for Bayesian gamlss fits (errors clearly).
- `GamModel.criterion` field storing the optimized GCV/UBRE criterion value
  for criterion-fitted models (NaN otherwise; note this adds a positional
  field to `GamModel`).

- **Nested effects** (Fasiolo et al. 2025 / gamFactory): `s_nest()` with
  `trans_linear`, `trans_nexpsm`, and `trans_mgks` inner transformations,
  fitted by `gam_nl()` (with `gam()` auto-routing); delta-method
  `predict(se=true)`; O(n·nn) kernel smoothing via fixed neighborhoods;
  `inner_coef`; live gamFactory comparison tests. `gam_nl` accepts
  `offset=` and `weights=` (forwarded by `gam()`; unsupported options
  error), handles categorical parametric terms, and fits with a
  Gauss–Newton/Fisher Hessian built from one η-Jacobian (exact chain-rule
  gradient, AD fallback) — roughly 5× faster per outer iteration cold and
  ~0.5 s warm on the three-effect benchmark, with adaptive EFS damping,
  scale-aware inner-ridge updates, and a unit-norm index reparameterization
  for well-conditioned standard errors.
- Influence measures: `StatsAPI.leverage` and `StatsAPI.cooksdistance` for
  `GamModel` fits.
- `predict(::GammModel, newdata; type=:link/:response, se=true)` with
  delta-method standard errors (conditional on BLUPs and smoothing
  parameters).
- `ginla(select=...)` convenience for per-smooth coefficient selection.
- Elementwise R-parity tests: smoothing parameters, coefficients, prediction
  SEs, and AIC vs mgcv; new `bs=:cc` and `bs=:fs` comparisons (Gaussian-CR
  agreement: sp log-diff 0.0000, coefficients 8.7e-8, SEs 5.1e-7); Fletcher
  scale vs `summary(m)$scale` (1.7%), GAMM prediction SEs vs `mgcv::gamm`
  (cor 0.966), and `appraise` reference bands vs
  `gratia::qq_plot(method="simulate")` (4.1e-6). A known scam-GCV optimizer
  divergence is pinned with `@test_broken` (see Fixed in later releases).
- Benchmarks regenerated against the fixed code (`benchmark/results.txt`,
  2026-08-23): overall geometric-mean speedup 11.16× vs R, with SCAM at
  1.98× reflecting that `scam()` now performs full GCV optimization like
  R's scam.
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

### Documentation

- The Documenter site (docs/) was fully refreshed against the current code:
  the BAM page no longer claims covariate discretization, the SCAM page
  documents the GCV default / `criterion` field / SE caveat, the diagnostics
  page covers the typed `PartialResiduals`, simulated `appraise` default,
  new k-index semantics, and influence measures, and new pages document
  nested effects and index the 12 vignettes. The mgcv-comparison page now
  reports the measured elementwise parity numbers.
