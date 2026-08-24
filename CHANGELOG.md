# Changelog

## Unreleased

Round-5 review follow-up. The review found the engines match mgcv more closely
than earlier rounds assumed, and relocated the real risks: a fix that had been
applied to only one of five P-IRLS loops, an identifiability constraint correct
on training data but not off it, and two undocumented conventions that make
correct side-by-side code compare different models.

### Breaking / behavior changes
- **`loglikelihood` now uses the maximum-likelihood dispersion `deviance/n`**
  for Normal, Gamma and InverseGaussian, matching R's `family$aic` convention
  (shared by `stats::glm` and mgcv) instead of the model's Pearson/Fletcher
  `scale`. `aic`, `bic` and `aicc` shift accordingly for those families; the
  Pearson/Fletcher estimate remains the scale used for inference (standard
  errors and intervals), which is what it is for. The previous docstring
  claimed the old behaviour matched mgcv — it did not.

- **`scam` now enforces the family mean domain** (mgcv's `validmu`) during
  step halving, as `gam` has since 0.2.0. Previously a Gamma model with its
  canonical inverse link could report convergence while a fraction of the
  fitted means lay outside the family's support, with a nonsensical scale and
  no warning (and, on some data, a hard LAPACK failure). Such a fit now warns,
  naming the violation and suggesting a log link, and reports
  `converged = false` rather than claiming success. `scasm` was affected by
  the same omission and is fixed with it.
- **`t2()` penalty-block ordering** now matches mgcv's (blocks ordered by
  descending range-mask), so `pen_ind` and the random-effect block order from
  `smooth2random` line up with `mgcv:::smooth2random` for unequal block sizes
  as well as equal ones. Investigation of a suspected constraint defect found
  none: our fitting-basis constraint is byte-for-byte mgcv's `C`. (mgcv builds
  a *second* constraint, `Cp`, used only for its separate prediction
  parameterization — comparing against that basis is what made the two look
  different. Documented in `_construct_t2`.)
- **`MultiParameterModel.Vc` renamed to `Ve`**, matching `GamModel` and
  avoiding a name collision with mgcv's `Vc` (which means the
  smoothing-parameter-uncertainty correction, something different).

### Added

- **Per-marginal `k` for tensor smooths**: `te(:x, :z, k = [4, 7])` sets the
  marginal basis dimensions directly, alongside the existing scalar form
  (which specifies the *total* dimension).

### Fixed

- Smoothing-parameter selection no longer walks to the clamp along a flat
  criterion in `scam` and the `:general` optimizer path (the fix `bam`
  received in 0.2.0).
- QGAM calibration is substantially faster: the stationarity polish is
  relaxed inside bootstrap replicates, which only need the loss value.
- Thin-plate basis construction — the dominant cost in a typical fit — no
  longer loses type inference during assembly (an `hcat` that dispatches
  generically once SparseArrays is loaded, causing ~580,000 boxed element
  accesses), and avoids per-column copies and a temporary per distance pair.
  Allocations drop 3.9–6.5× (15.2 → 3.9 MiB at n=5000, k=20). Fits are
  numerically unchanged (end-to-end agreement ≤2e-13; exact-mgcv parity
  assertions still hold).

### Documentation
- **`aic`'s exact relationship to mgcv is now documented and asserted.**
  GAM.jl's `aic(m)` is mgcv's `m$aic` field (`family$aic(...) + 2*sum(edf)`,
  set in `gam.outer`). mgcv's `AIC(m)` is *not* that value: `logLik.gam`
  reports a df based on `edf2`, the Wood, Pya & Säfken (2016) correction for
  smoothing-parameter uncertainty, so `AIC(m) = m$aic + 2*(sum(edf2) -
  sum(edf))`. GAM.jl does not compute `edf2` yet (it needs the corrected
  covariance `Vc`), so `aic(m)` is the conditional AIC. For `method="GCV.Cp"`
  fits mgcv leaves `edf2` unset and the two conventions coincide — measured
  agreement there is 8e-5. The smoothing penalty is accounted for in both,
  through the effective degrees of freedom.

- **Documented the two conventions that differ from mgcv**: `k` counts basis
  functions per margin in mgcv but in total in GAM.jl for tensor smooths
  (`te(x, z, k=5)` is 24 columns in mgcv, 8 here — use `k_julia = k_mgcv^d`
  or the new vector form), and mgcv's `gam()` defaults to `method="GCV.Cp"`
  while GAM.jl defaults to `:REML`. Both are now in the README, the mgcv
  comparison page, `smooths.md`, and the migration vignette.
- Documented the remaining algorithmic differences with their measured
  consequences (EFS vs mgcv's outer Newton; `scam`'s scan-and-refine vs R
  scam's BFGS; `qgam`'s frozen smoothing parameters during calibration;
  Gauss-Newton with automatic differentiation vs gamFactory's hand-coded
  blocks). Corrected the claim that mgcv also defaults to EFS — it defaults
  to outer Newton.
- Quantified `bam`'s crossover: about 21x slower than `gam` at n = 1,000,
  break-even near n ≈ 5,000-10,000, about 4x faster at n = 100,000.
- Marked the benchmark snapshot provisional: it predates several correctness
  fixes, and the SCAM and QGAM rows are now slower by design.
- Disclosed five further gaps against mgcv: `edf1`/`edf2` with the
  Wood-Pya-Säfken corrected AIC, `NCV`, `scat`, `mvn`, gamlss `SHASH`/`twlss`,
  and `paraPen`.

## 0.2.0 (2026-08-23)

A full review-and-fix release: four deep code reviews, three package-wide
fix passes, live validation against R (mgcv 1.9.4, scam, qgam, evgam,
gratia, egpd, gamFactory), and a new nested-effects feature. Commits
`ff81e2c..HEAD` on this branch.

### Breaking / behavior changes

- **`t2()` now uses mgcv's Wood, Scheipl & Faraway (2013) construction.**
  Each marginal is split into orthogonal null/range parts and every non-null
  block carries an identity penalty on its own columns, so the penalties are
  diagonal with disjoint support (previously they overlapped and were not the
  WSF construction). The number of penalties changes for d ≥ 3 (7 rather than
  4 for a 3-d smooth), and `t2` fits shift slightly. Verified against mgcv
  1.9.4: identical column count, penalty count, block ranks, disjoint supports
  and null-space dimension in 2-d and 3-d; spanned column space to 4.2e-15;
  fitted values correlate 1.000000 (max-abs 0.002).
- **`smooth2random` follows the mgcv convention.** A `t2` smooth now
  decomposes into one *independent* random-effect block per penalty (its
  `pen_ind`/`rind` match mgcv's exactly), so a `t2` term in the Turing
  extension gets one variance component per block instead of a single shared
  one. `te`/`ti` keep a single structured block — the `pdTens` analogue —
  since `mgcv:::smooth2random` refuses te decomposition outright ("te smooths
  not useable with gamm4: use t2 instead").

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
- Minimum Julia version raised to 1.11 (matching the CI matrix), and the
  `workflow_dispatch` CI trigger restored — both carried over from work that
  had been made directly on the published `main`.
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
