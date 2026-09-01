# Changelog

## Unreleased

## 0.3.0 (2026-08-31)

The release that closes the largest remaining gaps against mgcv: `bam`'s
covariate discretization, the Wood, Pya & Säfken smoothing-parameter-uncertainty
correction, REML/ML criteria evaluated the way mgcv evaluates them, a basis
that was a stub made real (Duchon splines, which had silently delegated
to a thin-plate spline) and one that was one-dimensional made spatial (`:gp`),
and two new vignettes.

Three review rounds fed into it. The first found the engines matching mgcv more
closely than earlier rounds had assumed, and relocated the real risks: a fix
applied to only one of five P-IRLS loops, an identifiability constraint correct
on training data but not off it, and two undocumented conventions that made
correct side-by-side code compare different models. The second was a
documentation round — writing a tutorial exercises feature *combinations* that
unit tests do not, and it surfaced five source bugs the suite had missed. The
third was the first exhaustive single-process run of the whole test suite,
which found three more that segmented runs had hidden.

**Upgrading from 0.2.0: read the breaking changes first.** The tensor `k`
convention changed, and a model written against the old one builds a different
basis size without raising an error.

### Breaking changes

- **Tensor scalar `k` is now per-marginal, matching mgcv.** `te(:x, :z, k=8)`
  builds an 8×8 tensor product; previously a scalar was a *total* dimension
  hint split as `round(Int, k^(1/d))` per margin, so the same call meant
  roughly 3×3. mgcv models now port with `k` unchanged; GAM.jl code written
  against the old convention should switch to the vector form
  (`k = [5, 5]`) to keep its basis size. Applies to `te`, `ti` and `t2`.
- **`smooth2random` yields one variance component per level for factor-`by`
  smooths** (mgcv's convention — `smoothCon` replicates the smooth per level,
  each with its own smoothing variance). Previously all levels were lumped
  into a single component, a strictly smaller model class. Breaking for
  downstream consumers that assumed one block per smooth; the Turing and
  MixedModels extensions iterate blocks generically and needed no change.
  Verified against mgcv: per-level `Z·Zᵀ` to ~1e-14, and a simulation with a
  413× spread in per-level variance recovers each to 4 significant figures.
- **`retain_X` defaults to `false` under `bam(...; discrete=true)`** (still
  `true` for dense fits): the `n × p` model matrix is never built — matching
  mgcv, which retains no model matrix — and `model_matrix(m)` reassembles it
  bitwise on demand. `m.X` is therefore empty on a discrete fit.
- **P-IRLS no longer declares convergence on a failed or heavily halved
  step.** Previously a step halved to 2⁻²³ of its length could satisfy the
  relative-deviance test and return a badly wrong fit flagged `converged`
  (InverseGaussian+log at high dispersion: deviance 427.7 against mgcv's
  299.6). `scam`/`scasm` gained the analogous `step_ok` guard.

- **`k_check` is now reproducible by default.** Its p-value comes from a
  randomization test, and the shuffles previously drew from the global RNG, so
  ten calls on one fitted model returned sixteen distinct p-values spanning
  0.41–0.70. `seed` now defaults to `11`. Note mgcv is *not* reproducible
  here: `k.check` calls bare `sample()` on R's global RNG (`R/plots.r:220`),
  so repeated mgcv calls differ unless the user sets `set.seed()` first. Pass
  `seed = nothing` for that behaviour, or any integer for a different fixed
  stream.
- **`appraise` and `derivatives` are reproducible by default too, on a stated
  rule.** `appraise`'s default `method = :simulate` draws its QQ reference
  quantiles from the global RNG, so two vignette figures were being redrawn
  differently on every render — ~850 changed path elements — with nothing in
  the printed output hinting anything was random. `derivatives` had the same
  unseeded default for its simultaneous-interval critical value. Both now
  default to `seed = 11`, matching `k_check`; pass `seed = nothing` for the old
  behaviour. The rule, now applied consistently: **randomness that is an
  implementation detail of a reported quantity is seeded by default; randomness
  the caller explicitly asked for is not.** So `posterior_samples`,
  `fitted_samples`, `smooth_samples` and `predicted_samples` keep drawing fresh
  values — seeding those would silently break every Monte Carlo workflow that
  calls them twice. A test now pins both halves of that split.
- **`discretize_covariates` now bins by mgcv's rule** — exact unique values
  where there are few enough, otherwise an equally spaced grid over the
  observed range — replacing quantile midpoints. The exported utility and what
  `bam(...; discrete=true)` actually fits against now share one implementation
  and agree by construction.
- **`bs=:sos` now reads latitude and longitude in DEGREES, matching mgcv.**
  Previously GAM.jl required radians while `mgcv::s(..., bs="sos")` takes
  degrees, so porting a model between the two silently rescaled the
  coordinates by 57×. mgcv converts inside `makeR`
  (`R/smooth.r`: `pi180 <- pi/180; la <- la*pi180; lo <- lo*pi180`), which is
  called from both its constructor and its `Predict.matrix` method; GAM.jl now
  converts at a single point too, and stores the resolved unit in the
  prediction cache so a fit and its predictions cannot disagree.

  **This will silently change results for existing code that passes radians.**
  To keep the old behaviour, pass `xt = Dict(:units => :radians)`:

  ```julia
  s(:lat, :lon, bs = :sos, k = 50)                              # degrees (new default)
  s(:lat, :lon, bs = :sos, k = 50, xt = Dict(:units => :radians))  # previous behaviour
  ```

  As a guard, a fit whose coordinates all fall within `|lat| ≤ π/2` and
  `|lon| ≤ π` — the range radian data occupies — warns that it is reading them
  as degrees and names the opt-out. An invalid `xt[:units]` is rejected in
  `s()` itself rather than being silently ignored. Note also that both
  packages expect **latitude first**; a `s(:lon, :lat, bs=:sos)` example in
  the smooth documentation had the arguments the wrong way round and has been
  corrected.
- **`bs=:sos` now uses mgcv's exact spherical-spline kernel.** The basis was
  previously an approximation: the planar thin-plate kernel
  `d^(2m-2)log(d)` applied to great-circle distance, keeping only positive
  eigenpairs and extending to the data by Nystrom. It is now a direct port of
  mgcv's construction — Wahba (1981) reproducing kernels via `makeR`
  (`R/smooth.r:2882-2988`), including the `m = 0` Wendelberger dilogarithm
  series from `src/misc.c:39-73`, the truncated eigendecomposition keeping the
  largest-*magnitude* eigenpairs (mgcv's `slanczos(R, k, -1)`, so negative
  eigenvalues are retained rather than discarded), the constraint absorption
  through `QR(U'Tc)`, and the `1/sd` column rescaling.

  Penalty orders `m = -2, -1, 0, 1, 2, 3, 4` are all supported, and the
  **default `m` is now 0** (mgcv's), where it was previously 2.

  Verification: `rksos` reproduces mgcv's C routine bit-for-bit (max relative
  error exactly 0 across `[-1, 1]`, including the branch discontinuity and the
  antipodal and coincident cases); `makeR` agrees to ≤2e-13 for every `m`; the
  generalized eigenvalues of `(S, X'X)` — which determine EDF as a function of
  the smoothing parameter — agree with mgcv's to 5e-8; and fitting at mgcv's
  selected `sp` reproduces mgcv's fit to 5e-9 (EDF 13.217991 in both). On the
  vignette's global dataset both packages now agree on **every printed digit**
  (EDF 45.955, AIC 358.802, scale 0.0865, RMSE 0.0642), where GAM.jl
  previously gave EDF 45.354 and RMSE 0.0756.

  A practical consequence: unlike `bs=:tp`, **`sp` values are portable between
  the packages** for `sos`. Freely-selected fits can still differ where the two
  smoothing-parameter optimizers (EFS here, outer Newton in mgcv) land
  differently — a difference in the optimizer, not the basis.

  The basis now yields `k-1` columns after the centering constraint (matching
  `smoothCon(..., absorb.cons=TRUE)`), where the old construction gave `k`.
- **The thin-plate (`bs=:tp`, `bs=:ts`) knot rule now follows mgcv's
  `max.knots`.** GAM.jl previously dropped to a rank-`k` Nyström approximation
  as soon as `n > max(3k, 200)`; mgcv subsamples only above `max.knots`
  (default 2000) and then keeps 2000 knots. Every thin-plate fit with more
  than 200 observations was therefore a different — and measurably worse —
  model than mgcv's. The effect is largest for multi-dimensional smooths: for
  `s(u, v, k=30)` at n = 300 the edf error against mgcv falls from 5.4e-2 to
  9.2e-8 and the maximum fitted-value error from 2.5e-2 to 1.0e-9. In 1-D at
  n = 500, k = 20 the edf was 11.35 against mgcv's 11.44 and is now 11.4386.
  The cap is settable per smooth via `s(x, ...)` with `xt[:max_knots]`,
  mirroring `s(..., xt = list(max.knots = ))`.

  Fits with `n > 200` will change. They are closer to mgcv than before, but
  pinned values derived from the old approximation will move. The eigenproblem
  is now solved at full size, so cost rises with `n`: a k = 10 fit at n = 5000
  goes from 0.003 s / 6 MiB to 0.09 s / 122 MiB. To keep that affordable the
  top-`k` eigenpairs are now extracted by Lanczos iteration with full
  reorthogonalization rather than a dense `eigen`, which is the same strategy
  mgcv uses (`slanczos`); GAM.jl remains 2.7–3.8× faster than mgcv across
  n = 200…5000.

- **Thin-plate smooths are now translation invariant.** Each covariate is
  mean-centred before the semi-kernel and polynomial null space are built, and
  the shift is stored and re-applied when predicting — matching mgcv's
  `shift`. Previously, adding a constant to a covariate changed the basis
  parameterization and the reported smoothing parameter (shifting `x` by 100
  moved `sp` from 0.010121 to 0.011088) while leaving the fit itself alone.

- **`bs=:gp` gains `xt[:corfun]` and `xt[:params]`**, and a new `:mgcv_m32`
  correlation function reproducing mgcv's `gp` default (`gpE` type 3),
  `(1 + E)·exp(-E)` — Matérn 3/2 with length-scale `rho/√3`, i.e. without the
  `√3` the standard parameterization carries. (This entry is superseded: `bs=:gp` has since become a direct port
  of mgcv's construction — default correlation, range, knots and null space
  all match, and edf agrees with mgcv to ~3e-12 at fixed `sp`; see the
  round above. The legacy named correlations remain available via
  `xt[:corfun]` for backward compatibility.)

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

- **`bs=:ad`: `m` now sets the number of adaptive sub-penalties, not a spline
  order.** This follows mgcv, where `m` is `p.order` (default 5) and the
  smoothing basis is always a cubic P-spline with a second-order difference
  penalty. Previously `m` was read as a spline order, so the same `m` built a
  different model in each package. Passing `m` to an adaptive smooth now emits
  a one-time warning; `xt = Dict(:n_penalties => n)` is the explicit spelling.

  Note that mgcv's default `k` for `bs="ad"` is **40** where GAM.jl's generic
  default is 10, so an out-of-the-box `s(x, bs=:ad)` still differs between the
  two packages — pass `k` explicitly when porting.

### Added

- **`bam(...; discrete=true)` — covariate discretization, as in
  `mgcv::bam(discrete=TRUE)`.** Each supported smooth is stored as its basis
  at the *unique* covariate values plus an integer index vector, instead of an
  `n × p` block. Discretized: 1-D smooths, `te` tensor smooths, `bs=:re`
  random effects (factor and random slope) and — since the follow-up round —
  factor-`by` smooths. Still dense: numeric-`by` terms and `ti`/`t2`, which
  carry per-marginal reparameterizations the discrete path does not
  implement; unsupported terms fall back silently, changing the
  representation and not the fit.

  **It is an approximation, by construction.** Covariates with at most
  `discrete` distinct values (default 1000; pass an integer to change it) are
  represented exactly; beyond that they are rounded onto an equally spaced
  grid. The rounding is the only source of error — the accumulation arithmetic
  is exact to 1e-15. Fitted values agree with the dense fit to ~1e-3 of their
  range and total EDF to ~1e-4 relative, but **individual smoothing parameters
  can move substantially** (Δlog λ up to 2.37 observed), so compare fitted
  values and EDF rather than `sp` elementwise. GAM.jl's discrete fit agrees
  with *mgcv's* discrete fit to 4.25e-4 of the fitted range — closer than
  either package agrees with its own dense fit (1.39e-3 and 2.32e-3).

  The `X'WX` kernel is ~60× faster at `k=20` and ~450× at `k=100` (n = 10⁶),
  but end-to-end gains are Amdahl-bounded and can be negative: a Normal fit
  with 4 × `s(k=20)` at n = 2·10⁵ measures **0.95×, slightly slower**, because
  Gaussian accumulates `X'WX` once so binning is pure overhead. The same shape
  with `Poisson()` is 1.71×, and 6.09× at `k=100`. Peak memory now falls too
  (superseding an earlier note here): the dense block is no longer built
  under `discrete=true` and smooths keep their reduced `m × k` bases, so a
  4 × `s(k=20)` cr fit at n = 10⁶ peaks at 1554 MB against 3053 dense, with
  retained design storage down ~50× (a tensor block is 8.13 MB against a
  1716.61 MB dense block; a 200-level random effect 766.75 → 6.03 MiB).

- **`bam(...; retain_X = false)` drops the model matrix after fitting**, from
  587.5 MiB to 7.6 MiB at n = 10⁶. `GamModel.X` duplicates data already held
  per smooth — every smooth block is bitwise identical to the corresponding
  `ConstructedSmooth.X` — so `model_matrix(m)` reassembles it bitwise on
  demand, with no basis re-evaluation. Default: `true` for dense fits,
  `false` under `discrete=true` (see the breaking-changes list above).

- **Per-marginal `k` for tensor smooths**: `te(:x, :z, k = [4, 7])` sets the
  marginal basis dimensions directly. (A scalar `k` is also per-marginal now
  — see the breaking-changes list above; it previously specified a total.)

- **`summary(m)` now prints an mgcv-style model summary.** It previously fell
  through to `Base.summary`'s type-name fallback (`"GamModel"`), which is the
  first thing an mgcv user types. The parametric and smooth tables now also
  carry R's significance-code column (`***`/`**`/`*`/`.`) with its legend, and
  the footer reports the selection criterion (`-REML`/`-ML`/`GCV`/`UBRE`)
  alongside the scale estimate and `n`, matching `summary.gam`'s layout.

- **`ScatFamily`** — mgcv's `scat()` scaled-t family for outlier-robust
  regression, with `ν` and `σ` estimated alongside the smooths. Derivatives
  are bit-identical to mgcv's `scat()$Dd`; on contaminated data RMSE against
  truth is ~3.8× better than a Gaussian fit.
- **`na_action`** on `gam`/`bam`/`gamm`/`gam_nl` with mgcv's `na.omit`
  semantics (`:fail` remains the default), plus `na_omit_rows` for aligning
  results with the original table. Weight and offset validation is wired at
  every entry point (negative weights now raise an informative error rather
  than a `DomainError` from `sqrt`).
- **Vector `sp` for multi-penalty smooths** (`:ad`, `te`/`t2`, factor-`by`,
  `fs`): each penalty gets its own fixed smoothing parameter, validated
  against the penalty count at fit time. Scalar `sp` still broadcasts.
- **Factor-`by` smooths discretize under `bam(...; discrete=true)`** via a
  shared per-level basis plus level/cell index vectors (the `X'WX` kernel is
  block-diagonal over levels — measured 126× faster than dense accumulation
  at L=8 and 895× at L=32), and their penalties are stored as `L` narrow
  `k×k` copies with offsets instead of a materialised `I_L ⊗ S_k`
  (2500× less penalty storage at L=50, and `L²` less penalty work per
  smoothing-parameter iteration). Numeric-`by`, `ti` and `t2` still fall back
  to dense.
- **`edf2`, `ref_df`, `vcov_corrected`, `has_vc` and `unconditional=true`**
  across `predict`/`smooth_estimates`/`derivatives`/`posterior_samples`: the
  Wood, Pya & Säfken (2016) smoothing-parameter-uncertainty correction `Vc`,
  computed lazily on first access. `aic(m)` now returns what mgcv's `AIC(m)`
  reports (the `edf2`-based df); `conditional_aic(m)` gives mgcv's `m$aic`.
- **REML/ML criteria matched to mgcv**: the REML score is evaluated at
  mgcv's profiled `reml.scale` rather than the Fletcher estimate (Gamma
  agreement 4.5e-4 → 3e-13); non-canonical links use mgcv's full-Newton
  working weights in `log|X'WX+S|` (Gamma+log ~330,000× closer); `:ML` uses
  the range-space determinant of `MLpenalty1` (was 1–8% off, now ~4e-16);
  and `log|S|₊` uses mgcv's `gam.reparam` similarity transform, stable to
  within-block smoothing-parameter ratios of 1e24 (previously NaN by 1e16).

- **`bs=:sz` accepts a configurable marginal basis**, matching mgcv, which
  takes it through `xt` (`smooth.construct.sz.smooth.spec`). GAM.jl's `xt` is
  always a `Dict`, so mgcv's list form is the one that maps:
  `xt = Dict(:bs => :cr)`; other `xt` keys are forwarded to the base as mgcv
  does. Supported: `:tp`, `:ts`, `:cr`, `:cs`, `:cc`, `:ps`, `:cps`, `:bs`,
  `:ds` — the singly-penalized bases. Multiply-penalized ones (`:ad`, `:fs`,
  tensors) are rejected with mgcv's own "wrong basis in xt" wording, as mgcv
  rejects them too; `:gp` is rejected for a different reason (it is singly
  penalized but has no unconstrained construction path yet). Parity with mgcv
  at free `sp`: `:cr` edf 8.2761 vs 8.2770, `:ps` 7.9197 vs 7.9199, `:cc`
  6.7406 vs 6.7386, deviances agreeing to ~1e-3. **The default is unchanged**,
  asserted three ways and verified to discriminate — moving the default to
  `:cr` produces six failures.

  This needed an `absorb_cons = true` keyword on the `cr` and P-spline
  constructors: `sz` needs the RAW marginal so per-level constants stay in the
  span (in mgcv, identifiability constraints are `smoothCon`'s job, not
  `smooth.construct`'s), and only TPRS and Duchon previously exposed an
  unconstrained path. The keyword is strictly additive and default-preserving,
  so every existing caller is unaffected.
- **Duchon splines (`bs=:ds`) are now real.** Previously `:ds` warned once per
  session and fitted an ordinary thin-plate spline — the only registered basis
  that did not do what its name said, and the reason the README carried a
  "one basis is a documented approximation" caveat. Now a direct port of
  mgcv's `smooth.construct.ds.smooth.spec`, `DuchonE`, `DuchonT` and
  `Predict.matrix.duchon.spline`, with the fractional-power kernel, the plain
  QR null-space rotation (not TPRS's QT), and no column rescaling — the three
  places mgcv's `ds` genuinely differs from its `tp`. Penalty scaling matches
  to 9-10 digits (`S.scale` 0.3957335051 against 0.3957335051; the old stub's
  tell was 0.427 against mgcv's 19.60), edf and deviance match at fixed `sp`,
  and smoothing parameters transfer both ways. `m = c(2, 0)` reduces to a
  thin-plate spline in 1-D as the theory requires, agreeing to 3.1e-8 — an
  independent check that does not route through mgcv.
  Duchon's second order goes in `xt`: mgcv's `m = c(2, 0.5)` is
  `m = 2, xt = Dict(:s => 0.5)`, and the vector-`m` error now names that
  spelling. Bases agree with mgcv up to per-column signs (R's `qr()` uses
  LINPACK, Julia's LAPACK), which is unobservable in fits, edf or penalties.
- **Multi-dimensional Gaussian-process smooths (`bs=:gp`)**, previously 1-D
  only — which left the basis missing its main use, spatial smoothing. Ported
  from mgcv's `smooth.construct.gp.smooth.spec`: knots are unique covariate
  *combinations* (not unique values per column), distances are Euclidean, and
  the default range is the largest knot-to-knot distance. 2-D fitted values
  agree with mgcv to 4.6e-13 at fixed `sp`, `rho` to full printed precision,
  and edf exactly (20.93675274 both); a known 2-D surface recovers at RMSE
  0.093 against noise sd 0.30. The 1-D path is bit-unchanged and still pinned
  by its 59-assertion parity suite.
  Two behaviours worth knowing, both asserted rather than merely documented:
  covariates are **centred but not scaled**, exactly as in mgcv, so the kernel
  is isotropic in the covariates' own units and rescaling one covariate is a
  genuinely different model; and GAM.jl's default `k` for a 2-D `s()` is 30
  where mgcv's is `d + 1 + 30`, so pass `k` explicitly when porting. The model
  space is invariant to covariate order (6.7e-16 at fixed `sp`), though free
  fits can stop ~2% apart in `sp` on the flat optimum. User-supplied `knots=`
  remains 1-D only and raises an informative error otherwise.
- **Neighbourhood cross validation (`method = :NCV`)**, a port of mgcv's
  `src/ncv.c` and the last outstanding item from the agreed backlog. GCV and
  REML assume independent observations and under-smooth badly on correlated
  data; NCV leaves out a *neighbourhood* of each point instead of the point
  alone. On AR(1) data with rho = 0.9 and a true edf near 3, GCV selects
  edf 27.2-27.7 across three seeds while NCV with a half-width-15 neighbourhood
  selects 10.7-19.2 — and is more accurate, RMSE 0.295-0.505 against GCV's
  0.359-0.531. Default `nei` is leave-one-out, which reproduces GCV-like
  behaviour as expected; `loo_neighbourhoods` and `interval_neighbourhoods`
  build the structures, in mgcv's own `k`/`m`/`ind`/`mi` encoding.
  Correctness rests on two independent checks: the criterion matches mgcv to
  every printed digit at fixed `sp`, and for a Gaussian identity model — where
  the Newton step is exact — it reproduces a brute-force leave-one-out refit to
  7.8e-16, which pins the algebra without reference to mgcv at all.
  **Scoped down deliberately**: mgcv computes analytic derivatives of the NCV
  score to drive a Newton optimizer; this port supplies the criterion and
  selects with the existing derivative-free optimizer — same optimum, more
  iterations, free-fit `sp` within ~0.03% on a flat optimum. Documented in the
  file header rather than left implicit. Vignette 14 gains a worked section
  with an mgcv companion, where the three-way comparison agrees closely across
  packages: GCV selects edf 28.10 (RMSE 0.9999) in both, leave-one-out NCV
  28.24 (1.0021) in both, and a half-width-15 neighbourhood recovers edf 4.47
  in both against a truth of about 3. Worth knowing when porting: mgcv's `nei`
  fields are `a`/`ma` (dropped) and `d`/`md` (predict), and mgcv **silently
  falls back to leave-one-out** if `a` or `ma` is missing, so a mis-named list
  looks like it worked.
- **A docstring-attachment guard** (`test/test_docstrings.jl`). Julia binds a
  docstring to whatever expression immediately follows it, so inserting a
  helper — or even two blank lines — between a docstring and its definition
  silently detaches it: `?name` goes empty and nothing else breaks. That
  happened three times in this branch (`@gamm_formula`/`cqcheck`/`check_qgam`
  documented on private helpers; a dangling docstring in `bam.jl`; and
  `_normalize_m` inserted between the `s` docstring and `function s`, orphaning
  the docs for the package's most-used function while the whole suite still
  passed). The guard asserts every exported binding the package owns has an
  attached docstring — 199 of them, all currently documented; re-exports from
  StatsModels/GLM/Distributions are excluded as upstream's to maintain. Verified
  to fail by re-injecting the `s` orphan, which it reports by name.

### Fixed

- **Bayesian sampling is reproducible on request; `Random.seed!` alone was not
  enough.** With `nchains > 1` the chains are sampled on threads, and
  AbstractMCMC does not derive the per-chain RNGs from the global one — so a
  script that called `Random.seed!` still got different posteriors run to run,
  and a convergence-diagnostics test failed on Windows at one commit and
  passed at the next with no code change between them. `gam(...; priors=...)`
  and `gamlss(...; priors=...)` now take `seed`, which passes an explicit rng
  to AbstractMCMC so each chain's seed is derived from it. Verified with two
  threads: the same seed reproduces the chains bit-for-bit, a different seed
  changes them, and the unseeded default still varies — MCMC draws are
  randomness the caller asked for, so `seed = nothing` remains the default.
  The `scam` and `gamm` Bayesian paths do not take `seed` yet.
- **`gam_nl` could return a much worse fit and report success.** The joint
  (index direction, smoothing parameter) problem is non-convex, and the EFS
  step halves when the score does not improve — which shrinks `max_change`
  until it passes the convergence test, so a single start can stop early with
  `converged = true`. Measured on the same data and the same seed: Windows
  returned `cor(fitted, y) = 0.808` where macOS returned `0.983`, with
  `log sp[1]` of −0.74 against −6.51, a factor of ~320 in λ. Nudging one
  covariate by 1 part in 10⁸ reached the better optimum on Windows, so the
  minimum was reachable and only the path to it was fragile. `gam_nl` now
  refits from fixed alternative starting directions and keeps the best LAML
  score (`nested_control(n_starts = 3)`, `n_starts = 1` restores the old
  behaviour), so the answer depends on the data rather than the arithmetic
  path. `NestedGamModel` gained a `criterion` field holding that score. Note
  this makes a nested fit ~`n_starts`× more expensive, which is the price of
  not silently returning the wrong answer.
- **The `:so` soap-film "approximation" caveat was wrong, and is removed.**
  The README, `index.md`, `smooths.md` and `mgcv.md` all described `:so` as a
  grid-PDE *approximation* of mgcv's exact method, whose "fits will differ".
  But mgcv's soap film **is itself a grid-PDE method** — `setup.soap`
  (`R/soap.r:171-268`) builds a square-celled grid, assembles a sparse PDE
  matrix and takes a sparse LU — and GAM.jl matches it on the grid rule, the
  5-point Laplacian, the cyclic boundary basis on arc length, the delta-forced
  interior knots and the bilinear grid-to-point interpolation. The two are
  different constructions of the same model class, not a port and a degradation
  of it. They genuinely differ in how boundary cells enter the solve, stencil
  scaling, `k` semantics, and interior knots — mgcv **errors** without
  user-supplied `knots=` (`soap.r:419`) where GAM.jl places them automatically
  — so fits are not elementwise comparable and `k` does not carry over.
  On Ramsay's horseshoe, mgcv's own canonical soap-film benchmark, GAM.jl was
  more accurate on all five seeds tested: mean RMSE 0.0801 against 0.1037 (23%
  lower) using about 43% of the effective degrees of freedom. That is one
  benchmark with REML-selected smoothing parameters and not a matched-edf
  comparison, so it supports "competitive to better on the standard test case",
  not a general claim. A port was assessed as feasible (mgcv's `soap.c` is 386
  self-contained lines) and **rejected**: it would trade a measurably better
  smoother for elementwise agreement and require breaking the interface to
  demand user knots.
  `test/test_soap_benchmark.jl` (19 assertions, no R needed — `fs.test` and
  `fs.boundary` are closed forms, ported with `soap.r` citations) pins RMSE at
  **mgcv's own level of 0.10**, so it fails if this basis ever regresses to
  merely matching mgcv. With this, **no basis in the package is a degraded
  port**.
- **NCV now selects smoothing parameters with analytic derivatives**, closing
  the divergence recorded when it landed. `ncv_score_grad` ports the `deriv > 0`
  branch of mgcv's `Rncv` (`ncv.c:311-320`, `368-397`, `389`), reconstructing
  `dbeta/drho` and `dw2/drho` from GAM.jl's own variance helpers where mgcv
  passes them in from `gam.fit3.r`, and reusing the Woodbury identity against a
  single Cholesky rather than mgcv's preconditioned CG. Selection moved from a
  derivative-free simplex to BFGS: **8.96× faster** (1.210 s to 0.135 s), and
  closer to mgcv on every axis — `log_sp` error 9.99e-4 to 7.7e-5, edf error
  1.54e-3 to 1.2e-4, and the criterion now agrees with mgcv at all 8 printed
  digits. The free-fit `sp` gap noted in the file header falls from ~0.03% to
  ~0.0018%. Still ~4× slower than mgcv's compiled C.
  The gradient is checked against central finite differences across Gaussian,
  Poisson and Binomial, single and multiple smoothing parameters, and interval
  neighbourhoods (worst relative error 1.7e-8), and against ForwardDiff of an
  INDEPENDENT reimplementation that shares no code with the analytic path — and
  asserts the values agree too, since two consistent mistakes would otherwise
  produce matching gradients. No Hessian: BFGS already removes ~9× of the
  iterations, so a full Newton step was judged not worth the validation burden.
- **`sp_optimizer = :newton` no longer throws on a third of model classes.**
  It errored with a `MethodError` on `te`/`ti`/`t2`, `bs=:ad`, `bs=:fs` and
  `select = true` — 14 of 42 model-by-method combinations — because the Newton
  step takes an autodiff Hessian and the penalty reparameterization
  (`_stable_penalty_factor`, mgcv's `gam.reparam`) is `Float64`-only, so
  ForwardDiff cannot pass through a multi-penalty block. It now attempts Newton
  and degrades to the EFS step on any failure, warning once and naming the
  affected constructions. The option had **no test coverage at all**, which is
  why this went unnoticed; `test/test_sp_optimizer.jl` (32 assertions) now pins
  the default, EFS/Newton agreement, the fallback on every previously-throwing
  class, and the shrinkage-basis case, and was verified to fail without the fix.
  The default is unchanged, so no existing fit moves.

  **EFS stays the default, on measured grounds.** Across those 42 combinations
  the two agree to within `4e-5` of criterion almost everywhere; Newton is
  materially better only on the shrinkage bases (`3.3e-2`), and EFS is never
  materially better. A Newton default would optimize `s(x)` by Newton and
  `te(x, z)` by EFS within one model — worse than a consistent default for a
  gain confined to two basis types. Making the reparameterization AD-compatible
  or porting mgcv's analytic REML derivatives is the prerequisite for
  revisiting it, and is recorded as the actual blocker.
- **Three defects found by the first exhaustive suite run.** Segmented runs
  (forced by this machine stopping any whole-suite attempt) had left gaps, so
  the suite was verified file-by-file against an enumerated, diff-checked
  partition of all 91 test files. That surfaced:
  - **A widened smoothing-parameter bound degraded SCAM.** `LOG_SP_BOUND` was
    raised 15 → 30 so the shrinkage bases could drop a term, but SCAM's coarse
    scan built its grid as `range(-BOUND, BOUND; length = 13)` — a fixed COUNT,
    so the resolution silently halved from 2.5 to 5.0 log units. Worse, the
    wider search let SCAM's golden-section bracket a poorer local minimum on a
    multimodal constrained GCV surface (0.10114 against mgcv's 0.10080, where
    GAM.jl had matched or beaten it). The grid is now specified by STEP, and
    SCAM has its own documented `SCAM_LOG_SP_BOUND = 15`: the wide bound exists
    for null-space-penalised shrinkage bases, which shape-constrained bases are
    not, so the search range and the representable range are kept separate.
  - **Constrained GAMLSS fits reported `converged = false` while being
    correct.** Convergence used an ABSOLUTE deviance tolerance; `c_crit = 1e-4`
    on a deviance of order 1 is reasonable but on a larger one is far stricter
    than intended, and shape-constrained fits jitter the deviance as the active
    constraint set flips. It is now relative to the deviance. One case remains
    (`:efs` on exactly-linear data under a monotone constraint, where the
    optimum is at λ → ∞) and is marked `@test_broken` with its diagnosis rather
    than hidden: the fit is right (correlation 1.0 with the truth), only the
    flag is wrong. It previously "passed" only because the old bound clamped
    the iteration and froze the deviance — convergence by clamping.
  - **A type-stability guard broke when GP smooths went multi-dimensional**
    (`_gp_E` and `GPPredictCache` take matrices and a per-covariate shift now).
    Updated, and extended to guard the multi-dimensional inner loop too, which
    is where a boxed accumulator would now appear.
- **A GP correlation-function test asserted on the wrong quantity.**
  `test_tprs_parity.jl` checked that the default `:gp` correlation (mgcv's
  √3-free type 3) differs from the √3-carrying `:matern32` by comparing
  **edf** — but the target there is a plain sine with `k = 10`, so every one
  of these bases fits to saturation and lands at edf ≈ 9 whatever the
  correlation function. The two differed by 1.9e-5 relative while their design
  matrices differed by 1.01 and their penalties by 13.1, so the check was
  failing (and would equally have passed) for reasons unrelated to its claim.
  It now asserts on the basis itself: the default is elementwise identical to
  `:mgcv_m32`, and `:matern32` is a genuinely different `X` and `S`. The
  underlying `corfun` handling was correct throughout; only the test was
  measuring the wrong thing.
- **`bs=:sz` now emits one penalty per factor level, as mgcv does.** GAM.jl
  built the deviation term with a single summed penalty, which turns out to be
  exactly mgcv's `id`-supplied branch (`R/smooth.r:2281-2286`) applied
  unconditionally — mgcv's *default* is one penalty per level. With one shared
  smoothing parameter the term could not shrink a weakly-deviating level while
  leaving a strongly-deviating one alone, so it reported markedly more
  effective degrees of freedom: on a three-region seasonal model, deviation
  edf 14.63 against mgcv's 10.2412. It now gives 10.244, deviance 158.456
  against 158.457, and fitted values within 4.09e-5 (1.2e-5 of range). The
  change is a strict *decomposition* of the previous penalty rather than a
  different model space — the per-level penalties sum back to the old one to
  ~1e-15 — so bases and coefficient counts are unchanged. Note that `:sz`
  smoothing parameters do **not** transfer between the packages: GAM.jl
  absorbs an orthonormal level contrast at construction where mgcv applies a
  non-orthonormal one afterwards, so λ is on a different scale and not by a
  constant factor. Vector `sp` for an `:sz` term now takes one value per level.

Fixed (all five found by writing the vignettes above):

- **Every extended family's estimated extra parameter was corrupted whenever
  an `offset=` was supplied.** `_null_deviance` fitted its intercept-plus-offset
  null model using the *caller's own mutable family object*, and
  `pirls_extended` re-estimates the extra parameter in place, so the null fit
  overwrote the value the real model had just converged to. Only offsets
  triggered it, because the offset-free branch returns a closed-form deviance
  without fitting. The fit itself was always correct — coefficients, fitted
  values, edf and deviance — but the *reported* parameter was not: NegBin θ was
  45% off (2.094 → 1.139), Beta φ 19.9% off, Tweedie `p` affected, and a wrong
  θ propagates into standard errors and AIC. With the fix, θ agrees to 2e-5 and
  fits are offset-invariant to 7.5e-8.
- **The `log λ` upper bound of 15 was too low for the shrinkage bases to
  shrink.** `:cs` stalled at edf 0.286 for an irrelevant term where mgcv reaches
  0.0002, with the optimizer pinned *exactly* at the bound; mgcv's own optima
  for these bases are `log λ` 16.8 (`ts`) and 22.61 (`cs`). Raised to a single
  documented `LOG_SP_BOUND = 30.0` replacing 12 literal occurrences across five
  optimizers, so the bound can no longer drift between methods. `:cs` now
  reaches edf 5.0e-5. Two pinned test values moved, both of which had been
  masking clamped fits — notably a dense-vs-discrete comparison that "agreed to
  1e-10" because *both* sides were clamped to exactly 15.0, agreement by
  construction rather than by accuracy; freed, they agree at 1.6e-10 honestly.
- **`smooth_estimates` and friends failed on any `by=` smooth.** The default
  evaluation grid was built from the smooth's own covariates only, so the `by`
  column was absent and prediction raised `FieldError`. Numeric `by` now
  evaluates at `z = 1`, returning the varying coefficient f(x) itself; factor
  `by` returns the grid per level with the level recorded in a new `by_level`
  field on `SmoothEstimates`. Applies to `smooth_estimates`, `derivatives`,
  `partial_residuals` and `data_slice`.
- **`gam()` and `bam()` gained `knots=`**, mgcv's mechanism for setting a
  smooth's knots explicitly — most importantly a cyclic smooth's period, which
  previously could only be controlled by coding the covariate so its observed
  range *was* the period. Two knots on a cyclic basis are read as the period
  endpoints with the interior filled in, matching mgcv. On data spanning 0–51
  weeks with a true period of 52, the fitted curve previously failed to join by
  0.2519; with `knots = Dict(:week => [0.0, 52.0])` it joins exactly. Per-margin
  tensor knots are not supported and are ignored rather than half-applied.
- **Vector `m` on a smooth raised a bare `MethodError`.** `s(:x; bs=:bs,
  m=[3,2])` — mgcv's documented form — now raises an `ArgumentError` naming the
  scalar convention and the mgcv mapping, on `s`/`te`/`ti`/`t2`.

- `bs=:gp` is now a direct port of mgcv's Kammann & Wand Matérn spline
  (correlation types via `m`, fixed range, `[1, x]` null space); at mgcv's
  own `sp`, edf agrees to ~3e-12 (was 0.37 off).
- `fs` factor smooths use one penalty per null-space dimension and receive
  mgcv's `scale.penalty` rescale (edf gaps vs mgcv fell from 0.14–0.33 to
  ~1e-4). Note `fs` smoothing parameters still do not transfer from mgcv —
  the `nat.param(type=1)` parameterisation differs; compare fits, not `sp`.
- `k_check` p-values are reproducible (`seed = 11` by default) and
  `gam_check` forwards the same default.

- **Adaptive smooths (`bs=:ad`, `bs=:scad`) now build mgcv's penalty.** The
  weight basis used order-3 splines where mgcv uses order-4, and normalized
  rows that mgcv leaves raw. Penalties now agree with mgcv elementwise to
  ~8e-16; mean `|Δedf|` against mgcv across 20 configurations fell from 1.14
  to 0.20, with 14 of 20 now matching to <0.01 (previously 4 of 20). An
  oversized penalty basis now errors as in mgcv rather than being silently
  clamped.
- **The achieved selection score is recorded for every `gam()` fit.**
  `m.criterion` was `NaN` on all of them — including a plain
  `s(x, bs=:cr)` — because the score was written to the `reml` field
  regardless of which criterion was optimized. `summary`'s footer reads
  `criterion` for GCV/UBRE fits, so those fits silently printed no score line
  at all, while REML/ML fits printed theirs.

  The two fields are now filled according to the criterion actually used:
  `reml` for `:REML`/`:ML`, `criterion` for `:GCV`/`:UBRE`, with the other
  `NaN` — the convention SCAM already followed. Together they are the
  analogue of mgcv's single `b$gcv.ubre`, which likewise stores whichever
  score the selection optimized. The new `sp_criterion(m)` accessor returns
  it without branching on `method`.

  Note this changes `m.reml` for `:GCV`/`:UBRE` fits from the GCV/UBRE value
  to `NaN`; read it via `sp_criterion(m)`, or `m.criterion` directly.

### Performance

- **`bam` performance: the default BLAS thread count can be over 5× slower
  than the best setting.** `bam`'s chunks are tall and thin, so BLAS threads
  beyond about four get too little work to cover synchronisation. Measured idle,
  `n = 100,000` Poisson with four `s(k=20)` smooths: 2.18 s at 1 thread, 1.75 s
  at 4, and **9.41 s at 8** (the default here). Now documented in `bam.md` and
  the `bam_control` docstring with the `BLAS.set_num_threads(4)` remedy. GAM.jl
  does not set this itself — it is global process state and a library should not
  silently change it. Julia-level threading of the accumulation was implemented
  and measured, then rejected: serial at 4 BLAS threads is fastest, Julia threads
  on top made it ~20% slower, and serial-vs-threaded results differed by 2.2e-13,
  uncomfortably close to this package's ~1e-12 mgcv parity tolerances.
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

- **New vignette 16, "Seasonal and Group-Varying Smooths"** — the first fitted
  examples anywhere of cyclic smooths (`bs=:cc`), factor-`by`, numeric-`by`
  (varying-coefficient) terms, and `ti()`. The cyclic wrap is verified exact
  (`|f(0) - f(52)| = 0` against `1.9e-2` for a non-cyclic `:cr` fit), per-level
  smoothing parameters come out monotone in signal strength (amplitude 1.40 →
  sp 1.93; 0.35 → 3.89, which a single shared λ cannot do), and fitted values
  agree with mgcv to `2.4e-6` on the factor-`by` model.
- **Vignette 04 gains a disease-counts arc** — `offset=` for log-exposure rate
  models (fitted nowhere previously, despite being the standard construction
  for incidence data), the naive alternatives shown failing, then
  `NegBinFamily` / `QuasiPoissonFamily` / `TweedieFamily` for overdispersion,
  with `rootogram` diagnostics. On the worked example a Poisson fit understates
  the smooth's standard errors by 2.2-2.6× and expects 1.4 zeros where 15 occur.
- **Vignette 02 gains eight previously-undemonstrated bases** — `:ts`/`:cs`
  (shrinkage, with the null-space penalty shown shrinking an irrelevant term to
  edf 0.0016 where `:tp` leaves 1.68, and the relationship to `select=true`
  spelled out), `:ad` (adaptive: 2.85× lower RMSE on the flat half of an
  inhomogeneous function while using 9 fewer effective df), `:bs`, `:cps`,
  `:fp`, `:lo`, and `:sz` conceptually.
- **Vignette 10 gains `bs=:fs`** — per-subject trajectories recovering the
  truth 2.5× better than a random intercept (RMSE 0.167 vs 0.425). Documents
  that `fs` shares a small *fixed set* of smoothing parameters (three:
  wiggliness, intercept, slope — verified invariant at 5/10/15/25 levels)
  rather than one, which is the real contrast with factor-`by`, and that `fs`
  smoothing parameters do not transfer from mgcv.
- **Vignette 05 gains the smoothing-parameter-uncertainty surface** —
  `vcov_corrected`, `edf1`/`edf2`, `conditional_aic`, `sp_criterion`,
  `data_slice`, `vis_gam`/`gamcontour`, with a *measured* coverage experiment
  rather than an assertion: paired over shared replicates, `Vc` improves
  coverage by `+0.005` (n=200) to `+0.043` (n=80), every difference 6-11 Monte
  Carlo standard errors from zero. It also states plainly that `Vc` is a
  partial fix — at n=80 it lifts coverage 0.85 → 0.89, short of nominal,
  because it corrects for estimating λ, not for smoothing bias.
- **Vignette 11 gains `ginla`** — compared directly against the MCMC posterior
  already in that vignette (22× faster; captures the posterior skewness the
  Gaussian approximation reports as exactly zero by construction), with a
  quadrature-convergence table showing the default `nk=16` understates the
  posterior SD by ~1.7%.
- **Vignette 14 gains its R companion**, the only vignette that lacked one.
- **Nine data generators added to `vignettes/generate_data.jl`**, so every
  dataset with a stated DGP is now reproducible. `data_gev.csv` was the last
  holdout and its original seed proved unrecoverable, so it is regenerated
  from the documented process (parameters still recover: location [2.75, 6.91]
  against a true [3.00, 7.00], shape 0.121 against 0.100). That also resolved
  a contradiction inside vignette 06, which said in one place that the GEV data
  was *not* script-generated and in another that it was.
- **`bs = :sz` gains a fitted example** (vignette 16, with an mgcv companion),
  the last basis demonstrated nowhere. It completes the three-way comparison of
  group-varying constructions: factor-`by` (a free curve and its own smoothing
  parameter per level), `:fs` (exchangeable levels sharing a fixed set), and
  `:sz` (a common curve plus per-level deviations constrained to sum to zero).
  The constraint holds to 6.1e-16, and the deviations read directly: the region
  whose seasonal amplitude sits at the average of the three deviates by
  essentially nothing, which three separately-fitted `by` curves cannot show.
  Deviance matches factor-`by` at equal `k` (157.999 vs 157.697). Noted for
  comparison: the *common* smooth's edf agrees with mgcv (8.45 vs 8.456) while
  the deviation term's does not (14.63 vs 10.241), so compare deviance and
  fitted curves for `:sz`, not per-term edf.
- **`ginla`'s `nk` accuracy is documented at the API**, not only in the
  vignette. The default 16 matches mgcv's own `ginla` default, but it is a
  quadrature resolution rather than an exact setting: measured against a model
  whose posterior is Gaussian in closed form, the posterior SD is understated
  ~1.7% at `nk=16`, halving with each doubling. The default stays at mgcv's
  value; the docstring now says when to raise it.

- **New vignette 15, "Large Data and Spatial Models"**, covering the two
  largest surfaces that previously had no vignette at all: `bam()` and the
  spatial bases. It measures the `gam`/`bam` crossover in time *and*
  allocations while the page renders (rather than asserting it), checks that
  the two fitters agree on EDF and fitted values, shows the cost scaling in
  `k` as well as `n`, verifies that `chunk_size` does not change the answer,
  and runs the same benchmark through `mgcv::gam`/`mgcv::bam` for an absolute
  comparison. It then fits a Markov random field on a 6×6 rook lattice and
  compares `bs=:sos` against `bs=:tp` on spherical data, including a seam test
  across the antimeridian. Each part has an mgcv companion in `R/` fitting the
  same CSVs. `sos` and `mrf` both match mgcv's EDF, AIC, scale and fit to the
  printed digits. (The vignette originally recorded `bam(discrete=TRUE)` as
  unimplemented; it has since been added — see **Added** — and the vignette
  has been corrected and re-rendered.)

- **Documented a reproducibility caveat in `vignettes/README.md`**: a fixed
  seed pins the RNG stream, not the values, so a generator that interleaves
  `rand(rng, Uniform(...))` with `rand(rng, Poisson(...))` can yield a
  different CSV after a Distributions.jl upgrade. `gen_poisson_gamm()` has that
  shape and no longer reproduces the checked-in
  `10_gamm/data_poisson_gamm.csv` byte-for-byte.
- **Two vignette figures were irreproducible by construction, and now are
  not.** The diagnostics and model-selection QQ panels were redrawn differently
  on every render. Fixed at the API rather than the call sites (see
  `appraise`/`derivatives` under Breaking changes), so the vignettes just call
  `appraise(m)` and inherit the reproducible default. Verified by rendering
  twice and diffing the panel.
- **Vignette 11 seeds the document.** The Turing extension samples with the
  global RNG and takes no `rng` argument, so every posterior summary on the
  page moved a little on each render. `Random.seed!` in the setup cell fixes
  the page; giving the extension an `rng` parameter is the better long-term
  fix and is not done here.
- **Re-rendered all sixteen vignettes against the release.** Ten came back
  byte-identical. Of the rest, vignette 01's `gam_check` p-value moved
  0.895 → 0.890 because its checked-in render predated the `k_check` seeding
  fix and was therefore an unreproducible draw; two vignettes had Julia's
  `Precompiling packages...` progress captured into their checked-in `.md` by
  a render that followed a source change, now removed and the warm-up step
  documented in `vignettes/README.md`; and vignette 11's MCMC output drifts in
  the third decimal because that vignette seeds nothing at all.

- **`aic`'s exact relationship to mgcv is now documented and asserted.**
  GAM.jl's `aic(m)` is mgcv's `m$aic` field (`family$aic(...) + 2*sum(edf)`,
  set in `gam.outer`). mgcv's `AIC(m)` is *not* that value: `logLik.gam`
  reports a df based on `edf2`, the Wood, Pya & Säfken (2016) correction for
  smoothing-parameter uncertainty, so `AIC(m) = m$aic + 2*(sum(edf2) -
  sum(edf))`. `edf2`/`Vc` have since shipped (see above), so `aic(m)` now IS
  mgcv's `AIC(m)` and `conditional_aic(m)` gives `m$aic`. For `method="GCV.Cp"`
  fits mgcv leaves `edf2` unset and the two conventions coincide — measured
  agreement there is 8e-5. The smoothing penalty is accounted for in both,
  through the effective degrees of freedom.

- **Documented the two conventions that differed from mgcv**: tensor `k`
  (since aligned to mgcv's per-marginal convention — see the breaking-changes
  list above) and mgcv's `gam()` defaulting to `method="GCV.Cp"` while GAM.jl
  defaults to `:REML`. Both are now in the README, the mgcv
  comparison page, `smooths.md`, and the migration vignette.
- Documented the remaining algorithmic differences with their measured
  consequences (EFS vs mgcv's outer Newton; `scam`'s scan-and-refine vs R
  scam's BFGS; `qgam`'s frozen smoothing parameters during calibration;
  Gauss-Newton with automatic differentiation vs gamFactory's hand-coded
  blocks). Corrected the claim that mgcv also defaults to EFS — it defaults
  to outer Newton.
- **Re-measured `bam`'s crossover, and corrected it.** The earlier figures
  (about 21x slower than `gam` at n = 1,000, break-even near n ≈ 5,000-10,000,
  about 4x faster at n = 100,000) predated the thin-plate allocation fix and do
  not reproduce: the worst case now measured is 1.18x slower, and on a
  single-smooth model `bam` is within 4% of `gam` at every n from 1,000 to
  100,000. The table in `bam.md` was also unqualified, which was the deeper
  problem — a crossover depends on the basis and on `p`, not on `n` alone. The
  same three smooths at `bs=:cr` instead of thin-plate cross over between
  n = 1,000 and 2,000 and reach 3.1x by n = 20,000, where at thin-plate they
  are still 1.18x *slower* at n = 10,000. Now stated per model, with the BLAS
  thread count named, and pointing at memory (and `discrete = true`) as the
  dependable reasons to reach for `bam`.
- Marked the benchmark snapshot provisional: it predates several correctness
  fixes, and the SCAM and QGAM rows are now slower by design.
- Disclosed further gaps against mgcv: `NCV`, `mvn`, gamlss
  `SHASH`/`twlss`, and `paraPen`. (`edf1`/`edf2` with the corrected AIC and
  `scat` were on this list and have since shipped — see above.)

- **The README Quick Start now runs verbatim.** It imported `StatsAPI` after
  `GAM`, which leaves `coef`, `fitted` and the other verbs ambiguous, so the
  advertised first example died on `UndefVarError: coef not defined`. It now
  uses `using StatsBase` and carries a note on the import options.

- **Two over-warnings corrected against measurement.** The SCAM standard-error
  caveat cited a bootstrap-to-analytic ratio as if it were a coverage
  shortfall; measured coverage of the analytic intervals is 0.948 +/- 0.006 at
  the nominal 0.95, and the ratio reflects coefficient-scale spread rather than
  coverage. The `testStat` note framed the simplified rank truncation as a
  general accuracy loss; measured rejection rates match mgcv within Monte Carlo
  error (0.003/0.037/0.077 vs 0.003/0.035/0.070 at alpha = 0.01/0.05/0.10), so
  the note now states the actual consequence: `Ref.df` and the F statistic
  differ from mgcv's edf1-based values, while test size does not.

- **`overall_uncertainty` documented as an estimand choice, not a bug.** The
  two settings correspond to mgcv's `iterms` and `terms` intervals, which
  target different quantities; measured coverages are 0.976 and 0.964.

- **The API reference covers every export and the docs build genuinely gates.** About
  65 exported bindings were missing from the manual, and six cross-references
  in shipped docstrings resolved to nothing. All six are fixed. `api.md` had
  grown past Documenter's hard HTML size limit, so the reference is now three
  pages (Core, Model Types, Diagnostics & Plotting) and the build exits 0.
  `warnonly` still lists `:missing_docs` (intentional: ~340 internal helpers
  stay out of the manual) and `:cross_references`, but the latter now carries a
  comment naming the only two remaining causes -- `statsbase.jl` links to `r2`,
  which is StatsAPI's binding and cannot be documented from this module, and
  `s_nest` links to `trans_linear`/`trans_nexpsm`/`trans_mgks`, whose
  docstrings sit on the types rather than the constructors -- so that the
  entry can be dropped once `src/nested.jl` gains those three docstrings.

- **Vignette reading order.** The sixteen vignettes are no longer presented as
  a flat list: the README gives a suggested order, and the diagnostics vignette
  now says it is a tour of the tools and points to the model-selection vignette
  for the decision workflow.

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
