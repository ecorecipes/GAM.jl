# GitHub Copilot instructions for GAM.jl

GAM.jl is a pure-Julia reimplementation of R's **mgcv**: penalized regression
spline GAMs with REML/ML/GCV/UBRE/NCV smoothness selection. 53 source files,
97 test files, 16 vignettes. Julia 1.11 is the minimum (`Project.toml`).

**Correctness here means agreement with mgcv**, asserted to a stated numeric
tolerance against values obtained from R. A change to a basis or a fitting path
needs to be justified in those terms, not by "tests pass". Many assertions
encode measured mgcv parity to ~1e-12; moving those numbers is a regression
even when nothing goes red.

## Verifying a change

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**This is the only run that counts as "the tests pass."** It is what CI runs.
Takes ~55 minutes and reports ~9618 assertions.

Three traps, all of which have produced a false "green" in this repo:

- **`julia --project=. test/runtests.jl` does not work.** Test-only
  dependencies (`StableRNGs`, `CSV`, `RCall`, …) live in `[extras]`, not the
  main project.
- **`Pkg.test()` runs with `--check-bounds=yes`**, which inhibits
  vectorization and changes floating-point reduction order. Assertions tighter
  than the arithmetic guarantees pass under a plain `julia` invocation and fail
  here. If something fails *only* under `Pkg.test()`, suspect an over-tight
  assertion before suspecting the engine.
- **Read the log, not the exit code.** Wrapping a command as `( cmd; echo $? )`
  or piping it to `tail` reports the wrapper's status. Grep the output for
  `Some tests did not pass`.

Faster subsets while iterating (never as evidence of green):

| variable | effect |
|---|---|
| `GAM_RCALL_ONLY=true` | 17 R-comparison files plus shared runner/probe checks (~836 assertions, ~6 min) |
| `GAM_SKIP_RCALL=true` | skip tests needing R |
| `GAM_REQUIRE_RCALL=true` | fail if required R comparisons are unavailable, empty, or skipped |
| `GAM_TEST_PROGRESS=true` | print each test file to stderr before running it |
| `GAM_DERIVATIVE_AUDIT=true` | opt in to the heavy Symbolics derivative audit |

## R integration

R tests run when R and the relevant package are available, and skip otherwise.
Strict runs set `GAM_REQUIRE_RCALL=true`; it cannot be combined with
`GAM_SKIP_RCALL=true`. `GAM_RCALL_LOG` names a file for the subprocess probe's
stdout/stderr, which CI also uploads as an artifact.
**Do not "simplify" the R availability probe in `test/r_test_support.jl`,
invoked by `test/runtests.jl`.** It runs `using RCall` in a *subprocess* on
purpose: loading RCall maps libR into the
running process, and where R is broken that **aborts** — which a `try`/`catch`
cannot catch, so the whole suite dies with no error and no summary. All three
RCall load sites are gated on that one probe. This is a real bug that was hit,
not defensive decoration.

CI's R job is required and rejects missing, empty, or skipped comparison
suites. Its Ubuntu setup puts `R_HOME/lib` on `LD_LIBRARY_PATH`: embedded R
does not pass through R's launcher, and without that path even `grDevices`
could not resolve `libR.so`. The probe log is retained as an Actions artifact.

## Docs

```bash
julia --project=docs docs/make.jl      # output: docs/build/ (gitignored)
```

Run this after touching any docstring. Julia attaches a docstring to the
**immediately following** expression, so inserting a definition — or a second
blank line — between a docstring and its function silently orphans it. The docs
build is what catches that; `test/test_docstrings.jl` separately requires every
exported symbol to carry an attached docstring.

## Vignettes

```bash
cd vignettes
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using GAM, Plots'        # warm the cache first
quarto render 01_introduction/01_introduction.qmd --to gfm
```

Warm the cache before rendering: Quarto captures whatever the first cell
prints, so a `using GAM` that triggers a rebuild writes Julia's
`Precompiling packages...` progress into the checked-in `.md`. Rendered GFM
(`.md` plus `*_files/`) is committed; HTML/PDF is not. `generate_data.jl`
regenerates the datasets from fixed seeds.

**Two diffs that look like changes and are not.** Plots.jl numbers `clipPath`
ids per process, so a re-render rewrites `clip550` → `clip820` throughout with
byte-identical path data — normalize (`sed -E 's/clip[0-9]+/clipN/g'`) before
judging whether a figure moved. And vignettes 11 and 15 print wall-clock
timings, which change every render and mean nothing.

## Conventions that matter here

**Tolerances are set from measurement, with stated headroom.** Do not loosen an
assertion to make a failure go away, and do not tighten one because it looks
loose. Record the measured value and the margin in a comment next to it. Where
a quantity is genuinely ill-conditioned — concurvity on a rank-deficient design,
for instance — assert the property that *is* well defined rather than picking a
threshold that will fail on someone else's machine.

**When a change alters which code path runs, re-derive the affected assertions
from scratch — don't adjust their tolerances.** Two bugs in this repo came from
assertions whose premise had quietly died: one asserted a fallback that had just
been removed, another asserted single-start fit quality after multi-start was
introduced to fix exactly that variability. Both kept passing locally while
testing the wrong thing.

**Randomness follows one rule:** randomness that is an implementation detail of
a reported quantity is seeded by default (`k_check`, `appraise`, `derivatives`
all default to `seed = 11`); randomness the caller explicitly asked for is not
(`posterior_samples`, `fitted_samples`, `smooth_samples`, `predicted_samples`,
and MCMC via `priors=` stay unseeded, with an optional `seed`). Tests pin both
halves — do not "fix" one without the other. Note `Random.seed!` alone does
**not** make `nchains > 1` MCMC reproducible: the chains sample on threads and
AbstractMCMC does not derive their RNGs from the global one, so only passing
`seed` works. Verify threaded reproducibility with `JULIA_NUM_THREADS=2`; a
single-threaded run cannot see the difference.

**CHANGELOG:** new work goes under `## Unreleased`. Released sections are dated
and must not gain entries afterwards — anchoring an insertion on the first
`### Fixed` heading writes into the released 0.3.0 section, which has happened.

## CI

Six gating jobs — Julia 1.11/ubuntu, Julia 1/ubuntu, macOS, Windows, the docs
build, and the strict R-comparison job.
Windows has historically been where platform-specific defects surface (a
single-index fit converged to a materially worse optimum there while reporting
success, and macOS never reproduced it), so a Windows-only failure is worth
taking seriously rather than retrying.

## Current state

Version 0.3.0 is cut in the CHANGELOG and `Project.toml` but **not registered**.
`sp_optimizer = :efs` is the default by measurement; `:newton` works across all
model classes and is a deliberate non-default, not a limitation.
