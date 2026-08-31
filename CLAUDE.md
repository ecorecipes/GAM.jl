# CLAUDE.md

GAM.jl is a pure-Julia reimplementation of R's mgcv: penalized regression
spline GAMs with REML/ML/GCV/UBRE/NCV smoothness selection. 53 source files,
96 test files, 16 vignettes. Julia 1.11 is the minimum (`Project.toml`).

Correctness here means **agreement with mgcv**, usually asserted to a stated
numeric tolerance against values obtained from R. When changing a basis or a
fitting path, expect to justify the change in those terms.

## Verifying

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**This is the only run that counts as "the suite passes."** It is what CI runs
(`.github/workflows/CI.yml` → `julia-actions/julia-runtest`), and it differs
from an ad-hoc invocation in ways that hide real failures:

- `Pkg.test()` uses `--check-bounds=yes`, which inhibits vectorization and
  changes floating-point reduction order. Assertions tighter than the
  arithmetic guarantees pass under a plain `julia` call and fail here.
- One process exposes load-order bugs a per-file harness hides (`CSV` was
  imported far below the first include that needed it, and only ever failed
  in a single-process run).
- `julia --project=. test/runtests.jl` does not work at all: test-only
  dependencies such as `StableRNGs` live in `[extras]`, not the main project.

Per-file or segmented runs are good for *iterating* on a known failure. They
are not evidence of green. Read the log for `Some tests did not pass` rather
than trusting an exit code — wrapping a command as `( cmd; echo $? )` or
piping it to `tail` reports the wrapper's status, not the run's.

R-comparison tests skip themselves when R or the relevant package is missing,
or with `GAM_SKIP_RCALL=true`. A green local run with R absent is a weaker
claim than CI's; say which one you mean.

When something fails only under `Pkg.test()`, suspect an over-tight assertion
(float `==`, or `rtol` at/below ~1e-9) before suspecting the engine.

## Docs

```bash
julia --project=docs docs/make.jl        # output: docs/build/ (gitignored)
```

Run this after touching any docstring. Julia attaches a docstring to the
*immediately following* expression, so inserting a definition (or a second
blank line) between a docstring and its function silently orphans it — the
docs build is what catches that. `test/test_docstrings.jl` separately requires
every GAM-owned export to carry an attached docstring.

## Vignettes

```bash
cd vignettes
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using GAM, Plots'          # warm the cache first
quarto render 01_introduction/01_introduction.qmd --to gfm
```

Warm the cache before rendering: Quarto captures whatever the first cell
prints, so a `using GAM` that triggers a rebuild writes Julia's
`Precompiling packages...` progress into the checked-in `.md`. It only bites
the first vignette rendered after a source change, which makes it easy to miss
in a batch. `generate_data.jl` regenerates the datasets with fixed seeds.

GFM renders (`.md` plus `*_files/` figures) are checked in; HTML/PDF are not.

### Diffs that look like changes and are not

- **SVG churn.** Plots.jl numbers its `clipPath` ids per process, so a
  re-render rewrites `clip550` → `clip820` throughout with byte-identical path
  data. Normalize (`sed -E 's/clip[0-9]+/clipN/g'`) before judging whether a
  figure actually moved.
- **Timings.** Vignettes 11 and 15 print wall-clock numbers; those move every
  render and say nothing about correctness.

## Randomness

The rule, applied consistently:

> Randomness that is an implementation detail of a reported quantity is seeded
> by default; randomness the caller explicitly asked for is not.

So `k_check`, `appraise` (its default `method = :simulate` draws the QQ
reference quantiles) and `derivatives` (simultaneous intervals) default to
`seed = 11`; pass `seed = nothing` for the unseeded behaviour. The sampling
functions — `posterior_samples`, `fitted_samples`, `smooth_samples`,
`predicted_samples` — stay unseeded, because a fixed default would silently
break any Monte Carlo workflow that calls one twice. `test/test_gratia.jl`
pins both halves of that split; do not "fix" one half without the other.
