# Contributing to GAM.jl

Thanks for contributing! This note records the project's conventions so the
codebase stays consistent.

## Code style

- **Formatting** is advisory: `.JuliaFormatter.toml` describes the de facto
  style (default JuliaFormatter style, 4-space indent, ~100-column lines, no
  alignment), but the formatter has not been applied wholesale. Format only
  the code you are already changing; avoid drive-by reformatting of untouched
  files — it obscures review diffs.
- **Naming**: `snake_case` for functions and variables, `CamelCase` for
  types, a leading underscore for internal (unexported) helpers.
- **Unicode**: Greek identifiers (`β`, `η`, `λ`, `ζ`, `σ`, …) are used for
  estimation-theory quantities in the specialized fitting code
  (`gamlss*.jl`, `mpfit.jl`, `nested.jl`; `qgam.jl` keeps ASCII identifiers
  with Greek confined to docstring math); the core engine and
  basis files use ASCII identifiers (`beta`, `eta`, `lambda`). Match the file
  you are editing; do not mix ASCII and Greek names for the same quantity in
  one file.
- **Docstrings**: every exported entry point gets a docstring with a
  signature line, a one-paragraph summary, and an `# Example`. Add an
  `# Arguments` section when a function takes more than four keyword
  arguments, and a `# Returns` section when the return shape is not obvious
  from the summary.
- **Errors**: user-facing validation throws `ArgumentError` with a message
  that states what was received, e.g.
  `"method must be :REML, :ML, :GCV, or :UBRE, got :$method"`. Use lowercase
  sentence fragments for argument errors and sentence case for data errors
  ("Response must be non-negative …"). Reserve bare `error()` for internal
  invariant failures.
- **Comments** explain *why*, not *what*; cite the mgcv/paper source for any
  algorithmic convention that is not obvious (e.g. "matches gam.fit3's
  step-halving baseline").

## Testing

- Run the suite with `julia --project=. -e 'using Pkg; Pkg.test()'`, or
  iterate on a single file via
  [TestEnv.jl](https://github.com/JuliaTesting/TestEnv.jl):

  ```julia
  using TestEnv; TestEnv.activate()
  include("test/test_general_fit.jl")
  ```

- R-comparison tests (via RCall) skip automatically when R or the relevant R
  package is missing, or when `GAM_SKIP_RCALL=true` (the default CI test job
  sets this). Use `GAM_REQUIRE_RCALL=true` to reject missing, empty, or skipped
  comparison suites instead. The R-enabled CI job uses this strict mode:

  ```bash
  GAM_RCALL_ONLY=true GAM_REQUIRE_RCALL=true GAM_TEST_PROGRESS=true \
    julia --project=. -e 'using Pkg; Pkg.test()'
  ```

  `GAM_SKIP_RCALL=true` and `GAM_REQUIRE_RCALL=true` are mutually exclusive.
  The load probe always runs in a subprocess, because a broken R library
  can abort rather than throw. Failures print the child's stdout/stderr,
  exit code, and signal; `GAM_RCALL_LOG=/path/to/probe.log` preserves that
  output. CI uploads the probe log as an artifact even on failure.
- New numerical behavior needs a quantitative test — prefer elementwise
  comparisons with explicit tolerances over correlation thresholds, and
  prefer a live R comparison (gated like the existing `*_rcall.jl` files)
  when an R counterpart exists.
- Behavior changes must update `CHANGELOG.md`.

## Vignettes

- Vignette datasets whose data-generating process is narrated in the text are
  produced by `vignettes/generate_data.jl` with fixed seeds — never edit
  those CSVs by hand; change the generator and regenerate, so data and
  narrative cannot drift apart.
- Rendered GFM outputs (`.md` + `*_files/`) are checked in; re-render with
  Quarto after changing a `.qmd` or the package (see `vignettes/README.md`).
  Each vignette has an R companion in its `R/` subdirectory that runs the
  same analysis on the same CSVs with the corresponding R package.

## Release preparation

Registration is a separate maintainer action; merging changes or dispatching
CI does not register the package or change its version.

1. Require a green CI run for the release commit, including the R comparison
   job's `17/17` coverage report. A successful workflow with skipped R
   comparisons is not release evidence.
2. Prepare the chosen version in `Project.toml` and a new dated CHANGELOG
   section from `Unreleased`. Leave already released sections unchanged.
   Make the version/notes change separately from workflow changes: GitHub
   restricts automated tagging of commits that modify workflow files.
3. Run the full `Pkg.test()` command and documentation build. Re-render
   affected vignettes after warming the Julia cache.
4. Enable the [Registrator GitHub app](https://github.com/JuliaRegistries/Registrator.jl)
   on the repository and request registration on the approved release commit,
   following its instructions. Do not create a tag for an unapproved version.
5. After registration, `.github/workflows/TagBot.yml` creates the corresponding
   Git tag and GitHub release. It accepts only Registrator's `JuliaTagBot`
   notifications or an explicit maintainer workflow dispatch.

TagBot has narrowly scoped write permissions for releases and diagnostic
issues. Configure a write-enabled SSH deploy key in `DOCUMENTER_KEY` if tag
pushes should trigger CI/documentation workflows; tags created using only
`GITHUB_TOKEN` do not trigger other workflows. See
[TagBot's SSH setup](https://github.com/JuliaRegistries/TagBot#ssh-deploy-keys).
No deploy key or registration is created by the repository workflows.
