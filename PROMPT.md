# Review prompt

A starter prompt for handing this repository to GitHub Copilot (or another
agent) for a comprehensive review. Point the agent at this file — "follow
PROMPT.md" — or paste the body below.

---

Read `.github/copilot-instructions.md` first, then `CLAUDE.md`, then perform a
comprehensive review of this repository. GAM.jl is a Julia reimplementation of
R's mgcv; correctness means numerical agreement with mgcv to stated tolerances,
not "the tests pass".

Prioritise these recently-changed, highest-risk areas:

- `src/reml.jl` — analytic first/second derivatives of `log|S_λ|₊` that let
  ForwardDiff chain through a reparameterization that is not itself
  differentiable
- `src/outer.jl` — Newton smoothness selection and its EFS fallback
- `src/nested.jl` — `gam_nl` multi-start over (index direction, smoothing
  parameter), and the LAML criterion used to pick a winner
- `src/mpfit.jl` — `mp_efs_outer`'s convergence score and penalty rank
  threshold
- `src/qgam.jl` — bootstrap stream derivation
- `ext/GAMTuringExt/` — seeded MCMC sampling
- `test/runtests.jl` — the out-of-process RCall probe

For each finding give: the file and line, why it is wrong, and either a
reproduction or the specific measurement that would settle it. Rank by
severity. Prefer a short list of substantiated findings over a long list of
impressions.

Pay particular attention to:

1. Assertions that cannot fail, or that no longer test what their comment
   claims — especially where a recent change altered which code path runs.
2. Numerical tolerances that are tighter than the arithmetic can guarantee, or
   loose enough to admit the bug they were written to catch.
3. Places where a fit can return a materially worse answer while reporting
   `converged = true`.
4. API inconsistencies in the seeded-by-default vs caller-supplied randomness
   split.
5. Claims in comments, docstrings or the CHANGELOG that the code no longer
   supports.

Constraints — these are deliberate, do not "fix" them:

- `sp_optimizer = :efs` is the default by measurement. Do not change it.
- Do not loosen a failing assertion to make it pass, and do not tighten one
  because it looks loose.
- Do not simplify the subprocess RCall probe in `test/runtests.jl`.
- Do not bump the version or register the package.

Verification. The authoritative command is:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

It takes ~55 minutes and needs test-only dependencies from `[extras]`;
`julia --project=. test/runtests.jl` will not work. If you cannot run it in
your environment, use `GAM_SKIP_RCALL=true` (R is likely unavailable to you) or
`GAM_RCALL_ONLY=true`, and say so explicitly.

Report exactly which checks you ran and which you could not. Do not describe
unverified reasoning as if it were tested — ending with "I could not verify X"
is a better answer than implying you did.
