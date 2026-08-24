# [Nested Effects](@id nested-effects)

Nested effects are smooths of **estimated** covariate transformations,

```math
s\bigl(\tilde{s}_{\mathbf{a}}(\mathbf{x}_i)\bigr),
```

where the inner transformation ``\tilde{s}`` has parameters ``\mathbf{a}``
estimated **jointly** with the outer spline coefficients. This follows
Fasiolo et al. (2025, [arXiv:2511.19234](https://arxiv.org/abs/2511.19234))
and mirrors the R package
[gamFactory](https://github.com/mfasiolo/gamFactory).

```@setup nested
using GAM, DataFrames, Random, LinearAlgebra, Statistics
Random.seed!(7)
n = 300
X = randn(n, 3)
a_true = normalize([0.7, 0.5, 0.2])
u = X * a_true
y = sin.(1.5 .* u) .+ 0.2 .* randn(n)
df = DataFrame(y=y, l1=X[:,1], l2=X[:,2], l3=X[:,3])
```

## Inner Transformations

| Constructor | Effect | Inner parameters |
|---|---|---|
| `trans_linear()` | Single-index / distributed-lag: ``\tilde{s}_i = \mathbf{a}^\top\mathbf{x}_i`` | index coefficients (second-difference or ridge penalty) |
| `trans_nexpsm()` | Adaptive exponential smoothing of a time-ordered series: ``\tilde{s}_i = \omega_i\tilde{s}_{i-1} + (1-\omega_i)x_i`` | logistic-weight coefficients |
| `trans_mgks()` | Gaussian-kernel smoothing of a covariate over coordinates | per-coordinate log-bandwidths |

`trans_mgks(nn=50)` uses fixed nearest-neighbor sets (the paper's
neighborhoods) for O(n·nn) evaluation; `nn=0` uses all points. Note that
`fitted` uses leave-one-out kernel smoothing during training while `predict`
on the training table includes the self point.

## Fitting

`s_nest` terms go directly in the formula; `gam_nl` fits the model (plain
`gam()` routes automatically when it sees an `s_nest` term):

```@example nested
m = gam_nl(@formula(y ~ s_nest(l1, l2, l3, trans=trans_linear(), k=10)), df)
```

All coefficients — including the inner transformation parameters — are
estimated by penalized Newton with a Gauss–Newton/Fisher Hessian built from
the η-Jacobian (exact chain-rule gradients, AD fallback); smoothing
parameters (one per penalty, inner and outer) by EFS with per-group
log-determinant targets. Identifiability follows the paper: the inner output
is standardized to mean 0 / variance 1, the outer cubic B-spline lives on a
fixed symmetric knot range with an ``s(0)=0`` constraint and linear
extrapolation, and single-index directions are returned unit-norm.

`gam_nl` supports `offset=` and `weights=` like `gam()`; supported families
are `Normal`, `Poisson`, `Bernoulli`/`Binomial`, and `Gamma` with
identity/log/logit links. `select=`, `start=`, and non-REML `method=` are
not supported on this route and error clearly.

## Inspecting the Fit

```@example nested
a_hat = inner_coef(m)          # estimated index direction (unit norm)
round.(a_hat, digits=3)
```

```@example nested
pred, se = GAM.predict(m, df; se=true);
round.((cor(pred, y), median(se)), digits=3)
```

`predict(...; se=true)` returns delta-method standard errors that propagate
the joint uncertainty of **all** coefficients — inner parameters included —
through the composition (differentiating through the training-data
standardization).

## Agreement with gamFactory

The test suite compares fits against R's gamFactory live (when installed):
single-index directions agree to ``|\mathbf{a}^\top\hat{\mathbf{a}}| > 0.999``
and fitted values to correlation > 0.999 for Gaussian and Poisson models;
the vignette below shows both implementations recovering the same index to
three decimals on a shared dataset.

## See Also

- Vignette [12_nested_effects](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/12_nested_effects)
  (with a gamFactory R companion on the same data)
- [Smooth Terms](@ref smooth-terms) for ordinary (fixed-transformation) smooths
- [API Reference](@ref api-reference) for `gam_nl`, `s_nest`, `trans_linear`,
  `trans_nexpsm`, `trans_mgks`, `inner_coef`, `NestedGamModel`
