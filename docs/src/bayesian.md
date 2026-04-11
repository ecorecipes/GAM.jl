# [Bayesian Inference](@id bayesian)

GAM.jl supports Bayesian GAM fitting through its Turing.jl extension. When
`Turing` is loaded, passing `priors = PriorSpec(...)` to `gam()`, `gamlss()`,
`scam()`, or `gamm()` switches the fit from penalized likelihood to full MCMC.

## Fitting a Bayesian GAM

```julia
using GAM, DataFrames, Turing, Distributions

n = 200
x = sort(rand(n))
y = sin.(2π .* x) .+ 0.3 .* randn(n)
df = DataFrame(x = x, y = y)

m = gam(@formulak(y ~ s(x, k = 10)), df;
    priors = PriorSpec(sds = Exponential(1.0)),
    nsamples = 1000,
    nchains = 2)
```

This returns a [`BayesGamModel`](@ref) whose `chains` field stores the
underlying `MCMCChains.Chains` object.

## Prior specification

`PriorSpec` controls the default priors for the Bayesian backend:

```julia
priors = PriorSpec(
    b = Normal(0, 10),                         # fixed effects
    sds = Exponential(1.0),                    # smooth SDs
    sigma = truncated(Normal(0, 2.5); lower = 0),  # Gaussian residual SD
    phi = truncated(Normal(0, 5); lower = 0),      # dispersion / precision
)
```

Specific parameters can be overridden by name:

```julia
priors = PriorSpec(
    sds = Exponential(1.0),
    specific = Dict(
        "sds_s(x)" => Exponential(0.5),
        "b_(Intercept)" => Normal(0, 100),
    ),
)
```

## Posterior summaries

Bayesian fits expose posterior summaries through the usual StatsAPI-style
methods:

```julia
coef(m)          # posterior means of fixed effects
vcov(m)          # posterior covariance of fixed effects
coeftable(m)     # posterior mean / sd / credible intervals
confint(m)       # equal-tail credible intervals
posterior_samples(m)  # raw fixed-effect draws
```

For model scoring, Bayesian fits also retain pointwise log-likelihood draws:

```julia
ll = pointwise_loglikelihood(m)  # n_draws × n_obs
l = loo(m)                       # PSIS-LOO by default
l_is = loo(m; method = :is)      # raw importance-sampling fallback
p = psis_loo(m)
d = pareto_k_diagnostic(p)
w = waic(m)

l.elpd_loo
l.p_loo
l.looic
l.pareto_k
l.n_eff
d.warning_indices
d.danger_indices
w.elpd_waic
w.p_waic
w.waic
```

`psis_loo` and the default `loo` path use Pareto-smoothed importance sampling to
stabilize leave-one-out estimates. The Pareto-k values diagnose whether the
importance ratios are reliable:

- `k ≤ 0.5`: good
- `0.5 < k ≤ 0.7`: okay
- `0.7 < k ≤ 1.0`: unstable
- `k > 1.0`: very unstable; prefer exact refits or K-fold CV

`pareto_k_diagnostic(...)` summarizes the observations with `k > 0.7` and
`k > 1.0`.

## `smooth2random`

The key bridge to Bayesian fitting is [`smooth2random`](@ref), which converts a
penalized smooth into a mixed-model representation:

```julia
sm = smooth_construct(s(:x, bs = :cr, k = 10), df)
smm = smooth2random(sm)
```

The null space becomes fixed effects (`smm.Xf`) and the penalized wiggliness
becomes one or more Gaussian random-effect blocks (`smm.Zs`).

## Composing custom Turing models

GAM.jl also exposes lower-level building blocks for hand-written Turing models:

```julia
X, smooths, labels = gam_matrices(@formulak(y ~ x + s(x, k = 10)), df)
smm = gam_smooth(:x, df; bs = :cr, k = 10)
```

Use [`smooth_prior`](@ref) inside a custom `@model` when you want the GAM
reparameterization without using the high-level `gam(...; priors=...)` entry
point.

## See also

- [API Reference](@ref api-reference) for `BayesGamModel`, `PriorSpec`,
  `pointwise_loglikelihood`, `psis_loo`, `pareto_k_diagnostic`, `loo`, and `waic`
- [Diagnostics](@ref diagnostics) for `posterior_samples` and `fitted_samples`
