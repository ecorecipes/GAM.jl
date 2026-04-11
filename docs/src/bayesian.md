# [Bayesian Inference](@id bayesian)

GAM.jl supports Bayesian GAM fitting through its Turing.jl extension. When
`Turing` is loaded, passing `priors = PriorSpec(...)` to `gam()` (including the
SCAM and GAMLSS dispatch paths) or `gamm()` switches the fit from penalized
likelihood to full MCMC.

```@setup bayesian
using GAM, DataFrames, Random, Turing, Distributions
Random.seed!(42)
Turing.setprogress!(false)

n = 80
x = sort(rand(n))
y = sin.(2π .* x) .+ 0.3 .* randn(n)
df = DataFrame(x = x, y = y)
```

## Fitting a Bayesian GAM

```@example bayesian
m = gam(@formula(y ~ s(x, k = 10)), df;
    priors = PriorSpec(sds = Exponential(1.0)),
    nsamples = 100,
    nchains = 1);
nothing
```

This returns a [`BayesGamModel`](@ref) whose `chains` field stores the
underlying `MCMCChains.Chains` object.

!!! note
    The documentation uses a short chain to keep the build practical. For real
    analyses, increase `nsamples` (and usually `nchains`) substantially.

## Prior specification

`PriorSpec` controls the default priors for the Bayesian backend:

```@example bayesian
priors = PriorSpec(
    b = Normal(0, 10),                              # fixed effects
    sds = Exponential(1.0),                         # smooth SDs
    sigma = truncated(Normal(0, 2.5); lower = 0),  # Gaussian residual SD
    phi = truncated(Normal(0, 5); lower = 0),      # dispersion / precision
);
nothing
```

Specific parameters can be overridden by name:

```@example bayesian
priors_specific = PriorSpec(
    sds = Exponential(1.0),
    specific = Dict(
        "sds_s(x)" => Exponential(0.5),
        "b_(Intercept)" => Normal(0, 100),
    ),
);
nothing
```

## Posterior summaries

Bayesian fits expose posterior summaries through the usual StatsAPI-style
methods:

```@example bayesian
GAM.coef(m);
GAM.vcov(m);
GAM.coeftable(m);
GAM.confint(m);
posterior_samples(m);
nothing
```

For model scoring, Bayesian fits also retain pointwise log-likelihood draws:

```@example bayesian
ll = pointwise_loglikelihood(m);
l = loo(m);
l_is = loo(m; method = :is);
p = psis_loo(m);
d = pareto_k_diagnostic(p);
w = waic(m);

l.elpd_loo;
l.p_loo;
l.looic;
l.pareto_k;
l.n_eff;
d.warning_indices;
d.danger_indices;
w.elpd_waic;
w.p_waic;
w.waic;
nothing
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

```@example bayesian
sm = smooth_construct(s(:x, bs = :cr, k = 10), df);
smm = smooth2random(sm);
nothing
```

The null space becomes fixed effects (`smm.Xf`) and the penalized wiggliness
becomes one or more Gaussian random-effect blocks (`smm.Zs`).

## Composing custom Turing models

GAM.jl also exposes lower-level building blocks for hand-written Turing models:

```@example bayesian
X, smooths, labels = gam_matrices(@formula(y ~ x + s(x, k = 10)), df);
smm2 = gam_smooth(:x, df; bs = :cr, k = 10);
nothing
```

Use [`smooth_prior`](@ref) inside a custom `@model` when you want the GAM
reparameterization without using the high-level `gam(...; priors=...)` entry
point.

## See also

- [API Reference](@ref api-reference) for `BayesGamModel`, `PriorSpec`,
  `pointwise_loglikelihood`, `psis_loo`, `pareto_k_diagnostic`, `loo`, and `waic`
- [Diagnostics](@ref diagnostics) for `posterior_samples` and `fitted_samples`
