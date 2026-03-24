# [Bayesian Inference](@id bayesian)

GAM.jl supports Bayesian GAM fitting via integration with
[Turing.jl](https://turing.ml). The key idea is to convert smooth terms
into their random-effects representation using `smooth2random`, then sample
from the posterior using MCMC.

## Key Components

### `smooth2random`

Converts a penalized smooth into a mixed-model (random effects) form suitable
for Bayesian priors. The smooth penalty becomes a Gaussian prior on the
random effects coefficients, with the smoothing parameter controlling the
prior precision.

```julia
re_smooth = smooth2random(constructed_smooth)
```

### `PriorSpec`

Specifies prior distributions for smoothing parameters and other model
components:

```julia
prior = PriorSpec(
    smoothing_parameter = InverseGamma(1.0, 0.005),   # prior on λ
    scale = InverseGamma(2.0, 1.0),                     # prior on σ²
)
```

### `BayesGamModel`

A Turing-compatible model object that wraps the GAM specification:

```julia
bm = BayesGamModel(formula, data;
    family = Gaussian(),
    prior = PriorSpec(),
)
```

## Example

### Bayesian GAM with Turing.jl

```julia
using GAM, DataFrames, Turing

n = 200
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# Create Bayesian model
bm = BayesGamModel(
    @gam_formula(y ~ s(x, k=15, bs=:cr)),
    df;
    prior=PriorSpec(
        smoothing_parameter=InverseGamma(1.0, 0.005),
    ),
)

# Sample with NUTS
chain = sample(bm, NUTS(), 2000)
```

### Posterior Summary

```julia
# Extract smooth function posterior
post = posterior_samples(bm, chain; n_samples=100)

# Credible intervals from posterior
using Statistics
mean_curve = mean(post.fitted; dims=2)
lower = [quantile(post.fitted[i, :], 0.025) for i in axes(post.fitted, 1)]
upper = [quantile(post.fitted[i, :], 0.975) for i in axes(post.fitted, 1)]
```

## How It Works

1. **smooth2random**: each smooth `s(x, k=K)` is split into a fixed-effects
   part (null space of the penalty) and a random-effects part (penalized
   coefficients)
2. **Priors**: the random effects get `Normal(0, σ_smooth)` priors where
   `σ_smooth` is controlled by the smoothing parameter prior
3. **Sampling**: Turing.jl's NUTS sampler explores the posterior jointly
   over regression coefficients and smoothing parameters

This approach produces fully Bayesian credible intervals that account for
smoothing parameter uncertainty — unlike the frequentist intervals from `gam()`.

## See Also

- [Diagnostics](@ref diagnostics) for `posterior_samples` and `fitted_samples`
- [API Reference](@ref api-reference) for `BayesGamModel`, `smooth2random`, `PriorSpec`
