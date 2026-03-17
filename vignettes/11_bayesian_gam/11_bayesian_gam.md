# Bayesian GAMs with Turing.jl
GAM.jl Contributors

- [Introduction](#introduction)
- [Setup](#setup)
- [Example 1: Gaussian GAM](#example-1-gaussian-gam)
  - [Data](#data)
  - [Frequentist reference](#frequentist-reference)
  - [Bayesian fit](#bayesian-fit)
  - [Understanding the priors](#understanding-the-priors)
  - [Posterior summaries](#posterior-summaries)
  - [Accessing posterior samples](#accessing-posterior-samples)
  - [Comparing posteriors: frequentist vs
    Bayesian](#comparing-posteriors-frequentist-vs-bayesian)
  - [Empirical CDF comparison](#empirical-cdf-comparison)
  - [Posterior predictive check](#posterior-predictive-check)
- [Example 2: Poisson GAM](#example-2-poisson-gam)
  - [Data](#data-1)
  - [Frequentist vs Bayesian](#frequentist-vs-bayesian)
  - [Posterior summary](#posterior-summary)
  - [ECDF comparison for intercept](#ecdf-comparison-for-intercept)
- [Example 3: Custom priors — effect on
  smoothing](#example-3-custom-priors--effect-on-smoothing)
- [Example 4: Building custom Turing
  models](#example-4-building-custom-turing-models)
- [Summary](#summary)
  - [Key design choices](#key-design-choices)

## Introduction

A **Bayesian GAM** replaces the penalized likelihood framework of a
standard GAM with a fully probabilistic model. Instead of choosing
smoothing parameters by REML or GCV, the Bayesian approach places
**priors** on the smooth function variability and uses **MCMC** (Markov
chain Monte Carlo) to obtain posterior distributions for all parameters.

The connection between penalized splines and Bayesian models is well
established:

- A penalized spline $f(x) = \mathbf{Z}\mathbf{b}$ with penalty
  $\lambda \|\mathbf{b}\|^2$ is equivalent to a random effect
  $\mathbf{b} \sim N(0, \sigma^2_s \mathbf{I})$ where
  $\sigma^2_s = \sigma^2 / \lambda$
- The **smooth2random** decomposition splits each smooth into a fixed
  null-space part (unpenalized) and a random-effects part (penalized)
- Priors on $\sigma_s$ control the amount of smoothing — this is the
  Bayesian analog of REML estimation of $\lambda$

GAM.jl implements this via a **Turing.jl package extension**: simply
pass `priors = PriorSpec(...)` to `gam()` and the model is automatically
reparameterized and sampled using NUTS (No-U-Turn Sampler).

## Setup

``` julia
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
```

``` julia
using GAM
using Turing
using CSV
using DataFrames
using Distributions
using Statistics: mean, std, var, cor, median, quantile
using StatsAPI: coef, coeftable, confint, fitted, nobs
using Printf
using LinearAlgebra: I
```

## Example 1: Gaussian GAM

### Data

Simulated data with $y = \sin(2\pi x) + \varepsilon$,
$\varepsilon \sim N(0, 0.3^2)$.

``` julia
dat = CSV.read("data_bayes_gaussian.csv", DataFrame)
@printf("n = %d, y range: [%.2f, %.2f]\n", nrow(dat), minimum(dat.y), maximum(dat.y))
```

    n = 200, y range: [-1.48, 1.61]

### Frequentist reference

``` julia
m_freq = gam(@gam_formula(y ~ s(x, k = 10)), dat)
freq_int = coef(m_freq)[1]
freq_σ = sqrt(m_freq.scale)
@printf("Frequentist: intercept = %.4f, σ = %.4f, edf = %.1f\n",
    freq_int, freq_σ, m_freq.edf_total)
```

    Frequentist: intercept = -0.1129, σ = 0.3134, edf = 7.8

### Bayesian fit

To trigger Bayesian fitting, pass a `PriorSpec` to `gam()`. The same
formula syntax is used — the dispatch is automatic:

``` julia
m_bayes = gam(@gam_formula(y ~ s(x, k = 10)), dat;
    priors = PriorSpec(sds = Exponential(1.0)),
    nsamples = 2000, nchains = 2)

show(stdout, MIME("text/plain"), m_bayes)
```

    ┌ Warning: Only a single thread available: MCMC chains are not sampled in parallel
    └ @ AbstractMCMC ~/.julia/packages/AbstractMCMC/oqm6Y/src/sample.jl:544
    Sampling (1 thread)   0%|                               |  ETA: N/A
    ┌ Info: Found initial step size
    └   ϵ = 0.025
    Sampling (1 thread)   0%|▏                              |  ETA: 0:25:05
    Sampling (1 thread)   1%|▎                              |  ETA: 0:12:31
    Sampling (1 thread)   2%|▌                              |  ETA: 0:08:25
    Sampling (1 thread)   2%|▋                              |  ETA: 0:06:22
    Sampling (1 thread)   2%|▊                              |  ETA: 0:05:07
    Sampling (1 thread)   3%|▉                              |  ETA: 0:04:18
    Sampling (1 thread)   4%|█▏                             |  ETA: 0:03:42
    Sampling (1 thread)   4%|█▎                             |  ETA: 0:03:15
    Sampling (1 thread)   4%|█▍                             |  ETA: 0:02:54
    Sampling (1 thread)   5%|█▌                             |  ETA: 0:02:37
    Sampling (1 thread)   6%|█▊                             |  ETA: 0:02:24
    Sampling (1 thread)   6%|█▉                             |  ETA: 0:02:12
    Sampling (1 thread)   6%|██                             |  ETA: 0:02:02
    Sampling (1 thread)   7%|██▏                            |  ETA: 0:01:54
    Sampling (1 thread)   8%|██▍                            |  ETA: 0:01:47
    Sampling (1 thread)   8%|██▌                            |  ETA: 0:01:40
    Sampling (1 thread)   8%|██▋                            |  ETA: 0:01:35
    Sampling (1 thread)   9%|██▊                            |  ETA: 0:01:30
    Sampling (1 thread)  10%|███                            |  ETA: 0:01:25
    Sampling (1 thread)  10%|███▏                           |  ETA: 0:01:21
    Sampling (1 thread)  10%|███▎                           |  ETA: 0:01:17
    Sampling (1 thread)  11%|███▍                           |  ETA: 0:01:14
    Sampling (1 thread)  12%|███▋                           |  ETA: 0:01:11
    Sampling (1 thread)  12%|███▊                           |  ETA: 0:01:08
    Sampling (1 thread)  12%|███▉                           |  ETA: 0:01:05
    Sampling (1 thread)  13%|████                           |  ETA: 0:01:03
    Sampling (1 thread)  14%|████▏                          |  ETA: 0:01:00
    Sampling (1 thread)  14%|████▍                          |  ETA: 0:00:58
    Sampling (1 thread)  14%|████▌                          |  ETA: 0:00:56
    Sampling (1 thread)  15%|████▋                          |  ETA: 0:00:54
    Sampling (1 thread)  16%|████▊                          |  ETA: 0:00:53
    Sampling (1 thread)  16%|█████                          |  ETA: 0:00:51
    Sampling (1 thread)  16%|█████▏                         |  ETA: 0:00:50
    Sampling (1 thread)  17%|█████▎                         |  ETA: 0:00:49
    Sampling (1 thread)  18%|█████▍                         |  ETA: 0:00:48
    Sampling (1 thread)  18%|█████▋                         |  ETA: 0:00:47
    Sampling (1 thread)  18%|█████▊                         |  ETA: 0:00:46
    Sampling (1 thread)  19%|█████▉                         |  ETA: 0:00:44
    Sampling (1 thread)  20%|██████                         |  ETA: 0:00:43
    Sampling (1 thread)  20%|██████▎                        |  ETA: 0:00:42
    Sampling (1 thread)  20%|██████▍                        |  ETA: 0:00:41
    Sampling (1 thread)  21%|██████▌                        |  ETA: 0:00:40
    Sampling (1 thread)  22%|██████▋                        |  ETA: 0:00:40
    Sampling (1 thread)  22%|██████▉                        |  ETA: 0:00:39
    Sampling (1 thread)  22%|███████                        |  ETA: 0:00:38
    Sampling (1 thread)  23%|███████▏                       |  ETA: 0:00:37
    Sampling (1 thread)  24%|███████▎                       |  ETA: 0:00:36
    Sampling (1 thread)  24%|███████▌                       |  ETA: 0:00:36
    Sampling (1 thread)  24%|███████▋                       |  ETA: 0:00:35
    Sampling (1 thread)  25%|███████▊                       |  ETA: 0:00:34
    Sampling (1 thread)  26%|███████▉                       |  ETA: 0:00:33
    Sampling (1 thread)  26%|████████                       |  ETA: 0:00:33
    Sampling (1 thread)  26%|████████▎                      |  ETA: 0:00:32
    Sampling (1 thread)  27%|████████▍                      |  ETA: 0:00:32
    Sampling (1 thread)  28%|████████▌                      |  ETA: 0:00:31
    Sampling (1 thread)  28%|████████▋                      |  ETA: 0:00:30
    Sampling (1 thread)  28%|████████▉                      |  ETA: 0:00:30
    Sampling (1 thread)  29%|█████████                      |  ETA: 0:00:29
    Sampling (1 thread)  30%|█████████▏                     |  ETA: 0:00:29
    Sampling (1 thread)  30%|█████████▎                     |  ETA: 0:00:28
    Sampling (1 thread)  30%|█████████▌                     |  ETA: 0:00:28
    Sampling (1 thread)  31%|█████████▋                     |  ETA: 0:00:27
    Sampling (1 thread)  32%|█████████▊                     |  ETA: 0:00:27
    Sampling (1 thread)  32%|█████████▉                     |  ETA: 0:00:27
    Sampling (1 thread)  32%|██████████▏                    |  ETA: 0:00:26
    Sampling (1 thread)  33%|██████████▎                    |  ETA: 0:00:26
    Sampling (1 thread)  34%|██████████▍                    |  ETA: 0:00:25
    Sampling (1 thread)  34%|██████████▌                    |  ETA: 0:00:25
    Sampling (1 thread)  34%|██████████▊                    |  ETA: 0:00:25
    Sampling (1 thread)  35%|██████████▉                    |  ETA: 0:00:24
    Sampling (1 thread)  36%|███████████                    |  ETA: 0:00:24
    Sampling (1 thread)  36%|███████████▏                   |  ETA: 0:00:23
    Sampling (1 thread)  36%|███████████▍                   |  ETA: 0:00:23
    Sampling (1 thread)  37%|███████████▌                   |  ETA: 0:00:23
    Sampling (1 thread)  38%|███████████▋                   |  ETA: 0:00:22
    Sampling (1 thread)  38%|███████████▊                   |  ETA: 0:00:22
    Sampling (1 thread)  38%|███████████▉                   |  ETA: 0:00:22
    Sampling (1 thread)  39%|████████████▏                  |  ETA: 0:00:21
    Sampling (1 thread)  40%|████████████▎                  |  ETA: 0:00:21
    Sampling (1 thread)  40%|████████████▍                  |  ETA: 0:00:21
    Sampling (1 thread)  40%|████████████▌                  |  ETA: 0:00:20
    Sampling (1 thread)  41%|████████████▊                  |  ETA: 0:00:20
    Sampling (1 thread)  42%|████████████▉                  |  ETA: 0:00:20
    Sampling (1 thread)  42%|█████████████                  |  ETA: 0:00:19
    Sampling (1 thread)  42%|█████████████▏                 |  ETA: 0:00:19
    Sampling (1 thread)  43%|█████████████▍                 |  ETA: 0:00:19
    Sampling (1 thread)  44%|█████████████▌                 |  ETA: 0:00:19
    Sampling (1 thread)  44%|█████████████▋                 |  ETA: 0:00:18
    Sampling (1 thread)  44%|█████████████▊                 |  ETA: 0:00:18
    Sampling (1 thread)  45%|██████████████                 |  ETA: 0:00:18
    Sampling (1 thread)  46%|██████████████▏                |  ETA: 0:00:18
    Sampling (1 thread)  46%|██████████████▎                |  ETA: 0:00:17
    Sampling (1 thread)  46%|██████████████▍                |  ETA: 0:00:17
    Sampling (1 thread)  47%|██████████████▋                |  ETA: 0:00:17
    Sampling (1 thread)  48%|██████████████▊                |  ETA: 0:00:17
    Sampling (1 thread)  48%|██████████████▉                |  ETA: 0:00:16
    Sampling (1 thread)  48%|███████████████                |  ETA: 0:00:16
    Sampling (1 thread)  49%|███████████████▎               |  ETA: 0:00:16
    Sampling (1 thread)  50%|███████████████▍               |  ETA: 0:00:16
    Sampling (1 thread)  50%|███████████████▌               |  ETA: 0:00:15
    ┌ Info: Found initial step size
    └   ϵ = 0.05
    Sampling (1 thread)  50%|███████████████▋               |  ETA: 0:00:18
    Sampling (1 thread)  51%|███████████████▊               |  ETA: 0:00:18
    Sampling (1 thread)  52%|████████████████               |  ETA: 0:00:17
    Sampling (1 thread)  52%|████████████████▏              |  ETA: 0:00:17
    Sampling (1 thread)  52%|████████████████▎              |  ETA: 0:00:17
    Sampling (1 thread)  53%|████████████████▍              |  ETA: 0:00:17
    Sampling (1 thread)  54%|████████████████▋              |  ETA: 0:00:16
    Sampling (1 thread)  54%|████████████████▊              |  ETA: 0:00:16
    Sampling (1 thread)  55%|████████████████▉              |  ETA: 0:00:16
    Sampling (1 thread)  55%|█████████████████              |  ETA: 0:00:16
    Sampling (1 thread)  56%|█████████████████▎             |  ETA: 0:00:15
    Sampling (1 thread)  56%|█████████████████▍             |  ETA: 0:00:15
    Sampling (1 thread)  56%|█████████████████▌             |  ETA: 0:00:15
    Sampling (1 thread)  57%|█████████████████▋             |  ETA: 0:00:15
    Sampling (1 thread)  57%|█████████████████▉             |  ETA: 0:00:14
    Sampling (1 thread)  58%|██████████████████             |  ETA: 0:00:14
    Sampling (1 thread)  58%|██████████████████▏            |  ETA: 0:00:14
    Sampling (1 thread)  59%|██████████████████▎            |  ETA: 0:00:14
    Sampling (1 thread)  60%|██████████████████▌            |  ETA: 0:00:13
    Sampling (1 thread)  60%|██████████████████▋            |  ETA: 0:00:13
    Sampling (1 thread)  60%|██████████████████▊            |  ETA: 0:00:13
    Sampling (1 thread)  61%|██████████████████▉            |  ETA: 0:00:13
    Sampling (1 thread)  62%|███████████████████▏           |  ETA: 0:00:12
    Sampling (1 thread)  62%|███████████████████▎           |  ETA: 0:00:12
    Sampling (1 thread)  62%|███████████████████▍           |  ETA: 0:00:12
    Sampling (1 thread)  63%|███████████████████▌           |  ETA: 0:00:12
    Sampling (1 thread)  64%|███████████████████▋           |  ETA: 0:00:12
    Sampling (1 thread)  64%|███████████████████▉           |  ETA: 0:00:11
    Sampling (1 thread)  64%|████████████████████           |  ETA: 0:00:11
    Sampling (1 thread)  65%|████████████████████▏          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▎          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▌          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▋          |  ETA: 0:00:10
    Sampling (1 thread)  67%|████████████████████▊          |  ETA: 0:00:10
    Sampling (1 thread)  68%|████████████████████▉          |  ETA: 0:00:10
    Sampling (1 thread)  68%|█████████████████████▏         |  ETA: 0:00:10
    Sampling (1 thread)  68%|█████████████████████▎         |  ETA: 0:00:10
    Sampling (1 thread)  69%|█████████████████████▍         |  ETA: 0:00:09
    Sampling (1 thread)  70%|█████████████████████▌         |  ETA: 0:00:09
    Sampling (1 thread)  70%|█████████████████████▊         |  ETA: 0:00:09
    Sampling (1 thread)  70%|█████████████████████▉         |  ETA: 0:00:09
    Sampling (1 thread)  71%|██████████████████████         |  ETA: 0:00:09
    Sampling (1 thread)  72%|██████████████████████▏        |  ETA: 0:00:09
    Sampling (1 thread)  72%|██████████████████████▍        |  ETA: 0:00:08
    Sampling (1 thread)  72%|██████████████████████▌        |  ETA: 0:00:08
    Sampling (1 thread)  73%|██████████████████████▋        |  ETA: 0:00:08
    Sampling (1 thread)  74%|██████████████████████▊        |  ETA: 0:00:08
    Sampling (1 thread)  74%|███████████████████████        |  ETA: 0:00:08
    Sampling (1 thread)  74%|███████████████████████▏       |  ETA: 0:00:07
    Sampling (1 thread)  75%|███████████████████████▎       |  ETA: 0:00:07
    Sampling (1 thread)  76%|███████████████████████▍       |  ETA: 0:00:07
    Sampling (1 thread)  76%|███████████████████████▌       |  ETA: 0:00:07
    Sampling (1 thread)  76%|███████████████████████▊       |  ETA: 0:00:07
    Sampling (1 thread)  77%|███████████████████████▉       |  ETA: 0:00:07
    Sampling (1 thread)  78%|████████████████████████       |  ETA: 0:00:06
    Sampling (1 thread)  78%|████████████████████████▏      |  ETA: 0:00:06
    Sampling (1 thread)  78%|████████████████████████▍      |  ETA: 0:00:06
    Sampling (1 thread)  79%|████████████████████████▌      |  ETA: 0:00:06
    Sampling (1 thread)  80%|████████████████████████▋      |  ETA: 0:00:06
    Sampling (1 thread)  80%|████████████████████████▊      |  ETA: 0:00:06
    Sampling (1 thread)  80%|█████████████████████████      |  ETA: 0:00:05
    Sampling (1 thread)  81%|█████████████████████████▏     |  ETA: 0:00:05
    Sampling (1 thread)  82%|█████████████████████████▎     |  ETA: 0:00:05
    Sampling (1 thread)  82%|█████████████████████████▍     |  ETA: 0:00:05
    Sampling (1 thread)  82%|█████████████████████████▋     |  ETA: 0:00:05
    Sampling (1 thread)  83%|█████████████████████████▊     |  ETA: 0:00:05
    Sampling (1 thread)  84%|█████████████████████████▉     |  ETA: 0:00:05
    Sampling (1 thread)  84%|██████████████████████████     |  ETA: 0:00:04
    Sampling (1 thread)  84%|██████████████████████████▎    |  ETA: 0:00:04
    Sampling (1 thread)  85%|██████████████████████████▍    |  ETA: 0:00:04
    Sampling (1 thread)  86%|██████████████████████████▌    |  ETA: 0:00:04
    Sampling (1 thread)  86%|██████████████████████████▋    |  ETA: 0:00:04
    Sampling (1 thread)  86%|██████████████████████████▉    |  ETA: 0:00:04
    Sampling (1 thread)  87%|███████████████████████████    |  ETA: 0:00:04
    Sampling (1 thread)  88%|███████████████████████████▏   |  ETA: 0:00:03
    Sampling (1 thread)  88%|███████████████████████████▎   |  ETA: 0:00:03
    Sampling (1 thread)  88%|███████████████████████████▍   |  ETA: 0:00:03
    Sampling (1 thread)  89%|███████████████████████████▋   |  ETA: 0:00:03
    Sampling (1 thread)  90%|███████████████████████████▊   |  ETA: 0:00:03
    Sampling (1 thread)  90%|███████████████████████████▉   |  ETA: 0:00:03
    Sampling (1 thread)  90%|████████████████████████████   |  ETA: 0:00:03
    Sampling (1 thread)  91%|████████████████████████████▎  |  ETA: 0:00:02
    Sampling (1 thread)  92%|████████████████████████████▍  |  ETA: 0:00:02
    Sampling (1 thread)  92%|████████████████████████████▌  |  ETA: 0:00:02
    Sampling (1 thread)  92%|████████████████████████████▋  |  ETA: 0:00:02
    Sampling (1 thread)  93%|████████████████████████████▉  |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████  |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████▏ |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████▎ |  ETA: 0:00:01
    Sampling (1 thread)  95%|█████████████████████████████▌ |  ETA: 0:00:01
    Sampling (1 thread)  96%|█████████████████████████████▋ |  ETA: 0:00:01
    Sampling (1 thread)  96%|█████████████████████████████▊ |  ETA: 0:00:01
    Sampling (1 thread)  96%|█████████████████████████████▉ |  ETA: 0:00:01
    Sampling (1 thread)  97%|██████████████████████████████▏|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▎|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▍|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▌|  ETA: 0:00:00
    Sampling (1 thread)  99%|██████████████████████████████▊|  ETA: 0:00:00
    Sampling (1 thread) 100%|██████████████████████████████▉|  ETA: 0:00:00
    Sampling (1 thread) 100%|███████████████████████████████| Time: 0:00:25
    Sampling (1 thread) 100%|███████████████████████████████| Time: 0:00:26
    Bayesian Generalized Additive Model

    Formula: y ~ 1
    Family:  Normal
    Link:    IdentityLink
    Sampler: Turing.Inference.NUTS{ADTypes.AutoForwardDiff{nothing, Nothing}, AdvancedHMC.DiagEuclideanMetric} (2000 samples × 2 chains)

    Parametric coefficients:
    ──────────────────────────────────────────────────────────
                    Estimate  Est.Error   l-95% CI    u-95% CI
    ──────────────────────────────────────────────────────────
    (Intercept)    -0.112893  0.0220463  -0.157117  -0.0687999
    s(x,bs=tp)_f1  -3.4648    1.00123    -5.42677   -1.49016
    ──────────────────────────────────────────────────────────

    Smooth terms: s(x,bs=tp)
    n = 200

### Understanding the priors

`PriorSpec` controls the prior distributions:

- **`sds`**: Prior on $\sigma_s$, the SD of smooth random effects. An
  `Exponential(1.0)` prior is weakly informative, allowing the data to
  determine smoothness.
- **`sigma`**: Prior on $\sigma_{obs}$, the residual SD (Gaussian family
  only). Default: `truncated(Normal(0, 2.5); lower=0)`.
- **`b`**: Prior on fixed-effect coefficients. Default: `Normal(0, 10)`.

``` julia
ps = PriorSpec(sds = Exponential(1.0))
show(stdout, MIME("text/plain"), ps)
```

    PriorSpec:
      b (fixed effects):    Distributions.Normal{Float64}(μ=0.0, σ=10.0)
      sds (smooth SDs):     Distributions.Exponential{Float64}(θ=1.0)
      sigma (residual SD):  Truncated(Distributions.Normal{Float64}(μ=0.0, σ=2.5); lower=0.0)
      phi (dispersion):     Truncated(Distributions.Normal{Float64}(μ=0.0, σ=5.0); lower=0.0)

### Posterior summaries

The `coeftable()` method returns posterior mean, SD, and 95% credible
intervals:

``` julia
ct = coeftable(m_bayes)
show(stdout, MIME("text/plain"), ct)
println()
```

    ──────────────────────────────────────────────────────────
                    Estimate  Est.Error   l-95% CI    u-95% CI
    ──────────────────────────────────────────────────────────
    (Intercept)    -0.112893  0.0220463  -0.157117  -0.0687999
    s(x,bs=tp)_f1  -3.4648    1.00123    -5.42677   -1.49016
    ──────────────────────────────────────────────────────────

Credible intervals at different levels:

``` julia
ci_95 = confint(m_bayes; level = 0.95)
ci_90 = confint(m_bayes; level = 0.90)
@printf("\nIntercept CIs:\n")
@printf("  90%%: [%.4f, %.4f]\n", ci_90[1, 1], ci_90[1, 2])
@printf("  95%%: [%.4f, %.4f]\n", ci_95[1, 1], ci_95[1, 2])
```


    Intercept CIs:
      90%: [-0.1484, -0.0771]
      95%: [-0.1571, -0.0688]

### Accessing posterior samples

The full MCMC chains are accessible via `m_bayes.chains`:

``` julia
chains = m_bayes.chains

# Residual SD posterior
σ_obs = vec(chains[Symbol("σ_obs")].data)
@printf("σ_obs posterior: mean = %.4f, sd = %.4f, median = %.4f\n",
    mean(σ_obs), std(σ_obs), median(σ_obs))
@printf("  95%% CI: [%.4f, %.4f]\n", quantile(σ_obs, 0.025), quantile(σ_obs, 0.975))
@printf("  Frequentist σ: %.4f\n", freq_σ)

# Smooth SD posterior (controls wiggliness)
σ_s = vec(chains[Symbol("σ_s[1]")].data)
@printf("\nσ_s[1] posterior: mean = %.4f, sd = %.4f\n", mean(σ_s), std(σ_s))
@printf("  Larger σ_s → more flexible smooth; smaller → smoother\n")
```

    σ_obs posterior: mean = 0.3157, sd = 0.0160, median = 0.3148
      95% CI: [0.2861, 0.3489]
      Frequentist σ: 0.3134

    σ_s[1] posterior: mean = 1.6165, sd = 0.4330
      Larger σ_s → more flexible smooth; smaller → smoother

### Comparing posteriors: frequentist vs Bayesian

``` julia
bayes_int = coef(m_bayes)[1]
@printf("Intercept: frequentist = %.4f, Bayesian posterior mean = %.4f\n",
    freq_int, bayes_int)
@printf("σ: frequentist = %.4f, Bayesian posterior mean = %.4f\n",
    freq_σ, mean(σ_obs))
```

    Intercept: frequentist = -0.1129, Bayesian posterior mean = -0.1129
    σ: frequentist = 0.3134, Bayesian posterior mean = 0.3157

### Empirical CDF comparison

We compare the posterior distribution of $\sigma_{obs}$ against the
frequentist point estimate using the empirical CDF:

``` julia
# ECDF of σ_obs posterior
σ_sorted = sort(σ_obs)
n_samples = length(σ_sorted)
ecdf_vals = (1:n_samples) ./ n_samples

# Where does the frequentist estimate sit in the posterior?
freq_rank = searchsortedfirst(σ_sorted, freq_σ) / n_samples
@printf("Frequentist σ = %.4f sits at %.1f%% of the posterior ECDF\n",
    freq_σ, 100 * freq_rank)
@printf("  (values near 50%% indicate good agreement)\n")

# Summary statistics of ECDF at key quantiles
for q in [0.025, 0.25, 0.5, 0.75, 0.975]
    idx = clamp(round(Int, q * n_samples), 1, n_samples)
    @printf("  ECDF %.1f%%: σ_obs = %.4f\n", 100q, σ_sorted[idx])
end
```

    Frequentist σ = 0.3134 sits at 45.5% of the posterior ECDF
      (values near 50% indicate good agreement)
      ECDF 2.5%: σ_obs = 0.2861
      ECDF 25.0%: σ_obs = 0.3048
      ECDF 50.0%: σ_obs = 0.3148
      ECDF 75.0%: σ_obs = 0.3261
      ECDF 97.5%: σ_obs = 0.3489

### Posterior predictive check

Compare the posterior mean fitted values to the frequentist fitted
values:

``` julia
# The Bayesian fitted values come from the posterior mean coefficients
bayes_coefs = coef(m_bayes)
freq_coefs = coef(m_freq)
@printf("Number of coefficients: frequentist = %d, Bayesian = %d\n",
    length(freq_coefs), length(bayes_coefs))
```

    Number of coefficients: frequentist = 10, Bayesian = 2

## Example 2: Poisson GAM

### Data

Count data with $\log(\lambda) = 1 + 1.5\sin(2\pi x)$.

``` julia
dat2 = CSV.read("data_bayes_poisson.csv", DataFrame)
@printf("n = %d, y range: [%.0f, %.0f]\n", nrow(dat2), minimum(dat2.y), maximum(dat2.y))
```

    n = 200, y range: [0, 17]

### Frequentist vs Bayesian

``` julia
m_freq2 = gam(@gam_formula(y ~ s(x, k = 10)), dat2;
    family = Poisson(), link = LogLink())

m_bayes2 = gam(@gam_formula(y ~ s(x, k = 10)), dat2;
    family = Poisson(), link = LogLink(),
    priors = PriorSpec(sds = Exponential(1.0)),
    nsamples = 2000, nchains = 2)
```

    ┌ Warning: Only a single thread available: MCMC chains are not sampled in parallel
    └ @ AbstractMCMC ~/.julia/packages/AbstractMCMC/oqm6Y/src/sample.jl:544
    Sampling (1 thread)   0%|                               |  ETA: N/A
    ┌ Info: Found initial step size
    └   ϵ = 0.025
    Sampling (1 thread)   0%|▏                              |  ETA: 0:11:52
    Sampling (1 thread)   1%|▎                              |  ETA: 0:06:19
    Sampling (1 thread)   2%|▌                              |  ETA: 0:04:26
    Sampling (1 thread)   2%|▋                              |  ETA: 0:03:29
    Sampling (1 thread)   2%|▊                              |  ETA: 0:02:53
    Sampling (1 thread)   3%|▉                              |  ETA: 0:02:29
    Sampling (1 thread)   4%|█▏                             |  ETA: 0:02:10
    Sampling (1 thread)   4%|█▎                             |  ETA: 0:01:56
    Sampling (1 thread)   4%|█▍                             |  ETA: 0:01:45
    Sampling (1 thread)   5%|█▌                             |  ETA: 0:01:36
    Sampling (1 thread)   6%|█▊                             |  ETA: 0:01:29
    Sampling (1 thread)   6%|█▉                             |  ETA: 0:01:23
    Sampling (1 thread)   6%|██                             |  ETA: 0:01:17
    Sampling (1 thread)   7%|██▏                            |  ETA: 0:01:13
    Sampling (1 thread)   8%|██▍                            |  ETA: 0:01:09
    Sampling (1 thread)   8%|██▌                            |  ETA: 0:01:06
    Sampling (1 thread)   8%|██▋                            |  ETA: 0:01:04
    Sampling (1 thread)   9%|██▊                            |  ETA: 0:01:01
    Sampling (1 thread)  10%|███                            |  ETA: 0:00:58
    Sampling (1 thread)  10%|███▏                           |  ETA: 0:00:56
    Sampling (1 thread)  10%|███▎                           |  ETA: 0:00:54
    Sampling (1 thread)  11%|███▍                           |  ETA: 0:00:53
    Sampling (1 thread)  12%|███▋                           |  ETA: 0:00:51
    Sampling (1 thread)  12%|███▊                           |  ETA: 0:00:49
    Sampling (1 thread)  12%|███▉                           |  ETA: 0:00:48
    Sampling (1 thread)  13%|████                           |  ETA: 0:00:46
    Sampling (1 thread)  14%|████▏                          |  ETA: 0:00:45
    Sampling (1 thread)  14%|████▍                          |  ETA: 0:00:44
    Sampling (1 thread)  14%|████▌                          |  ETA: 0:00:43
    Sampling (1 thread)  15%|████▋                          |  ETA: 0:00:42
    Sampling (1 thread)  16%|████▊                          |  ETA: 0:00:41
    Sampling (1 thread)  16%|█████                          |  ETA: 0:00:40
    Sampling (1 thread)  16%|█████▏                         |  ETA: 0:00:39
    Sampling (1 thread)  17%|█████▎                         |  ETA: 0:00:39
    Sampling (1 thread)  18%|█████▍                         |  ETA: 0:00:38
    Sampling (1 thread)  18%|█████▋                         |  ETA: 0:00:38
    Sampling (1 thread)  18%|█████▊                         |  ETA: 0:00:37
    Sampling (1 thread)  19%|█████▉                         |  ETA: 0:00:36
    Sampling (1 thread)  20%|██████                         |  ETA: 0:00:36
    Sampling (1 thread)  20%|██████▎                        |  ETA: 0:00:35
    Sampling (1 thread)  20%|██████▍                        |  ETA: 0:00:35
    Sampling (1 thread)  21%|██████▌                        |  ETA: 0:00:34
    Sampling (1 thread)  22%|██████▋                        |  ETA: 0:00:33
    Sampling (1 thread)  22%|██████▉                        |  ETA: 0:00:33
    Sampling (1 thread)  22%|███████                        |  ETA: 0:00:32
    Sampling (1 thread)  23%|███████▏                       |  ETA: 0:00:32
    Sampling (1 thread)  24%|███████▎                       |  ETA: 0:00:32
    Sampling (1 thread)  24%|███████▌                       |  ETA: 0:00:31
    Sampling (1 thread)  24%|███████▋                       |  ETA: 0:00:31
    Sampling (1 thread)  25%|███████▊                       |  ETA: 0:00:30
    Sampling (1 thread)  26%|███████▉                       |  ETA: 0:00:30
    Sampling (1 thread)  26%|████████                       |  ETA: 0:00:29
    Sampling (1 thread)  26%|████████▎                      |  ETA: 0:00:29
    Sampling (1 thread)  27%|████████▍                      |  ETA: 0:00:29
    Sampling (1 thread)  28%|████████▌                      |  ETA: 0:00:28
    Sampling (1 thread)  28%|████████▋                      |  ETA: 0:00:28
    Sampling (1 thread)  28%|████████▉                      |  ETA: 0:00:28
    Sampling (1 thread)  29%|█████████                      |  ETA: 0:00:27
    Sampling (1 thread)  30%|█████████▏                     |  ETA: 0:00:27
    Sampling (1 thread)  30%|█████████▎                     |  ETA: 0:00:27
    Sampling (1 thread)  30%|█████████▌                     |  ETA: 0:00:26
    Sampling (1 thread)  31%|█████████▋                     |  ETA: 0:00:26
    Sampling (1 thread)  32%|█████████▊                     |  ETA: 0:00:26
    Sampling (1 thread)  32%|█████████▉                     |  ETA: 0:00:25
    Sampling (1 thread)  32%|██████████▏                    |  ETA: 0:00:25
    Sampling (1 thread)  33%|██████████▎                    |  ETA: 0:00:25
    Sampling (1 thread)  34%|██████████▍                    |  ETA: 0:00:25
    Sampling (1 thread)  34%|██████████▌                    |  ETA: 0:00:24
    Sampling (1 thread)  34%|██████████▊                    |  ETA: 0:00:24
    Sampling (1 thread)  35%|██████████▉                    |  ETA: 0:00:24
    Sampling (1 thread)  36%|███████████                    |  ETA: 0:00:23
    Sampling (1 thread)  36%|███████████▏                   |  ETA: 0:00:23
    Sampling (1 thread)  36%|███████████▍                   |  ETA: 0:00:23
    Sampling (1 thread)  37%|███████████▌                   |  ETA: 0:00:23
    Sampling (1 thread)  38%|███████████▋                   |  ETA: 0:00:22
    Sampling (1 thread)  38%|███████████▊                   |  ETA: 0:00:22
    Sampling (1 thread)  38%|███████████▉                   |  ETA: 0:00:22
    Sampling (1 thread)  39%|████████████▏                  |  ETA: 0:00:22
    Sampling (1 thread)  40%|████████████▎                  |  ETA: 0:00:21
    Sampling (1 thread)  40%|████████████▍                  |  ETA: 0:00:21
    Sampling (1 thread)  40%|████████████▌                  |  ETA: 0:00:21
    Sampling (1 thread)  41%|████████████▊                  |  ETA: 0:00:21
    Sampling (1 thread)  42%|████████████▉                  |  ETA: 0:00:20
    Sampling (1 thread)  42%|█████████████                  |  ETA: 0:00:20
    Sampling (1 thread)  42%|█████████████▏                 |  ETA: 0:00:20
    Sampling (1 thread)  43%|█████████████▍                 |  ETA: 0:00:20
    Sampling (1 thread)  44%|█████████████▌                 |  ETA: 0:00:19
    Sampling (1 thread)  44%|█████████████▋                 |  ETA: 0:00:19
    Sampling (1 thread)  44%|█████████████▊                 |  ETA: 0:00:19
    Sampling (1 thread)  45%|██████████████                 |  ETA: 0:00:19
    Sampling (1 thread)  46%|██████████████▏                |  ETA: 0:00:19
    Sampling (1 thread)  46%|██████████████▎                |  ETA: 0:00:18
    Sampling (1 thread)  46%|██████████████▍                |  ETA: 0:00:18
    Sampling (1 thread)  47%|██████████████▋                |  ETA: 0:00:18
    Sampling (1 thread)  48%|██████████████▊                |  ETA: 0:00:18
    Sampling (1 thread)  48%|██████████████▉                |  ETA: 0:00:17
    Sampling (1 thread)  48%|███████████████                |  ETA: 0:00:17
    Sampling (1 thread)  49%|███████████████▎               |  ETA: 0:00:17
    Sampling (1 thread)  50%|███████████████▍               |  ETA: 0:00:17
    Sampling (1 thread)  50%|███████████████▌               |  ETA: 0:00:17
    ┌ Info: Found initial step size
    └   ϵ = 0.00078125
    Sampling (1 thread)  50%|███████████████▋               |  ETA: 0:00:17
    Sampling (1 thread)  51%|███████████████▊               |  ETA: 0:00:17
    Sampling (1 thread)  52%|████████████████               |  ETA: 0:00:17
    Sampling (1 thread)  52%|████████████████▏              |  ETA: 0:00:17
    Sampling (1 thread)  52%|████████████████▎              |  ETA: 0:00:17
    Sampling (1 thread)  53%|████████████████▍              |  ETA: 0:00:16
    Sampling (1 thread)  54%|████████████████▋              |  ETA: 0:00:16
    Sampling (1 thread)  54%|████████████████▊              |  ETA: 0:00:16
    Sampling (1 thread)  55%|████████████████▉              |  ETA: 0:00:16
    Sampling (1 thread)  55%|█████████████████              |  ETA: 0:00:15
    Sampling (1 thread)  56%|█████████████████▎             |  ETA: 0:00:15
    Sampling (1 thread)  56%|█████████████████▍             |  ETA: 0:00:15
    Sampling (1 thread)  56%|█████████████████▌             |  ETA: 0:00:15
    Sampling (1 thread)  57%|█████████████████▋             |  ETA: 0:00:15
    Sampling (1 thread)  57%|█████████████████▉             |  ETA: 0:00:14
    Sampling (1 thread)  58%|██████████████████             |  ETA: 0:00:14
    Sampling (1 thread)  58%|██████████████████▏            |  ETA: 0:00:14
    Sampling (1 thread)  59%|██████████████████▎            |  ETA: 0:00:14
    Sampling (1 thread)  60%|██████████████████▌            |  ETA: 0:00:14
    Sampling (1 thread)  60%|██████████████████▋            |  ETA: 0:00:13
    Sampling (1 thread)  60%|██████████████████▊            |  ETA: 0:00:13
    Sampling (1 thread)  61%|██████████████████▉            |  ETA: 0:00:13
    Sampling (1 thread)  62%|███████████████████▏           |  ETA: 0:00:13
    Sampling (1 thread)  62%|███████████████████▎           |  ETA: 0:00:13
    Sampling (1 thread)  62%|███████████████████▍           |  ETA: 0:00:12
    Sampling (1 thread)  63%|███████████████████▌           |  ETA: 0:00:12
    Sampling (1 thread)  64%|███████████████████▋           |  ETA: 0:00:12
    Sampling (1 thread)  64%|███████████████████▉           |  ETA: 0:00:12
    Sampling (1 thread)  64%|████████████████████           |  ETA: 0:00:12
    Sampling (1 thread)  65%|████████████████████▏          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▎          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▌          |  ETA: 0:00:11
    Sampling (1 thread)  66%|████████████████████▋          |  ETA: 0:00:11
    Sampling (1 thread)  67%|████████████████████▊          |  ETA: 0:00:11
    Sampling (1 thread)  68%|████████████████████▉          |  ETA: 0:00:10
    Sampling (1 thread)  68%|█████████████████████▏         |  ETA: 0:00:10
    Sampling (1 thread)  68%|█████████████████████▎         |  ETA: 0:00:10
    Sampling (1 thread)  69%|█████████████████████▍         |  ETA: 0:00:10
    Sampling (1 thread)  70%|█████████████████████▌         |  ETA: 0:00:10
    Sampling (1 thread)  70%|█████████████████████▊         |  ETA: 0:00:10
    Sampling (1 thread)  70%|█████████████████████▉         |  ETA: 0:00:09
    Sampling (1 thread)  71%|██████████████████████         |  ETA: 0:00:09
    Sampling (1 thread)  72%|██████████████████████▏        |  ETA: 0:00:09
    Sampling (1 thread)  72%|██████████████████████▍        |  ETA: 0:00:09
    Sampling (1 thread)  72%|██████████████████████▌        |  ETA: 0:00:09
    Sampling (1 thread)  73%|██████████████████████▋        |  ETA: 0:00:09
    Sampling (1 thread)  74%|██████████████████████▊        |  ETA: 0:00:08
    Sampling (1 thread)  74%|███████████████████████        |  ETA: 0:00:08
    Sampling (1 thread)  74%|███████████████████████▏       |  ETA: 0:00:08
    Sampling (1 thread)  75%|███████████████████████▎       |  ETA: 0:00:08
    Sampling (1 thread)  76%|███████████████████████▍       |  ETA: 0:00:08
    Sampling (1 thread)  76%|███████████████████████▌       |  ETA: 0:00:08
    Sampling (1 thread)  76%|███████████████████████▊       |  ETA: 0:00:07
    Sampling (1 thread)  77%|███████████████████████▉       |  ETA: 0:00:07
    Sampling (1 thread)  78%|████████████████████████       |  ETA: 0:00:07
    Sampling (1 thread)  78%|████████████████████████▏      |  ETA: 0:00:07
    Sampling (1 thread)  78%|████████████████████████▍      |  ETA: 0:00:07
    Sampling (1 thread)  79%|████████████████████████▌      |  ETA: 0:00:07
    Sampling (1 thread)  80%|████████████████████████▋      |  ETA: 0:00:06
    Sampling (1 thread)  80%|████████████████████████▊      |  ETA: 0:00:06
    Sampling (1 thread)  80%|█████████████████████████      |  ETA: 0:00:06
    Sampling (1 thread)  81%|█████████████████████████▏     |  ETA: 0:00:06
    Sampling (1 thread)  82%|█████████████████████████▎     |  ETA: 0:00:06
    Sampling (1 thread)  82%|█████████████████████████▍     |  ETA: 0:00:06
    Sampling (1 thread)  82%|█████████████████████████▋     |  ETA: 0:00:05
    Sampling (1 thread)  83%|█████████████████████████▊     |  ETA: 0:00:05
    Sampling (1 thread)  84%|█████████████████████████▉     |  ETA: 0:00:05
    Sampling (1 thread)  84%|██████████████████████████     |  ETA: 0:00:05
    Sampling (1 thread)  84%|██████████████████████████▎    |  ETA: 0:00:05
    Sampling (1 thread)  85%|██████████████████████████▍    |  ETA: 0:00:05
    Sampling (1 thread)  86%|██████████████████████████▌    |  ETA: 0:00:04
    Sampling (1 thread)  86%|██████████████████████████▋    |  ETA: 0:00:04
    Sampling (1 thread)  86%|██████████████████████████▉    |  ETA: 0:00:04
    Sampling (1 thread)  87%|███████████████████████████    |  ETA: 0:00:04
    Sampling (1 thread)  88%|███████████████████████████▏   |  ETA: 0:00:04
    Sampling (1 thread)  88%|███████████████████████████▎   |  ETA: 0:00:04
    Sampling (1 thread)  88%|███████████████████████████▍   |  ETA: 0:00:04
    Sampling (1 thread)  89%|███████████████████████████▋   |  ETA: 0:00:03
    Sampling (1 thread)  90%|███████████████████████████▊   |  ETA: 0:00:03
    Sampling (1 thread)  90%|███████████████████████████▉   |  ETA: 0:00:03
    Sampling (1 thread)  90%|████████████████████████████   |  ETA: 0:00:03
    Sampling (1 thread)  91%|████████████████████████████▎  |  ETA: 0:00:03
    Sampling (1 thread)  92%|████████████████████████████▍  |  ETA: 0:00:03
    Sampling (1 thread)  92%|████████████████████████████▌  |  ETA: 0:00:02
    Sampling (1 thread)  92%|████████████████████████████▋  |  ETA: 0:00:02
    Sampling (1 thread)  93%|████████████████████████████▉  |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████  |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████▏ |  ETA: 0:00:02
    Sampling (1 thread)  94%|█████████████████████████████▎ |  ETA: 0:00:02
    Sampling (1 thread)  95%|█████████████████████████████▌ |  ETA: 0:00:02
    Sampling (1 thread)  96%|█████████████████████████████▋ |  ETA: 0:00:01
    Sampling (1 thread)  96%|█████████████████████████████▊ |  ETA: 0:00:01
    Sampling (1 thread)  96%|█████████████████████████████▉ |  ETA: 0:00:01
    Sampling (1 thread)  97%|██████████████████████████████▏|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▎|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▍|  ETA: 0:00:01
    Sampling (1 thread)  98%|██████████████████████████████▌|  ETA: 0:00:00
    Sampling (1 thread)  99%|██████████████████████████████▊|  ETA: 0:00:00
    Sampling (1 thread) 100%|██████████████████████████████▉|  ETA: 0:00:00
    Sampling (1 thread) 100%|███████████████████████████████| Time: 0:00:30
    Sampling (1 thread) 100%|███████████████████████████████| Time: 0:00:30

    Bayesian Generalized Additive Model

    Formula: y ~ 1
    Family:  Poisson
    Link:    LogLink
    Sampler: Turing.Inference.NUTS{ADTypes.AutoForwardDiff{nothing, Nothing}, AdvancedHMC.DiagEuclideanMetric} (2000 samples × 2 chains)

    Parametric coefficients:
    ────────────────────────────────────────────────────────
                    Estimate  Est.Error   l-95% CI  u-95% CI
    ────────────────────────────────────────────────────────
    (Intercept)     0.907294  0.0527069   0.803447   1.00925
    s(x,bs=tp)_f1  -3.63885   1.6974     -6.88097   -0.32279
    ────────────────────────────────────────────────────────

    Smooth terms: s(x,bs=tp)
    n = 200

### Posterior summary

``` julia
ct2 = coeftable(m_bayes2)
show(stdout, MIME("text/plain"), ct2)
println()

# Intercept comparison (on log scale)
freq_int2 = coef(m_freq2)[1]
bayes_int2 = coef(m_bayes2)[1]
@printf("\nIntercept (log-scale): frequentist = %.4f, Bayesian = %.4f (true = 1.0)\n",
    freq_int2, bayes_int2)

# Smooth SD posterior
chains2 = m_bayes2.chains
σ_s2 = vec(chains2[Symbol("σ_s[1]")].data)
@printf("σ_s[1]: mean = %.4f, sd = %.4f\n", mean(σ_s2), std(σ_s2))
```

    ────────────────────────────────────────────────────────
                    Estimate  Est.Error   l-95% CI  u-95% CI
    ────────────────────────────────────────────────────────
    (Intercept)     0.907294  0.0527069   0.803447   1.00925
    s(x,bs=tp)_f1  -3.63885   1.6974     -6.88097   -0.32279
    ────────────────────────────────────────────────────────

    Intercept (log-scale): frequentist = 0.9094, Bayesian = 0.9073 (true = 1.0)
    σ_s[1]: mean = 2.2655, sd = 0.6382

### ECDF comparison for intercept

``` julia
β1_post = vec(chains2[Symbol("β[1]")].data)
β1_sorted = sort(β1_post)
n_s = length(β1_sorted)

freq_rank2 = searchsortedfirst(β1_sorted, freq_int2) / n_s
@printf("Frequentist intercept = %.4f sits at %.1f%% of Bayesian posterior\n",
    freq_int2, 100 * freq_rank2)
@printf("True intercept = 1.0 sits at %.1f%% of posterior\n",
    100 * searchsortedfirst(β1_sorted, 1.0) / n_s)
```

    Frequentist intercept = 0.9094 sits at 51.0% of Bayesian posterior
    True intercept = 1.0 sits at 96.1% of posterior

## Example 3: Custom priors — effect on smoothing

The prior on `sds` directly controls the amount of smoothing. A tighter
prior yields smoother fits:

``` julia
# Tight prior: Exponential(0.1) → small σ_s → smoother
m_tight = gam(@gam_formula(y ~ s(x, k = 10)), dat;
    priors = PriorSpec(sds = Exponential(0.1)),
    nsamples = 1000, nchains = 1)

# Wide prior: Exponential(5.0) → large σ_s → wigglier
m_wide = gam(@gam_formula(y ~ s(x, k = 10)), dat;
    priors = PriorSpec(sds = Exponential(5.0)),
    nsamples = 1000, nchains = 1)

σ_tight = mean(vec(m_tight.chains[Symbol("σ_s[1]")].data))
σ_wide = mean(vec(m_wide.chains[Symbol("σ_s[1]")].data))
@printf("Tight prior (Exp(0.1)): posterior mean σ_s = %.4f\n", σ_tight)
@printf("Wide prior  (Exp(5.0)): posterior mean σ_s = %.4f\n", σ_wide)
@printf("Ratio: %.1fx\n", σ_wide / σ_tight)
```

    Sampling   0%|                                          |  ETA: N/A
    ┌ Info: Found initial step size
    └   ϵ = 0.003125
    Sampling   1%|▎                                         |  ETA: 0:05:52
    Sampling   1%|▍                                         |  ETA: 0:03:10
    Sampling   2%|▋                                         |  ETA: 0:02:05
    Sampling   2%|▉                                         |  ETA: 0:01:36
    Sampling   3%|█▏                                        |  ETA: 0:01:17
    Sampling   3%|█▎                                        |  ETA: 0:01:07
    Sampling   4%|█▌                                        |  ETA: 0:00:57
    Sampling   4%|█▋                                        |  ETA: 0:00:51
    Sampling   5%|█▉                                        |  ETA: 0:00:45
    Sampling   5%|██▏                                       |  ETA: 0:00:42
    Sampling   6%|██▍                                       |  ETA: 0:00:38
    Sampling   6%|██▌                                       |  ETA: 0:00:35
    Sampling   7%|██▊                                       |  ETA: 0:00:33
    Sampling   7%|███                                       |  ETA: 0:00:31
    Sampling   8%|███▏                                      |  ETA: 0:00:29
    Sampling   8%|███▍                                      |  ETA: 0:00:27
    Sampling   9%|███▋                                      |  ETA: 0:00:25
    Sampling   9%|███▊                                      |  ETA: 0:00:24
    Sampling  10%|████                                      |  ETA: 0:00:23
    Sampling  10%|████▎                                     |  ETA: 0:00:22
    Sampling  11%|████▍                                     |  ETA: 0:00:20
    Sampling  11%|████▋                                     |  ETA: 0:00:20
    Sampling  12%|████▉                                     |  ETA: 0:00:19
    Sampling  12%|█████                                     |  ETA: 0:00:18
    Sampling  13%|█████▎                                    |  ETA: 0:00:17
    Sampling  13%|█████▌                                    |  ETA: 0:00:17
    Sampling  14%|█████▋                                    |  ETA: 0:00:16
    Sampling  14%|█████▉                                    |  ETA: 0:00:16
    Sampling  15%|██████▏                                   |  ETA: 0:00:15
    Sampling  15%|██████▎                                   |  ETA: 0:00:15
    Sampling  16%|██████▌                                   |  ETA: 0:00:14
    Sampling  16%|██████▊                                   |  ETA: 0:00:14
    Sampling  17%|███████                                   |  ETA: 0:00:13
    Sampling  17%|███████▏                                  |  ETA: 0:00:13
    Sampling  18%|███████▍                                  |  ETA: 0:00:12
    Sampling  18%|███████▌                                  |  ETA: 0:00:12
    Sampling  19%|███████▊                                  |  ETA: 0:00:12
    Sampling  19%|████████                                  |  ETA: 0:00:11
    Sampling  20%|████████▎                                 |  ETA: 0:00:11
    Sampling  20%|████████▍                                 |  ETA: 0:00:11
    Sampling  21%|████████▋                                 |  ETA: 0:00:11
    Sampling  21%|████████▉                                 |  ETA: 0:00:10
    Sampling  22%|█████████                                 |  ETA: 0:00:10
    Sampling  22%|█████████▎                                |  ETA: 0:00:10
    Sampling  23%|█████████▌                                |  ETA: 0:00:10
    Sampling  23%|█████████▋                                |  ETA: 0:00:09
    Sampling  24%|█████████▉                                |  ETA: 0:00:09
    Sampling  24%|██████████▏                               |  ETA: 0:00:09
    Sampling  25%|██████████▎                               |  ETA: 0:00:09
    Sampling  25%|██████████▌                               |  ETA: 0:00:08
    Sampling  26%|██████████▊                               |  ETA: 0:00:08
    Sampling  26%|██████████▉                               |  ETA: 0:00:08
    Sampling  27%|███████████▏                              |  ETA: 0:00:08
    Sampling  27%|███████████▍                              |  ETA: 0:00:08
    Sampling  28%|███████████▋                              |  ETA: 0:00:08
    Sampling  28%|███████████▊                              |  ETA: 0:00:07
    Sampling  29%|████████████                              |  ETA: 0:00:07
    Sampling  29%|████████████▏                             |  ETA: 0:00:07
    Sampling  30%|████████████▍                             |  ETA: 0:00:07
    Sampling  30%|████████████▋                             |  ETA: 0:00:07
    Sampling  31%|████████████▉                             |  ETA: 0:00:07
    Sampling  31%|█████████████                             |  ETA: 0:00:07
    Sampling  32%|█████████████▎                            |  ETA: 0:00:06
    Sampling  32%|█████████████▌                            |  ETA: 0:00:06
    Sampling  33%|█████████████▋                            |  ETA: 0:00:06
    Sampling  33%|█████████████▉                            |  ETA: 0:00:06
    Sampling  34%|██████████████▏                           |  ETA: 0:00:06
    Sampling  34%|██████████████▎                           |  ETA: 0:00:06
    Sampling  35%|██████████████▌                           |  ETA: 0:00:06
    Sampling  35%|██████████████▊                           |  ETA: 0:00:06
    Sampling  36%|██████████████▉                           |  ETA: 0:00:06
    Sampling  36%|███████████████▏                          |  ETA: 0:00:06
    Sampling  37%|███████████████▍                          |  ETA: 0:00:06
    Sampling  37%|███████████████▌                          |  ETA: 0:00:05
    Sampling  38%|███████████████▊                          |  ETA: 0:00:06
    Sampling  38%|████████████████                          |  ETA: 0:00:06
    Sampling  39%|████████████████▏                         |  ETA: 0:00:06
    Sampling  39%|████████████████▍                         |  ETA: 0:00:05
    Sampling  40%|████████████████▋                         |  ETA: 0:00:05
    Sampling  40%|████████████████▊                         |  ETA: 0:00:05
    Sampling  41%|█████████████████                         |  ETA: 0:00:05
    Sampling  41%|█████████████████▎                        |  ETA: 0:00:05
    Sampling  42%|█████████████████▌                        |  ETA: 0:00:05
    Sampling  42%|█████████████████▋                        |  ETA: 0:00:05
    Sampling  43%|█████████████████▉                        |  ETA: 0:00:05
    Sampling  43%|██████████████████                        |  ETA: 0:00:05
    Sampling  44%|██████████████████▎                       |  ETA: 0:00:05
    Sampling  44%|██████████████████▌                       |  ETA: 0:00:05
    Sampling  45%|██████████████████▊                       |  ETA: 0:00:05
    Sampling  45%|██████████████████▉                       |  ETA: 0:00:05
    Sampling  46%|███████████████████▏                      |  ETA: 0:00:04
    Sampling  46%|███████████████████▍                      |  ETA: 0:00:04
    Sampling  47%|███████████████████▌                      |  ETA: 0:00:04
    Sampling  47%|███████████████████▊                      |  ETA: 0:00:04
    Sampling  48%|████████████████████                      |  ETA: 0:00:04
    Sampling  48%|████████████████████▏                     |  ETA: 0:00:04
    Sampling  49%|████████████████████▍                     |  ETA: 0:00:04
    Sampling  49%|████████████████████▋                     |  ETA: 0:00:04
    Sampling  50%|████████████████████▊                     |  ETA: 0:00:04
    Sampling  50%|█████████████████████                     |  ETA: 0:00:04
    Sampling  51%|█████████████████████▎                    |  ETA: 0:00:04
    Sampling  51%|█████████████████████▍                    |  ETA: 0:00:04
    Sampling  52%|█████████████████████▋                    |  ETA: 0:00:04
    Sampling  52%|█████████████████████▉                    |  ETA: 0:00:04
    Sampling  53%|██████████████████████▏                   |  ETA: 0:00:03
    Sampling  53%|██████████████████████▎                   |  ETA: 0:00:03
    Sampling  54%|██████████████████████▌                   |  ETA: 0:00:03
    Sampling  54%|██████████████████████▋                   |  ETA: 0:00:03
    Sampling  55%|██████████████████████▉                   |  ETA: 0:00:03
    Sampling  55%|███████████████████████▏                  |  ETA: 0:00:03
    Sampling  56%|███████████████████████▍                  |  ETA: 0:00:03
    Sampling  56%|███████████████████████▌                  |  ETA: 0:00:03
    Sampling  57%|███████████████████████▊                  |  ETA: 0:00:03
    Sampling  57%|████████████████████████                  |  ETA: 0:00:03
    Sampling  58%|████████████████████████▏                 |  ETA: 0:00:03
    Sampling  58%|████████████████████████▍                 |  ETA: 0:00:03
    Sampling  59%|████████████████████████▋                 |  ETA: 0:00:03
    Sampling  59%|████████████████████████▊                 |  ETA: 0:00:03
    Sampling  60%|█████████████████████████                 |  ETA: 0:00:03
    Sampling  60%|█████████████████████████▎                |  ETA: 0:00:03
    Sampling  61%|█████████████████████████▍                |  ETA: 0:00:03
    Sampling  61%|█████████████████████████▋                |  ETA: 0:00:03
    Sampling  62%|█████████████████████████▉                |  ETA: 0:00:03
    Sampling  62%|██████████████████████████                |  ETA: 0:00:03
    Sampling  63%|██████████████████████████▎               |  ETA: 0:00:02
    Sampling  63%|██████████████████████████▌               |  ETA: 0:00:02
    Sampling  64%|██████████████████████████▋               |  ETA: 0:00:02
    Sampling  64%|██████████████████████████▉               |  ETA: 0:00:02
    Sampling  65%|███████████████████████████▏              |  ETA: 0:00:02
    Sampling  65%|███████████████████████████▎              |  ETA: 0:00:02
    Sampling  66%|███████████████████████████▌              |  ETA: 0:00:02
    Sampling  66%|███████████████████████████▊              |  ETA: 0:00:02
    Sampling  67%|████████████████████████████              |  ETA: 0:00:02
    Sampling  67%|████████████████████████████▏             |  ETA: 0:00:02
    Sampling  68%|████████████████████████████▍             |  ETA: 0:00:02
    Sampling  68%|████████████████████████████▌             |  ETA: 0:00:02
    Sampling  69%|████████████████████████████▊             |  ETA: 0:00:02
    Sampling  69%|█████████████████████████████             |  ETA: 0:00:02
    Sampling  70%|█████████████████████████████▎            |  ETA: 0:00:02
    Sampling  70%|█████████████████████████████▍            |  ETA: 0:00:02
    Sampling  71%|█████████████████████████████▋            |  ETA: 0:00:02
    Sampling  71%|█████████████████████████████▉            |  ETA: 0:00:02
    Sampling  72%|██████████████████████████████            |  ETA: 0:00:02
    Sampling  72%|██████████████████████████████▎           |  ETA: 0:00:02
    Sampling  73%|██████████████████████████████▌           |  ETA: 0:00:02
    Sampling  73%|██████████████████████████████▋           |  ETA: 0:00:02
    Sampling  74%|██████████████████████████████▉           |  ETA: 0:00:02
    Sampling  74%|███████████████████████████████▏          |  ETA: 0:00:02
    Sampling  75%|███████████████████████████████▎          |  ETA: 0:00:02
    Sampling  75%|███████████████████████████████▌          |  ETA: 0:00:01
    Sampling  76%|███████████████████████████████▊          |  ETA: 0:00:01
    Sampling  76%|███████████████████████████████▉          |  ETA: 0:00:01
    Sampling  77%|████████████████████████████████▏         |  ETA: 0:00:01
    Sampling  77%|████████████████████████████████▍         |  ETA: 0:00:01
    Sampling  78%|████████████████████████████████▋         |  ETA: 0:00:01
    Sampling  78%|████████████████████████████████▊         |  ETA: 0:00:01
    Sampling  79%|█████████████████████████████████         |  ETA: 0:00:01
    Sampling  79%|█████████████████████████████████▏        |  ETA: 0:00:01
    Sampling  80%|█████████████████████████████████▍        |  ETA: 0:00:01
    Sampling  80%|█████████████████████████████████▋        |  ETA: 0:00:01
    Sampling  81%|█████████████████████████████████▉        |  ETA: 0:00:01
    Sampling  81%|██████████████████████████████████        |  ETA: 0:00:01
    Sampling  82%|██████████████████████████████████▎       |  ETA: 0:00:01
    Sampling  82%|██████████████████████████████████▌       |  ETA: 0:00:01
    Sampling  83%|██████████████████████████████████▋       |  ETA: 0:00:01
    Sampling  83%|██████████████████████████████████▉       |  ETA: 0:00:01
    Sampling  84%|███████████████████████████████████▏      |  ETA: 0:00:01
    Sampling  84%|███████████████████████████████████▎      |  ETA: 0:00:01
    Sampling  85%|███████████████████████████████████▌      |  ETA: 0:00:01
    Sampling  85%|███████████████████████████████████▊      |  ETA: 0:00:01
    Sampling  86%|███████████████████████████████████▉      |  ETA: 0:00:01
    Sampling  86%|████████████████████████████████████▏     |  ETA: 0:00:01
    Sampling  87%|████████████████████████████████████▍     |  ETA: 0:00:01
    Sampling  87%|████████████████████████████████████▌     |  ETA: 0:00:01
    Sampling  88%|████████████████████████████████████▊     |  ETA: 0:00:01
    Sampling  88%|█████████████████████████████████████     |  ETA: 0:00:01
    Sampling  89%|█████████████████████████████████████▏    |  ETA: 0:00:01
    Sampling  89%|█████████████████████████████████████▍    |  ETA: 0:00:01
    Sampling  90%|█████████████████████████████████████▋    |  ETA: 0:00:01
    Sampling  90%|█████████████████████████████████████▊    |  ETA: 0:00:01
    Sampling  91%|██████████████████████████████████████    |  ETA: 0:00:01
    Sampling  91%|██████████████████████████████████████▎   |  ETA: 0:00:00
    Sampling  92%|██████████████████████████████████████▌   |  ETA: 0:00:00
    Sampling  92%|██████████████████████████████████████▋   |  ETA: 0:00:00
    Sampling  93%|██████████████████████████████████████▉   |  ETA: 0:00:00
    Sampling  93%|███████████████████████████████████████   |  ETA: 0:00:00
    Sampling  94%|███████████████████████████████████████▎  |  ETA: 0:00:00
    Sampling  94%|███████████████████████████████████████▌  |  ETA: 0:00:00
    Sampling  95%|███████████████████████████████████████▊  |  ETA: 0:00:00
    Sampling  95%|███████████████████████████████████████▉  |  ETA: 0:00:00
    Sampling  96%|████████████████████████████████████████▏ |  ETA: 0:00:00
    Sampling  96%|████████████████████████████████████████▍ |  ETA: 0:00:00
    Sampling  97%|████████████████████████████████████████▌ |  ETA: 0:00:00
    Sampling  97%|████████████████████████████████████████▊ |  ETA: 0:00:00
    Sampling  98%|█████████████████████████████████████████ |  ETA: 0:00:00
    Sampling  98%|█████████████████████████████████████████▏|  ETA: 0:00:00
    Sampling  99%|█████████████████████████████████████████▍|  ETA: 0:00:00
    Sampling  99%|█████████████████████████████████████████▋|  ETA: 0:00:00
    Sampling 100%|█████████████████████████████████████████▊|  ETA: 0:00:00
    Sampling 100%|██████████████████████████████████████████| Time: 0:00:05
    Sampling 100%|██████████████████████████████████████████| Time: 0:00:05
    Sampling   0%|                                          |  ETA: N/A
    ┌ Info: Found initial step size
    └   ϵ = 0.025
    Sampling   1%|▎                                         |  ETA: 0:00:01
    Sampling   1%|▍                                         |  ETA: 0:00:02
    Sampling   2%|▋                                         |  ETA: 0:00:05
    Sampling   2%|▉                                         |  ETA: 0:00:05
    Sampling   3%|█▏                                        |  ETA: 0:00:05
    Sampling   3%|█▎                                        |  ETA: 0:00:05
    Sampling   4%|█▌                                        |  ETA: 0:00:05
    Sampling   4%|█▋                                        |  ETA: 0:00:05
    Sampling   5%|█▉                                        |  ETA: 0:00:05
    Sampling   5%|██▏                                       |  ETA: 0:00:06
    Sampling   6%|██▍                                       |  ETA: 0:00:06
    Sampling   6%|██▌                                       |  ETA: 0:00:05
    Sampling   7%|██▊                                       |  ETA: 0:00:05
    Sampling   7%|███                                       |  ETA: 0:00:05
    Sampling   8%|███▏                                      |  ETA: 0:00:05
    Sampling   8%|███▍                                      |  ETA: 0:00:05
    Sampling   9%|███▋                                      |  ETA: 0:00:05
    Sampling   9%|███▊                                      |  ETA: 0:00:05
    Sampling  10%|████                                      |  ETA: 0:00:05
    Sampling  10%|████▎                                     |  ETA: 0:00:05
    Sampling  11%|████▍                                     |  ETA: 0:00:05
    Sampling  11%|████▋                                     |  ETA: 0:00:05
    Sampling  12%|████▉                                     |  ETA: 0:00:05
    Sampling  12%|█████                                     |  ETA: 0:00:05
    Sampling  13%|█████▎                                    |  ETA: 0:00:05
    Sampling  13%|█████▌                                    |  ETA: 0:00:05
    Sampling  14%|█████▋                                    |  ETA: 0:00:05
    Sampling  14%|█████▉                                    |  ETA: 0:00:05
    Sampling  15%|██████▏                                   |  ETA: 0:00:04
    Sampling  15%|██████▎                                   |  ETA: 0:00:04
    Sampling  16%|██████▌                                   |  ETA: 0:00:04
    Sampling  16%|██████▊                                   |  ETA: 0:00:04
    Sampling  17%|███████                                   |  ETA: 0:00:04
    Sampling  17%|███████▏                                  |  ETA: 0:00:04
    Sampling  18%|███████▍                                  |  ETA: 0:00:04
    Sampling  18%|███████▌                                  |  ETA: 0:00:04
    Sampling  19%|███████▊                                  |  ETA: 0:00:04
    Sampling  19%|████████                                  |  ETA: 0:00:04
    Sampling  20%|████████▎                                 |  ETA: 0:00:04
    Sampling  20%|████████▍                                 |  ETA: 0:00:04
    Sampling  21%|████████▋                                 |  ETA: 0:00:04
    Sampling  21%|████████▉                                 |  ETA: 0:00:04
    Sampling  22%|█████████                                 |  ETA: 0:00:04
    Sampling  22%|█████████▎                                |  ETA: 0:00:04
    Sampling  23%|█████████▌                                |  ETA: 0:00:04
    Sampling  23%|█████████▋                                |  ETA: 0:00:04
    Sampling  24%|█████████▉                                |  ETA: 0:00:04
    Sampling  24%|██████████▏                               |  ETA: 0:00:04
    Sampling  25%|██████████▎                               |  ETA: 0:00:04
    Sampling  25%|██████████▌                               |  ETA: 0:00:03
    Sampling  26%|██████████▊                               |  ETA: 0:00:03
    Sampling  26%|██████████▉                               |  ETA: 0:00:03
    Sampling  27%|███████████▏                              |  ETA: 0:00:03
    Sampling  27%|███████████▍                              |  ETA: 0:00:03
    Sampling  28%|███████████▋                              |  ETA: 0:00:03
    Sampling  28%|███████████▊                              |  ETA: 0:00:03
    Sampling  29%|████████████                              |  ETA: 0:00:03
    Sampling  29%|████████████▏                             |  ETA: 0:00:03
    Sampling  30%|████████████▍                             |  ETA: 0:00:03
    Sampling  30%|████████████▋                             |  ETA: 0:00:03
    Sampling  31%|████████████▉                             |  ETA: 0:00:03
    Sampling  31%|█████████████                             |  ETA: 0:00:03
    Sampling  32%|█████████████▎                            |  ETA: 0:00:03
    Sampling  32%|█████████████▌                            |  ETA: 0:00:03
    Sampling  33%|█████████████▋                            |  ETA: 0:00:03
    Sampling  33%|█████████████▉                            |  ETA: 0:00:03
    Sampling  34%|██████████████▏                           |  ETA: 0:00:03
    Sampling  34%|██████████████▎                           |  ETA: 0:00:03
    Sampling  35%|██████████████▌                           |  ETA: 0:00:03
    Sampling  35%|██████████████▊                           |  ETA: 0:00:03
    Sampling  36%|██████████████▉                           |  ETA: 0:00:03
    Sampling  36%|███████████████▏                          |  ETA: 0:00:03
    Sampling  37%|███████████████▍                          |  ETA: 0:00:03
    Sampling  37%|███████████████▌                          |  ETA: 0:00:03
    Sampling  38%|███████████████▊                          |  ETA: 0:00:03
    Sampling  38%|████████████████                          |  ETA: 0:00:03
    Sampling  39%|████████████████▏                         |  ETA: 0:00:03
    Sampling  39%|████████████████▍                         |  ETA: 0:00:03
    Sampling  40%|████████████████▋                         |  ETA: 0:00:03
    Sampling  40%|████████████████▊                         |  ETA: 0:00:03
    Sampling  41%|█████████████████                         |  ETA: 0:00:03
    Sampling  41%|█████████████████▎                        |  ETA: 0:00:03
    Sampling  42%|█████████████████▌                        |  ETA: 0:00:03
    Sampling  42%|█████████████████▋                        |  ETA: 0:00:03
    Sampling  43%|█████████████████▉                        |  ETA: 0:00:02
    Sampling  43%|██████████████████                        |  ETA: 0:00:02
    Sampling  44%|██████████████████▎                       |  ETA: 0:00:02
    Sampling  44%|██████████████████▌                       |  ETA: 0:00:02
    Sampling  45%|██████████████████▊                       |  ETA: 0:00:02
    Sampling  45%|██████████████████▉                       |  ETA: 0:00:02
    Sampling  46%|███████████████████▏                      |  ETA: 0:00:02
    Sampling  46%|███████████████████▍                      |  ETA: 0:00:02
    Sampling  47%|███████████████████▌                      |  ETA: 0:00:02
    Sampling  47%|███████████████████▊                      |  ETA: 0:00:02
    Sampling  48%|████████████████████                      |  ETA: 0:00:02
    Sampling  48%|████████████████████▏                     |  ETA: 0:00:02
    Sampling  49%|████████████████████▍                     |  ETA: 0:00:02
    Sampling  49%|████████████████████▋                     |  ETA: 0:00:02
    Sampling  50%|████████████████████▊                     |  ETA: 0:00:02
    Sampling  50%|█████████████████████                     |  ETA: 0:00:02
    Sampling  51%|█████████████████████▎                    |  ETA: 0:00:02
    Sampling  51%|█████████████████████▍                    |  ETA: 0:00:02
    Sampling  52%|█████████████████████▋                    |  ETA: 0:00:02
    Sampling  52%|█████████████████████▉                    |  ETA: 0:00:02
    Sampling  53%|██████████████████████▏                   |  ETA: 0:00:02
    Sampling  53%|██████████████████████▎                   |  ETA: 0:00:02
    Sampling  54%|██████████████████████▌                   |  ETA: 0:00:02
    Sampling  54%|██████████████████████▋                   |  ETA: 0:00:02
    Sampling  55%|██████████████████████▉                   |  ETA: 0:00:02
    Sampling  55%|███████████████████████▏                  |  ETA: 0:00:02
    Sampling  56%|███████████████████████▍                  |  ETA: 0:00:02
    Sampling  56%|███████████████████████▌                  |  ETA: 0:00:02
    Sampling  57%|███████████████████████▊                  |  ETA: 0:00:02
    Sampling  57%|████████████████████████                  |  ETA: 0:00:02
    Sampling  58%|████████████████████████▏                 |  ETA: 0:00:02
    Sampling  58%|████████████████████████▍                 |  ETA: 0:00:02
    Sampling  59%|████████████████████████▋                 |  ETA: 0:00:02
    Sampling  59%|████████████████████████▊                 |  ETA: 0:00:02
    Sampling  60%|█████████████████████████                 |  ETA: 0:00:02
    Sampling  60%|█████████████████████████▎                |  ETA: 0:00:02
    Sampling  61%|█████████████████████████▍                |  ETA: 0:00:02
    Sampling  61%|█████████████████████████▋                |  ETA: 0:00:02
    Sampling  62%|█████████████████████████▉                |  ETA: 0:00:02
    Sampling  62%|██████████████████████████                |  ETA: 0:00:02
    Sampling  63%|██████████████████████████▎               |  ETA: 0:00:02
    Sampling  63%|██████████████████████████▌               |  ETA: 0:00:02
    Sampling  64%|██████████████████████████▋               |  ETA: 0:00:02
    Sampling  64%|██████████████████████████▉               |  ETA: 0:00:02
    Sampling  65%|███████████████████████████▏              |  ETA: 0:00:02
    Sampling  65%|███████████████████████████▎              |  ETA: 0:00:01
    Sampling  66%|███████████████████████████▌              |  ETA: 0:00:01
    Sampling  66%|███████████████████████████▊              |  ETA: 0:00:01
    Sampling  67%|████████████████████████████              |  ETA: 0:00:01
    Sampling  67%|████████████████████████████▏             |  ETA: 0:00:01
    Sampling  68%|████████████████████████████▍             |  ETA: 0:00:01
    Sampling  68%|████████████████████████████▌             |  ETA: 0:00:01
    Sampling  69%|████████████████████████████▊             |  ETA: 0:00:01
    Sampling  69%|█████████████████████████████             |  ETA: 0:00:01
    Sampling  70%|█████████████████████████████▎            |  ETA: 0:00:01
    Sampling  70%|█████████████████████████████▍            |  ETA: 0:00:01
    Sampling  71%|█████████████████████████████▋            |  ETA: 0:00:01
    Sampling  71%|█████████████████████████████▉            |  ETA: 0:00:01
    Sampling  72%|██████████████████████████████            |  ETA: 0:00:01
    Sampling  72%|██████████████████████████████▎           |  ETA: 0:00:01
    Sampling  73%|██████████████████████████████▌           |  ETA: 0:00:01
    Sampling  73%|██████████████████████████████▋           |  ETA: 0:00:01
    Sampling  74%|██████████████████████████████▉           |  ETA: 0:00:01
    Sampling  74%|███████████████████████████████▏          |  ETA: 0:00:01
    Sampling  75%|███████████████████████████████▎          |  ETA: 0:00:01
    Sampling  75%|███████████████████████████████▌          |  ETA: 0:00:01
    Sampling  76%|███████████████████████████████▊          |  ETA: 0:00:01
    Sampling  76%|███████████████████████████████▉          |  ETA: 0:00:01
    Sampling  77%|████████████████████████████████▏         |  ETA: 0:00:01
    Sampling  77%|████████████████████████████████▍         |  ETA: 0:00:01
    Sampling  78%|████████████████████████████████▋         |  ETA: 0:00:01
    Sampling  78%|████████████████████████████████▊         |  ETA: 0:00:01
    Sampling  79%|█████████████████████████████████         |  ETA: 0:00:01
    Sampling  79%|█████████████████████████████████▏        |  ETA: 0:00:01
    Sampling  80%|█████████████████████████████████▍        |  ETA: 0:00:01
    Sampling  80%|█████████████████████████████████▋        |  ETA: 0:00:01
    Sampling  81%|█████████████████████████████████▉        |  ETA: 0:00:01
    Sampling  81%|██████████████████████████████████        |  ETA: 0:00:01
    Sampling  82%|██████████████████████████████████▎       |  ETA: 0:00:01
    Sampling  82%|██████████████████████████████████▌       |  ETA: 0:00:01
    Sampling  83%|██████████████████████████████████▋       |  ETA: 0:00:01
    Sampling  83%|██████████████████████████████████▉       |  ETA: 0:00:01
    Sampling  84%|███████████████████████████████████▏      |  ETA: 0:00:01
    Sampling  84%|███████████████████████████████████▎      |  ETA: 0:00:01
    Sampling  85%|███████████████████████████████████▌      |  ETA: 0:00:01
    Sampling  85%|███████████████████████████████████▊      |  ETA: 0:00:01
    Sampling  86%|███████████████████████████████████▉      |  ETA: 0:00:01
    Sampling  86%|████████████████████████████████████▏     |  ETA: 0:00:01
    Sampling  87%|████████████████████████████████████▍     |  ETA: 0:00:01
    Sampling  87%|████████████████████████████████████▌     |  ETA: 0:00:01
    Sampling  88%|████████████████████████████████████▊     |  ETA: 0:00:01
    Sampling  88%|█████████████████████████████████████     |  ETA: 0:00:01
    Sampling  89%|█████████████████████████████████████▏    |  ETA: 0:00:00
    Sampling  89%|█████████████████████████████████████▍    |  ETA: 0:00:00
    Sampling  90%|█████████████████████████████████████▋    |  ETA: 0:00:00
    Sampling  90%|█████████████████████████████████████▊    |  ETA: 0:00:00
    Sampling  91%|██████████████████████████████████████    |  ETA: 0:00:00
    Sampling  91%|██████████████████████████████████████▎   |  ETA: 0:00:00
    Sampling  92%|██████████████████████████████████████▌   |  ETA: 0:00:00
    Sampling  92%|██████████████████████████████████████▋   |  ETA: 0:00:00
    Sampling  93%|██████████████████████████████████████▉   |  ETA: 0:00:00
    Sampling  93%|███████████████████████████████████████   |  ETA: 0:00:00
    Sampling  94%|███████████████████████████████████████▎  |  ETA: 0:00:00
    Sampling  94%|███████████████████████████████████████▌  |  ETA: 0:00:00
    Sampling  95%|███████████████████████████████████████▊  |  ETA: 0:00:00
    Sampling  95%|███████████████████████████████████████▉  |  ETA: 0:00:00
    Sampling  96%|████████████████████████████████████████▏ |  ETA: 0:00:00
    Sampling  96%|████████████████████████████████████████▍ |  ETA: 0:00:00
    Sampling  97%|████████████████████████████████████████▌ |  ETA: 0:00:00
    Sampling  97%|████████████████████████████████████████▊ |  ETA: 0:00:00
    Sampling  98%|█████████████████████████████████████████ |  ETA: 0:00:00
    Sampling  98%|█████████████████████████████████████████▏|  ETA: 0:00:00
    Sampling  99%|█████████████████████████████████████████▍|  ETA: 0:00:00
    Sampling  99%|█████████████████████████████████████████▋|  ETA: 0:00:00
    Sampling 100%|█████████████████████████████████████████▊|  ETA: 0:00:00
    Sampling 100%|██████████████████████████████████████████| Time: 0:00:04
    Sampling 100%|██████████████████████████████████████████| Time: 0:00:04
    Tight prior (Exp(0.1)): posterior mean σ_s = 1.0070
    Wide prior  (Exp(5.0)): posterior mean σ_s = 1.7921
    Ratio: 1.8x

## Example 4: Building custom Turing models

For full control, use `gam_matrices()` and `gam_smooth()` to extract the
basis matrices, then write your own `@model`:

``` julia
# Extract matrices from a formula
gf = @gam_formula(y ~ s(x, k = 10))
X, sms, labels = gam_matrices(gf, dat)
println("Fixed matrix X: ", size(X), " (intercept)")
println("Smooth '$(labels[1])':")
println("  Xf (null space): ", size(sms[1].Xf))
println("  Zs (penalized):  ", [size(Z) for Z in sms[1].Zs])

# Build custom model
Xf = sms[1].Xf
Zs = sms[1].Zs[1]
X_fixed = hcat(X, Xf)

@model function my_gam(y_obs, X_f, Z)
    n_f = size(X_f, 2)
    n_z = size(Z, 2)

    # Priors
    β ~ MvNormal(zeros(n_f), 10.0 * I)
    σ ~ truncated(Normal(0, 2.5); lower = 0.0)
    σ_s ~ Exponential(1.0)

    # Random effects (non-centered parameterization)
    z ~ MvNormal(zeros(n_z), I)
    η = X_f * β .+ σ_s .* (Z * z)

    # Likelihood
    y_obs ~ MvNormal(η, σ^2 * I)
end

custom_chains = sample(my_gam(dat.y, X_fixed, Zs), NUTS(), 1000; progress = false)
σ_custom = mean(vec(custom_chains[:σ].data))
@printf("Custom model σ posterior mean: %.4f (compare to %.4f from gam())\n",
    σ_custom, mean(vec(m_bayes.chains[Symbol("σ_obs")].data)))
```

    Fixed matrix X: (200, 1) (intercept)
    Smooth 's(x,bs=tp)':
      Xf (null space): (200, 1)
      Zs (penalized):  [(200, 8)]
    ┌ Info: Found initial step size
    └   ϵ = 0.2
    Custom model σ posterior mean: 0.3148 (compare to 0.3157 from gam())

## Summary

| Feature | Syntax |
|----|----|
| Default Bayesian GAM | `gam(formula, data; priors = PriorSpec())` |
| Custom priors | `PriorSpec(sds = Exponential(1.0), sigma = InverseGamma(2, 3))` |
| Per-smooth priors | `PriorSpec(specific = Dict("sds_s(x2)" => Exponential(0.5)))` |
| Poisson | `gam(formula, data; family = Poisson(), priors = PriorSpec())` |
| Coefficient table | `coeftable(m)` — posterior mean, SD, 95% CI |
| Credible intervals | `confint(m; level = 0.95)` |
| Full posterior | `m.chains[Symbol("σ_obs")]` |
| Custom model | `gam_matrices()` + `@model` + `sample()` |

### Key design choices

1.  **Dispatch-based API**: The same `gam()` function handles both
    frequentist and Bayesian fitting — the presence of `priors=`
    triggers Bayesian mode via multiple dispatch.

2.  **smooth2random decomposition**: Smooth terms are split into a fixed
    null-space (unpenalized, estimated via `β`) and random-effects
    blocks (penalized, estimated via `σ_s × z`).

3.  **Non-centered parameterization**: Random effects use
    $\mathbf{b} = \sigma_s \cdot \mathbf{z}$ where
    $\mathbf{z} \sim N(0, I)$, which improves NUTS sampling geometry.

4.  **Package extension**: Turing.jl is only loaded when needed
    (`using Turing`), keeping the base GAM.jl lightweight.
