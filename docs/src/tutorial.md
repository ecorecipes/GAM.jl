# [Getting Started](@id getting-started)

This tutorial walks through practical GAM fitting in Julia, from basic models
to diagnostics and model comparison. All examples use synthetic data so you can
run them directly.

```julia
using GAM, DataFrames, Random, Distributions
Random.seed!(42)
```

## Basic Gaussian GAM

The simplest use case: fitting a smooth function to noisy continuous data.

```julia
n = 300
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# Fit with cubic regression splines
m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)
```

The `@gam_formula` macro works like StatsModels' `@formula` but supports
`s()`, `te()`, `ti()`, and `t2()` smooth terms with keyword arguments.

## Multiple Smooths

When the response depends on several covariates, add multiple smooth terms:

```julia
x2 = randn(n)
y2 = sin.(x) .+ 0.5 .* x2.^2 .+ 0.3 .* randn(n)
df2 = DataFrame(x=x, x2=x2, y=y2)

m2 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)
```

You can mix parametric and smooth terms freely:

```julia
z = randn(n)
y3 = sin.(x) .+ 1.5 .* z .+ 0.3 .* randn(n)
df3 = DataFrame(x=x, z=z, y=y3)

m3 = gam(@gam_formula(y ~ z + s(x, k=15, bs=:cr)), df3)
```

## Poisson GAM

For count data, specify a Poisson family with log link (equivalent to R's
`gam(y ~ s(x), family=poisson)`):

```julia
mu = exp.(0.5 .* sin.(x) .+ 0.5)
counts = [rand(Poisson(m)) for m in mu]
df_pois = DataFrame(x=x, y=Float64.(counts))

m_pois = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink())
```

## Binomial GAM

For binary outcomes or proportions (equivalent to R's
`gam(y ~ s(x), family=binomial)`):

```julia
p = 1.0 ./ (1.0 .+ exp.(-2.0 .* sin.(x)))
y_bin = Float64.([rand(Bernoulli(pi)) for pi in p])
df_bin = DataFrame(x=x, y=y_bin)

m_bin = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_bin;
    family=Binomial(), link=LogitLink())
```

## Gamma GAM

For positive continuous data with variance proportional to the mean squared
(equivalent to R's `gam(y ~ s(x), family=Gamma(link="log"))`):

```julia
shape = 5.0
mu_gamma = exp.(1.0 .+ 0.5 .* sin.(x))
y_gamma = [rand(Gamma(shape, m / shape)) for m in mu_gamma]
df_gamma = DataFrame(x=x, y=y_gamma)

m_gamma = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_gamma;
    family=Gamma(), link=LogLink())
```

## Choosing Smooth Types

The basis type (`bs`) controls the shape of basis functions. Different types
suit different problems:

```julia
n = 400
x = range(0, 1; length=n) |> collect
y = sin.(4π .* x) .+ 0.3 .* randn(n)
df_smooth = DataFrame(x=x, y=y)

# Thin plate regression spline — the default, good general-purpose choice
m_tp = gam(@gam_formula(y ~ s(x, k=20, bs=:tp)), df_smooth)

# Cubic regression spline — fast, good for 1D smooths
m_cr = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df_smooth)

# P-spline — B-spline basis with difference penalty, popular in biostatistics
m_ps = gam(@gam_formula(y ~ s(x, k=20, bs=:ps)), df_smooth)
```

**When to use each:**

| Basis | When to use |
|-------|------------|
| `:tp` | Default choice; optimal smoothness properties; good for 1–3 dimensions |
| `:ts` | Like `:tp` but with shrinkage — smooth can be penalized to zero |
| `:cr` | Fast for 1D; knots at quantiles; slightly less optimal than `:tp` |
| `:cs` | Like `:cr` with shrinkage |
| `:cc` | Periodic data (time of day, angle) — endpoints match |
| `:ps` | When you want explicit control over penalty order via `m` |
| `:cps` | Periodic P-spline |
| `:bs` | B-spline with integrated derivative penalty |
| `:gp` | When a Gaussian process interpretation is desired |

## Choosing `k` (Basis Dimension)

`k` sets the upper bound on complexity. If `k` is too low the smooth cannot
capture the true signal; too high is usually fine (the penalty prevents
overfitting), but wastes computation.

```julia
# Too low — underfits
m_low = gam(@gam_formula(y ~ s(x, k=4, bs=:cr)), df_smooth)

# Adequate
m_ok = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df_smooth)

# Generous — fine, penalty handles it
m_high = gam(@gam_formula(y ~ s(x, k=50, bs=:cr)), df_smooth)
```

Use [`k_check`](@ref) to test whether `k` is large enough (see
[Diagnostics](@ref diagnostics) below).

## Tensor Product Smooths

For interactions between covariates measured on different scales, tensor
products (`te`, `ti`, `t2`) are preferred over isotropic 2D smooths.

### `te()` — Full Tensor Product

Includes both main effects and interaction. Equivalent to R's `te(x1, x2)`:

```julia
n = 500
x1 = rand(n)
x2 = rand(n)
y_te = sin.(2π .* x1) .* cos.(2π .* x2) .+ 0.3 .* randn(n)
df_te = DataFrame(x1=x1, x2=x2, y=y_te)

m_te = gam(@gam_formula(y ~ te(x1, x2, k=8)), df_te)
```

### `ti()` — Tensor Product Interaction

For an ANOVA-style decomposition separating main effects from interaction.
Equivalent to R's `ti(x1, x2)`:

```julia
m_anova = gam(
    @gam_formula(y ~ s(x1, k=10) + s(x2, k=10) + ti(x1, x2, k=6)),
    df_te,
)
```

### `t2()` — Alternative Tensor Product

Like `te()` but with independent marginal penalties, giving finer control per
marginal direction. Equivalent to R's `t2(x1, x2)`:

```julia
m_t2 = gam(@gam_formula(y ~ t2(x1, x2, k=8)), df_te)
```

## Model Diagnostics

### `gam_check` — Residual Diagnostics

Returns QQ plot data, residuals vs fitted, histogram of residuals, and
response vs fitted (equivalent to R's `gam.check()`):

```julia
m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)
gc = gam_check(m)
```

### `k_check` — Basis Dimension Adequacy

Tests whether `k` is large enough. A significant p-value suggests you should
increase `k` (equivalent to the basis dimension test in R's `gam.check()`):

```julia
kc = k_check(m)
```

Rule of thumb: if the effective degrees of freedom (EDF) is close to `k - 1`,
increase `k`.

### `concurvity` — Smooth Collinearity

Measures concurvity between smooth terms (the nonlinear analogue of
collinearity). Values close to 1 indicate identifiability issues:

```julia
m_multi = gam(@gam_formula(y ~ s(x, k=15) + s(x2, k=10)), df2)
c = concurvity(m_multi)
```

### `anova_gam` — Smooth Significance and Model Comparison

**Single model** — tests significance of each smooth term using the Bayesian
test of Wood (2013). Equivalent to R's `summary.gam()` smooth significance
table or `anova.gam(m)`:

```julia
a = anova_gam(m_multi)
```

**Multiple models** — sequential deviance comparison of nested models.
Equivalent to R's `anova(m1, m2, test="F")`:

```julia
m_small = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df2)
m_full = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)
a_comp = anova_gam(m_small, m_full)
```

Uses F-test for scale-estimated families, χ² for known-scale families. You
can force a specific test:

```julia
a_chisq = anova_gam(m_small, m_full; test=:Chisq)
```

## Prediction on New Data

Use `predict()` to evaluate the model at new covariate values (equivalent to
R's `predict.gam()`):

```julia
m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)

# New data as a DataFrame
newdf = DataFrame(x=range(0, 2π; length=100) |> collect)

# Predictions on the link scale (default)
eta = predict(m, newdf)

# Predictions on the response scale
mu = predict(m, newdf; type=:response)

# With standard errors
mu, se = predict(m, newdf; type=:response, se=true)
```

For Poisson or Binomial models, response-scale predictions are on the natural
scale (counts or probabilities):

```julia
m_pois = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink())

newdf = DataFrame(x=range(0, 2π; length=50) |> collect)
lambda, se_lambda = predict(m_pois, newdf; type=:response, se=true)
```

## Inspecting the Model

GAM.jl implements the full StatsBase interface:

```julia
using StatsBase

coef(m)           # coefficients
vcov(m)           # variance-covariance matrix
stderror(m)       # standard errors
deviance(m)       # model deviance
nobs(m)           # number of observations
dof(m)            # degrees of freedom

# GAM-specific fields
m.edf             # EDF per smooth
m.edf_total       # total EDF
m.scale           # scale parameter
m.sp              # log smoothing parameters
m.smooths         # constructed smooth terms
```

## Controlling the Fit

Fine-tune convergence criteria and optimization behaviour:

```julia
ctrl = gam_control(
    epsilon = 1e-8,      # convergence tolerance
    maxit = 500,         # max PIRLS iterations
    outer_maxit = 100,   # max outer iterations
    trace = true,        # print progress
    gamma = 1.4,         # extra smoothing (>1 = smoother)
)

m = gam(@gam_formula(y ~ s(x)), df; control=ctrl)
```

Use `gamma > 1` to encourage smoother fits (useful for exploratory analysis).

## Smooth Evaluation and Derivatives

Evaluate smooth functions on a grid for plotting, and compute derivatives to
identify regions of significant change:

```julia
m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)

# Evaluate smooth on a regular grid with confidence intervals
se = smooth_estimates(m)

# First derivatives (finite differences)
d = derivatives(m)

# Second derivatives
d2 = derivatives(m; order=2)

# Partial residuals for each smooth term
pr = partial_residuals(m)
```

## Complete Workflow Example

Putting it all together — fit, check, compare, and predict:

```julia
using GAM, DataFrames, Random, Distributions
Random.seed!(123)

# Simulate data with two smooth effects
n = 500
x1 = range(0, 1; length=n) |> collect
x2 = rand(n)
y = sin.(4π .* x1) .+ 0.8 .* x2.^2 .+ 0.3 .* randn(n)
df = DataFrame(x1=x1, x2=x2, y=y)

# Fit candidate models
m1 = gam(@gam_formula(y ~ s(x1, k=20, bs=:cr)), df)
m2 = gam(@gam_formula(y ~ s(x1, k=20, bs=:cr) + s(x2, k=10, bs=:cr)), df)

# Diagnostics
gam_check(m2)
k_check(m2)
concurvity(m2)

# Smooth significance
anova_gam(m2)

# Compare nested models — is x2 needed?
anova_gam(m1, m2)

# Predict on new data
newdf = DataFrame(
    x1 = range(0, 1; length=200) |> collect,
    x2 = fill(0.5, 200),
)
mu, se = predict(m2, newdf; type=:response, se=true)
```

## GAMLSS: Location-Scale Models

Model both the mean and variance as smooth functions of covariates. See
[GAMLSS](gamlss.md) for full details.

```julia
n = 500
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ (0.1 .+ 0.3 .* abs.(cos.(x))) .* randn(n)
df = DataFrame(x=x, y=y)

m = gamlss(
    [
        @gam_formula(y ~ s(x, k=15, bs=:cr)),     # location (μ)
        @gam_formula(y ~ s(x, k=10, bs=:cr)),     # scale (σ)
    ],
    df,
    family=GaussianLS(),
)
```

## SCAM: Shape-Constrained Models

Enforce monotonicity or convexity constraints. See [Shape Constraints (SCAM)](scam.md).

```julia
# Monotone increasing fit
m = scam(@gam_formula(y ~ s(x, k=15, bs=:mpi)), df)

# Convex fit
m = scam(@gam_formula(y ~ s(x, k=15, bs=:cx)), df)
```

## QGAM: Quantile Regression

Estimate conditional quantiles. See [Quantile Regression (QGAM)](qgam.md).

```julia
# Fit median regression
m50 = qgam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df, 0.5)

# Fit multiple quantiles at once
fits = mqgam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df,
    [0.1, 0.25, 0.5, 0.75, 0.9])
m10 = qdo(fits, 0.1)   # extract individual quantile model
```

## BAM: Large Datasets

Memory-efficient fitting for large n. See [Large Data (BAM)](bam.md).

```julia
n = 100_000
x = rand(n)
y = sin.(2π .* x) .+ 0.3 .* randn(n)
big_df = DataFrame(x=x, y=y)

m = bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), big_df)
```

## Next Steps

- [Smooth Terms](@ref smooth-terms) — full reference for all 28 smooth basis types
- [Formula Syntax](@ref formula-syntax) — details on `@gam_formula`
- [Diagnostics](@ref diagnostics) — comprehensive diagnostic functions
- [Mixed Models (GAMM)](@ref gamm) — hierarchical data with random effects
- [API Reference](@ref api-reference) — complete function signatures
