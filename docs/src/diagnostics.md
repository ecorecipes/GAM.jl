# [Diagnostics](@id diagnostics)

GAM.jl provides comprehensive model diagnostics inspired by mgcv's built-in
checks and the [gratia](https://gavinsimpson.github.io/gratia/) R package.

## Model Checking

### `gam_check`

Produces residual diagnostic information similar to R's `gam.check()`:

```julia
using GAM, DataFrames, Random, Distributions
Random.seed!(42)

n = 500
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)
gc = gam_check(m)
```

Returns a `GamCheck` object with:
- QQ plot data (theoretical vs observed quantiles)
- Residuals vs fitted values
- Histogram of residuals
- Response vs fitted values

### `k_check`

Tests whether the basis dimension `k` is adequate for each smooth term:

```julia
kc = k_check(m)
```

A significant p-value suggests that `k` should be increased. Rule of thumb:
if the effective degrees of freedom (EDF) is close to `k - 1`, increase `k`.

### `concurvity`

Measures concurvity (the smooth analogue of collinearity) between smooth terms:

```julia
x2 = randn(n)
y2 = sin.(x) .+ 0.5 .* x2 .+ 0.3 .* randn(n)
df2 = DataFrame(x=x, x2=x2, y=y2)

m2 = gam(@gam_formula(y ~ s(x, k=15) + s(x2, k=10)), df2)
c = concurvity(m2)
```

Returns worst-case and observed concurvity measures for each smooth.
Values close to 1 indicate potential identifiability issues.

## ANOVA for GAMs (`anova_gam`)

`anova_gam` provides two modes of operation: single-model smooth significance
testing and multi-model comparison.

### Single-Model Smooth Significance

Tests the significance of each smooth term using the Bayesian test of
Wood (2013). This is equivalent to the smooth significance table printed by
R's `summary.gam()` or `anova.gam(m)` with a single model:

```julia
m = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)
a = anova_gam(m)
```

The returned `AnovaGamResult` contains a `smooth_table` with:

| Column | Description |
|--------|-------------|
| `label` | Smooth term name (e.g., `"s(x)"`) |
| `edf` | Effective degrees of freedom |
| `ref_df` | Reference degrees of freedom for the test |
| `statistic` | Wald-type test statistic (F or χ²) |
| `p_value` | Approximate p-value |

The test uses the Bayesian posterior covariance matrix with reference degrees
of freedom based on effective degrees of freedom. An F-test is used when the
scale parameter is estimated (e.g., Gaussian); a χ² test when it is known
(e.g., Poisson, Binomial).

### Multi-Model Comparison

Compare two or more nested GAM models via sequential deviance tests.
Equivalent to R's `anova(m1, m2, test="F")`:

```julia
m_small = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df2)
m_full = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)

a = anova_gam(m_small, m_full)
```

Models are automatically sorted by increasing total EDF. The result includes
a `model_table` with columns for deviance, EDF difference, test statistic,
and p-value.

You can compare more than two models:

```julia
m1 = gam(@gam_formula(y ~ s(x, k=10, bs=:cr)), df2)
m2 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df2)
m3 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)

a = anova_gam(m1, m2, m3)
```

### Controlling the Test Type

By default, `anova_gam` selects the test automatically (`:auto`): F-test for
families with estimated scale, χ² for known scale. You can override this:

```julia
a_f = anova_gam(m_small, m_full; test=:F)
a_chi = anova_gam(m_small, m_full; test=:Chisq)
```

### Example: Poisson Model Comparison

```julia
mu = exp.(0.5 .* sin.(x) .+ 0.3 .* x2 .+ 0.5)
counts = Float64.([rand(Poisson(m)) for m in mu])
df_pois = DataFrame(x=x, x2=x2, y=counts)

m_p1 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink())
m_p2 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink())

# Uses χ² test automatically for Poisson
a = anova_gam(m_p1, m_p2)
```

## Smooth Evaluation

### `smooth_estimates`

Evaluates smooth functions on a regular grid, returning estimates with
confidence intervals:

```julia
se = smooth_estimates(m)
```

Returns a DataFrame with columns for the covariate value, estimated smooth
value, standard error, and confidence bounds. Useful for plotting.

### `derivatives`

Computes first (and optionally higher-order) derivatives of smooth functions
using finite differences:

```julia
d = derivatives(m)                    # first derivatives
d2 = derivatives(m; order=2)          # second derivatives
```

Useful for identifying regions of significant change.

### `partial_residuals`

Computes partial residuals for each smooth term — the residuals plus the
smooth contribution:

```julia
pr = partial_residuals(m)
```

Useful for assessing smooth fit: plot partial residuals against the covariate
and overlay the estimated smooth.

## Posterior Inference

### `posterior_samples`

Draws samples from the approximate posterior distribution of the model
coefficients (using the Bayesian covariance matrix):

```julia
ps = posterior_samples(m; n_samples=1000)
```

### `fitted_samples`

Draws samples from the posterior distribution of the fitted values:

```julia
fs = fitted_samples(m; n_samples=1000)
```

Useful for computing posterior credible intervals on predictions.

## Summary Diagnostics

### `appraise`

Produces a multi-panel diagnostic summary (analogous to gratia's `appraise`):

```julia
ap = appraise(m)
```

Returns data for four diagnostic panels:
1. QQ plot of residuals
2. Residuals vs linear predictor
3. Histogram of residuals
4. Observed vs fitted values

### `rootogram`

For count data models, computes a rootogram comparing observed and expected
frequencies:

```julia
m_pois = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink())
rg = rootogram(m_pois)
```

Useful for diagnosing fit of Poisson, negative binomial, and other count models.

## Data Exploration

### `data_slice`

Creates a data slice for evaluating model predictions while holding some
variables at fixed values:

```julia
ds = data_slice(m2; x=range(0, 2π; length=100), x2=0.5)
```

Useful for generating prediction grids for visualization.

## Example Workflow

```julia
using GAM, DataFrames, Random, Distributions
Random.seed!(123)

n = 500
x1 = range(0, 1; length=n) |> collect
x2 = rand(n)
y = sin.(4π .* x1) .+ 0.8 .* x2.^2 .+ 0.3 .* randn(n)
df = DataFrame(x1=x1, x2=x2, y=y)

# Fit two candidate models
m1 = gam(@gam_formula(y ~ s(x1, k=20, bs=:cr)), df)
m2 = gam(@gam_formula(y ~ s(x1, k=20, bs=:cr) + s(x2, k=10, bs=:cr)), df)

# 1. Check model adequacy
gc = gam_check(m2)
kc = k_check(m2)

# 2. Check concurvity between smooths
c = concurvity(m2)

# 3. Test smooth significance
a = anova_gam(m2)

# 4. Compare models — is x2 needed?
a_comp = anova_gam(m1, m2)

# 5. Evaluate smooth on a grid
se = smooth_estimates(m2)

# 6. Compute derivatives to find where the function changes
d = derivatives(m2)

# 7. Get posterior uncertainty
ps = posterior_samples(m2; n_samples=500)
fs = fitted_samples(m2; n_samples=500)

# 8. Multi-panel diagnostic
ap = appraise(m2)
```

## 3D Surface Visualization

### `vis_gam`

Creates a 3D surface/perspective visualization of a 2D smooth term,
analogous to R's `vis.gam()`:

```julia
using GAM, DataFrames, Plots

n = 500
x1 = rand(n); x2 = rand(n)
y = sin.(2π .* x1) .* cos.(2π .* x2) .+ 0.3 .* randn(n)
df = DataFrame(x1=x1, x2=x2, y=y)

m = gam(@gam_formula(y ~ te(x1, x2, k=10)), df)

# Basic surface plot
plot(vis_gam(m; select=1, n_grid=40))

# With standard errors and too-far masking
v = vis_gam(m; select=1, n_grid=30, se=true, too_far=0.1)
plot(v)

# Response-scale predictions (useful for non-Gaussian families)
plot(vis_gam(m; select=1, type=:response))
```

Returns a `VisGamData` struct containing:
- `x1`, `x2`: grid coordinate vectors
- `z`: predicted values matrix (`n_grid × n_grid`)
- `se`: standard error matrix (if requested)
- Label strings for axes and title

The `too_far` argument masks grid points far from observed data (0 = no
masking, 0.1 = mask if scaled distance to nearest data point > 10% of
range). Masked points are set to `NaN` and appear as gaps in the surface.

## See Also

- [Getting Started](@ref getting-started) for basic diagnostic usage
- [Bayesian Inference](@ref bayesian) for full posterior inference via Turing.jl
- [API Reference](@ref api-reference) for all diagnostic function signatures
