# Diagnostics

GAM.jl provides comprehensive model diagnostics inspired by mgcv's built-in
checks and the [gratia](https://gavinsimpson.github.io/gratia/) R package.

## Model Checking

### `gam_check`

Produces residual diagnostic information similar to `gam.check()` in R:

```julia
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
c = concurvity(m)
```

Returns worst-case and observed concurvity measures for each smooth.
Values close to 1 indicate potential identifiability issues.

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
rg = rootogram(m)
```

Useful for diagnosing fit of Poisson, negative binomial, and other count models.

## Data Exploration

### `data_slice`

Creates a data slice for evaluating model predictions while holding some
variables at fixed values:

```julia
ds = data_slice(m; x1=range(0, 1; length=100), x2=0.5)
```

Useful for generating prediction grids for visualization.

## Example Workflow

```julia
using GAM, DataFrames

n = 500
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)

# 1. Check model adequacy
gc = gam_check(m)
kc = k_check(m)

# 2. Evaluate smooth
se = smooth_estimates(m)

# 3. Compute derivatives to find where the function changes
d = derivatives(m)

# 4. Get posterior uncertainty
ps = posterior_samples(m; n_samples=500)
fs = fitted_samples(m; n_samples=500)

# 5. Multi-panel diagnostic
ap = appraise(m)
```

## See Also

- [Getting Started](@ref) for basic diagnostic usage
- [Bayesian Inference](@ref) for full posterior inference via Turing.jl
- [API Reference](@ref) for all diagnostic function signatures
