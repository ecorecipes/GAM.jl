# [Diagnostics](@id diagnostics)

GAM.jl provides comprehensive model diagnostics inspired by mgcv's built-in
checks and the [gratia](https://gavinsimpson.github.io/gratia/) R package.

```@setup diagnostics
using GAM, DataFrames, Random, Distributions, Plots
using GLM: LogLink
Random.seed!(42)

n = 180
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)
```

## Model Checking

### `gam_check`

Produces residual diagnostic information similar to R's `gam.check()`:

```@example diagnostics
m = gam(@formula(y ~ s(x, k=20, bs=:cr)), df);
gc = gam_check(m);
nothing
```

Returns a `GamCheck` object with:
- QQ plot data (theoretical vs observed quantiles)
- Residuals vs fitted values
- Histogram of residuals
- Response vs fitted values

### `k_check`

Tests whether the basis dimension `k` is adequate for each smooth term:

```@example diagnostics
kc = k_check(m);
nothing
```

A significant p-value suggests that `k` should be increased. Rule of thumb:
if the effective degrees of freedom (EDF) is close to `k - 1`, increase `k`.

### `concurvity`

Measures concurvity (the smooth analogue of collinearity) between smooth terms:

```@example diagnostics
x2 = randn(n)
y2 = sin.(x) .+ 0.5 .* x2 .+ 0.3 .* randn(n)
df2 = DataFrame(x=x, x2=x2, y=y2)

m2 = gam(@formula(y ~ s(x, k=15) + s(x2, k=10)), df2);
c = concurvity(m2);
nothing
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

```@example diagnostics
m_single = gam(@formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2);
a = anova_gam(m_single);
nothing
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

```@example diagnostics
m_small = gam(@formula(y ~ s(x, k=15, bs=:cr)), df2);
m_full = gam(@formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2);

a = anova_gam(m_small, m_full);
nothing
```

Models are automatically sorted by increasing total EDF. The result includes
a `model_table` with columns for deviance, EDF difference, test statistic,
and p-value.

You can compare more than two models:

```@example diagnostics
m1 = gam(@formula(y ~ s(x, k=10, bs=:cr)), df2);
m2_seq = gam(@formula(y ~ s(x, k=15, bs=:cr)), df2);
m3 = gam(@formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2);

a = anova_gam(m1, m2_seq, m3);
nothing
```

### Controlling the Test Type

By default, `anova_gam` selects the test automatically (`:auto`): F-test for
families with estimated scale, χ² for known scale. You can override this:

```@example diagnostics
a_f = anova_gam(m_small, m_full; test=:F);
a_chi = anova_gam(m_small, m_full; test=:Chisq);
nothing
```

### Example: Poisson Model Comparison

```@example diagnostics
mu = exp.(0.5 .* sin.(x) .+ 0.3 .* x2 .+ 0.5)
counts = Float64.(rand.(Poisson.(mu)))
df_pois = DataFrame(x=x, x2=x2, y=counts)

m_p1 = gam(@formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink());
m_p2 = gam(@formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink());

a_pois = anova_gam(m_p1, m_p2);
nothing
```

## Smooth Evaluation

### `smooth_estimates`

Evaluates smooth functions on a regular grid, returning estimates with
confidence intervals:

```@example diagnostics
se = smooth_estimates(m);
nothing
```

Returns a DataFrame with columns for the covariate value, estimated smooth
value, standard error, and confidence bounds. Useful for plotting.

### `derivatives`

Computes first (and optionally higher-order) derivatives of smooth functions
using finite differences:

```@example diagnostics
d = derivatives(m);           # first derivatives
d2 = derivatives(m; order=2); # second derivatives
nothing
```

Useful for identifying regions of significant change.

### `partial_residuals`

Computes partial residuals for each smooth term — the residuals plus the
smooth contribution:

```@example diagnostics
pr = partial_residuals(m);
nothing
```

Useful for assessing smooth fit: plot partial residuals against the covariate
and overlay the estimated smooth.

## Posterior Inference

### `posterior_samples`

Draws samples from the approximate posterior distribution of the model
coefficients (using the Bayesian covariance matrix):

```@example diagnostics
ps = posterior_samples(m; n=200);
nothing
```

### `fitted_samples`

Draws samples from the posterior distribution of the fitted values:

```@example diagnostics
fs = fitted_samples(m; n=200);
nothing
```

Useful for computing posterior credible intervals on predictions.

## Summary Diagnostics

### `appraise`

Produces a multi-panel diagnostic summary (analogous to gratia's `appraise`):

```@example diagnostics
ap = appraise(m);
nothing
```

Returns data for four diagnostic panels:
1. QQ plot of residuals
2. Residuals vs linear predictor
3. Histogram of residuals
4. Observed vs fitted values

### `rootogram`

For count data models, computes a rootogram comparing observed and expected
frequencies:

```@example diagnostics
m_root = gam(@formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink());
rg = rootogram(m_root);
nothing
```

Useful for diagnosing fit of Poisson, negative binomial, and other count models.

## Data Exploration

### `data_slice`

Creates a data slice for evaluating model predictions while holding some
variables at fixed values:

```@example diagnostics
ds = data_slice(m2; var=:x, n=100, x2=0.5);
nothing
```

Useful for generating prediction grids for visualization.

## Example Workflow

```@example diagnostics
Random.seed!(123)
x1 = range(0, 1; length=160) |> collect
x2_workflow = rand(160)
y_workflow = sin.(4π .* x1) .+ 0.8 .* x2_workflow.^2 .+ 0.3 .* randn(160)
df_workflow = DataFrame(x1=x1, x2=x2_workflow, y=y_workflow)

m1_workflow = gam(@formula(y ~ s(x1, k=20, bs=:cr)), df_workflow);
m2_workflow = gam(@formula(y ~ s(x1, k=20, bs=:cr) + s(x2, k=10, bs=:cr)), df_workflow);

gc = gam_check(m2_workflow);
kc = k_check(m2_workflow);
c = concurvity(m2_workflow);
a = anova_gam(m2_workflow);
a_comp = anova_gam(m1_workflow, m2_workflow);
se = smooth_estimates(m2_workflow);
d = derivatives(m2_workflow);
ps = posterior_samples(m2_workflow; n=150);
fs = fitted_samples(m2_workflow; n=150);
ap = appraise(m2_workflow);
nothing
```

## 3D Surface Visualization

### `vis_gam`

Creates a 3D surface/perspective visualization of a 2D smooth term,
analogous to R's `vis.gam()`:

```@example diagnostics
x1_surface = rand(140)
x2_surface = rand(140)
y_surface = sin.(2π .* x1_surface) .* cos.(2π .* x2_surface) .+ 0.3 .* randn(140)
df_surface = DataFrame(x1=x1_surface, x2=x2_surface, y=y_surface)

m_surface = gam(@formula(y ~ te(x1, x2, k=10)), df_surface);

plot(vis_gam(m_surface; select=1, n_grid=30));
v = vis_gam(m_surface; select=1, n_grid=25, se=true, too_far=0.1);
plot(v);
plot(vis_gam(m_surface; select=1, type=:response));
nothing
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
