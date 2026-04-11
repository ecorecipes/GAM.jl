# [Smooth Term Reference](@id smooth-terms)

GAM.jl provides 28 smooth basis types, covering all commonly used options from
R's mgcv, plus shape-constrained bases from scam and several additional types
including loess, fractional polynomials, spherical splines, SPDE Matérn, and
constrained factor smooths.

## Specifying Smooths

All smooth terms are specified using the `s()` function (or `te()`/`ti()`/`t2()`
for tensor products):

```julia
s(:x)                          # TPRS, default k=10
s(:x, bs=:cr, k=20)           # cubic regression spline, k=20
s(:x, :y)                     # 2d TPRS
s(:x, bs=:ps, m=3)            # P-spline with 3rd-order difference penalty
s(:x, bs=:cps, k=12)          # cyclic P-spline
s(:x, :y, bs=:gp)             # Gaussian process smooth
s(:x, fx=true, k=5)           # unpenalized (fixed df)
s(:group, bs=:re)              # random effect
s(:region, bs=:mrf, xt=nb)    # Markov random field
s(:x, bs=:lo, k=15)           # loess smooth
s(:x, bs=:fp)                 # fractional polynomial
s(:lon, :lat, bs=:sos, k=50)  # spherical spline
s(:x, :y, bs=:spde)           # SPDE Matérn
te(:x, :y, k=5)               # tensor product
ti(:x, :y, k=5)               # tensor product interaction
t2(:x, :y, k=5)               # alternative tensor product
```

## Available Basis Types

### Thin Plate Regression Splines (`bs=:tp`, `bs=:ts`)

The default smooth type. Optimal in a certain sense among all smoothers of
a given dimension and penalty order.

- `:tp` — standard TPRS
- `:ts` — TPRS with shrinkage penalty (adds small penalty on null space)

**When to use:** General-purpose default. Works well in 1–3 dimensions.
`:ts` is useful for variable selection since it allows terms to shrink to zero.

**Default k**: 10 for 1D, 30 for 2D.

```julia
s(:x)               # default TPRS
s(:x, bs=:ts)       # with shrinkage — smooth can be penalized to zero
s(:x, :y, bs=:tp)   # 2D thin plate spline
```

### Cubic Regression Splines (`bs=:cr`, `bs=:cs`, `bs=:cc`)

Natural cubic regression splines with knots placed at quantiles.

- `:cr` — standard cubic spline
- `:cs` — cubic spline with shrinkage
- `:cc` — cyclic cubic spline (endpoints match)

**When to use:** `:cr` is fast and well-understood for 1D smooths. `:cc` is
the natural choice for periodic covariates (time of day, day of year, angle).

**Default k**: 10.

```julia
s(:x, bs=:cr, k=20)
s(:time, bs=:cc, k=12)   # cyclic for periodic data
```

### P-Splines (`bs=:ps`)

B-spline basis with difference penalty (Eilers & Marx, 1996).

- `m` parameter controls the penalty order (default 2 = second-order difference)
- The B-spline order is `m + 2` (default = cubic, degree 3)

**When to use:** Popular in biostatistics and demography. Good when you want
explicit control over penalty order.

**Default k**: 10.

```julia
s(:x, bs=:ps, m=3)   # third-order difference penalty
```

### Cyclic P-Splines (`bs=:cps`)

P-spline basis with periodic boundary conditions — the basis wraps so that
the function value and its derivatives match at the endpoints.

**When to use:** Periodic covariates where you prefer a P-spline basis over
cubic splines (e.g., hour of day, month of year).

**Default k**: 10.

```julia
s(:hour, bs=:cps, k=12)   # smooth over 24-hour cycle
```

### B-Splines (`bs=:bs`)

B-spline basis with integrated squared derivative penalty.

- Similar to P-splines but uses a continuous derivative penalty
- `m` parameter controls the derivative order (default 2)

**When to use:** When you prefer a continuous derivative penalty over the
discrete difference penalty of P-splines.

**Default k**: 10.

```julia
s(:x, bs=:bs, m=2)
```

### Gaussian Process Smooth (`bs=:gp`)

A smooth based on a Gaussian process covariance kernel.

- Supports multiple kernel types via `xt` (e.g., Matérn, squared exponential)
- Automatically estimates the length-scale parameter

**When to use:** Spatial data, or when a GP interpretation is desired (e.g.,
for uncertainty quantification with a specific correlation structure).

**Default k**: 10.

```julia
s(:x, bs=:gp)                             # default kernel
s(:x, :y, bs=:gp, k=50)                   # 2D GP smooth
```

### Loess Smooth (`bs=:lo`)

Local polynomial regression smooth (Cleveland, 1979). Fits local polynomial
regressions using a kernel weighting scheme.

**When to use:** When you want a non-parametric smooth that adapts locally
without a global basis representation. Useful for exploratory analysis.
Equivalent to R's `lo()` in gam/mgcv.

**Default k**: 10.

```julia
s(:x, bs=:lo)
s(:x, bs=:lo, k=15)
```

### Fractional Polynomial (`bs=:fp`)

Fractional polynomial smooth — fits polynomial terms with powers selected from
a predefined set (including negative and fractional powers).

**When to use:** When the relationship is well-described by a low-dimensional
polynomial-like function with potentially non-integer powers. Common in
epidemiology and dose-response modelling. Equivalent to R's `fp()` in mfp.

```julia
s(:x, bs=:fp)
```

### Duchon Splines (`bs=:ds`)

Generalization of thin plate splines that allow fractional derivative penalties.

- `m` parameter controls the penalty order (can be non-integer via a tuple)

**When to use:** When the standard TPS penalty order is not ideal and you want
more flexibility in the smoothness penalty.

**Default k**: 10.

```julia
s(:x, bs=:ds)
s(:x, :y, bs=:ds, m=(1, 0.5))   # custom penalty specification
```

### Adaptive Smooth (`bs=:ad`)

Adaptive smooth with spatially varying smoothness — the effective penalty
changes along the covariate range, allowing more flexibility in regions
with more rapid change.

**When to use:** When the underlying function has regions of rapid change and
regions of slow change. The penalty adapts so you don't oversmooth or
undersmooth locally.

```julia
s(:x, bs=:ad, k=20)
```

### Spherical Splines (`bs=:sos`)

Splines on the sphere for data defined on the surface of a sphere (e.g.,
global spatial data with latitude/longitude coordinates).

**When to use:** Geospatial data on the globe where you need smoothing that
respects spherical geometry. Avoids edge effects at poles and the date line.
Equivalent to R's `s(lon, lat, bs="sos")` in mgcv.

**Default k**: 50.

```julia
s(:lon, :lat, bs=:sos, k=50)
```

### SPDE Matérn Smooth (`bs=:spde`)

Stochastic Partial Differential Equation approach to Matérn Gaussian process
smoothing. Uses a sparse precision matrix representation via the SPDE
approach of Lindgren et al. (2011).

**When to use:** Spatial data where you want Matérn covariance with
computational efficiency from sparse precision matrices. Especially useful
for large spatial datasets where a dense GP (`bs=:gp`) is too slow.

```julia
s(:x, :y, bs=:spde)
```

### Markov Random Field (`bs=:mrf`)

For discrete spatial or network data. The penalty is defined by a neighbourhood
matrix passed via `xt`.

- Requires a neighbourhood list or adjacency matrix in `xt`
- The covariate should be a factor/categorical variable identifying regions

**When to use:** Areal/lattice data (e.g., disease mapping by region).
Equivalent to R's `s(region, bs="mrf", xt=list(nb=nb))`.

```julia
# nb is a Dict mapping region => [neighbours...]
s(:region, bs=:mrf, xt=nb, k=20)
```

### Soap Film Smooth (`bs=:so`)

For smoothing over complex domains with boundaries (e.g., an estuary, a lake).
Uses a soap-film PDE approach to respect domain boundaries.

- Requires boundary specification via `xt`
- Typically used for 2D spatial smooths

**When to use:** 2D spatial smoothing where the domain has complex boundaries
and you don't want the smooth to "leak" across boundaries. Equivalent to R's
`s(x, y, bs="so", xt=list(bnd=bnd))`.

```julia
# bnd defines the domain boundary
s(:x, :y, bs=:so, xt=bnd, k=30)
```

### Factor-Smooth Interaction (`bs=:fs`)

A smooth-factor interaction that produces a separate smooth curve for each
level of a factor, sharing a common smoothing parameter.

**When to use:** Random smooth effects in multilevel data — each group gets its
own smooth, but a shared smoothing parameter prevents overfitting. Equivalent
to R's `s(x, group, bs="fs")`.

```julia
s(:x, :group, bs=:fs, k=10)   # separate smooth per group level
```

### Constrained Factor Smooth (`bs=:sz`)

A factor-smooth interaction with sum-to-zero constraints. Like `bs=:fs` but
the group-specific smooths are constrained to sum to zero at each covariate
value, ensuring identifiability with a population-level smooth.

**When to use:** When you have a population-level smooth plus group deviations
and want to ensure the deviations are identifiable (sum to zero). Equivalent
to R's `s(x, group, bs="sz")`.

```julia
s(:x, :group, bs=:sz, k=10)
```

### Random Effects (`bs=:re`)

Identity penalty matrix — equivalent to a random intercept or random slope.

**When to use:** Simple random effects (intercepts or slopes) within a `gam()`
call. For more complex random effects structures, use [`gamm()`](@ref).

```julia
s(:group, bs=:re)   # random intercept for `group`
```

### Tensor Products (`te()`, `ti()`, `t2()`)

Tensor product smooths for interactions between variables on different scales.

- `te(:x, :y)` — full tensor product (main effects + interaction)
- `ti(:x, :y)` — interaction only (for ANOVA decomposition)
- `t2(:x, :y)` — alternative tensor product with independent marginal penalties

**When to use:** When interacting variables are on different scales (e.g., space
and time), isotropic smooths (`s(:x, :y)`) are inappropriate because they
assume the same smoothness in all directions. Tensor products handle this.

```julia
# Full tensor product (equivalent to R's te(x1, x2))
te(:x, :y, k=8)

# ANOVA-style decomposition (equivalent to R's ti(x1, x2))
@formulak(y ~ s(x1) + s(x2) + ti(x1, x2))

# Alternative tensor product with independent marginal penalties (R's t2())
t2(:x, :y, k=8)
```

`t2()` produces more penalties than `te()` (one per marginal direction plus a
full interaction penalty), but each penalty is simpler. This can give finer
control over smoothing in each marginal direction.

### Linear-Constraint Bases (`bs=:sc`, `bs=:scad`)

These bases impose general linear inequality or equality constraints on spline
coefficients. Use them through [`gam()`](@ref), [`gamm()`](@ref), or
[`gamlss()`](@ref); they dispatch automatically to the constrained fitting
backend.

- `:sc` — single-penalty constrained spline
- `:scad` — adaptive constrained spline with multiple penalties
- `pc=...` — additional point or weighted-average linear constraints appended to
  a smooth

```julia
# Monotone increasing via explicit linear constraints
gam(@formulak(y ~ s(x, bs=:sc, xt=["m+"], k=12)), df)

# Positive smooth with no intercept
gam(@formulak(y ~ 0 + s(x, bs=:sc, xt=["+"], k=12)), df)
```

These smooths are the closest analogue to `mgcv::scasm()`, but GAM.jl does not
currently mirror mgcv's `pcls()` optimizer line-for-line. Instead, it uses a
Julia-native constrained PIRLS / quadratic-programming backend that preserves
the `gam(...)`-centric API and supports the same linear-constraint basis
families across GAM, GAMLSS, and GAMM workflows.

### SCAM Shape-Constrained Bases

These basis types impose monotonicity and/or convexity constraints on the
smooth. They are used with the [`scam()`](@ref) function. See
[Shape Constraints (SCAM)](scam.md) for details.

| Basis | Constraint | When to use |
|-------|-----------|-------------|
| `:mpi` | Monotone increasing | Dose-response, age effects |
| `:mpd` | Monotone decreasing | Decay curves, survival |
| `:cx` | Convex | U-shaped relationships |
| `:cv` | Concave | Diminishing returns |
| `:micx` | Monotone increasing & convex | Accelerating growth |
| `:micv` | Monotone increasing & concave | Saturating growth |
| `:mdcx` | Monotone decreasing & convex | Decelerating decline |
| `:mdcv` | Monotone decreasing & concave | Accelerating decline |

```julia
# Monotone increasing smooth (use with scam())
s(:x, bs=:mpi, k=10)

# Convex smooth
s(:x, bs=:cx, k=15)
```

## Quick Reference Table

| Symbol | Type | Dimensions | Use case |
|--------|------|-----------|----------|
| `:tp` | Thin plate spline | 1D+ | General-purpose default |
| `:ts` | Thin plate + shrinkage | 1D+ | Variable selection |
| `:cr` | Cubic regression | 1D | Fast, well-understood |
| `:cs` | Cubic + shrinkage | 1D | Cubic with variable selection |
| `:cc` | Cyclic cubic | 1D | Periodic data |
| `:ps` | P-spline | 1D | Explicit penalty control |
| `:cps` | Cyclic P-spline | 1D | Periodic, P-spline variant |
| `:bs` | B-spline | 1D | Continuous derivative penalty |
| `:gp` | Gaussian process | 1D+ | Spatial, GP interpretation |
| `:lo` | Loess | 1D | Local polynomial, exploratory |
| `:fp` | Fractional polynomial | 1D | Dose-response, epidemiology |
| `:ds` | Duchon spline | 1D+ | Flexible TPS generalization |
| `:ad` | Adaptive | 1D | Varying smoothness |
| `:sos` | Spherical spline | 2D (sphere) | Global geospatial data |
| `:spde` | SPDE Matérn | 2D | Large spatial data, sparse GP |
| `:mrf` | Markov random field | Discrete | Areal/lattice data |
| `:so` | Soap film | 2D | Complex domain boundaries |
| `:fs` | Factor-smooth | 1D+ | Random smooth effects |
| `:sz` | Constrained factor | 1D+ | Sum-to-zero group deviations |
| `:re` | Random effect | — | Random intercepts/slopes |
| `te` | Tensor product | 2D+ | Interactions, different scales |
| `ti` | Tensor interaction | 2D+ | ANOVA decomposition |
| `t2` | Alt tensor product | 2D+ | Finer marginal penalty control |
| `:mpi` | SCAM: mono. increasing | 1D | Dose-response |
| `:mpd` | SCAM: mono. decreasing | 1D | Decay curves |
| `:cx` | SCAM: convex | 1D | U-shaped |
| `:cv` | SCAM: concave | 1D | Diminishing returns |
| `:micx` | SCAM: mono. inc. + convex | 1D | Accelerating growth |
| `:micv` | SCAM: mono. inc. + concave | 1D | Saturating growth |
| `:mdcx` | SCAM: mono. dec. + convex | 1D | Decelerating decline |
| `:mdcv` | SCAM: mono. dec. + concave | 1D | Accelerating decline |

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bs` | Symbol | Basis type (see table above) |
| `k` | Int | Basis dimension (number of basis functions) |
| `m` | Int/Tuple | Penalty order (basis-type specific) |
| `fx` | Bool | If true, no penalty (fixed df) |
| `by` | Symbol | Varying coefficient variable |
| `sp` | Float64 | Fixed smoothing parameter |
| `id` | Symbol | Link smoothing parameters across terms |
| `xt` | Any | Extra information (e.g., neighbourhood list for `:mrf`, boundary for `:so`) |

## API Reference

See [API Reference](@ref api-reference) for full documentation of all smooth types
and constructors (`s`, `te`, `ti`, `t2`, `smooth_construct`, `predict_matrix`, etc.).
