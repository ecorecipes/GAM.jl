# Smooth Term Reference

GAM.jl provides 18 smooth basis types, covering all commonly used options from
R's mgcv plus shape-constrained bases from scam.

## Specifying Smooths

All smooth terms are specified using the `s()` function (or `te()`/`ti()` for
tensor products):

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
te(:x, :y, k=5)               # tensor product
ti(:x, :y, k=5)               # tensor product interaction
```

## Available Basis Types

### Thin Plate Regression Splines (`bs=:tp`, `bs=:ts`)

The default smooth type. Optimal in a certain sense among all smoothers of
a given dimension and penalty order.

- `:tp` — standard TPRS
- `:ts` — TPRS with shrinkage penalty (adds small penalty on null space)

**Default k**: 10 for 1D, 30 for 2D.

```julia
s(:x)               # default TPRS
s(:x, bs=:ts)       # with shrinkage
```

### Cubic Regression Splines (`bs=:cr`, `bs=:cs`, `bs=:cc`)

Natural cubic regression splines with knots placed at quantiles.

- `:cr` — standard cubic spline
- `:cs` — cubic spline with shrinkage
- `:cc` — cyclic cubic spline (endpoints match)

**Default k**: 10.

```julia
s(:x, bs=:cr, k=20)
s(:time, bs=:cc, k=12)   # cyclic for periodic data
```

### P-Splines (`bs=:ps`)

B-spline basis with difference penalty (Eilers & Marx, 1996).

- `m` parameter controls the penalty order (default 2 = second-order difference)
- The B-spline order is `m + 2` (default = cubic, degree 3)

**Default k**: 10.

```julia
s(:x, bs=:ps, m=3)   # third-order difference penalty
```

### Cyclic P-Splines (`bs=:cps`)

P-spline basis with periodic boundary conditions — the basis wraps so that
the function value and its derivatives match at the endpoints.

- Use for periodic covariates (time of day, day of year, angle)
- `m` parameter controls the penalty order as with `:ps`

**Default k**: 10.

```julia
s(:hour, bs=:cps, k=12)   # smooth over 24-hour cycle
```

### B-Splines (`bs=:bs`)

B-spline basis with integrated squared derivative penalty.

- Similar to P-splines but uses a continuous derivative penalty
- `m` parameter controls the derivative order (default 2)

**Default k**: 10.

```julia
s(:x, bs=:bs, m=2)
```

### Gaussian Process Smooth (`bs=:gp`)

A smooth based on a Gaussian process covariance kernel. Useful for spatial
data and when a GP interpretation is desired.

- Supports multiple kernel types via `xt` (e.g., Matérn, squared exponential)
- Automatically estimates the length-scale parameter

**Default k**: 10.

```julia
s(:x, bs=:gp)                             # default kernel
s(:x, :y, bs=:gp, k=50)                   # 2D GP smooth
```

### Duchon Splines (`bs=:ds`)

Generalization of thin plate splines that allow fractional derivative penalties.
Useful when the standard TPS penalty order is not ideal.

- `m` parameter controls the penalty order (can be non-integer via a tuple)

**Default k**: 10.

```julia
s(:x, bs=:ds)
s(:x, :y, bs=:ds, m=(1, 0.5))   # custom penalty specification
```

### Markov Random Field (`bs=:mrf`)

For discrete spatial or network data. The penalty is defined by a neighbourhood
matrix passed via `xt`.

- Requires a neighbourhood list or adjacency matrix in `xt`
- The covariate should be a factor/categorical variable identifying regions

```julia
# nb is a Dict mapping region => [neighbours...]
s(:region, bs=:mrf, xt=nb, k=20)
```

### Soap Film Smooth (`bs=:so`)

For smoothing over complex domains with boundaries (e.g., an estuary, a lake).
Uses a soap-film PDE approach to respect domain boundaries.

- Requires boundary specification via `xt`
- Typically used for 2D spatial smooths

```julia
# bnd defines the domain boundary
s(:x, :y, bs=:so, xt=bnd, k=30)
```

### Factor-Smooth Interaction (`bs=:fs`)

A smooth-factor interaction that produces a separate smooth curve for each
level of a factor, sharing a common smoothing parameter. Useful for
random smooth effects in multilevel data.

```julia
s(:x, :group, bs=:fs, k=10)   # separate smooth per group level
```

### Random Effects (`bs=:re`)

Identity penalty matrix — equivalent to a random intercept or random slope.

```julia
s(:group, bs=:re)   # random intercept for `group`
```

### Tensor Products (`te()`, `ti()`)

Tensor product smooths for interactions between variables on different scales.

- `te(:x, :y)` — full tensor product (main effects + interaction)
- `ti(:x, :y)` — interaction only (for ANOVA decomposition)

```julia
# ANOVA-style decomposition
@gam_formula(y ~ s(x1) + s(x2) + ti(x1, x2))
```

### SCAM Shape-Constrained Bases

These basis types impose monotonicity and/or convexity constraints on the
smooth. They are used with the [`scam()`](@ref) function. See
[Shape Constraints (SCAM)](@ref) for details.

| Basis | Constraint |
|-------|-----------|
| `:mpi` | Monotone increasing |
| `:mpd` | Monotone decreasing |
| `:cx` | Convex |
| `:cv` | Concave |
| `:micx` | Monotone increasing & convex |
| `:micv` | Monotone increasing & concave |
| `:mdcx` | Monotone decreasing & convex |
| `:mdcv` | Monotone decreasing & concave |

```julia
# Monotone increasing smooth (use with scam())
s(:x, bs=:mpi, k=10)

# Convex smooth
s(:x, bs=:cx, k=15)
```

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bs` | Symbol | Basis type (`:tp`, `:cr`, `:ps`, `:cps`, `:bs`, `:gp`, `:ds`, `:mrf`, `:so`, `:fs`, `:re`, etc.) |
| `k` | Int | Basis dimension (number of basis functions) |
| `m` | Int/Tuple | Penalty order (basis-type specific) |
| `fx` | Bool | If true, no penalty (fixed df) |
| `by` | Symbol | Varying coefficient variable |
| `sp` | Float64 | Fixed smoothing parameter |
| `id` | Symbol | Link smoothing parameters across terms |
| `xt` | Any | Extra information (e.g., neighbourhood list for `:mrf`, boundary for `:so`) |

## API Reference

```@docs
s
te
ti
smooth_construct
predict_matrix
SmoothSpec
ConstructedSmooth
CyclicPSpline
GPSmooth
DuchonSpline
SoapFilm
MarkovRandomField
FactorSmooth
```
