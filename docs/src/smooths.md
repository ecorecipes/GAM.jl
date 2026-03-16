# Smooth Term Reference

GAM.jl provides several smooth basis types, matching the most commonly used
options from R's mgcv.

## Specifying Smooths

All smooth terms are specified using the `s()` function (or `te()`/`ti()` for
tensor products):

```julia
s(:x)                          # TPRS, default k=10
s(:x, bs=:cr, k=20)           # cubic regression spline, k=20
s(:x, :y)                     # 2d TPRS
s(:x, bs=:ps, m=3)            # P-spline with 3rd-order difference penalty
s(:x, fx=true, k=5)           # unpenalized (fixed df)
s(:group, bs=:re)              # random effect
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

### Cubic Regression Splines (`bs=:cr`, `bs=:cs`, `bs=:cc`)

Natural cubic regression splines with knots placed at quantiles.

- `:cr` — standard cubic spline
- `:cs` — cubic spline with shrinkage
- `:cc` — cyclic cubic spline (endpoints match)

**Default k**: 10.

### P-Splines (`bs=:ps`)

B-spline basis with difference penalty (Eilers & Marx, 1996).

- `m` parameter controls the penalty order (default 2 = second-order difference)
- The B-spline order is `m + 2` (default = cubic, degree 3)

**Default k**: 10.

### B-Splines (`bs=:bs`)

B-spline basis with integrated squared derivative penalty.

- Similar to P-splines but uses a continuous derivative penalty
- `m` parameter controls the derivative order (default 2)

**Default k**: 10.

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

## Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bs` | Symbol | Basis type (`:tp`, `:cr`, `:ps`, `:bs`, `:re`, etc.) |
| `k` | Int | Basis dimension (number of basis functions) |
| `m` | Int | Penalty order (basis-type specific) |
| `fx` | Bool | If true, no penalty (fixed df) |
| `by` | Symbol | Varying coefficient variable |
| `sp` | Float64 | Fixed smoothing parameter |
| `id` | Symbol | Link smoothing parameters across terms |

## API Reference

```@docs
s
te
ti
smooth_construct
predict_matrix
SmoothSpec
ConstructedSmooth
```
