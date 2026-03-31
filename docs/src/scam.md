# [Shape Constraints (SCAM)](@id scam)

Shape-Constrained Additive Models (SCAM) extend GAMs by enforcing
monotonicity and/or convexity constraints on smooth terms. This is useful when
domain knowledge dictates that a relationship must be, e.g., monotonically
increasing or convex.

GAM.jl's SCAM implementation follows Pya & Wood (2015).

For explicit linear-constraint smooths (`bs=:sc`, `bs=:scad`, or `pc`
constraints), use `gam(...)` instead. Those terms go through GAM.jl's separate
constrained-QP backend, whereas SCAM basis types use reparameterized
SCOP-splines.

## Constraint Types

| Basis | Constraint | Description |
|-------|-----------|-------------|
| `:mpi` | Monotone increasing | f'(x) ≥ 0 everywhere |
| `:mpd` | Monotone decreasing | f'(x) ≤ 0 everywhere |
| `:cx` | Convex | f''(x) ≥ 0 everywhere |
| `:cv` | Concave | f''(x) ≤ 0 everywhere |
| `:micx` | Monotone increasing & convex | f'(x) ≥ 0 and f''(x) ≥ 0 |
| `:micv` | Monotone increasing & concave | f'(x) ≥ 0 and f''(x) ≤ 0 |
| `:mdcx` | Monotone decreasing & convex | f'(x) ≤ 0 and f''(x) ≥ 0 |
| `:mdcv` | Monotone decreasing & concave | f'(x) ≤ 0 and f''(x) ≤ 0 |

## Interface

```julia
scam(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,
    control = scam_control(),
)
```

The interface is identical to `gam()` except that smooth terms use
shape-constrained basis types.

## Examples

### Monotone Increasing

```julia
using GAM, DataFrames

n = 200
x = sort(randn(n))
y = 2.0 .* atan.(x) .+ 0.3 .* randn(n)   # monotone increasing truth
df = DataFrame(x=x, y=y)

m = scam(@gam_formula(y ~ s(x, k=15, bs=:mpi)), df)
```

### Convex Fit

```julia
y = x.^2 .+ 0.5 .* randn(n)   # convex truth
df = DataFrame(x=x, y=y)

m = scam(@gam_formula(y ~ s(x, k=15, bs=:cx)), df)
```

### Combined Constraints

You can mix constrained and unconstrained smooths in the same model:

```julia
x2 = randn(n)
y = 2.0 .* atan.(x) .+ sin.(x2) .+ 0.3 .* randn(n)
df = DataFrame(x=x, x2=x2, y=y)

# x must be monotone increasing, x2 is unconstrained
m = scam(@gam_formula(y ~ s(x, k=15, bs=:mpi) + s(x2, k=10, bs=:cr)), df)
```

### Monotone Decreasing & Concave

```julia
y = -log.(1.0 .+ exp.(x)) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

m = scam(@gam_formula(y ~ s(x, k=15, bs=:mdcv)), df)
```

## ScamControl Options

```julia
ctrl = scam_control(
    epsilon = 1e-7,      # convergence tolerance
    maxit = 200,         # max iterations
    trace = true,        # print progress
)

m = scam(@gam_formula(y ~ s(x, bs=:mpi)), df; control=ctrl)
```

## See Also

- [Smooth Terms](@ref smooth-terms) for the full list of basis types including SCAM bases
- [Comparison with mgcv](@ref mgcv-comparison) for how SCAM compares to R's scam package
- [API Reference](@ref api-reference) for `scam`, `scam_control`, `ScamControl`
