# [Shape Constraints (SCAM)](@id scam)

Shape-Constrained Additive Models (SCAM) extend GAMs by enforcing
monotonicity and/or convexity constraints on smooth terms. This is useful when
domain knowledge dictates that a relationship must be, e.g., monotonically
increasing or convex.

GAM.jl's SCAM implementation follows Pya & Wood (2015).

```@setup scam
using GAM, DataFrames, Random
Random.seed!(42)

n = 120
x = sort(randn(n))
y = 2.0 .* atan.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)
```

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

```text
gam(formula, data;
    family = Normal(),
    link = IdentityLink(),
    method = :REML,
    control = gam_control(),
)
```

The preferred interface is ordinary `gam(...)` with SCAM basis types such as
`bs=:mpi` or `bs=:cx`. The unified `gam(...)` surface reuses
`gam_control(...)`; the legacy `scam(...)` wrapper remains available for
compatibility and still accepts `scam_control(...)` for the SCAM-specific
`not_exp` option.

## Examples

### Monotone Increasing

```@example scam
m = gam(@formula(y ~ s(x, k=15, bs=:mpi)), df);
nothing
```

### Convex Fit

```@example scam
y_cx = x .^ 2 .+ 0.3 .* randn(n)
df_cx = DataFrame(x=x, y=y_cx)

m_cx = gam(@formula(y ~ s(x, k=15, bs=:cx)), df_cx);
nothing
```

### Combined Constraints

You can mix constrained and unconstrained smooths in the same model:

```@example scam
x2 = randn(n)
y_mix = 2.0 .* atan.(x) .+ sin.(x2) .+ 0.3 .* randn(n)
df_mix = DataFrame(x=x, x2=x2, y=y_mix)

m_mix = gam(@formula(y ~ s(x, k=15, bs=:mpi) + s(x2, k=10, bs=:cr)), df_mix);
nothing
```

### Monotone Decreasing & Concave

```@example scam
y_mdcv = -log.(1.0 .+ exp.(x)) .+ 0.3 .* randn(n)
df_mdcv = DataFrame(x=x, y=y_mdcv)

m_mdcv = gam(@formula(y ~ s(x, k=15, bs=:mdcv)), df_mdcv);
nothing
```

## Control Options

```@example scam
ctrl = gam_control(
    epsilon = 1e-7,      # convergence tolerance
    maxit = 100,         # max iterations
    trace = false,
)

m_ctrl = gam(@formula(y ~ s(x, bs=:mpi)), df; control=ctrl);
nothing
```

If you prefer the legacy `scam(...)` wrapper, it still accepts
`scam_control(...)` directly.

## See Also

- [Smooth Terms](@ref smooth-terms) for the full list of basis types including SCAM bases
- [Comparison with mgcv](@ref mgcv-comparison) for how SCAM compares to R's scam package
- [API Reference](@ref api-reference) for `gam`, `gam_control`, and the compatibility wrappers `scam`, `scam_control`, `ScamControl`
