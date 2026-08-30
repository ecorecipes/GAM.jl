# [Smooth Term Reference](@id smooth-terms)

GAM.jl provides 31 registered smooth basis types, covering all commonly used
options from R's mgcv, plus shape-constrained bases from scam and several
additional types including loess, fractional polynomials, spherical splines,
SPDE Matérn, and constrained factor smooths. Two bases (`:so`, `:ds`) are
documented **approximations** of their mgcv namesakes — see the per-basis
notes below. For smooths of
*estimated* covariate transformations (single-index effects and friends),
see [Nested Effects](@ref nested-effects).

```@setup smooths
using GAM, DataFrames, Random
Random.seed!(42)
x = range(0, 1; length=40) |> collect
y = 2 .* x .+ 0.1 .* randn(length(x))
df = DataFrame(x=x, y=y)
```

## Specifying Smooths

All smooth terms are specified using the `s()` function (or `te()`/`ti()`/`t2()`
for tensor products):

```@example smooths
s(:x);                          # TPRS, default k=10
s(:x, bs=:cr, k=20);            # cubic regression spline, k=20
s(:x, :y);                      # 2d TPRS
s(:x, bs=:ps, m=3);             # P-spline with 3rd-order difference penalty
s(:x, bs=:cps, k=12);           # cyclic P-spline
s(:x, :y, bs=:gp);              # Gaussian process smooth
s(:x, fx=true, k=5);            # unpenalized (fixed df)
s(:group, bs=:re);              # random effect
s(:x, bs=:lo, k=15);            # loess smooth
s(:x, bs=:fp);                  # fractional polynomial
s(:lat, :lon, bs=:sos, k=50);   # spherical spline (LATITUDE first, degrees)
s(:x, :y, bs=:spde);            # SPDE Matérn
te(:x, :y, k=5);                # tensor product
ti(:x, :y, k=5);                # tensor product interaction
t2(:x, :y, k=5);                # ANOVA-style tensor product
nothing
```

For neighbourhood- and boundary-based smooths, pass the auxiliary structure
through `xt`:

```text
s(:region, bs=:mrf, xt=nb, k=20)    # Markov random field
s(:x, :y, bs=:so, xt=bnd, k=30)     # soap film smooth
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

```@example smooths
s(:x);               # default TPRS
s(:x, bs=:ts);       # with shrinkage — smooth can be penalized to zero
s(:x, :y, bs=:tp);   # 2D thin plate spline
nothing
```

### Cubic Regression Splines (`bs=:cr`, `bs=:cs`, `bs=:cc`)

Natural cubic regression splines with knots placed at quantiles.

- `:cr` — standard cubic spline
- `:cs` — cubic spline with shrinkage
- `:cc` — cyclic cubic spline (endpoints match)

**When to use:** `:cr` is fast and well-understood for 1D smooths. `:cc` is
the natural choice for periodic covariates (time of day, day of year, angle).

**Default k**: 10.

```@example smooths
s(:x, bs=:cr, k=20);
s(:time, bs=:cc, k=12);   # cyclic for periodic data
nothing
```

### P-Splines (`bs=:ps`)

B-spline basis with difference penalty (Eilers & Marx, 1996).

- `m` parameter controls the penalty order (default 2 = second-order difference)
- The B-spline order is `m + 2` (default = cubic, degree 3)

**When to use:** Popular in biostatistics and demography. Good when you want
explicit control over penalty order.

**Default k**: 10.

```@example smooths
s(:x, bs=:ps, m=3);   # third-order difference penalty
nothing
```

### Cyclic P-Splines (`bs=:cps`)

P-spline basis with periodic boundary conditions — the basis wraps so that
the function value and its derivatives match at the endpoints.

**When to use:** Periodic covariates where you prefer a P-spline basis over
cubic splines (e.g., hour of day, month of year).

**Default k**: 10.

```@example smooths
s(:hour, bs=:cps, k=12);   # smooth over 24-hour cycle
nothing
```

### B-Splines (`bs=:bs`)

B-spline basis with integrated squared derivative penalty.

- Similar to P-splines but uses a continuous derivative penalty
- `m` parameter controls the derivative order (default 2)

**When to use:** When you prefer a continuous derivative penalty over the
discrete difference penalty of P-splines.

**Default k**: 10.

```@example smooths
s(:x, bs=:bs, m=2);
nothing
```

### Gaussian Process Smooth (`bs=:gp`)

A Kammann & Wand (2003) Matérn spline — a direct port of mgcv's `bs="gp"`,
matching it in correlation function, range parameter, knot selection and null
space.

`m` selects the correlation family, defaulting to **3** as in mgcv:

| `m` | correlation (`e = distance / rho`) | |
|-----|-----------------------------------|--|
| 1 | `(1 - 1.5e + 0.5e³)·1(e ≤ 1)` | spherical |
| 2 | `exp(-e^κ)` | power exponential |
| 3 | `(1 + e)·exp(-e)` | Matérn κ = 1.5 (default) |
| 4 | `exp(-e) + e·exp(-e)(1 + e/3)` | Matérn κ = 2.5 |
| 5 | `exp(-e) + e·exp(-e)(1 + 0.4e + e²/15)` | Matérn κ = 3.5 |

A **negative** `m` selects mgcv's *stationary* variant. Note that type 3 is
not the textbook Matérn 3/2: mgcv omits the `√3` the standard
parameterization carries, and this is a port of mgcv's convention.

The range `rho` is **not estimated** — it is fixed at the largest pairwise
distance between knots, the Kammann & Wand choice, exactly as mgcv does.
Knots are the unique covariate values, capped at `max_knots` (2000).

The null space is unpenalized `[1, x]` (or `[1]` when stationary), so a `gp`
smooth shrinks toward a **straight line**, not toward zero.

**When to use:** Spatial data, or when a GP interpretation is desired (e.g.
uncertainty quantification with a specific correlation structure).

**Default k**: 10. (mgcv's own default for `bs="gp"` is `ncol + 1 + 10` = 12
in one dimension; GAM.jl applies its generic default of 10 instead, so set `k`
explicitly when porting a model that relied on mgcv's default.)

`xt` options: `:rho` (range), `:kappa` (κ for `m = 2`), `:max_knots`. Legacy
named correlations remain available through `:corfun` (`:matern32`,
`:matern52`, `:exponential`, `:gaussian`/`:sqexp`, `:power_exp`,
`:mgcv_m32`); these are not mgcv-compatible and are kept for backward
compatibility only.

```@example smooths
s(:x, bs=:gp);                                  # mgcv default (Matérn κ = 1.5)
s(:x, bs=:gp, m=5);                             # Matérn κ = 3.5
s(:x, bs=:gp, m=-3);                            # stationary variant
s(:x, bs=:gp, k=20, xt=Dict(:rho => 0.5));      # fixed range
nothing
```

!!! note "Currently one-dimensional"
    `bs=:gp` supports a single covariate. mgcv also supports multi-dimensional
    GP smooths; use `bs=:tp`, `bs=:spde` or `bs=:sos` for multivariate
    spatial smoothing in the meantime.

### Loess Smooth (`bs=:lo`)

Local polynomial regression smooth (Cleveland, 1979). Fits local polynomial
regressions using a kernel weighting scheme.

**When to use:** When you want a non-parametric smooth that adapts locally
without a global basis representation. Useful for exploratory analysis.
Equivalent to R's `lo()` in gam/mgcv.

**Default k**: 10.

```@example smooths
s(:x, bs=:lo);
s(:x, bs=:lo, k=15);
nothing
```

### Fractional Polynomial (`bs=:fp`)

Fractional polynomial smooth — fits polynomial terms with powers selected from
a predefined set (including negative and fractional powers).

**When to use:** When the relationship is well-described by a low-dimensional
polynomial-like function with potentially non-integer powers. Common in
epidemiology and dose-response modelling. Equivalent to R's `fp()` in mfp.

```@example smooths
s(:x, bs=:fp);
nothing
```

### Duchon Splines (`bs=:ds`)

!!! warning "Approximation"
    `:ds` is currently an **alias for the thin plate spline** (`:tp`) —
    Duchon's fractional-order generalization is not yet implemented.

**Default k**: 10.

```@example smooths
s(:x, bs=:ds);
nothing
```

### Adaptive Smooth (`bs=:ad`)

Adaptive smooth with spatially varying smoothness — the effective penalty
changes along the covariate range, allowing more flexibility in regions
with more rapid change.

**When to use:** When the underlying function has regions of rapid change and
regions of slow change. The penalty adapts so you don't oversmooth or
undersmooth locally.

`m` sets the **number of adaptive sub-penalties** (mgcv's convention, default
5), not a spline order — this changed in 0.2 with a one-time warning when `m`
is passed; `xt = Dict(:n_penalties => n)` is an explicit alias.

```@example smooths
s(:x, bs=:ad, k=20);
nothing
```

### Spherical Splines (`bs=:sos`)

Splines on the sphere for data defined on the surface of a sphere (e.g.,
global spatial data with latitude/longitude coordinates).

**Units.** Latitude and longitude are read as **degrees**, matching mgcv,
whose `makeR` converts internally with `pi/180`. Pass
`xt = Dict(:units => :radians)` to supply radians instead. The unit is
resolved once when the smooth is constructed and reused at prediction, so a
fit and its predictions can never disagree about the scale.

**Penalty order.** `m` selects the reproducing kernel, exactly as in mgcv:

| `m` | kernel | null space |
|---|---|---|
| `-2` | Duchon first-derivative semi-kernel | 1 |
| `-1` | Duchon semi-kernel `z²log(z)/(8π)` | 4 |
| `0` | Wendelberger order 2 (**default**) | 1 |
| `1`–`4` | Wahba pseudospline closed forms | 1 |

Values below `-2` are mapped to `-1` and values above `4` to `4`, matching
mgcv's own normalisation.

!!! note "Matches mgcv's basis exactly"
    This is a direct port of mgcv's construction — Wahba (1981) spherical
    reproducing kernels on great-circle angles, the same truncated
    eigendecomposition keeping the largest-*magnitude* eigenpairs, the same
    constraint absorption and column rescaling. The generalized eigenvalues
    of `(S, XᵀX)`, which determine edf as a function of the smoothing
    parameter, agree with mgcv's to 5×10⁻⁸.

    Consequently `sp` values are **portable**: fitting at mgcv's selected
    `sp` reproduces mgcv's fit to 5×10⁻⁹ (edf 13.217991 in both, on a ±30°,
    `n=300`, `k=20` test). This is not true of `bs=:tp`, where the two
    packages use different — though equivalent — parameterizations.

    Freely-selected fits can still differ slightly, because GAM.jl selects
    smoothing parameters by extended Fellner–Schall where mgcv uses outer
    Newton on the LAML. That is a difference in the optimizer, not the basis.

**When to use:** Geospatial data on a sphere where smoothing should respect
spherical geometry rather than treating lat/lon as a plane. Avoids edge
effects at the poles and the date line.

**Default k**: 50.

```@example smooths
s(:lat, :lon, bs=:sos, k=50);                              # degrees (default)
s(:lat, :lon, bs=:sos, k=50, xt=Dict(:units => :radians)); # radians
nothing
```

### SPDE Matérn Smooth (`bs=:spde`)

Stochastic Partial Differential Equation approach to Matérn Gaussian process
smoothing. Uses a sparse precision matrix representation via the SPDE
approach of Lindgren et al. (2011).

**When to use:** Spatial data where you want Matérn covariance with
computational efficiency from sparse precision matrices. Especially useful
for large spatial datasets where a dense GP (`bs=:gp`) is too slow.

```@example smooths
s(:x, :y, bs=:spde);
nothing
```

### Markov Random Field (`bs=:mrf`)

For discrete spatial or network data. The penalty is defined by a neighbourhood
matrix passed via `xt`.

- Requires a neighbourhood list or adjacency matrix in `xt`
- The covariate should be a factor/categorical variable identifying regions

**When to use:** Areal/lattice data (e.g., disease mapping by region).
Equivalent to R's `s(region, bs="mrf", xt=list(nb=nb))`.

```text
# nb is a Dict mapping region => [neighbours...]
s(:region, bs=:mrf, xt=nb, k=20)
```

### Soap Film Smooth (`bs=:so`)

!!! warning "Approximation"
    The soap-film construction uses a grid-PDE approximation of mgcv's
    exact method; fits will differ from mgcv's `bs="so"`.


For smoothing over complex domains with boundaries (e.g., an estuary, a lake).
Uses a soap-film PDE approach to respect domain boundaries.

- Requires boundary specification via `xt`
- Typically used for 2D spatial smooths

**When to use:** 2D spatial smoothing where the domain has complex boundaries
and you don't want the smooth to "leak" across boundaries. Equivalent to R's
`s(x, y, bs="so", xt=list(bnd=bnd))`.

```text
# bnd defines the domain boundary
s(:x, :y, bs=:so, xt=bnd, k=30)
```

### Factor-Smooth Interaction (`bs=:fs`)

A smooth-factor interaction that produces a separate smooth curve for each
level of a factor, sharing a common smoothing parameter.

**When to use:** Random smooth effects in multilevel data — each group gets its
own smooth, but a shared smoothing parameter prevents overfitting. Equivalent
to R's `s(x, group, bs="fs")`.

```@example smooths
s(:x, :group, bs=:fs, k=10);   # separate smooth per group level
nothing
```

### Constrained Factor Smooth (`bs=:sz`)

A factor-smooth interaction with sum-to-zero constraints. Like `bs=:fs` but
the group-specific smooths are constrained to sum to zero at each covariate
value, ensuring identifiability with a population-level smooth.

**When to use:** When you have a population-level smooth plus group deviations
and want to ensure the deviations are identifiable (sum to zero). Equivalent
to R's `s(x, group, bs="sz")`.

```@example smooths
s(:x, :group, bs=:sz, k=10);
nothing
```

### Random Effects (`bs=:re`)

Identity penalty matrix — equivalent to a random intercept or random slope.

**When to use:** Simple random effects (intercepts or slopes) within a `gam()`
call. For more complex random effects structures, use [`gamm()`](@ref).

```@example smooths
s(:group, bs=:re);   # random intercept for `group`
nothing
```

### Tensor Products (`te()`, `ti()`, `t2()`)

Tensor product smooths for interactions between variables on different scales.

- `te(:x, :y)` — full tensor product (main effects + interaction)
- `ti(:x, :y)` — interaction only (for ANOVA decomposition)
- `t2(:x, :y)` — ANOVA-style tensor product with independent penalty blocks

**When to use:** When interacting variables are on different scales (e.g., space
and time), isotropic smooths (`s(:x, :y)`) are inappropriate because they
assume the same smoothness in all directions. Tensor products handle this.

```@example smooths
te(:x, :y, k=8);                               # full tensor product
@formula(y ~ s(x1) + s(x2) + ti(x1, x2));     # ANOVA-style decomposition
t2(:x, :y, k=8);                              # ANOVA-style tensor product
nothing
```

`t2()` follows mgcv's Wood, Scheipl & Faraway (2013) construction: each
marginal is split into orthogonal null and range parts, and the tensor columns
partition into blocks that each carry their own identity penalty on their own
columns. The penalties are therefore **diagonal with non-overlapping support**,
which is what lets a `t2()` smooth be written as independent random-effect
blocks (one variance component per penalty) — the property `te()` lacks, since
its overlapping penalties require mgcv's `pdTens` class. Use `t2()` when you
need that mixed-model decomposition (as gamm4 does); use `te()` otherwise.

The construction is verified against mgcv: for `t2(x, z, k=4)` both produce 15
columns, 3 penalties of rank 4 with identical disjoint supports, and a
null-space dimension of 3.

!!! warning "Scalar `k` is per-marginal, as in mgcv (changed in 0.2)"
    For tensor smooths a scalar `k` is the dimension of **each marginal**
    basis, recycled across margins — exactly mgcv's convention. So
    `te(x, z, k=5)` builds a 5×5 (24-column, post-constraint) smooth in both
    packages, and an mgcv model ports with its `k` unchanged.

    Before GAM.jl 0.2 a scalar `k` was a *total* dimension hint, split as
    `round(Int, k^(1/d))` per margin — `te(x, z, k=25)` used to mean 5×5 and
    now means 25×25. Models written against the old behaviour should switch
    to the explicit vector form to keep their basis size:

    ```julia
    te(:x, :z, k = 5)        # 5 per margin — same model as mgcv's te(x, z, k = 5)
    te(:x, :z, k = [5, 5])   # identical, margins given explicitly
    te(:x, :z, k = [4, 7])   # unequal marginal dimensions
    ```

    Plain `s()` smooths are unaffected — `k` has always meant the same thing
    in both packages there.

### Linear-Constraint Bases (`bs=:sc`, `bs=:scad`)

These bases impose general linear inequality or equality constraints on spline
coefficients. Use them through [`gam()`](@ref), `gam(..., family)` for
multi-parameter models, or [`gamm()`](@ref); they dispatch automatically to
the constrained fitting backend.

- `:sc` — single-penalty constrained spline
- `:scad` — adaptive constrained spline with multiple penalties
- `pc=...` — additional point or weighted-average linear constraints appended to
  a smooth

```@example smooths
gam(@formula(y ~ s(x, bs=:sc, xt=["m+"], k=12)), df);
gam(@formula(y ~ 0 + s(x, bs=:sc, xt=["+"], k=12)), df);
nothing
```

These smooths are the closest analogue to `mgcv::scasm()`, but GAM.jl does not
currently mirror mgcv's `pcls()` optimizer line-for-line. Instead, it uses a
Julia-native constrained PIRLS / quadratic-programming backend that preserves
the `gam(...)`-centric API and supports the same linear-constraint basis
families across GAM, GAMLSS, and GAMM workflows.

### SCAM Shape-Constrained Bases

These basis types impose monotonicity and/or convexity constraints on the
smooth. They dispatch automatically through [`gam()`](@ref); the legacy
[`scam()`](@ref) wrapper remains available. See
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

```@example smooths
s(:x, bs=:mpi, k=10);   # monotone increasing smooth
s(:x, bs=:cx, k=15);    # convex smooth
nothing
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
| `:ds` | Duchon spline | 1D+ | **Alias for `:tp`** (warns) — not yet implemented |
| `:ad` | Adaptive | 1D | Varying smoothness |
| `:sos` | Spherical spline | 2D (sphere) | Geospatial data on a sphere (degrees; matches mgcv) |
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
| `k` | Int or Vector | Basis dimension. For `s()`, the number of basis functions. For `te`/`ti`/`t2`, the **per-marginal** dimension recycled across margins (mgcv's convention — see the warning above); a vector gives unequal marginal dimensions |
| `m` | Int/Tuple | Penalty order (basis-type specific) |
| `fx` | Bool | If true, no penalty (fixed df) |
| `by` | Symbol | Varying coefficient variable |
| `sp` | Float64 or Vector | Fixed smoothing parameter(s); a vector fixes each penalty of a multi-penalty smooth (`:ad`, `te`/`t2`, factor-`by`) individually |
| `id` | Symbol | Link smoothing parameters across terms |
| `xt` | Any | Extra information (e.g., neighbourhood list for `:mrf`, boundary for `:so`) |

## API Reference

See [API Reference](@ref api-reference) for full documentation of all smooth types
and constructors (`s`, `te`, `ti`, `t2`, `smooth_construct`, `predict_matrix`, etc.).
