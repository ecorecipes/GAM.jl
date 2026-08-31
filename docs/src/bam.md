# [Large Data (BAM)](@id bam)

`bam()` is the large-dataset counterpart to `gam()`. It accumulates the
normal equations in row chunks, so peak memory beyond the design matrix is
bounded by the chunk size, and reuses the same EFS smoothing-parameter
machinery as `gam()`.

```@setup bam
using GAM, DataFrames, Random, Distributions
using GLM: LogLink
Random.seed!(42)

n = 12_000
x = rand(n)
y = sin.(2π .* x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

n2 = 15_000
x1 = rand(n2)
x2 = rand(n2)
y2 = sin.(2π .* x1) .+ cos.(2π .* x2) .+ 0.3 .* randn(n2)
df2 = DataFrame(x1=x1, x2=x2, y=y2)

n_pois = 10_000
x_pois = rand(n_pois) .* 4
mu = exp.(0.5 .+ sin.(x_pois))
y_pois = Float64.(rand.(Poisson.(mu)))
df_pois = DataFrame(x=x_pois, y=y_pois)
```

## When to Use BAM

`bam()` carries fixed setup overhead that only pays for itself once there are
enough rows. Measured against `gam()` on the same model:

| n | `bam` vs `gam` | Recommended |
|---|----------------|-------------|
| 1,000 | ~21x **slower** | `gam()` |
| 10,000 | ~1.9x faster | either |
| 100,000 | ~3.9x faster | `bam()` |

The crossover is around **n ≈ 5,000–10,000**. Below it, prefer `gam()`;
`bam()` is not simply a drop-in speedup.

BAM produces results equivalent to `gam()` (fitted-value agreement is asserted
in the test suite) with reduced peak memory on large datasets.

## How It Works

BAM builds the dense n × p design matrix once, then accumulates `X'WX` and
`X'Wz` in row chunks (BLAS `syrk` per chunk), never forming the full weighted
design. Covariate discretization is available separately via
`discrete=true` (see below). Note also that
solving the normal equations squares the condition number relative to mgcv's
QR updating; poorly scaled bases are less stable here than in `gam()`-grade
QR approaches, though near-singular models are protected by ridge recovery.

Key features:
- **Chunked accumulation**: processes data in chunks to limit memory
- **Fast Gaussian path**: for Gaussian identity models, precomputed `X'X` and `X'y`
- **Same EFS smoothing selection as `gam()`**, including score-based convergence
- **`offset=` and `select=` supported** as in `gam()`

## Interface

```text
bam(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,        # :REML or :ML (:GCV/:UBRE are not supported by bam)
    discrete = false,      # or `true`, or an Int grid resolution (default 1000)
    retain_X = true,       # `false` drops the n x p model matrix after fitting
    bam_ctrl = bam_control(),
)
```

## Covariate Discretization (`discrete=true`)

`bam(...; discrete=true)` mirrors `mgcv::bam(..., discrete=TRUE)`: each
supported smooth is stored as its basis evaluated at the *unique* covariate
values, plus an integer index vector, instead of an `n × p` block.

| term | under `discrete=true` |
|---|---|
| 1-D smooths (`s(x)`) | discretized |
| `te(x, z)` tensor smooths | discretized |
| `s(g, bs=:re)` (factor or random slope) | discretized |
| `by=` terms, factor or numeric | dense |
| `ti`, `t2` | dense |

Unsupported terms fall back silently — the fit is unchanged, only its
representation.

**It is an approximation, by design.** Covariates with at most `discrete`
distinct values (default 1000; pass an integer to change it) are represented
exactly. Beyond that they are rounded onto an equally spaced grid, and the
rounding — not the arithmetic, which is exact to 1e-15 — makes the fit
approximate. Fitted values agree with the dense fit to about 1e-3 of their
range and total EDF to about 1e-4 relative, but **individual smoothing
parameters can move substantially** (Δlog λ up to 2.37 observed). Compare
fitted values and EDF against a dense fit, not `sp` elementwise.

Reassuringly, GAM.jl's discrete fit agrees with *mgcv's* discrete fit to
4.25e-4 of the fitted range — closer than either package agrees with its own
dense fit (1.39e-3 and 2.32e-3 respectively).

**When it pays.** The `X'WX` kernel is ~60× faster at `k=20` and ~450× at
`k=100` (n = 10⁶), but end-to-end gains are Amdahl-bounded:

| model, n = 2·10⁵ | speedup vs dense |
|---|---|
| Normal, 4 × `s(k=20)` | **0.95× — slightly slower** |
| Poisson, 4 × `s(k=20)` | 1.71× |
| Poisson, 2 × `s(k=100)` | 6.09× |
| `te(15,15)`, n = 10⁶ | within noise of dense |

A Gaussian fit accumulates `X'WX` once, so binning is pure overhead; the win
scales with the number of inner solves, i.e. with non-Gaussian families.

**Peak memory is largely unchanged today.** The compact representations are
dramatic in isolation — a tensor block is 8.13 MB against a 1716.61 MB dense
block, and a 200-level random effect 766.75 → 6.03 MiB — but `setup_gam`
still materializes the dense block before the discrete design replaces it,
and `ConstructedSmooth.X` is re-expanded to `n` rows for downstream
consumers. Do not enable `discrete=true` expecting a large peak-RSS
reduction today.

## Dropping the Model Matrix (`retain_X=false`)

`GamModel.X` duplicates data already held per smooth: every smooth block of
`X` is bitwise identical to the corresponding `ConstructedSmooth.X`. Passing
`retain_X = false` keeps only the parametric columns (587.5 MiB → 7.6 MiB at
n = 10⁶); [`model_matrix`](@ref) reassembles the full matrix bitwise on
demand, with no basis re-evaluation.

## BamControl Options

`bam_control` takes a single option, the accumulation chunk size.
Discretization is a keyword on `bam` itself (`bam(...; discrete=true)`), not
on `bam_control`; the former `discrete`, `max_unique` and `nthreads`
arguments to `bam_control` are deprecated and warn. Threading the
accumulation is still not implemented.

```@example bam
ctrl = bam_control(chunk_size = 4000)

m_ctrl = bam(@formula(y ~ s(x, k=20, bs=:cr)), df; bam_ctrl=ctrl);
nothing
```

## Performance: set the BLAS thread count

`bam` accumulates `X'WX` chunk by chunk, and each chunk's contribution is a
BLAS rank-`k` update on a **tall, thin** matrix (`chunk_size × p`, with `p`
typically in the tens). That shape gives each BLAS thread too little work to
cover synchronisation, and beyond about four threads the overhead dominates
badly. Measured on an idle 8-core machine, a Poisson fit with `n = 100,000`
and four `s(k=20)` smooths:

| BLAS threads | time |
|---|---|
| 1 | 2.18 s |
| 2 | 1.90 s |
| **4** | **1.75 s** |
| 8 (the default here) | **9.41 s** |

So the default can be **over 5× slower** than the best setting. If `bam` seems
slow, this is the first thing to check:

```julia
using LinearAlgebra
BLAS.set_num_threads(4)
```

GAM.jl deliberately does **not** set this for you: `BLAS.set_num_threads` is
global process state, and a library silently changing it would affect every
other computation in the session.

Threading the accumulation at the Julia level was implemented and measured,
then rejected: the fastest configuration is serial at four BLAS threads, and
adding Julia threads on top made it ~20% slower while costing per-thread
scratch memory. The accumulation is already parallel — through BLAS.

## Examples

### Basic Large Dataset

```@example bam
m = bam(@formula(y ~ s(x, k=20, bs=:cr)), df);
nothing
```

### Multiple Smooths

```@example bam
m2 = bam(@formula(y ~ s(x1, k=20, bs=:cr) + s(x2, k=20, bs=:cr)), df2);
nothing
```

### Poisson BAM

```@example bam
m_pois = bam(@formula(y ~ s(x, k=15, bs=:cr)), df_pois;
    family=Poisson(), link=LogLink());
nothing
```

## See Also

- [Getting Started](@ref getting-started) for a quick BAM example
- [API Reference](@ref api-reference) for `bam`, `bam_control`, `BamControl`
