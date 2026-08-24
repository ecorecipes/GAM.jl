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
design. Unlike mgcv's `bam(discrete=TRUE)`, covariates are **not**
discretized — that machinery is a possible future addition. Note also that
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
    bam_ctrl = bam_control(),
)
```

## BamControl Options

`bam_control` takes a single option, the accumulation chunk size. (The former
`discrete`, `max_unique`, and `nthreads` keywords are deprecated no-ops.)

```@example bam
ctrl = bam_control(chunk_size = 4000)

m_ctrl = bam(@formula(y ~ s(x, k=20, bs=:cr)), df; bam_ctrl=ctrl);
nothing
```

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
