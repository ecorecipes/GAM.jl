# [Large Data (BAM)](@id bam)

`bam()` is the large-dataset counterpart to `gam()`. It uses discretization
and efficient matrix operations to handle datasets with hundreds of thousands
to millions of observations.

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

| Dataset size | Recommended |
|-------------|-------------|
| n < 10,000 | `gam()` |
| n > 10,000 | `bam()` — faster, lower memory |

BAM produces equivalent results to `gam()` but with substantially reduced
computation time and memory usage on large datasets.

## How It Works

BAM discretizes each covariate into a grid and works with the discretized
representation. This avoids forming the full n × p model matrix, reducing
both memory and computation from O(n) to O(grid size).

Key optimizations:
- **Discretized covariates**: covariates are binned; basis evaluation happens on unique values
- **Fast Gaussian path**: for Gaussian family, uses precomputed X'X and X'y
- **Chunked accumulation**: processes data in chunks to limit memory

## Interface

```text
bam(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :REML,
    bam_ctrl = bam_control(),
)
```

## BamControl Options

```@example bam
ctrl = bam_control(
    chunk_size = 4000,
    discrete = true,
    nthreads = 1,
)

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
