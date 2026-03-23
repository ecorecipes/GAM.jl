# Large Data (BAM)

`bam()` is the large-dataset counterpart to `gam()`. It uses discretization
and efficient matrix operations to handle datasets with hundreds of thousands
to millions of observations.

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

```julia
bam(formula, data;
    family = Gaussian(),
    link = IdentityLink(),
    method = :fREML,
    control = bam_control(),
)
```

## BamControl Options

```julia
ctrl = bam_control(
    chunk_size = 10000,     # rows per chunk
    discrete = true,        # enable discretization (default)
    nthreads = 4,           # threads for parallel accumulation
    trace = true,           # print progress
)

m = bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), big_df; control=ctrl)
```

## Examples

### Basic Large Dataset

```julia
using GAM, DataFrames

n = 100_000
x = rand(n)
y = sin.(2π .* x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

m = bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)
```

### Multiple Smooths

```julia
n = 200_000
x1 = rand(n)
x2 = rand(n)
y = sin.(2π .* x1) .+ cos.(2π .* x2) .+ 0.3 .* randn(n)
df = DataFrame(x1=x1, x2=x2, y=y)

m = bam(@gam_formula(y ~ s(x1, k=20, bs=:cr) + s(x2, k=20, bs=:cr)), df)
```

### Poisson BAM

```julia
using Distributions

n = 50_000
x = rand(n) .* 4
mu = exp.(0.5 .+ sin.(x))
y = Float64.([rand(Poisson(m)) for m in mu])
df = DataFrame(x=x, y=y)

m = bam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df;
    family=Poisson(), link=LogLink())
```

## See Also

- [Getting Started](@ref) for a quick BAM example
- [API Reference](@ref) for `bam`, `bam_control`, `BamControl`
