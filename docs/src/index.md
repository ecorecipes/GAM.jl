# GAM.jl

*Generalized Additive Models for Julia*

GAM.jl is a Julia port of R's [mgcv](https://cran.r-project.org/package=mgcv) package, providing
penalized regression spline GAMs with automatic smoothness estimation.

## Features

- **Multiple smooth types**: thin plate regression splines (TPRS), cubic regression splines,
  P-splines, B-splines, Gaussian process, tensor products, random effects
- **Automatic smoothing**: REML, ML, and GCV smoothing parameter estimation via
  Extended Fellner-Schall (EFS) method
- **Multiple families**: Gaussian, Poisson, Binomial, Gamma, Inverse Gaussian,
  plus extended families (Negative Binomial, Tweedie, Beta)
- **Large datasets**: `bam()` for memory-efficient fitting with chunked accumulation
  and a fast Gaussian path (precomputed X'X)
- **JuliaStats integration**: follows StatsModels.jl/GLM.jl conventions with
  `@gam_formula` syntax and full StatsBase interface
- **Tested against R**: comprehensive integration tests comparing results against mgcv

## Quick Start

```julia
using GAM, DataFrames

# Generate data
n = 200
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# Fit a GAM
m = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df)

# Inspect the fit
m                    # pretty-printed summary
coef(m)              # coefficients
deviance(m)          # model deviance
m.edf                # effective degrees of freedom per smooth
m.scale              # estimated scale parameter
```

## Installation

```julia
using Pkg
Pkg.add("GAM")
```

## Contents

```@contents
Pages = ["tutorial.md", "smooths.md", "formulas.md", "families.md", "mgcv.md", "api.md"]
Depth = 2
```
