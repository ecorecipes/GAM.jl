# Getting Started

## Basic Gaussian GAM

The simplest use case: fitting a smooth function to noisy data.

```julia
using GAM, DataFrames

# Simulate data
n = 300
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# Fit with cubic regression splines
m = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df)
```

The `@gam_formula` macro works like StatsModels' `@formula` but supports
`s()`, `te()`, and `ti()` smooth terms with keyword arguments.

## Multiple Smooths

```julia
x2 = randn(n)
y2 = sin.(x) .+ 0.5 .* x2.^2 .+ 0.3 .* randn(n)
df2 = DataFrame(x=x, x2=x2, y=y2)

m2 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), df2)
```

## Poisson GAM

```julia
using Distributions

mu = exp.(0.5 .* sin.(x) .+ 0.5)
counts = [rand(Poisson(m)) for m in mu]
df3 = DataFrame(x=x, y=Float64.(counts))

m3 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df3;
    family=Poisson(), link=LogLink())
```

## Binomial GAM

```julia
p = 1.0 ./ (1.0 .+ exp.(-2.0 .* sin.(x)))
y_bin = Float64.([rand(Bernoulli(pi)) for pi in p])
df4 = DataFrame(x=x, y=y_bin)

m4 = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df4;
    family=Binomial(), link=LogitLink())
```

## Inspecting the Model

GAM.jl implements the full StatsBase interface:

```julia
using StatsBase

coef(m)           # coefficients
vcov(m)           # variance-covariance matrix
stderror(m)       # standard errors
deviance(m)       # model deviance
nobs(m)           # number of observations
dof(m)            # degrees of freedom

# GAM-specific
m.edf             # EDF per smooth
m.edf_total       # total EDF
m.scale           # scale parameter
m.sp              # log smoothing parameters
m.smooths         # constructed smooth terms
```

## Prediction

```julia
# Prediction at new points
x_new = range(0, 2π; length=50) |> collect
sm = m.smooths[1]
X_new = hcat(ones(length(x_new)), predict_matrix(sm, (x=x_new,)))
y_pred = X_new * coef(m)
```

## Controlling the Fit

```julia
ctrl = gam_control(
    epsilon = 1e-8,      # convergence tolerance
    maxit = 500,         # max PIRLS iterations
    outer_maxit = 100,   # max outer iterations
    trace = true,        # print progress
    gamma = 1.4,         # extra smoothing (>1 = smoother)
)

m = gam(@gam_formula(y ~ s(x)), df; control=ctrl)
```
