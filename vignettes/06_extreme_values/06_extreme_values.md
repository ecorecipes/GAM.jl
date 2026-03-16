# Extreme Value GAMs
GAM.jl Contributors

- [Introduction](#introduction)
- [Setup](#setup)
- [GEV model](#gev-model)
  - [Simulate GEV data](#simulate-gev-data)
  - [Fit the GEV model](#fit-the-gev-model)
  - [Examine parameter estimates](#examine-parameter-estimates)
  - [Compare fitted vs true
    functions](#compare-fitted-vs-true-functions)
- [GPD model](#gpd-model)
  - [Simulate GPD data](#simulate-gpd-data)
  - [Fit the GPD model](#fit-the-gpd-model)
  - [Examine GPD estimates](#examine-gpd-estimates)
- [Model structure](#model-structure)
- [Summary](#summary)

## Introduction

Extreme value theory (EVT) provides a principled framework for modeling
the tails of distributions. Two key models are:

- **Generalized Extreme Value (GEV)** distribution for block maxima:
  $Y \sim \text{GEV}(\mu, \sigma, \xi)$ with CDF
  $$F(y) = \exp\left\{-\left[1 + \xi\left(\frac{y-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

- **Generalized Pareto Distribution (GPD)** for threshold exceedances:
  $Y \mid Y > u \sim \text{GPD}(\sigma, \xi)$ with survival function
  $$\bar{F}(y) = \left[1 + \xi\left(\frac{y-u}{\sigma}\right)\right]^{-1/\xi}$$

GAM.jl’s `evgam` function fits **multi-parameter GAMs** where each
distribution parameter can depend on covariates through smooth
functions, following the approach of the R
[evgam](https://cran.r-project.org/package=evgam) package.

## Setup

``` julia
using GAM
using StatsAPI: fitted
using LinearAlgebra: diag
using DataFrames
using CSV
using Random
using Statistics
```

## GEV model

### Simulate GEV data

We generate block maxima where the location and scale vary smoothly with
a covariate $x$:

- Location: $\mu(x) = 5 + 2\sin(2\pi x)$
- Log-scale: $\log\sigma(x) = -0.5 + 0.5x$, so
  $\sigma(x) = \exp(-0.5 + 0.5x)$
- Shape: $\xi = 0.1$ (constant, light upper tail)

``` julia
df_gev = CSV.read("data_gev.csv", DataFrame)
n = nrow(df_gev)
x = df_gev.x
y_gev = df_gev.y

mu_true = 5.0 .+ 2.0 .* sin.(2π .* x)
logsigma_true = -0.5 .+ 0.5 .* x
sigma_true = exp.(logsigma_true)
xi_true = 0.1

println("GEV data: n = $(nrow(df_gev)), y range = [$(round(minimum(y_gev), digits=2)), $(round(maximum(y_gev), digits=2))]")
```

    GEV data: n = 500, y range = [1.79, 17.11]

### Fit the GEV model

We specify one formula per distribution parameter. The GEV has three
parameters: location $\mu$, log-scale $\psi = \log\sigma$, and shape
$\xi$.

``` julia
m_gev = evgam(
    [@gam_formula(y ~ s(x, k=10, bs=:cr)),   # location μ(x)
     @gam_formula(y ~ s(x, k=8, bs=:cr)),    # log-scale ψ(x)
     @gam_formula(y ~ 1)],                    # shape ξ (constant)
    df_gev,
    GEVFamily()
)
```

    MultiParameterModel{GEVFamily}(GEVFamily(), [5.036640685469203, 1.0409439168652266, 1.8136737382830233, 1.7356293154236417, 0.7079870912017758, -0.6410794276850689, -1.8735541767682284, -1.8841523019469104, -1.0432461868196545, 0.07853204449777734, -0.2812086774098698, -0.09264862435288806, -0.03990371018725843, 0.048814831526631214, 0.13848355373114948, 0.23029835826939168, 0.32649864283215624, 0.3431386634483198, 0.1411597716364006], [[4.137955659365179, 4.38632220969695, 6.951839425683556, 3.4335366732803765, 3.228337159263553, 4.589641465387023, 3.0438859986426396, 6.381699999667373, 3.1402804707028076, 3.0183511091114013  …  4.908245311593589, 5.065586880507828, 3.5314457089531004, 5.932608393027157, 4.9952697927642395, 6.796723589610126, 3.101341725991767, 6.874765156039117, 6.6641444564074765, 5.392326884882849], [-0.02725317062653011, -0.013934646807622611, -0.40263897079638783, -0.07769497541533879, -0.19044537591539146, -0.2636565729988389, -0.13379218467854478, -0.4930508947521607, -0.18134019355330896, -0.15262610449207165  …  0.011628508785062044, -0.28449556600685905, -0.068759873338572, -0.3249858034680847, -0.281408342048468, -0.46402467982517176, -0.12030353088037075, -0.39129918128324964, -0.4745368347963723, -0.5538704096723862], [0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006  …  0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006, 0.1411597716364006]], [[1.0 -0.12291896228955582 … 0.8438166805691195 0.0447707000334424; 1.0 -0.10509884017847809 … 0.7007569184507677 0.2480101534139755; … ; 1.0 0.2449317510546628 … -0.09659396556034537 -0.029036447114687874; 1.0 0.21491894137543813 … -0.2890406423684066 -0.08668591615307703], [1.0 -0.14143801811295187 … 0.700577309129319 0.24661821243013488; 1.0 -0.12337120529887255 … 0.535191584651037 0.42519514492899374; … ; 1.0 0.7263085982326665 … -0.13936080706666798 -0.05448994539787703; 1.0 -0.01898755923276018 … -0.34094434872632795 -0.13153300201052215], [1.0; 1.0; … ; 1.0; 1.0;;]], Vector{ConstructedSmooth}[[ConstructedSmooth(s(x,bs=cr), k=9, rank=8)], [ConstructedSmooth(s(x,bs=cr), k=7, rank=6)], []], [-6.303016525558759, 5.891463215493163], [0.9999999999999994, 0.7976489181297176, 0.9060831212675853, 0.8657301953417887, 0.831392730571631, 0.8359491665516708, 0.8351024797505057, 0.8540170511008168, 0.8378610495076205, 0.7794316042352466, 1.0, 0.09536774811657191, 0.013208445116270527, -0.0018407267911483157, 0.05360983301053432, 0.1576885059012074, 0.4277186306114169, 0.2588189557196346, 0.9999999999999998], [0.0015511399448254598 -0.00027324306292036233 … 0.00022921732191574258 -0.000447766386601689; -0.00027324306292036233 0.009896579255207878 … -0.0013292270718602305 -0.00012818198843896934; … ; 0.00022921732191574258 -0.0013292270718602305 … 0.005537244201920306 1.3976715887157893e-5; -0.000447766386601689 -0.00012818198843896934 … 1.3976715887157893e-5 0.001255401854944482], [0.0015511399448254598 -0.00027324306292036233 … 0.00022921732191574258 -0.000447766386601689; -0.00027324306292036233 0.009896579255207878 … -0.0013292270718602305 -0.00012818198843896934; … ; 0.00022921732191574258 -0.0013292270718602305 … 0.005537244201920306 1.3976715887157893e-5; -0.000447766386601689 -0.00012818198843896934 … 1.3976715887157893e-5 0.001255401854944482], 689.26865779351, 724.9246182033163, [3.61243275935754, 3.67482461810157, 9.62757006978756, 3.69647266955379, 2.71978270287675, 4.41916781818956, 5.22956108002974, 6.91351275491862, 2.94614214388403, 3.05242715532112  …  6.6881523752481, 6.13741568442759, 2.67682063492149, 5.60200152317484, 5.3446107515215, 7.51096177983951, 5.13960141854121, 7.06772185929518, 6.75388645855366, 5.87620827444264], 500, true, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3], [0, 10, 18, 19])

### Examine parameter estimates

``` julia
println("Number of parameters: ", nparams(m_gev))
println("Converged: ", m_gev.converged)
println("Negative log-likelihood: ", round(m_gev.nll, digits=2))
println("REML score: ", round(m_gev.reml, digits=2))
```

    Number of parameters: 3
    Converged: true
    Negative log-likelihood: 689.27
    REML score: 724.92

Location parameter coefficients and fitted values:

``` julia
mu_coefs = param_coef(m_gev, 1)
mu_hat = param_eta(m_gev, 1)
println("Location coefficients (first 5): ", round.(mu_coefs[1:min(5, length(mu_coefs))], digits=3))
println("Location fitted range: [$(round(minimum(mu_hat), digits=2)), $(round(maximum(mu_hat), digits=2))]")
println("Location true range: [$(round(minimum(mu_true), digits=2)), $(round(maximum(mu_true), digits=2))]")
```

    Location coefficients (first 5): [5.037, 1.041, 1.814, 1.736, 0.708]
    Location fitted range: [3.02, 7.01]
    Location true range: [3.0, 7.0]

Log-scale parameter:

``` julia
psi_coefs = param_coef(m_gev, 2)
psi_hat = param_eta(m_gev, 2)
println("Log-scale coefficients (first 5): ", round.(psi_coefs[1:min(5, length(psi_coefs))], digits=3))
println("Log-scale fitted range: [$(round(minimum(psi_hat), digits=2)), $(round(maximum(psi_hat), digits=2))]")
println("Log-scale true range: [$(round(minimum(logsigma_true), digits=2)), $(round(maximum(logsigma_true), digits=2))]")
```

    Log-scale coefficients (first 5): [-0.281, -0.093, -0.04, 0.049, 0.138]
    Log-scale fitted range: [-0.57, 0.02]
    Log-scale true range: [-0.5, -0.0]

Shape parameter (constant):

``` julia
xi_coefs = param_coef(m_gev, 3)
println("Shape coefficient: ", round(xi_coefs[1], digits=4))
println("True shape: ", xi_true)
```

    Shape coefficient: 0.1412
    True shape: 0.1

### Compare fitted vs true functions

``` julia
ord = sortperm(x)
x_sorted = x[ord]

cor_mu = cor(mu_hat, mu_true)
cor_psi = cor(psi_hat, logsigma_true)
println("Correlation (fitted vs true location): ", round(cor_mu, digits=4))
println("Correlation (fitted vs true log-scale): ", round(cor_psi, digits=4))
println("RMSE (location): ", round(sqrt(mean((mu_hat .- mu_true).^2)), digits=3))
println("RMSE (log-scale): ", round(sqrt(mean((psi_hat .- logsigma_true).^2)), digits=3))
```

    Correlation (fitted vs true location): 0.9958
    Correlation (fitted vs true log-scale): 1.0
    RMSE (location): 0.13
    RMSE (log-scale): 0.039

## GPD model

### Simulate GPD data

For threshold exceedances, we simulate GPD data with covariate-dependent
scale:

- Log-scale: $\log\sigma(x) = 0.5\sin(2\pi x)$
- Shape: $\xi = 0.15$ (constant)

``` julia
df_gpd = CSV.read("data_gpd.csv", DataFrame)
n_gpd = nrow(df_gpd)
x_gpd = df_gpd.x
y_gpd = df_gpd.y

logsigma_gpd_true = 0.5 .* sin.(2π .* x_gpd)
sigma_gpd_true = exp.(logsigma_gpd_true)
xi_gpd_true = 0.15

println("GPD data: n = $(nrow(df_gpd)), y range = [$(round(minimum(y_gpd), digits=2)), $(round(maximum(y_gpd), digits=2))]")
```

    GPD data: n = 500, y range = [0.0, 12.35]

### Fit the GPD model

The GPD has two parameters: log-scale $\psi = \log\sigma$ and shape
$\xi$.

``` julia
m_gpd = evgam(
    [@gam_formula(y ~ s(x, k=10, bs=:cr)),   # log-scale ψ(x)
     @gam_formula(y ~ 1)],                    # shape ξ (constant)
    df_gpd,
    GPDFamily()
)
```

    MultiParameterModel{GPDFamily}(GPDFamily(0.0), [0.5450726792010139, 0.013086632597933507, -0.08368522714797137, -0.1823945581257055, -0.23534058424211302, -0.2083219220217687, -0.11654555430015713, 0.013229783991115919, 0.1670740675863881, 0.3333888200891796, 0.06247172999298983], [[0.3577938124598328, 0.7794059674316478, 0.6544650805057429, 0.5075821471249369, 0.4149267772635219, 0.37431660744135425, 0.4785104516993892, 0.361874039974107, 0.5381704783255097, 0.7259922538666129  …  0.4227635193181231, 0.4839242814238836, 0.3861322376634336, 0.45114196601708617, 0.5093397607942516, 0.37336032510763856, 0.6137643511860356, 0.5968936831649606, 0.36166372527611934, 0.358534446395492], [0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983  …  0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983, 0.06247172999298983]], [[1.0 -0.12422550358987869 … -0.12041743435729219 -0.04021065844661173; 1.0 -0.13042251125561038 … 0.8517930129845935 0.012142638611563268; … ; 1.0 -0.12187198223679759 … -0.1213315197755695 -0.03914430141811147; 1.0 -0.10861130400117934 … -0.11520961231045121 -0.042781265610324345], [1.0; 1.0; … ; 1.0; 1.0;;]], Vector{ConstructedSmooth}[[ConstructedSmooth(s(x,bs=cr), k=9, rank=8)], []], [-1.9119430721778603], [0.9999999999999998, 0.1640482073026282, 0.2243790012486489, 0.2370412220135193, 0.31094974614464155, 0.2740062206522669, 0.3032657811723351, 0.23199555344311792, 0.47029091912344245, 0.38417629493421324, 0.9999999999999998], [0.00437409145598854 0.000307501813176535 … -0.00036918467627315975 -0.002251355396027057; 0.000307501813176535 0.002687977231016012 … -0.0017151679670224824 -0.0003399696135800265; … ; -0.00036918467627315975 -0.0017151679670224824 … 0.01928444754901849 0.00031686445315068536; -0.002251355396027057 -0.0003399696135800265 … 0.00031686445315068536 0.0023876892360751955], [0.00437409145598854 0.000307501813176535 … -0.00036918467627315975 -0.002251355396027057; 0.000307501813176535 0.002687977231016012 … -0.0017151679670224824 -0.0003399696135800265; … ; -0.00036918467627315975 -0.0017151679670224824 … 0.01928444754901849 0.00031686445315068536; -0.002251355396027057 -0.0003399696135800265 … 0.00031686445315068536 0.0023876892360751955], 803.7722049810695, 817.5679072960863, [0.298876567416734, 0.524216785894067, 1.04822076082263, 0.0130278358764623, 0.29082640401221, 1.19023184531972, 3.99792235758352, 1.25837987311792, 2.0204555276891, 6.86338808021889  …  0.770574164299274, 0.00416939151483722, 1.78205624511358, 0.311453594119774, 0.684430309093354, 0.599251905263417, 3.09788959300914, 3.16043401450683, 0.707524055537864, 0.205472579356741], 500, true, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [0, 10, 11])

### Examine GPD estimates

``` julia
println("Converged: ", m_gpd.converged)
println("Negative log-likelihood: ", round(m_gpd.nll, digits=2))

psi_gpd_hat = param_eta(m_gpd, 1)
xi_gpd_hat = param_coef(m_gpd, 2)

println("Log-scale fitted range: [$(round(minimum(psi_gpd_hat), digits=2)), $(round(maximum(psi_gpd_hat), digits=2))]")
println("Log-scale true range: [$(round(minimum(logsigma_gpd_true), digits=2)), $(round(maximum(logsigma_gpd_true), digits=2))]")
println("Shape estimate: ", round(xi_gpd_hat[1], digits=4))
println("True shape: ", xi_gpd_true)

cor_gpd = cor(psi_gpd_hat, logsigma_gpd_true)
println("Correlation (fitted vs true log-scale): ", round(cor_gpd, digits=4))
```

    Converged: true
    Negative log-likelihood: 803.77
    Log-scale fitted range: [0.36, 0.9]
    Log-scale true range: [-0.5, 0.5]
    Shape estimate: 0.0625
    True shape: 0.15
    Correlation (fitted vs true log-scale): -0.2538

## Model structure

The `MultiParameterModel` returned by `evgam` stores:

``` julia
println("Type: ", typeof(m_gev))
println("Total coefficients: ", length(m_gev.coefficients))
println("EDFs: ", round.(m_gev.edf, digits=2))
println("Log smoothing parameters: ", round.(m_gev.sp, digits=2))
println("Covariance matrix size: ", size(m_gev.Vp))
```

    Type: MultiParameterModel{GEVFamily}
    Total coefficients: 19
    EDFs: [1.0, 0.8, 0.91, 0.87, 0.83, 0.84, 0.84, 0.85, 0.84, 0.78, 1.0, 0.1, 0.01, -0.0, 0.05, 0.16, 0.43, 0.26, 1.0]
    Log smoothing parameters: [-6.3, 5.89]
    Covariance matrix size: (19, 19)

Standard errors via the posterior covariance:

``` julia
se_all = sqrt.(diag(m_gev.Vp))
println("Standard errors (first 5): ", round.(se_all[1:min(5, length(se_all))], digits=4))
```

    Standard errors (first 5): [0.0394, 0.0995, 0.0853, 0.0923, 0.0956]

## Summary

GAM.jl’s `evgam` provides:

| Feature                 | Description                                  |
|-------------------------|----------------------------------------------|
| `GEVFamily()`           | GEV distribution (3 parameters: μ, log σ, ξ) |
| `GPDFamily()`           | GPD distribution (2 parameters: log σ, ξ)    |
| Multi-formula interface | One formula per distribution parameter       |
| REML smoothing          | Automatic smoothing parameter estimation     |
| `param_coef(m, k)`      | Coefficients for parameter k                 |
| `param_eta(m, k)`       | Fitted linear predictor for parameter k      |
