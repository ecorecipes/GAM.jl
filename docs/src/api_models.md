# [API Reference — Model Types](@id api-models)

Specialised fitters and the model objects they return.

!!! tip "Other API pages"
    [Core](@ref api-reference) · [Model Types](@ref api-models) · [Diagnostics & Plotting](@ref api-diagnostics)

## BAM (Large Data)

```@docs
bam
bam_control
BamControl
```

```@docs
discretize_covariates
DiscretizedData
```

## GAMLSS (Distributional Regression)

```@docs
gamlss
GamlssControl
GAM.MPFitControl
GAM.MultiParameterModel
GAM.mp_control
GAM.GaussianLS
GAM.GammaLocationScale
GAM.BetaRegression
GAM.NegativeBinomialLocationScale
GAM.InverseGaussianLocationScale
GAM.DistFamily
```

```@docs
gamlss_control
GammaLS
BetaLS
NegBinLS
nparams
param_names
param_coef
param_eta
nll_total
```

## Nested Effects

```@docs
gam_nl
s_nest
trans_linear
trans_nexpsm
trans_mgks
GAM.TransLinear
GAM.TransExpSmooth
GAM.TransMGKS
inner_coef
NestedGamModel
```

```@docs
NestedControl
nested_control
GAM.NestedBasis
GAM.NestedTransform
```

## SCAM (Shape Constraints)

```@docs
scam
scam_control
ScamControl
```

## QGAM (Quantile Regression)

```@docs
qgam
mqgam
qdo
GAM.ELFFamily
GAM.ELFLSSFamily
```

```@docs
pinball_loss
quantile_residuals
cqcheck
CQCheckResult
check_qgam
QGamCheck
```

## evgam (Extreme Values)

```@docs
evgam
GAM.GEVFamily
GAM.GPDFamily
GAM.MultiParameterFamily
```

```@docs
evgam_control
EGPD1Family
EGPD2Family
EGPD3Family
EGPD4Family
```

## GAMM (Mixed Models)

```@docs
gamm
GammModel
ranef
VarCorr
re
GAM.@gamm_formula
```

```@docs
RandomEffectSpec
ConstructedRandomEffect
VarCorrResult
```

## GINLA

```@docs
ginla
GinlaResult
```

## Bayesian Inference

```@docs
BayesGamModel
PSISKDiagnostic
LOOResult
WAICResult
smooth2random
PriorSpec
GAM.smooth_prior
pointwise_loglikelihood
psis_loo
pareto_k_diagnostic
loo
waic
```

```@docs
get_prior
default_priors
gam_smooth
gam_matrices
GAM.smooth_predictive
SmoothMixedModel
```

