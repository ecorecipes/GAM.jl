# [API Reference](@id api-reference)

## Main Interface

```@docs
gam
gam_control
GamControl
GamModel
```

## Formula Macros

```@docs
GAM.@formula
GamFormula
GammFormula
```

## Smooth Specification

```@docs
s
te
ti
t2
SmoothSpec
GAM.ConstructedSmooth
smooth_construct
predict_matrix
```

## Basis Types

```@docs
ThinPlateSpline
ThinPlateShrink
CubicSpline
CubicShrink
CyclicCubic
PSpline
CyclicPSpline
BSplineBasis
GPSmooth
LoessSmooth
FractionalPolynomial
DuchonSpline
AdaptiveSmooth
SphericalSpline
SPDESmooth
ConstrainedFactorSmooth
MarkovRandomField
SoapFilm
FactorSmooth
RandomEffect
TensorProduct
TensorInteraction
T2TensorProduct
```

## Extended Families

```@docs
GAM.ExtendedFamily
NegBinFamily
QuasiPoissonFamily
QuasiBinomialFamily
TweedieFamily
BetaFamily
```

## BAM (Large Data)

```@docs
bam
bam_control
BamControl
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

## evgam (Extreme Values)

```@docs
evgam
GAM.GEVFamily
GAM.GPDFamily
GAM.MultiParameterFamily
```

## GAMM (Mixed Models)

```@docs
gamm
GammModel
ranef
VarCorr
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

## Diagnostics

```@docs
gam_check
k_check
concurvity
anova_gam
AnovaGamResult
```

## Gratia-Style Diagnostics

```@docs
smooth_estimates
GAM.SmoothEstimates
derivatives
GAM.DerivativeEstimates
partial_residuals
posterior_samples
fitted_samples
appraise
GAM.AppraiseData
rootogram
GAM.RootogramData
data_slice
```

## Visualization

```@docs
gamplot
gamcontour
```

## Smoothing Methods

```@docs
REML
ML
GCV
UBRE
```

## Side Constraints

Side constraints (`gam.side`) are applied internally to enforce identifiability
when smooth terms overlap with parametric terms or with each other. They are not
typically called directly by users.
