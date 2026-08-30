# [API Reference — Core](@id api-reference)

Fitting entry points, formula macros, smooth specification, basis types,
families and smoothing-method selectors.

!!! tip "Other API pages"
    [Core](@ref api-reference) · [Model Types](@ref api-models) · [Diagnostics & Plotting](@ref api-diagnostics)

## Main Interface

```@docs
gam
gam_control
GamControl
GamModel
na_omit_rows
```

### Model Accessors

```@docs
edf
deviance_explained
lpmatrix
overview
OverviewTable
model_edf
sp_criterion
model_matrix
has_model_matrix
GAM.drop_model_matrix!
```

### Smoothing-Parameter Uncertainty

The Wood, Pya & Säfken (2016) correction: `edf2` degrees of freedom and the
corrected covariance `Vc`, which is what `unconditional = true` uses in
`predict`, `smooth_estimates`, `derivatives` and `posterior_samples`.

```@docs
edf2
ref_df
has_vc
vcov_corrected
conditional_aic
conditional_dof
GAM.force_vc!
GAM.edf1_from_F
GAM.corrected_covariance
```

### Formula Macros (extra)

```@docs
@formulak
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

### Basis-Alias Constructors

Shorthand constructors equivalent to `s(x; bs=...)`.

```@docs
tp
ts
cr
cs
cc
ps
cps
```

### Smooth Term Internals

```@docs
SmoothTerm
penalty_matrix
null_space_dim
has_shape_constraints
has_nested_effects
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

### Shape-Constrained Basis Types

```@docs
MonoIncBasis
MonoDecBasis
ConcaveBasis
ConvexBasis
MonoIncConvexBasis
MonoIncConcaveBasis
MonoDecConvexBasis
MonoDecConcaveBasis
ShapeConstrainedBSpline
ShapeConstrainedAdaptive
```

## Extended Families

```@docs
GAM.ExtendedFamily
NegBinFamily
QuasiPoissonFamily
QuasiBinomialFamily
TweedieFamily
BetaFamily
ScatFamily
```

## Smoothing Methods

```@docs
REML
ML
GCV
UBRE
```

### Neighbourhood cross validation

`method = :NCV` selects smoothing parameters by leaving out a *neighbourhood*
of each observation rather than the observation alone, which is what you want
when the data are correlated and GCV/REML under-smooth. The default
neighbourhood is leave-one-out; `nei=` supplies another.

```@docs
NeighbourhoodStructure
loo_neighbourhoods
interval_neighbourhoods
validate_neighbourhoods
ncv_score
```

