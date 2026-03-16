# API Reference

## Main Interface

```@docs
gam
gam_control
GamControl
GamModel
GamFormula
GAM.@gam_formula
```

## Smooth Specification

```@docs
s
te
ti
SmoothSpec
ConstructedSmooth
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
BSplineBasis
RandomEffect
TensorProduct
TensorInteraction
```

## Extended Families

```@docs
GAM.ExtendedFamily
NegBinFamily
TweedieFamily
BetaFamily
```

## Diagnostics

```@docs
gam_check
k_check
concurvity
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
