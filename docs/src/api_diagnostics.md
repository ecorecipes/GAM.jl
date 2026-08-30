# [API Reference — Diagnostics & Plotting](@id api-diagnostics)

Model checking, influence measures, gratia-style tabular results and
visualisation helpers.

!!! tip "Other API pages"
    [Core](@ref api-reference) · [Model Types](@ref api-models) · [Diagnostics & Plotting](@ref api-diagnostics)

## Diagnostics

```@docs
gam_check
k_check
concurvity
anova_gam
AnovaGamResult
```

## Influence Measures

```@docs
GAM.leverage(::GamModel)
GAM.cooksdistance(::GamModel)
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
PartialResiduals
```

```@docs
smooth_samples
predicted_samples
```

## Visualization

```@docs
gamplot
gamcontour
```

```@docs
vis_gam
VisGamData
```

## Side Constraints

Side constraints (`gam.side`) are applied internally to enforce identifiability
when smooth terms overlap with parametric terms or with each other. They are not
typically called directly by users.

## Internals

Documented internal helpers that are referenced from public docstrings.
They are not part of the public API and may change without notice.

```@docs
GAM._smooth2random_tensor
GAM._smooth2random_disjoint
GAM.s2r_predict
```
