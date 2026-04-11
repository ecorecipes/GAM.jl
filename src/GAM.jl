"""
    GAM

Generalized Additive Models for Julia. A port of R's mgcv package following
JuliaStats conventions (StatsModels.jl/GLM.jl patterns).

Provides `gam()` for fitting penalized regression spline GAMs with automatic
smoothness estimation via REML, GCV, or ML.

# Key types
- `GamModel`: fitted GAM (implements StatsBase interface)
- `SmoothSpec`: smooth term specification before data (e.g., `s(x, bs=:cr, k=10)`)
- `SmoothTerm`: smooth term in a formula (integrates with StatsModels.jl `@formula`)

# Key functions
- `gam`: fit a GAM to data
- `s`: specify a smooth term
- `te`, `ti`: specify tensor product smooth terms
- `smooth_construct`: construct a smooth basis from data
- `predict_matrix`: prediction matrix for a smooth at new data

# Example
```julia
using GAM, DataFrames

df = DataFrame(x = randn(200), y = sin.(randn(200)) .+ 0.1 .* randn(200))
m = gam(@formula(y ~ s(x)), df)
```
"""
module GAM

using Distributions
using GLM
using LinearAlgebra
using LinearAlgebra: BLAS
using OSQP
import PSIS
using Printf: @sprintf, @printf
using Reexport
using SparseArrays
using SpecialFunctions
using Statistics
using StatsAPI
using StatsBase
using StatsBase: CoefTable, StatisticalModel, RegressionModel
using Tables

@reexport using StatsModels
@reexport using Distributions: Normal, Binomial, Poisson, Gamma, InverseGaussian
@reexport using GLM: LogitLink, LogLink, IdentityLink, InverseLink, ProbitLink,
    CauchitLink, CloglogLink, SqrtLink

import Base: show, size
import StatsAPI: coef, coeftable, coefnames, confint, deviance, nulldeviance,
    aic, aicc, bic, dof, dof_residual, loglikelihood, nobs, stderror, vcov, residuals,
    predict, fitted, fit, response, r2, adjr2
import StatsModels: apply_schema, modelcols, schema, hasintercept

export
    # Main interface
    gam,
    gam_check,
    gam_control,
    GamControl,
    GamModel,
    GamFormula,
    @formula,
    @formulak,

    # Smooth specification
    s,
    te,
    ti,
    t2,
    cr,
    tp,
    ts,
    cs,
    cc,
    ps,
    cps,
    SmoothSpec,
    SmoothTerm,

    # Smooth types
    ThinPlateSpline,
    ThinPlateShrink,
    CubicSpline,
    CubicShrink,
    CyclicCubic,
    PSpline,
    CyclicPSpline,
    BSplineBasis,
    RandomEffect,
    AdaptiveSmooth,
    TensorProduct,
    TensorInteraction,
    T2TensorProduct,
    GPSmooth,
    LoessSmooth,
    FractionalPolynomial,
    DuchonSpline,
    FactorSmooth,
    SoapFilm,
    MarkovRandomField,
    SphericalSpline,
    ConstrainedFactorSmooth,
    SPDESmooth,
    ShapeConstrainedBSpline,
    ShapeConstrainedAdaptive,

    # Smooth construction
    smooth_construct,
    predict_matrix,
    penalty_matrix,
    null_space_dim,
    ConstructedSmooth,

    # Diagnostics
    edf,
    k_check,
    concurvity,
    anova_gam,
    AnovaGamResult,

    # Visualization
    gamplot,
    gamcontour,
    vis_gam,
    VisGamData,

    # Smoothing methods
    REML,
    ML,
    GCV,
    UBRE,

    # Extended families
    ExtendedFamily,
    NegBinFamily,
    QuasiPoissonFamily,
    QuasiBinomialFamily,
    TweedieFamily,
    BetaFamily,

    # BAM (Big Additive Models)
    bam,
    bam_control,
    BamControl,
    DiscretizedData,
    discretize_covariates,

    # GINLA (Integrated Nested Laplace Approximation)
    ginla,
    GinlaResult,

    # Multi-parameter models (evgam)
    MultiParameterFamily,
    MultiParameterModel,
    GEVFamily,
    GPDFamily,
    evgam,
    evgam_control,
    nparams,
    param_names,
    param_coef,
    param_eta,
    nll_total,
    nll_derivs!,
    nll_obs,

    # GAMLSS families and interface
    DistFamily,
    GammaLocationScale,
    BetaRegression,
    NegativeBinomialLocationScale,
    InverseGaussianLocationScale,
    GaussianLS,
    GammaLS,
    BetaLS,
    NegBinLS,
    gamlss,
    GamlssControl,
    gamlss_control,
    gamlss_rs!,
    gamlss_cg!,
    mp_laml,

    # EGPD families
    EGPD1Family,
    EGPD2Family,
    EGPD3Family,
    EGPD4Family,

    # Quantile GAM (qgam)
    ELFFamily,
    ELFLSSFamily,
    qgam,
    mqgam,
    qdo,
    pinball_loss,
    deviance_explained,
    cqcheck,
    CQCheckResult,
    check_qgam,
    QGamCheck,
    quantile_residuals,

    # Shape-constrained smooths (SCAM)
    MonoIncBasis,
    MonoDecBasis,
    ConcaveBasis,
    ConvexBasis,
    MonoIncConvexBasis,
    MonoIncConcaveBasis,
    MonoDecConvexBasis,
    MonoDecConcaveBasis,
    scam,
    scam_control,
    ScamControl,
    softplus,
    has_shape_constraints,

    # Gratia-like diagnostics & visualization
    smooth_estimates,
    SmoothEstimates,
    partial_residuals,
    data_slice,
    derivatives,
    DerivativeEstimates,
    posterior_samples,
    fitted_samples,
    smooth_samples,
    predicted_samples,
    appraise,
    AppraiseData,
    rootogram,
    RootogramData,
    model_edf,
    overview,
    OverviewTable,

    # GAMM (Generalized Additive Mixed Models)
    gamm,
    GammModel,
    GammFormula,
    @gamm_formula,
    RandomEffectSpec,
    ConstructedRandomEffect,
    construct_random_effect,
    predict_re_matrix,
    ranef,
    VarCorr,
    VarCorrResult,
    re,

    # Bayesian GAM support (Turing.jl extension)
    PriorSpec,
    get_prior,
    default_priors,
    BayesGamModel,
    PSISKDiagnostic,
    LOOResult,
    WAICResult,
    smooth2random,
    SmoothMixedModel,
    gam_matrices,
    gam_smooth,
    smooth_prior,
    smooth_predictive,
    s2r_predict,
    pointwise_loglikelihood,
    psis_loo,
    pareto_k_diagnostic,
    loo,
    waic

# Type aliases matching GLM.jl convention
const FP = AbstractFloat
const FPVector{T<:FP} = AbstractArray{T, 1}

include("types.jl")
include("smoothspec.jl")
include("knots.jl")
include("basis_common.jl")
include("basis_tprs.jl")
include("basis_cr.jl")
include("basis_ps.jl")
include("basis_adaptive.jl")
include("basis_re.jl")
include("basis_tensor.jl")
include("basis_gp.jl")
include("basis_loess.jl")
include("basis_fp.jl")
include("basis_extra.jl")
include("basis_sphere.jl")
include("basis_spde.jl")
include("basis_sz.jl")
include("basis_scam.jl")
include("basis_scasm.jl")
include("penalty.jl")
include("smooth2random.jl")
include("formula.jl")
include("pirls.jl")
include("extended_families.jl")
include("pirls_extended.jl")
include("reml.jl")
include("general_fit.jl")
include("outer.jl")
include("multiparameter.jl")
include("evfamilies.jl")
include("egpd_families.jl")
include("mpfit.jl")
include("priors.jl")
include("bayes_types.jl")
include("gamlss.jl")
include("gamlss_solvers.jl")
include("gamfit.jl")
include("gamm.jl")
include("bam.jl")
include("ginla.jl")
include("qgam.jl")
include("scam.jl")
include("scasm.jl")
include("validation.jl")
include("statsbase.jl")
include("diagnostics.jl")
include("gratia.jl")
include("plots.jl")
include("show.jl")

# Register basis-alias functions so @formula recognizes them as smooth terms
_register_smooth_aliases()

end # module GAM
