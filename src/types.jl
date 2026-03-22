# Core type hierarchy for GAM.jl

# ============================================================================
# Smoothing method selectors
# ============================================================================

"""Smoothing parameter estimation methods."""
abstract type SmoothingMethod end

"""Restricted Maximum Likelihood smoothing parameter estimation."""
struct REML <: SmoothingMethod end

"""Maximum Likelihood smoothing parameter estimation."""
struct ML <: SmoothingMethod end

"""Generalized Cross Validation smoothing parameter estimation."""
struct GCV <: SmoothingMethod end

"""Un-Biased Risk Estimator (for known scale parameter)."""
struct UBRE <: SmoothingMethod end

# ============================================================================
# Smooth basis types — dispatch target for smooth_construct / predict_matrix
# ============================================================================

"""
    AbstractBasisType

Abstract supertype for all smooth basis types. Subtypes identify which
basis construction algorithm to use via dispatch on `smooth_construct`.
"""
abstract type AbstractBasisType end

"""Thin plate regression spline basis (mgcv `bs="tp"`)."""
struct ThinPlateSpline <: AbstractBasisType end

"""Thin plate regression spline with shrinkage penalty on null space (mgcv `bs="ts"`)."""
struct ThinPlateShrink <: AbstractBasisType end

"""Natural cubic regression spline (mgcv `bs="cr"`)."""
struct CubicSpline <: AbstractBasisType end

"""Cubic regression spline with shrinkage (mgcv `bs="cs"`)."""
struct CubicShrink <: AbstractBasisType end

"""Cyclic cubic regression spline (mgcv `bs="cc"`)."""
struct CyclicCubic <: AbstractBasisType end

"""P-spline: B-spline basis with difference penalty (mgcv `bs="ps"`)."""
struct PSpline <: AbstractBasisType end

"""B-spline basis with derivative penalty (mgcv `bs="bs"`)."""
struct BSplineBasis <: AbstractBasisType end

"""Random effect smooth — identity penalty (mgcv `bs="re"`)."""
struct RandomEffect <: AbstractBasisType end

"""Tensor product smooth basis (mgcv `te()`)."""
struct TensorProduct <: AbstractBasisType end

"""Tensor product interaction basis (mgcv `ti()`)."""
struct TensorInteraction <: AbstractBasisType end

# Shape-constrained smooth types (scam package)
"""Abstract base for shape-constrained spline basis types."""
abstract type AbstractConstrainedBasis <: AbstractBasisType end

"""Monotone increasing B-spline (scam `bs="mpi"`)."""
struct MonoIncBasis <: AbstractConstrainedBasis end

"""Monotone decreasing B-spline (scam `bs="mpd"`)."""
struct MonoDecBasis <: AbstractConstrainedBasis end

"""Concave B-spline (scam `bs="cv"`)."""
struct ConcaveBasis <: AbstractConstrainedBasis end

"""Convex B-spline (scam `bs="cx"`)."""
struct ConvexBasis <: AbstractConstrainedBasis end

"""Monotone increasing + convex B-spline (scam `bs="micx"`)."""
struct MonoIncConvexBasis <: AbstractConstrainedBasis end

"""Monotone increasing + concave B-spline (scam `bs="micv"`)."""
struct MonoIncConcaveBasis <: AbstractConstrainedBasis end

"""Monotone decreasing + convex B-spline (scam `bs="mdcx"`)."""
struct MonoDecConvexBasis <: AbstractConstrainedBasis end

"""Monotone decreasing + concave B-spline (scam `bs="mdcv"`)."""
struct MonoDecConcaveBasis <: AbstractConstrainedBasis end

"""Map from symbol to basis type."""
const BASIS_TYPES = Dict{Symbol, AbstractBasisType}(
    :tp => ThinPlateSpline(),
    :ts => ThinPlateShrink(),
    :cr => CubicSpline(),
    :cs => CubicShrink(),
    :cc => CyclicCubic(),
    :ps => PSpline(),
    :bs => BSplineBasis(),
    :re => RandomEffect(),
    :mpi => MonoIncBasis(),
    :mpd => MonoDecBasis(),
    :cv => ConcaveBasis(),
    :cx => ConvexBasis(),
    :micx => MonoIncConvexBasis(),
    :micv => MonoIncConcaveBasis(),
    :mdcx => MonoDecConvexBasis(),
    :mdcv => MonoDecConcaveBasis(),
)

function resolve_basis_type(bs::Symbol)
    haskey(BASIS_TYPES, bs) || throw(ArgumentError("Unknown basis type: $bs. " *
        "Available types: $(join(sort(collect(keys(BASIS_TYPES))), ", "))"))
    return BASIS_TYPES[bs]
end

# ============================================================================
# Smooth specification — parsed from formula, before seeing data
# ============================================================================

"""
    SmoothSpec{B<:AbstractBasisType}

Specification of a smooth term parsed from a formula, before data is available.
Contains all user-specified options. Type parameter `B` determines which basis
construction algorithm is dispatched.

# Fields
- `term_vars`: variable names for this smooth (e.g., `[:x]` or `[:x, :y]`)
- `basis`: basis type instance
- `k`: basis dimension (number of basis functions before constraint absorption)
- `by`: optional `by` variable for varying-coefficient models
- `id`: optional identifier for linking smooths
- `sp`: optional fixed smoothing parameter (nothing = estimate)
- `fx`: if true, do not penalize (fixed df smooth)
- `m`: penalty order (meaning depends on basis type)
- `label`: human-readable label for the smooth
"""
struct SmoothSpec{B<:AbstractBasisType}
    term_vars::Vector{Symbol}
    basis::B
    k::Int
    by::Union{Symbol, Nothing}
    id::Union{Symbol, Nothing}
    sp::Union{Float64, Nothing}
    fx::Bool
    m::Union{Int, Nothing}
    label::String
end

# ============================================================================
# Constructed smooth — after basis construction from data
# ============================================================================

"""
    ConstructedSmooth{B<:AbstractBasisType}

A smooth term after basis construction. Contains the model matrix columns,
penalty matrix/matrices, and all metadata needed for fitting and prediction.

# Fields
- `spec`: the original specification
- `X`: model matrix for this smooth (n × k_eff after constraint absorption)
- `S`: list of penalty matrices (each k_eff × k_eff)
- `knots`: knot locations used
- `null_dim`: dimension of the penalty null space
- `rank`: penalty rank
- `constraint`: constraint matrix (if any) used for identifiability
- `qrc`: QR factorization used for constraint absorption
- `first_para`: index of first parameter in full model matrix
- `last_para`: index of last parameter in full model matrix
- `Sigma`: constraint matrix for shape-constrained smooths (scam)
- `cmX`: column means for centering (scam)
- `p_ident`: boolean mask — which coefficients must be positive (scam)
"""
mutable struct ConstructedSmooth{B<:AbstractBasisType}
    spec::SmoothSpec{B}
    X::Matrix{Float64}
    S::Vector{Matrix{Float64}}
    knots::Vector{Float64}
    null_dim::Int
    rank::Int
    constraint::Union{Matrix{Float64}, Nothing}
    qrc::Union{LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}, Nothing}
    first_para::Int
    last_para::Int
    # Shape constraint metadata (scam) — nothing for unconstrained smooths
    Sigma::Union{Matrix{Float64}, Nothing}
    cmX::Union{Vector{Float64}, Nothing}
    p_ident::Union{BitVector, Nothing}
end

# ============================================================================
# GAM control parameters
# ============================================================================

"""
    GamControl

Control parameters for GAM fitting.

# Fields
- `epsilon`: convergence tolerance for P-IRLS inner iteration
- `maxit`: maximum P-IRLS iterations
- `outer_maxit`: maximum outer iterations for smoothing parameter estimation
- `trace`: print iteration progress
- `gamma`: inflation factor for GCV/UBRE degrees of freedom (>1 = more smoothing)
- `scale_est`: scale parameter estimate method (:fletcher, :pearson, :deviance)
- `edge_correct`: apply edge correction to smoothing parameters
"""
struct GamControl
    epsilon::Float64
    maxit::Int
    outer_maxit::Int
    trace::Bool
    gamma::Float64
    scale_est::Symbol
    edge_correct::Bool
    sp_optimizer::Symbol
end

"""
    gam_control(; epsilon=1e-7, maxit=200, outer_maxit=200, trace=false,
                  gamma=1.0, scale_est=:fletcher, edge_correct=true,
                  sp_optimizer=:efs)

Construct a [`GamControl`](@ref) with the given parameters.

# Smoothing parameter optimizers
- `:efs` (default) — Extended Fellner-Schall (Wood & Fasiolo 2017). Fast,
  monotonically convergent, one PIRLS call per outer iteration.
- `:newton` — Newton's method with autodiff Hessian. Uses ForwardDiff to
  differentiate the REML score w.r.t. log(sp). More expensive but may
  converge in fewer iterations for difficult problems.
"""
function gam_control(;
    epsilon::Real = 1e-7,
    maxit::Int = 200,
    outer_maxit::Int = 200,
    trace::Bool = false,
    gamma::Real = 1.0,
    scale_est::Symbol = :fletcher,
    edge_correct::Bool = true,
    sp_optimizer::Symbol = :efs,
)
    sp_optimizer in (:efs, :newton) ||
        throw(ArgumentError("sp_optimizer must be :efs or :newton, got :$sp_optimizer"))
    return GamControl(Float64(epsilon), maxit, outer_maxit, trace,
        Float64(gamma), scale_est, edge_correct, sp_optimizer)
end

# ============================================================================
# Penalty block structure — block-diagonal penalty for multi-smooth models
# ============================================================================

"""
    PenaltyBlock

One block of the block-diagonal penalty structure. Represents all penalties
for a single smooth term.

# Fields
- `S`: list of penalty matrices for this block
- `rS`: square root penalty matrices (rS[i] * rS[i]' = λ[i] * S[i])
- `rank`: penalty rank
- `start`: first parameter index in the full coefficient vector
- `stop`: last parameter index
- `repara`: should reparameterization be applied
"""
struct PenaltyBlock
    S::Vector{Matrix{Float64}}
    rS::Vector{Matrix{Float64}}
    rank::Int
    start::Int
    stop::Int
    repara::Bool
end

"""
    PenaltySetup

Complete block-diagonal penalty structure for a GAM.
Equivalent to mgcv's `Sl.setup` output.

# Fields
- `blocks`: individual penalty blocks
- `sp`: current smoothing parameters (log scale)
- `E`: square root of total penalty for rank detection
"""
mutable struct PenaltySetup
    blocks::Vector{PenaltyBlock}
    sp::Vector{Float64}
    E::Matrix{Float64}
end

# ============================================================================
# GAM model type
# ============================================================================

"""
    GamModel{D<:UnivariateDistribution, L<:GLM.Link}

A fitted generalized additive model. Implements the StatsBase interface
(`coef`, `vcov`, `predict`, `deviance`, etc.).

# Fields
- `formula`: the formula used to fit the model
- `y`: response vector
- `X`: full model matrix (parametric + smooth columns)
- `coefficients`: fitted coefficient vector
- `fitted_values`: fitted values on response scale (μ)
- `linear_predictor`: fitted values on link scale (η)
- `weights`: prior weights
- `family`: distribution family
- `link`: link function
- `smooths`: list of constructed smooth terms
- `penalty`: penalty structure
- `sp`: estimated log smoothing parameters
- `edf`: effective degrees of freedom per smooth
- `edf_total`: total effective degrees of freedom (parametric + smooth)
- `scale`: estimated or fixed scale parameter
- `deviance_val`: model deviance
- `null_deviance`: null model deviance
- `reml`: REML/ML/GCV score at convergence
- `method`: smoothing method used (:REML, :ML, :GCV, :UBRE)
- `Vp`: Bayesian posterior covariance of parameters
- `Ve`: frequentist covariance of parameters
- `hat_matrix_diag`: diagonal of the hat/influence matrix
- `R`: R factor from QR of augmented model matrix
- `converged`: did the iteration converge
- `iterations`: number of outer iterations
- `n_smooth`: number of smooth terms
- `n_parametric`: number of parametric coefficients (including intercept)
- `control`: fitting control parameters
"""
mutable struct GamModel{D, L<:GLM.Link}
    formula::Union{FormulaTerm, Nothing}
    y::Vector{Float64}
    X::Matrix{Float64}
    coefficients::Vector{Float64}
    fitted_values::Vector{Float64}
    linear_predictor::Vector{Float64}
    weights::Vector{Float64}
    family::D
    link::L
    smooths::Vector{ConstructedSmooth}
    penalty::PenaltySetup
    sp::Vector{Float64}
    edf::Vector{Float64}
    edf_total::Float64
    scale::Float64
    deviance_val::Float64
    null_deviance::Float64
    reml::Float64
    method::Symbol
    Vp::Matrix{Float64}
    Ve::Matrix{Float64}
    hat_matrix_diag::Vector{Float64}
    R::Matrix{Float64}
    converged::Bool
    iterations::Int
    n_smooth::Int
    n_parametric::Int
    control::GamControl
    data::Any  # original data (for gratia-like smooth evaluation grids)
end
