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

"""Cyclic P-spline: periodic B-spline basis with cyclic difference penalty."""
struct CyclicPSpline <: AbstractBasisType end

"""B-spline basis with derivative penalty (mgcv `bs="bs"`)."""
struct BSplineBasis <: AbstractBasisType end

"""Random effect smooth — identity penalty (mgcv `bs="re"`)."""
struct RandomEffect <: AbstractBasisType end

"""Tensor product smooth basis (mgcv `te()`)."""
struct TensorProduct <: AbstractBasisType end

"""Tensor product interaction basis (mgcv `ti()`)."""
struct TensorInteraction <: AbstractBasisType end

"""Alternative tensor product smooth basis (mgcv `t2()`)."""
struct T2TensorProduct <: AbstractBasisType end

# mgcv::scasm basis types
"""Shape-constrained B-spline smooth (mgcv `bs="sc"`)."""
struct ShapeConstrainedBSpline <: AbstractBasisType end

"""Shape-constrained adaptive smooth (mgcv `bs="scad"`)."""
struct ShapeConstrainedAdaptive <: AbstractBasisType end

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
    :cps => CyclicPSpline(),
    :bs => BSplineBasis(),
    :re => RandomEffect(),
    :sc => ShapeConstrainedBSpline(),
    :scad => ShapeConstrainedAdaptive(),
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
- `sp`: optional fixed smoothing parameter (nothing = estimate). A scalar fixes
  every penalty of the smooth at that value; a vector fixes them individually
  and must have one entry per penalty, which lets multi-penalty smooths
  (`bs=:ad`, `t2`, `bs=:fs`) round-trip mgcv's per-penalty `sp` vector. The
  length is checked against the constructed penalty count in `setup_penalties`,
  since for several bases that count is not known until after construction.
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
    sp::Union{Float64, Vector{Float64}, Nothing}
    fx::Bool
    m::Union{Int, Nothing}
    label::String
    xt::Dict{Symbol,Any}  # extra type-specific data (e.g., :nb for MRF)
end

# Convenience constructor with default empty xt for backward compatibility
function SmoothSpec(term_vars, basis::B, k, by, id, sp, fx, m, label) where {B<:AbstractBasisType}
    return SmoothSpec{B}(term_vars, basis, k, by, id, sp, fx, m, label, Dict{Symbol,Any}())
end

# ============================================================================
# Constructed smooth — after basis construction from data
# ============================================================================

"""Internal marker type for smooth-specific prediction caches."""
abstract type AbstractSmoothPredictCache end

# ----------------------------------------------------------------------------
# Prediction caches for smooth types whose prediction needs construction-time
# metadata. Stored in `ConstructedSmooth.predict_cache` so the data travels
# with the model through serialization (rather than module-level objectid dicts).
# The concrete `_predict_matrix` methods live alongside each basis constructor.
# ----------------------------------------------------------------------------

"""
Prediction cache for tensor product smooths (te/ti/t2).

Stores the raw (unconstrained) marginal bases so prediction can rebuild each
marginal at new data. For ti() smooths, `marginal_Zs` holds the per-marginal
sum-to-zero null-space bases that were absorbed into each marginal before the
tensor product was formed (empty for te/t2).
"""
struct TensorPredictCache <: AbstractSmoothPredictCache
    raw_marginals::Vector  # Vector{RawMarginalBasis} (defined in basis_tensor.jl)
    marginal_Zs::Vector{Matrix{Float64}}
end

"""
Prediction cache for Markov random field smooths (bs="mrf"): stores the
region labels (in basis-column order) seen at construction time.
"""
struct MRFPredictCache <: AbstractSmoothPredictCache
    levels::Vector
end

"""
Prediction cache for factor-smooth interactions (bs="fs"): stores the factor
levels, the marginal smooth, and the factor variable name.
"""
struct FactorSmoothPredictCache <: AbstractSmoothPredictCache
    levels::Vector
    marginal_smooth  # ConstructedSmooth
    factor_var::Symbol
end

"""
Prediction cache for splines-on-the-sphere smooths (bs="sos"): stores the knot
lat/lon, knot eigenvectors, eigenvalues, penalty order, and basis dimension.
"""
struct SOSPredictCache <: AbstractSmoothPredictCache
    lat_k::Vector{Float64}   # knot latitudes, ALWAYS stored in radians
    lon_k::Vector{Float64}   # knot longitudes, ALWAYS stored in radians
    U_k::Matrix{Float64}
    lambda_k::Vector{Float64}
    m_order::Int
    k::Int
    # Angle units of the *user-facing* covariates (:degrees or :radians).
    # Resolved once at construction and read back here at prediction, so the
    # two paths cannot disagree even if `spec.xt` is mutated in between.
    units::Symbol
end

"""
Prediction cache for SPDE Matérn smooths (bs="spde"): stores the mesh geometry
and dimension info needed to rebuild the interpolation matrix at new data.
`extra` carries the remaining metadata (e.g. mesh nodes, grid dims, L matrix).
"""
struct SPDEPredictCache <: AbstractSmoothPredictCache
    info::Dict{Symbol, Any}
end

"""
Prediction cache for soap-film smooths (bs="so"): stores the grid geometry and
per-basis-function grid values needed to interpolate at new data.
"""
struct SoapPredictCache <: AbstractSmoothPredictCache
    data::Dict{Symbol, Any}
end

"""
    ConstructedSmooth{B<:AbstractBasisType}

A smooth term after basis construction. Contains the model matrix columns,
penalty matrix or matrices, and metadata needed for fitting and prediction.

# Fields
- `spec`: the original smooth specification
- `X`: model matrix for this smooth (n x k_eff after constraint absorption)
- `S`: list of penalty matrices (each k_eff x k_eff)
- `knots`: knot locations used to build the basis
- `null_dim`: dimension of the penalty null space
- `rank`: penalty rank
- `constraint`: identifiability constraint matrix, if any
- `qrc`: QR factorization used for constraint absorption, if any
- `first_para`, `last_para`: coefficient index range in the full model
- `Sigma`, `cmX`, `p_ident`: shape-constraint metadata used by SCAM smooths
- `del_index`: columns removed by side constraints
- `Ain`, `bin`, `Aeq`, `beq`: optional linear inequality/equality constraints
- `predict_cache`: cached metadata needed to build prediction matrices
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
    # Side constraint tracking — columns removed by side_constrain!
    del_index::Vector{Int}
    # General linear constraints (mgcv::scasm / pc constraints)
    Ain::Union{Matrix{Float64}, Nothing}
    bin::Union{Vector{Float64}, Nothing}
    Aeq::Union{Matrix{Float64}, Nothing}
    beq::Union{Vector{Float64}, Nothing}
    predict_cache::Union{AbstractSmoothPredictCache, Nothing}
end

function ConstructedSmooth(
    spec::SmoothSpec{B},
    X::Matrix{Float64},
    S::Vector{Matrix{Float64}},
    knots::Vector{Float64},
    null_dim::Int,
    rank::Int,
    constraint::Union{Matrix{Float64}, Nothing},
    qrc::Union{LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}, Nothing},
    first_para::Int,
    last_para::Int,
    Sigma::Union{Matrix{Float64}, Nothing},
    cmX::Union{Vector{Float64}, Nothing},
    p_ident::Union{BitVector, Nothing},
    del_index::Vector{Int},
    ;
    predict_cache::Union{AbstractSmoothPredictCache, Nothing} = nothing,
) where {B<:AbstractBasisType}
    return ConstructedSmooth{B}(
        spec, X, S, knots, null_dim, rank, constraint, qrc,
        first_para, last_para, Sigma, cmX, p_ident, del_index,
        nothing, nothing, nothing, nothing, predict_cache,
    )
end

function ConstructedSmooth(
    spec::SmoothSpec{B},
    X::Matrix{Float64},
    S::Vector{Matrix{Float64}},
    knots::Vector{Float64},
    null_dim::Int,
    rank::Int,
    constraint::Union{Matrix{Float64}, Nothing},
    qrc::Union{LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}}, Nothing},
    first_para::Int,
    last_para::Int,
    Sigma::Union{Matrix{Float64}, Nothing},
    cmX::Union{Vector{Float64}, Nothing},
    p_ident::Union{BitVector, Nothing},
    del_index::Vector{Int},
    Ain::Union{Matrix{Float64}, Nothing},
    bin::Union{Vector{Float64}, Nothing},
    Aeq::Union{Matrix{Float64}, Nothing},
    beq::Union{Vector{Float64}, Nothing};
    predict_cache::Union{AbstractSmoothPredictCache, Nothing} = nothing,
) where {B<:AbstractBasisType}
    return ConstructedSmooth{B}(
        spec, X, S, knots, null_dim, rank, constraint, qrc,
        first_para, last_para, Sigma, cmX, p_ident, del_index,
        Ain, bin, Aeq, beq, predict_cache,
    )
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
- `scale_est`: scale parameter estimator for the reported scale/covariances
  (`:fletcher` — Fletcher 2012 corrected Pearson, mgcv's default;
   `:pearson` — Pearson/(n−edf); `:deviance` — deviance/(n−edf))
"""
struct GamControl
    epsilon::Float64
    maxit::Int
    outer_maxit::Int
    trace::Bool
    gamma::Float64
    scale_est::Symbol
    sp_optimizer::Symbol
end

"""
    gam_control(; epsilon=1e-7, maxit=200, outer_maxit=200, trace=false,
                  gamma=1.0, scale_est=:fletcher, sp_optimizer=:efs)

Construct a [`GamControl`](@ref) with the given parameters.

# Smoothing parameter optimizers
- `:efs` (default) — Extended Fellner-Schall (Wood & Fasiolo 2017). Fast,
  monotonically convergent, one PIRLS call per outer iteration.
- `:newton` — Newton's method with autodiff Hessian. Differentiates a
  *conditional* REML score (β, W, and scale held fixed at their current
  P-IRLS values), a performance-iteration-style approximation of the exact
  Wood (2011) REML derivative; its fixed point can differ slightly from the
  exact REML optimum. More expensive per step but may converge in fewer
  iterations for difficult problems.

Note: `gamma` inflates the effective degrees of freedom in the GCV/UBRE
criteria and enters the EFS step-acceptance test; unlike mgcv it does not
reshape the EFS update itself. Under the EFS optimizer, `method=:ML` differs
from `:REML` only in the acceptance test and the reported score.
"""
function gam_control(;
    epsilon::Real = 1e-7,
    maxit::Int = 200,
    outer_maxit::Int = 200,
    trace::Bool = false,
    gamma::Real = 1.0,
    scale_est::Symbol = :fletcher,
    sp_optimizer::Symbol = :efs,
)
    sp_optimizer in (:efs, :newton) ||
        throw(ArgumentError("sp_optimizer must be :efs or :newton, got :$sp_optimizer"))
    scale_est in (:fletcher, :pearson, :deviance) ||
        throw(ArgumentError("scale_est must be :fletcher, :pearson, or :deviance, got :$scale_est"))
    return GamControl(Float64(epsilon), maxit, outer_maxit, trace,
        Float64(gamma), scale_est, sp_optimizer)
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
- `rank`: rank of this block's TOTAL penalty `Σⱼ λⱼSⱼ`, not the sum of the
  individual penalties' ranks. `Mp = p - Σ_blocks rank` is then the penalty
  null-space dimension, matching mgcv's `G\$Mp <- ncol(Ssp\$Z)` (`R/mgcv.r:1924`),
  where `Ssp\$Z` is the null space of the Frobenius-normalised total penalty
  (`totalPenaltySpace`, `R/gam.fit3.r:2673-2683`). mgcv separately keeps a
  per-penalty rank VECTOR (`object\$rank`), but that is a different quantity —
  for an adaptive k=40 smooth it is `[8,15,23,30,30,23,15,8]`, summing to 152
  in a 39-column basis — and it feeds `mini.roots` (`R/mgcv.r:1917`), not `Mp`.
  GAM.jl's equivalent per-penalty ranks are derived where they are needed, by
  `_stable_penalty_factor` in `reml.jl`, and match mgcv's vector exactly.
- `start`: first parameter index in the full coefficient vector
- `stop`: last parameter index
"""
struct PenaltyBlock
    S::Vector{Matrix{Float64}}
    rank::Int
    start::Int
    stop::Int
end

"""
    PenaltySetup

Complete block-diagonal penalty structure for a GAM.
Equivalent to mgcv's `Sl.setup` output.

# Fields
- `blocks`: individual penalty blocks
- `sp`: current smoothing parameters (log scale)
- `fixed`: per-penalty flag — `true` entries hold a user-supplied `sp=` value
  and are excluded from smoothing parameter optimization
"""
mutable struct PenaltySetup
    blocks::Vector{PenaltyBlock}
    sp::Vector{Float64}
    fixed::BitVector
end

# Backward-compatible constructor: default to all smoothing parameters free.
PenaltySetup(blocks::Vector{PenaltyBlock}, sp::Vector{Float64}) =
    PenaltySetup(blocks, sp, falses(length(sp)))

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
- `reml`: REML/ML score at convergence when `method` is `:REML` or `:ML`;
  NaN otherwise. Minimized, so it is the *negative* log marginal likelihood —
  the same sign convention as mgcv, whose `b\$gcv.ubre` holds this quantity for
  `method="REML"`/`"ML"` (`R/gam.fit3.r:611-614`).
- `criterion`: achieved GCV/UBRE score when `method` is `:GCV` or `:UBRE`;
  NaN otherwise. Also used by SCAM.

  Between them these two fields are the analogue of mgcv's single
  `b\$gcv.ubre`, which likewise stores whichever score the smoothness selection
  actually optimized (`R/mgcv.r:1669, 1689, 1714`; mgcv picks the criterion
  from `method` at `R/mgcv.r:1946-1965`, using GCV when the scale is estimated
  and UBRE when it is known). Use [`sp_criterion`](@ref) to read whichever one
  applies without branching on `method`.
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
    formula::Any
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
    criterion::Float64
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
    # Smoothing-parameter uncertainty (mgcv's edf1/edf2/Vc). Per-coefficient
    # vectors of length p and a p×p matrix, matching mgcv's convention — note
    # that `edf` above is per-*smooth*. Left empty by fitters that do not
    # compute them (see the constructor below); consumers must treat empty as
    # "unavailable" and fall back to the conditional quantities.
    edf1::Vector{Float64}
    edf2::Vector{Float64}
    Vc::Matrix{Float64}
    # Thunk producing `(Vc, edf2, Vρ)`, or `nothing` once resolved. Computing
    # them costs O(M²) extra P-IRLS refits — enough to dominate the fit itself
    # — and most fits never read them, so the work is deferred to the first
    # `vcov_corrected`/`edf2`/`has_vc`/`dof` call and cached into the two
    # fields above. `edf1` is *not* deferred: it is a pure function of F, which
    # the fit has already formed.
    vc_thunk::Any
end

# Constructor for the fitters that do not supply the smoothing-parameter
# uncertainty quantities (bam, scam, scasm, the GAMM PQL path). Adding the
# three fields positionally would otherwise break every one of those call
# sites, and a missed site is exactly how the `criterion` field regressed the
# non-Gaussian GAMM paths. Empty defaults are deliberate: they are detectable
# by `has_vc`, and every consumer degrades to the conditional quantity rather
# than silently using a zero correction.
function GamModel(formula, y, X, coefficients, fitted_values, linear_predictor,
    weights, family::D, link::L, smooths, penalty, sp, edf, edf_total, scale,
    deviance_val, null_deviance, reml, criterion, method, Vp, Ve,
    hat_matrix_diag, R, converged, iterations, n_smooth, n_parametric,
    control, data) where {D, L <: GLM.Link}

    return GamModel{D, L}(formula, y, X, coefficients, fitted_values,
        linear_predictor, weights, family, link, smooths, penalty, sp, edf,
        edf_total, scale, deviance_val, null_deviance, reml, criterion, method,
        Vp, Ve, hat_matrix_diag, R, converged, iterations, n_smooth,
        n_parametric, control, data,
        Float64[], Float64[], Matrix{Float64}(undef, 0, 0), nothing)
end

"""
    force_vc!(m) -> GamModel

Resolve a deferred `Vc`/`edf2` computation, caching the result on `m`. Called
by every consumer; direct field access (`m.Vc`, `m.edf2`) bypasses it and may
see an empty placeholder, so prefer [`vcov_corrected`](@ref) and [`edf2`](@ref).
"""
function force_vc!(m::GamModel)
    th = m.vc_thunk
    th === nothing && return m
    m.vc_thunk = nothing
    result = th()
    if result !== nothing
        m.Vc = result[1]
        m.edf2 = result[2]
    end
    return m
end

"""
    has_vc(m) -> Bool

Whether `m` carries the smoothing-parameter uncertainty correction — the
`Vc` covariance of Wood, Pya & Säfken (2016) and the `edf2` degrees of freedom
derived from it.

It is available for `gam` fits selected by REML or ML with at least one free
smoothing parameter. It is absent when every `sp` was fixed (there is then no
uncertainty to propagate), when the criterion was GCV or UBRE (mgcv likewise
leaves `Vc` unset there), and for the `bam`, `scam`, `scasm` and GAMM fitters.
"""
function has_vc(m::GamModel)
    force_vc!(m)
    return !isempty(m.edf2) && size(m.Vc, 1) == length(m.coefficients)
end

"""
    sp_criterion(m) -> Float64

The achieved smoothness-selection score at convergence — the value of whichever
criterion `m.method` names. This is the analogue of mgcv's `b\$gcv.ubre`, which
similarly holds the REML/ML score under `method="REML"`/`"ML"` and the GCV/UBRE
score under `method="GCV.Cp"` (`R/mgcv.r:1669, 1689, 1714`).

All four are minimized, so lower is better and REML/ML values are *negative*
log marginal likelihoods, matching mgcv's sign. Comparisons are only meaningful
between fits to the same response with the same criterion.

Reads [`GamModel`](@ref)'s `reml` field for `:REML`/`:ML` and `criterion` for
`:GCV`/`:UBRE`, so callers need not branch on `method`. Returns `NaN` when no
score was recorded.

# Examples
```julia
m = gam(@formula(y ~ s(x)), df; method = :GCV)
sp_criterion(m)          # the achieved GCV score
```
"""
function sp_criterion(m::GamModel)
    return m.method in (:REML, :ML) ? m.reml : m.criterion
end

"""
    edf2(m) -> Vector{Float64}

Per-coefficient `edf2`, the Wood, Pya & Säfken (2016) degrees of freedom that
account for having estimated the smoothing parameters:
`edf2 = rowSums(Vc ∘ X'WX)/φ`, capped at `edf1`. `sum(edf2)` is the df behind
mgcv's `AIC()`. Falls back to the per-coefficient `edf` when unavailable
(see [`has_vc`](@ref)); triggers the deferred computation on first call.
"""
function edf2(m::GamModel)
    has_vc(m) && return m.edf2
    return isempty(m.edf1) ? Float64[] : copy(m.edf1)
end

"""
    vcov_corrected(m) -> Matrix{Float64}

The smoothing-parameter-uncertainty-corrected covariance `Vc` of Wood, Pya &
Säfken (2016) — mgcv's `m\$Vc`, the matrix behind `predict.gam`'s
`unconditional = TRUE`. Wider than the Bayesian `Vp`, which conditions on the
selected smoothing parameters as if they were known. Falls back to `Vp` when
unavailable; triggers the deferred computation on first call.
"""
vcov_corrected(m::GamModel) = has_vc(m) ? m.Vc : m.Vp

"""
    _covariance_for(m, unconditional) -> Matrix{Float64}

Covariance to propagate into intervals: `Vc` when `unconditional = true` and
the fit provides it, `Vp` otherwise. Mirrors mgcv's `unconditional` argument to
`predict.gam`/`plot.gam` and gratia's to `smooth_estimates`/`derivatives`.
Warns once when the correction is asked for but unavailable for this fit,
rather than silently returning the narrower `Vp`.
"""
function _covariance_for(m::GamModel, unconditional::Bool)
    unconditional || return m.Vp
    has_vc(m) && return m.Vc
    @warn "unconditional=true requested but this fit has no smoothing-" *
          "parameter-corrected covariance (mgcv's Vc); using Vp. Vc is " *
          "available for gam() fits selected by :REML or :ML with at least " *
          "one free smoothing parameter." maxlog = 1
    return m.Vp
end


"""
    ref_df(m) -> Vector{Float64}

Per-smooth reference degrees of freedom — the `Ref.df` column of mgcv's
`summary.gam`, obtained by summing `edf1` over each smooth's coefficients.
Falls back to the per-smooth `edf` for fits that do not compute `edf1`.
"""
function ref_df(m::GamModel)
    isempty(m.edf1) && return copy(m.edf)
    return smooth_edf(m.edf1, m.smooths)
end
