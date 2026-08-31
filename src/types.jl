# Core type hierarchy for GAM.jl

# ============================================================================
# Smoothing parameter bounds
# ============================================================================

"""
Bounds on `log λ` shared by every smoothing-parameter optimizer (EFS, Newton,
GCV/UBRE, `bam`, `scam`, `gamlss`, `mpfit`).

`30.0`, not the former `15.0`. The old bound was too low to shrink a term out
through a shrinkage basis: `bs=:cs` cascades its null-space eigenvalues down to
`shrink^2`, so it needs roughly 100x more λ than `bs=:ts` for the same amount
of shrinkage, and mgcv's own optima on a term that should vanish sit at
`log λ = 16.8` (`ts`) and `22.61` (`cs`) — both above 15. Terms pinned exactly
at the bound and could not shrink further: an irrelevant covariate held
`edf = 0.286` where mgcv reached `0.0002`, and fixing λ by hand walked it to
`0.0026` at `log λ = 20` and `1.8e-5` at `25`.

Keep every optimizer on this constant rather than a literal, so the bound
cannot drift between them — a term that can shrink out under one method and
not another is a silent modelling difference.
"""
const LOG_SP_BOUND = 30.0

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
    # Row map for a REDUCED basis. Empty (the default) means `X` already has
    # one row per observation. Non-empty means `X` is the `m x k` basis at the
    # unique covariate values and row `i` of the full block is `X[rowmap[i], :]`
    # -- the storage `bam(...; discrete=true)` builds before it is re-expanded.
    # Use `smooth_matrix(sm)` rather than `sm.X` wherever `n` rows are needed.
    rowmap::Vector{Int32}
    # Per-sub-penalty offsets for NARROW penalty storage. Empty (the default)
    # means every matrix in `S` spans the smooth's full column width. Non-empty
    # means `S[i]` occupies columns `S_offsets[i] .+ (1:size(S[i], 1))` of the
    # smooth's own column space (0-based offsets, mirroring
    # `PenaltyBlock.offsets`). A factor-`by` penalty is `I_L ⊗ S_k`; storing it
    # as `L` narrow `k×k` copies is O(L·k²) instead of O(L³k²) — 186.92 MiB →
    # 0.075 MiB at L=50, k=14 — and every penalty hot path bounds its loops by
    # `size(Si,1)`, so the same L² factor comes off `total_penalty!` and
    # `_log_penalty_det` per sp-iteration. mgcv's own factor-`by` storage is
    # already narrow (R/smooth.r:3980 replicates per level, each keeping its
    # k×k S). Consumers that need the widened form call `penalty_matrices(sm)`,
    # which returns `S` itself — no copy — on the empty (common) path.
    S_offsets::Vector{Int}
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
        nothing, nothing, nothing, nothing, predict_cache, Int32[], Int[],
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
        Ain, bin, Aeq, beq, predict_cache, Int32[], Int[],
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

  **`:newton` applies to single-penalty smooths only.** Multi-penalty blocks —
  `te`/`ti`/`t2` tensors, `bs=:ad`, `bs=:fs`, and every smooth under
  `select=true` — go through a stable penalty reparameterization
  (`_stable_penalty_factor`, a port of mgcv's `gam.reparam`) that is
  `Float64`-only, so the ForwardDiff Hessian cannot be taken through it. Those
  models silently *fall back to `:efs`*, warning once; the fit is still
  correct, it is simply an EFS fit. mgcv avoids this by computing the REML
  derivatives analytically rather than by autodiff. Because that fallback would
  mean a single model optimizing `s(x)` by Newton and `te(x,z)` by EFS, `:efs`
  remains the default.

  Where Newton does run it reaches an equal or slightly better criterion than
  EFS. The gap is negligible (≤4e-5) on ordinary bases but material on the
  **shrinkage bases**, which need very large log-λ to drop a term and are
  exactly where the EFS fixed point stops short: on a `bs=:ts` fit the REML
  score is 126.2084 (Newton) against 126.2417 (EFS). Prefer `:newton` there.
  Only REML and ML are affected — GCV, UBRE and NCV are optimized directly and
  ignore `sp_optimizer` entirely.

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
- `offsets`: per-sub-penalty 0-based offset within the block. `S[i]` occupies
  rows/columns `start + offsets[i] .+ (0:size(S[i],1)-1)`. All-zero (with each
  `S[i]` the full block width) is the ordinary case and the default.

  A non-zero offset lets a structured penalty be stored without materialising
  it. A factor-`by` smooth's penalty is `I_L ⊗ S_k`, so it can be held as `L`
  copies of the narrow `S_k` at offsets `0, k, 2k, …` instead of `L` dense
  `kL × kL` matrices — `O(L·k²)` rather than `O(L³k²)`, which is 0.9 MiB at
  `L=8, k=15` but 225 MiB at `L=50, k=15`.

  The inner constructor validates that every `S[i]` is square and fits inside
  the block at its offset. That check is what makes the `@inbounds`
  accumulation in `total_penalty` safe: the loop runs over `size(S[i],1)`, so
  a narrow sub-penalty is now a construction-time error rather than an
  out-of-bounds read. Before this field existed the loop ran over the full
  block width and a narrow `S[i]` silently produced a finite, wrong penalty.
"""
struct PenaltyBlock
    S::Vector{Matrix{Float64}}
    rank::Int
    start::Int
    stop::Int
    offsets::Vector{Int}

    function PenaltyBlock(S::Vector{Matrix{Float64}}, rank::Int, start::Int,
        stop::Int, offsets::Vector{Int})
        width = stop - start + 1
        length(offsets) == length(S) || throw(DimensionMismatch(
            "PenaltyBlock has $(length(S)) penalties but $(length(offsets)) " *
            "offsets; supply one offset per penalty"))
        for (i, Si) in enumerate(S)
            size(Si, 1) == size(Si, 2) || throw(DimensionMismatch(
                "penalty $i is $(size(Si,1))x$(size(Si,2)); penalties must be square"))
            off = offsets[i]
            off >= 0 || throw(ArgumentError(
                "penalty $i has negative offset $off"))
            off + size(Si, 1) <= width || throw(DimensionMismatch(
                "penalty $i is $(size(Si,1))x$(size(Si,1)) at offset $off, " *
                "which does not fit in a block of width $width " *
                "(parameters $start:$stop). A sub-penalty narrower than its " *
                "block needs an offset saying where it sits."))
        end
        return new(S, rank, start, stop, offsets)
    end
end

# Ordinary case: every sub-penalty spans the whole block.
#
# This form requires full width rather than defaulting to offset 0, and the
# distinction is the point. A narrow sub-penalty at offset 0 *fits*, so it
# would be accepted and would then penalize only the first `size(Si,1)`
# coefficients of the block — well defined, but almost never what a caller who
# never thought about placement intended. Supplying offsets explicitly is the
# way to say a narrow penalty is deliberate.
function PenaltyBlock(S::Vector{Matrix{Float64}}, rank::Int, start::Int, stop::Int)
    width = stop - start + 1
    for (i, Si) in enumerate(S)
        size(Si, 1) == width || throw(DimensionMismatch(
            "penalty $i is $(size(Si,1))x$(size(Si,2)) but its block is " *
            "$(width) wide (parameters $start:$stop). Sub-penalties must span " *
            "the block unless you pass explicit offsets saying where each one " *
            "sits — e.g. PenaltyBlock(S, rank, start, stop, [0, k, 2k, ...]) " *
            "to store a factor-by penalty as L copies of a narrow S_k."))
    end
    return PenaltyBlock(S, rank, start, stop, zeros(Int, length(S)))
end

"""
    _sub_penalty_idx(block::PenaltyBlock, i::Int) -> UnitRange{Int}

Absolute coefficient indices occupied by sub-penalty `i` of `block`.

Every consumer of `block.S[i]` must index through this rather than through
`block.start:block.stop`. A sub-penalty may be narrower than its block — a
factor-`by` penalty is `L` copies of a narrow `S_k` at disjoint offsets — and
iterating the block width while indexing `Si[j,k]` reads past the end of `Si`.
Several such loops carried `@inbounds`, so the read did not raise: it returned
a plausible, finite, wrong answer. `total_penalty` was fixed first; this helper
exists so the remaining consumers cannot reintroduce the same bug.
"""
@inline function _sub_penalty_idx(block::PenaltyBlock, i::Int)
    off = block.offsets[i]
    m = size(block.S[i], 1)
    return (block.start + off):(block.start + off + m - 1)
end

"""
    _penalties_disjoint(block::PenaltyBlock) -> Bool

Whether `block`'s sub-penalties occupy pairwise non-overlapping coefficient
ranges, as a factor-`by` block does (`I_L ⊗ S_k` stored as `L` copies of `S_k`).

This is worth detecting because the log pseudo-determinant is then *separable*:
`log|Σⱼ λⱼSⱼ|₊ = Σⱼ log|λⱼSⱼ|₊`, and `∂/∂ρⱼ` is exactly `rank(Sⱼ)`. Both are
exact and need no eigen-decomposition. The similarity-transform path
(`_stable_penalty_factor`) cannot be used on such a block: it works in the
sub-penalty's own `k`-dimensional space, which for disjoint penalties is not
the block's space at all, and silently returns a determinant for the wrong
object. Overlapping multi-penalty blocks — `te`, `ti`, `t2`, adaptive, `fs` —
are unaffected and keep that path.
"""
function _penalties_disjoint(block::PenaltyBlock)
    length(block.S) > 1 || return false
    width = block.stop - block.start + 1
    # Cheap early-out that keeps this allocation-free for every block that
    # exists today: `te`, `ti`, `t2`, adaptive and `fs` all store block-width
    # sub-penalties, which with more than one penalty must overlap. Only a
    # genuinely narrow block reaches the bitmap below.
    for Si in block.S
        size(Si, 1) == width && return false
    end
    covered = falses(width)
    for i in eachindex(block.S)
        off = block.offsets[i]
        for j in 1:size(block.S[i], 1)
            covered[off + j] && return false
            covered[off + j] = true
        end
    end
    return true
end

"""
    _block_width_penalties(block::PenaltyBlock) -> Vector{Matrix{Float64}}

`block.S` widened so every sub-penalty spans the block, returning the stored
vector itself — no copy — in the common case where they already do.

`_stable_penalty_factor` (mgcv's `gam.reparam` transform) assumes all its inputs
share one coordinate system. That holds for `te`/`ti`/`t2`/adaptive/`fs`, whose
sub-penalties are all block-width, but not for a block mixing widths: under
`select = true` a factor-`by` block carries `L` narrow penalties at disjoint
offsets *plus* a block-width null-space penalty. Such a block is not disjoint,
so the separable shortcut does not apply either, and handing ragged matrices to
the transform raises `DimensionMismatch`. Widening first is correct and costs a
transient only for the blocks that need it.
"""
function _block_width_penalties(block::PenaltyBlock)
    width = block.stop - block.start + 1
    all(size(Si, 1) == width for Si in block.S) && return block.S
    out = Vector{Matrix{Float64}}(undef, length(block.S))
    for (i, Si) in enumerate(block.S)
        if size(Si, 1) == width
            out[i] = Matrix{Float64}(Si)
        else
            W = zeros(Float64, width, width)
            off = block.offsets[i]
            m = size(Si, 1)
            @inbounds for j in 1:m, k in 1:m
                W[off + j, off + k] = Si[j, k]
            end
            out[i] = W
        end
    end
    return out
end

"""
    penalty_matrices(sm::ConstructedSmooth) -> Vector{Matrix{Float64}}

`sm.S` with every sub-penalty widened to the smooth's full column width.

Returns the stored vector ITSELF — no copy — on the common path where
`sm.S_offsets` is empty (all penalties already block-width). When the smooth
carries narrow factor-`by` penalties (`L` copies of a `k×k` `S_k` at offsets
`(l-1)k`), each is zero-padded to `size(sm.X, 2)` — the `ConstructedSmooth`
analogue of `_block_width_penalties`. Consumers that only need the
*count* of penalties, or that index through offsets themselves, should keep
reading `sm.S` directly.
"""
function penalty_matrices(sm::ConstructedSmooth)
    isempty(sm.S_offsets) && return sm.S
    width = size(sm.X, 2)
    out = Vector{Matrix{Float64}}(undef, length(sm.S))
    for (i, Si) in enumerate(sm.S)
        m = size(Si, 1)
        if m == width
            out[i] = Matrix{Float64}(Si)
        else
            W = zeros(Float64, width, width)
            off = sm.S_offsets[i]
            @inbounds for j in 1:m, k in 1:m
                W[off + j, off + k] = Si[j, k]
            end
            out[i] = W
        end
    end
    return out
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
- `criterion`: achieved GCV/UBRE/NCV score when `method` is `:GCV`, `:UBRE`
  or `:NCV`; NaN otherwise. Also used by SCAM.

  Between them these two fields are the analogue of mgcv's single
  `b\$gcv.ubre`, which likewise stores whichever score the smoothness selection
  actually optimized (`R/mgcv.r:1669, 1689, 1714`; mgcv picks the criterion
  from `method` at `R/mgcv.r:1946-1965`, using GCV when the scale is estimated
  and UBRE when it is known). Use [`sp_criterion`](@ref) to read whichever one
  applies without branching on `method`.
- `method`: smoothing method used (:REML, :ML, :GCV, :UBRE, :NCV)
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
    # Parametric columns of the model matrix, retained ONLY once `X` has been
    # dropped by `drop_model_matrix!`. `X` is pure duplication of data already
    # held elsewhere — every smooth block of `X` is bitwise identical to the
    # corresponding `ConstructedSmooth.X` — so the only part not recoverable
    # from `smooths` is the parametric block, which is `n × n_parametric`
    # (usually one intercept column) against `X`'s `n × p`. Empty while `X` is
    # retained, following the same empty-as-unavailable convention as `edf1`,
    # `edf2` and `Vc` above; that keeps the field concretely typed rather than
    # a `Union`, which matters because `GamModel` is on every hot path.
    X_par::Matrix{Float64}
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
        Float64[], Float64[], Matrix{Float64}(undef, 0, 0), nothing,
        Matrix{Float64}(undef, 0, 0))
end

"""
    model_matrix(m::GamModel) -> Matrix{Float64}

The `n × p` model matrix. Returns `m.X` when it is retained (the default), and
otherwise reassembles it exactly from the parametric block and the smooth
bases — see [`drop_model_matrix!`](@ref).

Reassembly is **bitwise** identical to the retained matrix, not merely close.
It concatenates stored blocks rather than re-evaluating any basis: every
smooth block of `X` was verified bitwise equal to the corresponding
`ConstructedSmooth.X` across plain, tensor, random-effect, side-constrained
and parametric models. Re-evaluating instead (via `_gam_prediction_matrix`)
would drift by ~2.7e-13 on thin-plate smooths, which is why that route is not
used.

Prefer this over direct `m.X` access: a model whose matrix has been dropped
has an empty `X`, and reading the field would silently yield a `0×0` matrix.
"""
function model_matrix(m::GamModel)
    size(m.X, 1) > 0 && return m.X
    n = length(m.y)
    npar = m.n_parametric
    p = npar
    for sm in m.smooths
        p = max(p, sm.last_para)
    end
    X = Matrix{Float64}(undef, n, p)
    if npar > 0
        size(m.X_par, 2) == npar || throw(ArgumentError(
            "model_matrix: the model matrix was dropped but its parametric " *
            "block is missing ($(size(m.X_par, 2)) columns, expected $npar). " *
            "This model was not produced by `drop_model_matrix!`."))
        copyto!(view(X, :, 1:npar), m.X_par)
    end
    for sm in m.smooths
        _scatter_block!(view(X, :, sm.first_para:sm.last_para), sm)
    end
    return X
end

"""
    smooth_matrix(sm::ConstructedSmooth) -> Matrix{Float64}

The smooth's model-matrix block with one row per observation.

Returns `sm.X` directly when it is already stored that way (the default). When
the smooth holds a REDUCED basis -- `bam(...; discrete=true)` builds each
1-D smooth at the `m` unique covariate values -- this scatters it back out,
allocating an `n x k` matrix. Prefer working with `sm.X` and `sm.rowmap`
directly in per-row loops; this exists for callers that genuinely need the
expanded block.
"""
function smooth_matrix(sm::ConstructedSmooth)
    isempty(sm.rowmap) && return sm.X
    out = Matrix{Float64}(undef, length(sm.rowmap), size(sm.X, 2))
    _scatter_block!(out, sm)
    return out
end

"""
    is_reduced(sm::ConstructedSmooth) -> Bool

Whether `sm.X` is a reduced basis needing `sm.rowmap` to expand.
"""
is_reduced(sm::ConstructedSmooth) = !isempty(sm.rowmap)

# Scatter `sm`'s block into `dest`, which must be `n x k`. Column-major so the
# destination is written contiguously; `Xd` rows are gathered by the index.
function _scatter_block!(dest::AbstractMatrix{Float64}, sm::ConstructedSmooth)
    if isempty(sm.rowmap)
        copyto!(dest, sm.X)
        return dest
    end
    Xd = sm.X
    kmap = sm.rowmap
    @inbounds for j in axes(Xd, 2)
        col = view(dest, :, j)
        for i in eachindex(kmap)
            col[i] = Xd[kmap[i], j]
        end
    end
    return dest
end

"""
    has_model_matrix(m::GamModel) -> Bool

Whether `m` retains its model matrix directly. `false` means
[`model_matrix`](@ref) will reassemble it on each call.
"""
has_model_matrix(m::GamModel) = size(m.X, 1) > 0

"""
    drop_model_matrix!(m::GamModel) -> GamModel

Drop the retained `n × p` model matrix, keeping only its parametric columns,
so that [`model_matrix`](@ref) reassembles it on demand from the smooth bases.

This is what mgcv does — `bam` sets `G\$smooth <- G\$X <- NULL` and
`model.matrix.gam` recomputes — and it is worth `n × (p − n_parametric)`
doubles: 587 MB on a measured `n = 10⁶` fit.

!!! warning "Internal until consumers migrate"
    Several call sites still read `m.X` directly (`concurvity` and the
    `k_check` helper in `diagnostics.jl`, three sites in `gratia.jl`,
    `ginla.jl`, and the plotting extension). Those must move to
    `model_matrix(m)` before dropping is safe to expose as a user-facing
    option; until then a dropped model will silently give them a `0×0`
    matrix.
"""
function drop_model_matrix!(m::GamModel)
    size(m.X, 1) > 0 || return m
    npar = m.n_parametric
    m.X_par = npar > 0 ? m.X[:, 1:npar] :
              Matrix{Float64}(undef, size(m.X, 1), 0)
    m.X = Matrix{Float64}(undef, 0, 0)
    return m
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
`:GCV`/`:UBRE`/`:NCV`, so callers need not branch on `method`. Returns `NaN`
when no score was recorded.

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
    # Fallback: per-coefficient edf = diag(F), matching mgcv, which builds Vc
    # even at fully fixed sp — where Vc collapses to Vp and its
    # `edf2 = rowSums(Vc ∘ X'WX)/φ` collapses to diag(F). diag(F) is not
    # stored, but F = A⁻¹X'WX = A⁻¹(A − S_λ) gives
    # diag(F) = 1 − diag(Vp·S_λ)/φ from fields the model does carry; the sum
    # reproduces `edf_total` exactly (verified to machine precision). The
    # previous fallback returned edf1, which overstates the df of every
    # penalized fit (edf1 ≥ edf strictly) and contradicted both mgcv and this
    # docstring.
    p = length(m.coefficients)
    (isempty(m.Vp) || p == 0 || !(m.scale > 0)) && return Float64[]
    S_total = total_penalty(m.penalty, m.sp, p)
    return 1.0 .- vec(sum(m.Vp .* S_total; dims = 2)) ./ m.scale
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
