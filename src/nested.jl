# Nested effects — smooths of estimated covariate transformations
#
# Implements the nested-effect framework of Fasiolo et al. (2025),
# "Scalable smoothing with nested models" (arXiv:2511.19234), as provided by
# the R package gamFactory: effects of the form s(s̃_a(x)) where the inner
# transformation s̃ has parameters `a` estimated jointly with the outer
# spline coefficients.
#
# Supported inner transformations:
#   * trans_linear()  — single-index / distributed-lag: s̃ᵢ = Σⱼ aⱼ xᵢⱼ
#   * trans_nexpsm()  — adaptive exponential smoothing of a time-ordered
#                       series: s̃ᵢ = ωᵢ s̃ᵢ₋₁ + (1−ωᵢ) xᵢ, ωᵢ = logistic(a'zᵢ)
#   * trans_mgks()    — multivariate Gaussian-kernel smoothing of a covariate
#                       over coordinates: s̃ᵢ = Σⱼ K_a(cᵢ,cⱼ) zⱼ / Σⱼ K_a(cᵢ,cⱼ)
#
# Following the paper, the inner output is standardized (mean 0, variance 1)
# so the outer basis lives on a fixed symmetric range; the outer smooth is a
# cubic B-spline with an s(0) = 0 constraint, a second-order difference
# penalty, and linear extrapolation beyond the boundary knots. The inner
# parameters get their own quadratic penalty (ridge, or a second-difference
# penalty across lags for trans_linear). All coefficients are estimated by
# penalized Newton with the exact gradient (chain rule through the η-Jacobian,
# computed by ForwardDiff) and a Gauss–Newton/Fisher (expected-information)
# Hessian H = Jη'WJη + S_λ — an order of magnitude cheaper than AD of the
# full Hessian, with an exact-AD fallback when a Gauss–Newton step fails.
# Smoothing parameters are selected by EFS on the deviance-scale criterion
# with an adaptive step multiplier and a conditional-LAML acceptance check,
# mirroring the conventions of the core engine.

using ForwardDiff: ForwardDiff

# ============================================================================
# Inner transformations
# ============================================================================

"""
Abstract supertype for nested-effect inner transformations (`trans_linear`,
`trans_nexpsm`, `trans_mgks`).
"""
abstract type NestedTransform end

"""
    trans_linear(; penalty = :diff2)

Single-index / distributed-lag inner transformation for [`s_nest`](@ref):
`s̃ᵢ = Σⱼ aⱼ xᵢⱼ` over the variables given to `s_nest`. `penalty` is the
quadratic penalty on the index coefficients `a`: `:diff2` (second-order
difference across the variables, natural when they are consecutive lags) or
`:ridge`.
"""
struct TransLinear <: NestedTransform
    penalty::Symbol
    function TransLinear(penalty::Symbol)
        penalty in (:diff2, :ridge) || throw(ArgumentError(
            "trans_linear penalty must be :diff2 or :ridge, got :$penalty"))
        new(penalty)
    end
end
trans_linear(; penalty::Symbol = :diff2) = TransLinear(penalty)

"""
    trans_nexpsm()

Adaptive exponential smoothing inner transformation for [`s_nest`](@ref).
The first variable given to `s_nest` is the series to smooth (rows must be
in time order); any further variables drive the smoothing weight:
`s̃ᵢ = ωᵢ s̃ᵢ₋₁ + (1−ωᵢ) xᵢ` with `ωᵢ = logistic(a₁ + a₂ z₁ᵢ + …)` and
`s̃₁ = x₁`. The weight coefficients `a` carry a ridge penalty.
"""
struct TransExpSmooth <: NestedTransform end
trans_nexpsm() = TransExpSmooth()

"""
    trans_mgks()

Multivariate Gaussian-kernel smoothing inner transformation for
[`s_nest`](@ref). The first variable given to `s_nest` is the covariate `z`
to smooth; the remaining variables are the coordinates. For observation `i`,
`s̃ᵢ = Σⱼ∈Nᵢ K_a(cᵢ,cⱼ) zⱼ / Σⱼ∈Nᵢ K_a(cᵢ,cⱼ)` with a Gaussian kernel whose
per-coordinate log-bandwidths are the inner parameters `a` (ridge penalty).
`Nᵢ` is the neighborhood of the `nn` nearest training points (excluding `i`
during fitting), fixed at setup as in Fasiolo et al. (2025), giving O(n·nn)
evaluation; `nn = 0` uses all points (O(n²)). Prediction smooths over the
stored training `(c, z)`.

Note on training-point semantics: `fitted(m)` uses the leave-one-out smoother
(each observation excluded from its own neighborhood, exactly as during
fitting), whereas `predict(m, data)` treats every row as a new location and
smooths over *all* stored training points — so `predict` on the training
table includes each point's own `z` value and generally differs from
`fitted`. This matches the estimation/prediction distinction in the paper.

    trans_mgks(; nn = 50)
"""
struct TransMGKS <: NestedTransform
    nn::Int    # neighborhood size (0 = use all training points)
    function TransMGKS(nn::Int)
        nn >= 0 || throw(ArgumentError("trans_mgks nn must be >= 0, got $nn"))
        new(nn)
    end
end
trans_mgks(; nn::Int = 50) = TransMGKS(nn)

# Number of inner parameters for a transform given the s_nest data columns
_n_inner(::TransLinear, ncols::Int) = ncols
_n_inner(::TransExpSmooth, ncols::Int) = ncols        # a₁ + one per extra var
_n_inner(::TransMGKS, ncols::Int) = ncols - 1         # one log-bandwidth per coordinate

function _inner_penalty(t::TransLinear, na::Int)
    if t.penalty == :diff2 && na >= 3
        D = zeros(na - 2, na)
        for i in 1:(na - 2)
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        end
        # Small ridge keeps the (standardization-unidentified) scale of `a` proper
        return D' * D + 1e-4 * I(na)
    end
    return Matrix{Float64}(I, na, na)
end
_inner_penalty(::TransExpSmooth, na::Int) = Matrix{Float64}(I, na, na)
_inner_penalty(::TransMGKS, na::Int) = Matrix{Float64}(I, na, na)

_default_inner_start(::TransLinear, na::Int) = fill(1.0 / sqrt(na), na)
_default_inner_start(::TransExpSmooth, na::Int) = zeros(na)
_default_inner_start(::TransMGKS, na::Int) = zeros(na)

# Evaluate s̃ at inner parameters `a` for the n×d data matrix `Xin`
# (AD-generic: eltype of `a` may be a ForwardDiff.Dual).
function _trans_eval(::TransLinear, a::AbstractVector, Xin::AbstractMatrix)
    return Xin * a
end

function _trans_eval(::TransExpSmooth, a::AbstractVector, Xin::AbstractMatrix)
    n = size(Xin, 1)
    T = promote_type(eltype(a), eltype(Xin))
    st = Vector{T}(undef, n)
    st[1] = Xin[1, 1]
    for i in 2:n
        lin = a[1]
        for j in 2:length(a)
            lin += a[j] * Xin[i, j]
        end
        ω = 1.0 / (1.0 + exp(-lin))
        st[i] = ω * st[i - 1] + (1.0 - ω) * Xin[i, 1]
    end
    return st
end

function _trans_eval(::TransMGKS, a::AbstractVector, Xin::AbstractMatrix)
    return _mgks_smooth(a, Xin, view(Xin, :, 2:size(Xin, 2)), view(Xin, :, 1);
        exclude_self = true)
end

# Kernel-smooth training values `z` (with coordinates `Ctr`) onto the query
# coordinates in columns 2:end of Xq. `exclude_self=true` performs
# leave-one-out smoothing (training); prediction uses all training points.
function _mgks_smooth(a::AbstractVector, Xq::AbstractMatrix,
    Ctr::AbstractMatrix, z::AbstractVector; exclude_self::Bool,
    neigh::Union{Nothing, Matrix{Int}} = nothing)
    nq = size(Xq, 1)
    ntr = size(Ctr, 1)
    d = size(Ctr, 2)
    T = promote_type(eltype(a), eltype(Xq))
    out = Vector{T}(undef, nq)
    invh2 = [exp(-2.0 * a[j]) for j in 1:d]
    for i in 1:nq
        num = zero(T)
        den = zero(T)
        cand = neigh === nothing ? (1:ntr) : view(neigh, i, :)
        for j in cand
            # NOTE: `i == j` compares a query index with a training index, so
            # exclude_self is only meaningful when the query rows ARE the
            # training rows in the same order (the leave-one-out training
            # smoother). Prediction always passes exclude_self = false.
            exclude_self && i == j && continue
            q = zero(T)
            for l in 1:d
                δ = Xq[i, 1 + l] - Ctr[j, l]
                q += δ * δ * invh2[l]
            end
            w = exp(-0.5 * q)
            num += w * z[j]
            den += w
        end
        out[i] = num / (den + 1e-300)
    end
    return out
end

# Fixed nearest-neighbor sets (std-scaled Euclidean metric on coordinates),
# computed once at setup: query rows of Cq against training rows of Ctr.
# `exclude_self=true` skips index i (training leave-one-out).
function _mgks_neighbors(Cq::AbstractMatrix, Ctr::AbstractMatrix, nn::Int;
    exclude_self::Bool)
    nq, d = size(Cq, 1), size(Cq, 2)
    ntr = size(Ctr, 1)
    scale = [max(std(view(Ctr, :, l)), 1e-12) for l in 1:d]
    nn_eff = min(nn, exclude_self ? ntr - 1 : ntr)
    neigh = Matrix{Int}(undef, nq, nn_eff)
    d2 = Vector{Float64}(undef, ntr)
    for i in 1:nq
        for j in 1:ntr
            q = 0.0
            for l in 1:d
                δ = (Cq[i, l] - Ctr[j, l]) / scale[l]
                q += δ * δ
            end
            d2[j] = q
        end
        exclude_self && (d2[i] = Inf)
        neigh[i, :] .= partialsortperm(d2, 1:nn_eff)
    end
    return neigh
end

# ============================================================================
# Basis type and s_nest()
# ============================================================================

"""
    NestedBasis <: AbstractBasisType

Basis marker for nested effects (`s_nest`). Carries the inner
[`NestedTransform`](@ref). Nested smooths are fitted by the dedicated joint
Newton/EFS routine ([`gam_nl`](@ref)); `gam()` routes to it automatically.
"""
struct NestedBasis{T<:NestedTransform} <: AbstractBasisType
    trans::T
end

"""
    s_nest(vars...; trans = trans_linear(), k = 10, sp = nothing)

Specify a nested effect `s(s̃_a(x))` (Fasiolo et al. 2025 / gamFactory): a
cubic-spline outer smooth of an inner covariate transformation whose
parameters `a` are estimated jointly with the spline. `trans` selects the
inner transformation ([`trans_linear`](@ref), [`trans_nexpsm`](@ref),
[`trans_mgks`](@ref)); the meaning of `vars` depends on it. `k` is the outer
basis dimension; `sp` optionally fixes the outer smoothing parameter.
Usable inside `@formula`/`@formulak`; such formulas are fitted with
[`gam_nl`](@ref) (or `gam()`, which routes automatically).

# Example
```julia
m = gam_nl(@formula(y ~ s(x0) + s_nest(l1, l2, l3, trans=trans_linear(), k=10)), df)
```
"""
function s_nest(vars::Union{Symbol, StatsModels.AbstractTerm}...;
    trans::NestedTransform = trans_linear(), k::Int = 10,
    sp = nothing, by = nothing, id = nothing, fx::Bool = false)
    syms = Symbol[v isa Symbol ? v : v.sym for v in vars]
    isempty(syms) && throw(ArgumentError("s_nest() requires at least one variable"))
    by === nothing || throw(ArgumentError("s_nest does not support by="))
    id === nothing || throw(ArgumentError("s_nest does not support id="))
    fx && throw(ArgumentError("s_nest does not support fx=true"))
    if trans isa TransMGKS && length(syms) < 2
        throw(ArgumentError("trans_mgks needs s_nest(z, coord1, ...): a value variable plus coordinates"))
    end
    k == -1 && (k = 10)   # default from the positional @formula path
    k >= 4 || throw(ArgumentError("s_nest requires k >= 4, got $k"))
    label = "s_nest(" * join(string.(syms), ",") * ")"
    return SmoothSpec(syms, NestedBasis(trans), k, nothing, nothing,
        sp === nothing ? nothing : Float64(sp), false, nothing, label)
end

"""Whether any smooth spec in a collection is a nested effect."""
has_nested_effects(specs) = any(sp -> sp.basis isa NestedBasis, specs)

# Nested smooths never enter the standard construction pipeline
function _smooth_construct(::NestedBasis, spec::SmoothSpec, data, knots)
    throw(ArgumentError(
        "s_nest terms are fitted by gam_nl()/gam(), not smooth_construct()"))
end

# ============================================================================
# Outer basis: cubic B-spline on a fixed range, s(0)=0, linear extrapolation
# ============================================================================

# Range of the (standardized) inner variable covered by the outer basis.
const _NESTED_RANGE = 3.0

# Not an AbstractSmoothPredictCache: those caches are attached to a
# ConstructedSmooth and consumed by predict_matrix dispatch, whereas the
# nested outer basis is evaluated directly by the gam_nl fitter/predict code
# (its argument u depends on the estimated inner parameters, so there is no
# fixed prediction matrix to cache).
struct NestedOuterBasis
    knots::Vector{Float64}
    drop::Int              # column removed to absorb the s(0)=0 constraint
    ncols::Int             # columns after the drop
end

function _nested_outer_basis(k::Int)
    order = 4
    nknots = k + order         # k basis functions
    interior = range(-_NESTED_RANGE, _NESTED_RANGE; length = nknots - 2 * (order - 1))
    dk = step(interior)
    knots = vcat([interior[1] - dk * i for i in (order - 1):-1:1], collect(interior),
        [interior[end] + dk * i for i in 1:(order - 1)])
    row0 = _bspline_row_interior(0.0, knots, order, k)
    drop = argmax(abs.(row0))
    return NestedOuterBasis(knots, drop, k - 1)
end

# Cox–de Boor evaluation of all k cubic B-splines at u, AD-generic, with
# linear extrapolation (value + first derivative at the boundary) outside
# the knot range, matching the package's B-spline predict convention.
function _bspline_row_generic(u, knots::Vector{Float64}, order::Int, k::Int)
    lo = knots[order]
    hi = knots[k + 1]
    if u < lo
        b0 = _bspline_row_interior(lo, knots, order, k)
        d0 = _bspline_row_deriv(lo, knots, order, k)
        return b0 .+ (u - lo) .* d0
    elseif u > hi
        b1 = _bspline_row_interior(hi, knots, order, k)
        d1 = _bspline_row_deriv(hi, knots, order, k)
        return b1 .+ (u - hi) .* d1
    end
    return _bspline_row_interior(u, knots, order, k)
end

function _bspline_row_interior(u, knots::Vector{Float64}, order::Int, k::Int)
    T = typeof(float(u))
    row = zeros(T, k)
    for j in 1:k
        row[j] = _bspline_basis_one(u, knots, j, order)
    end
    return row
end

function _bspline_row_deriv(u, knots::Vector{Float64}, order::Int, k::Int)
    T = typeof(float(u))
    row = zeros(T, k)
    for j in 1:k
        d1 = knots[j + order - 1] - knots[j]
        d2 = knots[j + order] - knots[j + 1]
        v = zero(T)
        if d1 > 0
            v += (order - 1) / d1 * _bspline_basis_one(u, knots, j, order - 1)
        end
        if d2 > 0
            v -= (order - 1) / d2 * _bspline_basis_one(u, knots, j + 1, order - 1)
        end
        row[j] = v
    end
    return row
end

function _bspline_basis_one(u, knots::Vector{Float64}, j::Int, order::Int)
    if order == 1
        if knots[j] <= u < knots[j + 1]
            return one(float(u))
        elseif u == knots[j + 1] && knots[j + 1] == knots[end]
            return one(float(u))
        else
            return zero(float(u))
        end
    end
    v = zero(float(u))
    d1 = knots[j + order - 1] - knots[j]
    if d1 > 0
        v += (u - knots[j]) / d1 * _bspline_basis_one(u, knots, j, order - 1)
    end
    d2 = knots[j + order] - knots[j + 1]
    if d2 > 0
        v += (knots[j + order] - u) / d2 * _bspline_basis_one(u, knots, j + 1, order - 1)
    end
    return v
end

# Design row for the constrained basis: (B(u) − B(0)) with column `drop` removed
function _nested_design_row(ob::NestedOuterBasis, u, row0::Vector{Float64}, k::Int)
    row = _bspline_row_generic(u, ob.knots, 4, k)
    out = similar(row, ob.ncols)
    idx = 1
    for j in 1:k
        j == ob.drop && continue
        out[idx] = row[j] - row0[j]
        idx += 1
    end
    return out
end

# Second-difference penalty on the outer coefficients, with the dropped
# column removed
function _nested_outer_penalty(k::Int, drop::Int)
    D = zeros(k - 2, k)
    for i in 1:(k - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    keep = setdiff(1:k, drop)
    Dk = D[:, keep]
    return Dk' * Dk
end

# ============================================================================
# Unit deviances (AD-generic) and link helpers
# ============================================================================

_nested_linkinv(::IdentityLink, η) = η
_nested_linkinv(::LogLink, η) = exp(η)
_nested_linkinv(::LogitLink, η) = 1.0 / (1.0 + exp(-η))

function _nested_unit_dev(::Normal, y, μ)
    r = y - μ
    return r * r
end
function _nested_unit_dev(::Poisson, y, μ)
    return 2.0 * ((y > 0 ? y * log(y / μ) : zero(μ)) - (y - μ))
end
function _nested_unit_dev(::Union{Bernoulli, Binomial}, y, μ)
    ϵ = 1e-10
    μc = min(max(μ, ϵ), 1 - ϵ)
    t1 = y > 0 ? y * log(y / μc) : zero(μc)
    t2 = y < 1 ? (1 - y) * log((1 - y) / (1 - μc)) : zero(μc)
    return 2.0 * (t1 + t2)
end
function _nested_unit_dev(::Gamma, y, μ)
    return 2.0 * (-log(y / μ) + (y - μ) / μ)
end

_nested_default_link(::Normal) = IdentityLink()
_nested_default_link(::Poisson) = LogLink()
_nested_default_link(::Union{Bernoulli, Binomial}) = LogitLink()
_nested_default_link(::Gamma) = LogLink()

# dμ/dη and the variance function, for the exact chain-rule gradient and the
# Gauss–Newton (expected-information) Hessian
_nested_mueta(::IdentityLink, η) = 1.0
_nested_mueta(::LogLink, η) = exp(η)
function _nested_mueta(::LogitLink, η)
    p = 1.0 / (1.0 + exp(-η))
    return p * (1.0 - p)
end

_nested_varfun(::Normal, μ) = 1.0
_nested_varfun(::Poisson, μ) = max(μ, 1e-10)
_nested_varfun(::Union{Bernoulli, Binomial}, μ) = max(μ * (1.0 - μ), 1e-10)
_nested_varfun(::Gamma, μ) = max(μ * μ, 1e-20)

# ============================================================================
# Model type
# ============================================================================

"""
    NestedGamModel

A GAM with one or more nested effects (`s_nest`), fitted by [`gam_nl`](@ref).
Coefficients are ordered `[parametric; standard-smooth; outer₁; inner₁; …]`
(see the `coef_ranges`, `outer_ranges`, and `inner_ranges` fields). Supports
`coef`, `fitted`, `deviance`, `nulldeviance`, `nobs`, `predict`, and `show`;
[`inner_coef`](@ref) returns the estimated inner parameters.
"""
struct NestedGamModel
    formula::GamFormula
    family::UnivariateDistribution
    link::GLM.Link
    y::Vector{Float64}
    wts::Vector{Float64}                  # prior weights (ones if unweighted)
    offset::Vector{Float64}               # training offset (zeros if none)
    ζ::Vector{Float64}                    # all coefficients
    coef_ranges::Dict{Symbol, UnitRange{Int}}   # :parametric, :smooth
    para_terms::Vector{StatsModels.AbstractTerm}  # schema-applied parametric terms
    smooths::Vector{ConstructedSmooth}    # standard smooths
    X_fixed::Matrix{Float64}              # parametric + standard smooth design
    nested_specs::Vector{SmoothSpec}
    nested_data::Vector{Matrix{Float64}}  # training columns per nested effect
    nested_aux::Vector{Union{Nothing, Matrix{Int}}}  # e.g. mgks neighborhoods
    outer_bases::Vector{NestedOuterBasis}
    inner_ranges::Vector{UnitRange{Int}}  # ranges of a within ζ
    outer_ranges::Vector{UnitRange{Int}}  # ranges of b within ζ
    standardize::Vector{Tuple{Float64, Float64}}  # (μ̂, σ̂) of s̃ at fit
    sp::Vector{Float64}                   # log smoothing parameters
    penalties::Vector{Matrix{Float64}}    # full-ζ penalty matrices, one per sp
    edf_total::Float64
    scale::Float64
    deviance::Float64
    null_deviance::Float64
    Vp::Matrix{Float64}
    converged::Bool
    iterations::Int
end

# ============================================================================
# Fitting
# ============================================================================

"""
    gam_nl(formula, data; family = Normal(), link = nothing,
           weights = nothing, offset = nothing, control...)

Fit a GAM containing nested effects ([`s_nest`](@ref)) alongside ordinary
parametric terms and smooths, following Fasiolo et al. (2025) / gamFactory's
`gam_nl`. All coefficients — including the inner transformation parameters —
are estimated by penalized Newton with the exact chain-rule gradient and a
Gauss–Newton/Fisher (expected-information) Hessian built from the η-Jacobian,
falling back to the exact ForwardDiff Hessian if a Gauss–Newton step fails.
Smoothing parameters (one per standard-smooth penalty, plus one for each
outer spline and each inner parameter vector) are selected by EFS with an
adaptive step multiplier and a conditional-LAML acceptance check.

The stored covariance `Vp` (used by `predict(se = true)`) is the
expected-information Bayesian covariance `φ·(Jη'WJη + S_λ)⁻¹` — mgcv's
convention.

# Arguments
- `family`: `Normal` (default), `Poisson`, `Bernoulli`/`Binomial`, or `Gamma`
- `link`: `IdentityLink`, `LogLink`, or `LogitLink` (default: canonical-ish
  per family — identity, log, logit, log)
- `weights`: optional non-negative prior observation weights
- `offset`: optional known additive term on the link scale (e.g.
  log-exposure); supply the matching `offset` to `predict` as well
- `outer_maxit`, `newton_maxit`, `tol`: iteration controls

Categorical parametric covariates are dummy-coded with a schema built from
the training data, as in `gam()`.

# Example
```julia
m = gam_nl(@formula(y ~ s(x0, k=8) + s_nest(l1, l2, l3, trans=trans_linear())), df)
predict(m, newdata; type = :response)
inner_coef(m)     # estimated index direction
```
"""
function gam_nl(gf::GamFormula, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing,
    outer_maxit::Int = 100, newton_maxit::Int = 200, tol::Float64 = 1e-7)

    family isa Union{Normal, Poisson, Bernoulli, Binomial, Gamma} ||
        throw(ArgumentError("gam_nl supports Normal, Poisson, Bernoulli/Binomial, and Gamma families; got $(typeof(family))"))
    link_eff = link === nothing ? _nested_default_link(family) : link
    link_eff isa Union{IdentityLink, LogLink, LogitLink} ||
        throw(ArgumentError("gam_nl supports IdentityLink, LogLink, and LogitLink"))

    tbl = Tables.columntable(data)
    y = Float64.(Tables.getcolumn(tbl, gf.response))
    n = length(y)
    _validate_response(y, family)

    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(ArgumentError(
        "offset length $(length(off)) does not match number of observations $n"))
    wt = weights === nothing ? ones(n) : Float64.(weights)
    length(wt) == n || throw(ArgumentError(
        "weights length $(length(wt)) does not match number of observations $n"))
    all(>=(0.0), wt) || throw(ArgumentError("weights must be non-negative"))
    n_eff_obs = count(>(0.0), wt)

    nested_specs = [sp for sp in gf.smooth_specs if sp.basis isa NestedBasis]
    std_specs = [sp for sp in gf.smooth_specs if !(sp.basis isa NestedBasis)]
    isempty(nested_specs) && throw(ArgumentError(
        "gam_nl requires at least one s_nest term; use gam() otherwise"))

    # ── Standard part: parametric columns + constructed smooths ─────────────
    # Parametric terms are schema-applied against the training data so that
    # categorical/string covariates are dummy-coded, as in gam().
    gf_std = GamFormula(gf.response, gf.parametric, gf.has_intercept, std_specs)
    para_terms = collect(StatsModels.AbstractTerm,
        _apply_parametric_schema(
            StatsModels.AbstractTerm[Term(v) for v in gf.parametric], tbl))
    X_para = _nested_parametric_matrix(para_terms, gf.has_intercept, tbl, n)
    smooths = ConstructedSmooth[]
    Xs_blocks = Matrix{Float64}[]
    for sp in std_specs
        sm = smooth_construct(sp, tbl)
        push!(smooths, sm)
        push!(Xs_blocks, sm.X)
    end
    X_fixed = isempty(Xs_blocks) ? X_para : hcat(X_para, Xs_blocks...)
    p_para = size(X_para, 2)
    p_std = size(X_fixed, 2) - p_para

    # ── Nested blocks ──────────────────────────────────────────────────────
    n_eff = length(nested_specs)
    nested_data = Matrix{Float64}[]
    nested_aux = Union{Nothing, Matrix{Int}}[]
    outer_bases = NestedOuterBasis[]
    for sp in nested_specs
        cols = [Float64.(Tables.getcolumn(tbl, v)) for v in sp.term_vars]
        Xin = reduce(hcat, cols)
        push!(nested_data, Xin)
        t = sp.basis.trans
        if t isa TransMGKS && t.nn > 0
            C = Xin[:, 2:end]
            push!(nested_aux, _mgks_neighbors(C, C, t.nn; exclude_self = true))
        else
            push!(nested_aux, nothing)
        end
        push!(outer_bases, _nested_outer_basis(sp.k))
    end

    # Coefficient layout: [γ (p_para) | β_std (p_std) | b₁,a₁ | b₂,a₂ | …]
    col_off = p_para + p_std
    outer_ranges = UnitRange{Int}[]
    inner_ranges = UnitRange{Int}[]
    for (j, sp) in enumerate(nested_specs)
        nb = outer_bases[j].ncols
        na = _n_inner(sp.basis.trans, size(nested_data[j], 2))
        na >= 1 || throw(ArgumentError("$(sp.label): transform needs at least one inner parameter"))
        push!(outer_ranges, (col_off + 1):(col_off + nb))
        col_off += nb
        push!(inner_ranges, (col_off + 1):(col_off + na))
        col_off += na
    end
    p_tot = col_off

    # ── Penalties (full-ζ matrices, one per smoothing parameter) ───────────
    penalties = Matrix{Float64}[]
    pen_ranks = Float64[]
    if !isempty(std_specs)
        pen_std = setup_penalties(smooths, p_para)
        for block in pen_std.blocks
            for S in block.S
                Sfull = zeros(p_tot, p_tot)
                idx = block.start:block.stop
                Sfull[idx, idx] .= S
                push!(penalties, Sfull)
                push!(pen_ranks, Float64(rank(S)))
            end
        end
    end
    for (j, sp) in enumerate(nested_specs)
        S_out = _nested_outer_penalty(sp.k, outer_bases[j].drop)
        Sfull = zeros(p_tot, p_tot)
        Sfull[outer_ranges[j], outer_ranges[j]] .= S_out
        push!(penalties, Sfull)
        push!(pen_ranks, Float64(rank(S_out)))
        S_in = _inner_penalty(sp.basis.trans, length(inner_ranges[j]))
        Sfull2 = zeros(p_tot, p_tot)
        Sfull2[inner_ranges[j], inner_ranges[j]] .= S_in
        push!(penalties, Sfull2)
        push!(pen_ranks, Float64(rank(S_in)))
    end
    nsp = length(penalties)
    log_sp = zeros(nsp)
    # honor user-fixed sp on nested terms (applies to the outer penalty)
    sp_fixed = falses(nsp)
    # For trans_linear effects the index SCALE is unidentified (standardized
    # away), so their inner-ridge EFS update must work on the normalized
    # direction: bSb evaluated at a/‖a‖ and rank reduced by the one
    # unidentified dimension — otherwise λ runs away to the boundary,
    # over-penalizing the direction itself.
    scale_free_pen = falses(nsp)
    scale_free_rng = Dict{Int, UnitRange{Int}}()
    si = nsp - 2 * n_eff
    for (j, spc) in enumerate(nested_specs)
        if spc.sp !== nothing
            log_sp[si + 2 * (j - 1) + 1] = log(spc.sp)
            sp_fixed[si + 2 * (j - 1) + 1] = true
        end
        if spc.basis.trans isa TransLinear
            g_in = si + 2 * (j - 1) + 2
            scale_free_pen[g_in] = true
            scale_free_rng[g_in] = inner_ranges[j]
        end
    end

    # Penalty coordinate supports and overlap groups, for the generalized
    # log|S_λ|₊ in the sp-acceptance score. Penalties on disjoint coordinate
    # blocks contribute rank·log λ (up to a λ-independent constant that
    # cancels in comparisons); overlapping groups (e.g. a tensor smooth's
    # margins in the standard part) get a small local eigendecomposition.
    supports = [findall(i -> any(!iszero, view(penalties[g], :, i)), 1:p_tot)
                for g in 1:nsp]
    group_of = collect(1:nsp)
    for g1 in 1:nsp, g2 in (g1 + 1):nsp
        if !isempty(intersect(supports[g1], supports[g2]))
            old_id, new_id = group_of[g2], group_of[g1]
            for g in 1:nsp
                group_of[g] == old_id && (group_of[g] = new_id)
            end
        end
    end
    pen_groups = [findall(==(gid), group_of) for gid in unique(group_of)]
    group_idx = [sort(union([supports[g] for g in gs]...)) for gs in pen_groups]

    function _logdetS(lsp)
        v = 0.0
        for (gs, idx) in zip(pen_groups, group_idx)
            if length(gs) == 1
                v += pen_ranks[gs[1]] * lsp[gs[1]]
            else
                Sl = zeros(length(idx), length(idx))
                for g in gs
                    Sl .+= exp(lsp[g]) .* penalties[g][idx, idx]
                end
                ev = eigvals(Symmetric(Sl))
                thr = maximum(abs, ev) * 1e-12
                for e in ev
                    e > thr && (v += log(e))
                end
            end
        end
        return v
    end

    # ── Objective: 0.5·Σ wᵢ·unit_dev + 0.5·ζ\'S_λ ζ  (deviance scale) ──────
    trans_list = NestedTransform[sp.basis.trans for sp in nested_specs]
    row0s = [_bspline_row_interior(0.0, ob.knots, 4, nested_specs[j].k)
             for (j, ob) in enumerate(outer_bases)]
    k_outer = [sp.k for sp in nested_specs]

    function _train_st(j, a)
        t = trans_list[j]
        if t isa TransMGKS
            return _mgks_smooth(a, nested_data[j],
                view(nested_data[j], :, 2:size(nested_data[j], 2)),
                view(nested_data[j], :, 1);
                exclude_self = true, neigh = nested_aux[j])
        end
        return _trans_eval(t, a, nested_data[j])
    end

    function _eta(ζ)
        η = X_fixed * ζ[1:(p_para + p_std)]
        η .+= off
        for j in 1:n_eff
            a = ζ[inner_ranges[j]]
            b = ζ[outer_ranges[j]]
            st = _train_st(j, a)
            μs = sum(st) / n
            σs = sqrt(sum(abs2, st .- μs) / n + 1e-12)
            ob = outer_bases[j]
            for i in 1:n
                u = (st[i] - μs) / σs
                row = _nested_design_row(ob, u, row0s[j], k_outer[j])
                η[i] += dot(row, b)
            end
        end
        return η
    end

    function _dev(ζ)
        η = _eta(ζ)
        d = zero(eltype(η))
        for i in 1:n
            μ = _nested_linkinv(link_eff, η[i])
            d += wt[i] * _nested_unit_dev(family, y[i], μ)
        end
        return d
    end

    function _pen_obj(ζ, lsp)
        obj = 0.5 * _dev(ζ)
        for g in 1:nsp
            obj += 0.5 * exp(lsp[g]) * dot(ζ, penalties[g] * ζ)
        end
        return obj
    end

    # Exact chain-rule gradient and Gauss–Newton/Fisher Hessian:
    #   grad = Jη'gη + S_λζ  (exact),  H = Jη'WJη + S_λ  (expected information)
    # with gη,i = −wᵢ(yᵢ−μᵢ)(dμ/dη)ᵢ/V(μᵢ) and Wᵢᵢ = wᵢ(dμ/dη)ᵢ²/V(μᵢ).
    # One n×p Jacobian per call — an order of magnitude cheaper than AD of
    # the full Hessian at moderate p.
    function _gn_parts(ζ, lsp)
        η = _eta(ζ)
        Jη = ForwardDiff.jacobian(_eta, ζ)
        gη = Vector{Float64}(undef, n)
        w_gn = Vector{Float64}(undef, n)
        for i in 1:n
            μ = _nested_linkinv(link_eff, η[i])
            dμ = _nested_mueta(link_eff, η[i])
            V = _nested_varfun(family, μ)
            gη[i] = -wt[i] * (y[i] - μ) * dμ / V
            w_gn[i] = wt[i] * dμ * dμ / V
        end
        Sλ = zeros(p_tot, p_tot)
        for g in 1:nsp
            Sλ .+= exp(lsp[g]) .* penalties[g]
        end
        JtWJ = Jη' * (Diagonal(w_gn) * Jη)
        grad = Jη' * gη .+ Sλ * ζ
        H = JtWJ .+ Sλ
        return grad, H, JtWJ
    end

    function _ridge_chol(H0)
        F = _safe_cholesky(Symmetric(H0))
        F !== nothing && return F
        base = max(1e-6 * maximum(abs, diag(H0)), 1e-8)
        for attempt in 0:8
            Hp = copy(H0)
            for i in 1:p_tot
                Hp[i, i] += base * 10.0^attempt
            end
            F = _safe_cholesky(Symmetric(Hp))
            F !== nothing && return F
        end
        return nothing
    end

    # Conditional LAML-like acceptance score for sp steps, at fixed ζ and
    # fixed working information JtWJ (mirrors the core's _conditional_reml):
    #   V(λ) = pen_obj(ζ,λ)/φ + ½log|JtWJ + S_λ| − ½log|S_λ|₊
    function _cond_score(lsp, JtWJ, φ)
        Sλ = zeros(p_tot, p_tot)
        for g in 1:nsp
            Sλ .+= exp(lsp[g]) .* penalties[g]
        end
        F = _ridge_chol(JtWJ .+ Sλ)
        F === nothing && return Inf
        ld = 2.0 * sum(log, diag(F.U))
        return _pen_obj(ζ, lsp) / φ + 0.5 * ld - 0.5 * _logdetS(lsp)
    end

    # ── Initialization ─────────────────────────────────────────────────────
    ζ = zeros(p_tot)
    if p_para + p_std > 0 && !isempty(std_specs)
        # Warm start γ/β from the standard-part fit
        m0 = try
            gam(gf_std, data; family = family, link = link_eff,
                weights = weights, offset = offset)
        catch
            nothing
        end
        if m0 !== nothing && length(coef(m0)) == p_para + p_std
            ζ[1:(p_para + p_std)] .= coef(m0)
        end
    elseif p_para > 0 && gf.has_intercept
        # Offset-aware null intercept (positive-domain clamps only where the
        # link needs them, inside the helper)
        ζ[1] = _nested_null_intercept(family, link_eff, y, wt, off)
    end
    for j in 1:n_eff
        ζ[inner_ranges[j]] .= _default_inner_start(trans_list[j],
            length(inner_ranges[j]))
    end

    # ── EFS outer loop with penalized Gauss–Newton inner loop ──────────────
    S_λ = zeros(p_tot, p_tot)
    converged = false
    iterations = 0
    efs_mult = 1.0
    score_prev = Inf
    for outer in 1:outer_maxit
        iterations = outer

        # inner Newton at current λ (Gauss–Newton H, exact-AD fallback)
        for it in 1:newton_maxit
            obj0 = _pen_obj(ζ, log_sp)
            grad, Hgn, _ = _gn_parts(ζ, log_sp)
            maximum(abs, grad) < tol * (1.0 + abs(obj0)) && break
            stepped = false
            small_improve = false
            for hess_try in 1:2
                Hcur = hess_try == 1 ? Hgn :
                       ForwardDiff.hessian(z -> _pen_obj(z, log_sp), ζ)
                F = _ridge_chol(Hcur)
                F === nothing && continue
                δ = F \ grad
                step = 1.0
                ζ_new = ζ .- step .* δ
                obj1 = _pen_obj(ζ_new, log_sp)
                halves = 0
                while (!isfinite(obj1) || obj1 > obj0) && halves < 30
                    step *= 0.5
                    ζ_new = ζ .- step .* δ
                    obj1 = _pen_obj(ζ_new, log_sp)
                    halves += 1
                end
                if isfinite(obj1) && obj1 <= obj0
                    ζ .= ζ_new
                    stepped = true
                    small_improve = (obj0 - obj1) < tol * (abs(obj0) + 0.1)
                    break
                end
            end
            stepped || break
            small_improve && break
        end

        # EFS update with adaptive step multiplier + conditional acceptance
        grad, Hgn, JtWJ = _gn_parts(ζ, log_sp)
        F = _ridge_chol(Hgn)
        F === nothing && break
        Ainv = inv(F)

        fill!(S_λ, 0.0)
        for g in 1:nsp
            S_λ .+= exp(log_sp[g]) .* penalties[g]
        end
        edf = p_tot - tr(Ainv * S_λ)
        dev_cur = _dev(ζ)
        φ = family isa Union{Poisson, Bernoulli, Binomial} ? 1.0 :
            max(dev_cur / max(n_eff_obs - edf, 1.0), 1e-10)

        target = copy(log_sp)
        for g in 1:nsp
            sp_fixed[g] && continue
            bSb = dot(ζ, penalties[g] * ζ)
            rank_g = pen_ranks[g]
            if scale_free_pen[g]
                a_g = ζ[scale_free_rng[g]]
                bSb /= max(dot(a_g, a_g), 1e-300)   # direction-only curvature
                rank_g -= 1.0                       # scale dim is unidentified
            end
            trAS = exp(log_sp[g]) * sum(Ainv .* penalties[g])
            num = max(0.0, rank_g - trAS)
            if num > 0 && bSb > 1e-12
                target[g] = clamp(log(max(φ * num / bSb, 1e-15)), -12.0, 12.0)
            end
        end
        log_sp_new = log_sp .+ efs_mult .* (target .- log_sp)
        max_change = maximum(abs.(log_sp_new .- log_sp))

        score_cur = _cond_score(log_sp, JtWJ, φ)
        if max_change > 1e-10
            score_new = _cond_score(log_sp_new, JtWJ, φ)
            if score_new > score_cur + 1e-7 * abs(score_cur)
                for _halve in 1:4
                    efs_mult *= 0.5
                    log_sp_new = log_sp .+ efs_mult .* (target .- log_sp)
                    score_new = _cond_score(log_sp_new, JtWJ, φ)
                    score_new <= score_cur + 1e-7 * abs(score_cur) && break
                end
                max_change = maximum(abs.(log_sp_new .- log_sp))
            else
                efs_mult = min(1.0, efs_mult * 2.0)
            end
        end
        log_sp .= log_sp_new

        # Convergence: sp movement small, or the conditional score flat
        # (flat-ridge protection, as in the core EFS loops)
        if max_change < 1e-3 ||
           (outer > 1 && abs(score_cur - score_prev) <
                         1e-7 * (abs(score_prev) + 0.1))
            converged = true
            break
        end
        score_prev = score_cur
    end

    # ── Post-fit ‖a‖ = 1 reparameterization for single-index effects ───────
    # η is invariant to rescaling a trans_linear index (up to the tiny
    # standardization regularizer), so normalize for a well-conditioned
    # delta method, then polish with a few Newton steps at the new
    # parameterization; the final covariance is computed there.
    renormed = false
    for j in 1:n_eff
        trans_list[j] isa TransLinear || continue
        s_a = norm(ζ[inner_ranges[j]])
        if s_a > 1e-10 && abs(s_a - 1.0) > 1e-12
            ζ[inner_ranges[j]] ./= s_a
            renormed = true
        end
    end
    if renormed
        for it in 1:20
            obj0 = _pen_obj(ζ, log_sp)
            grad, Hgn, _ = _gn_parts(ζ, log_sp)
            maximum(abs, grad) < tol * (1.0 + abs(obj0)) && break
            F = _ridge_chol(Hgn)
            F === nothing && break
            δ = F \ grad
            step = 1.0
            ζ_new = ζ .- step .* δ
            obj1 = _pen_obj(ζ_new, log_sp)
            halves = 0
            while (!isfinite(obj1) || obj1 > obj0) && halves < 30
                step *= 0.5
                ζ_new = ζ .- step .* δ
                obj1 = _pen_obj(ζ_new, log_sp)
                halves += 1
            end
            (isfinite(obj1) && obj1 <= obj0) || break
            # keep the unit-norm parameterization exactly
            ζ .= ζ_new
            for j in 1:n_eff
                trans_list[j] isa TransLinear || continue
                s_a = norm(ζ[inner_ranges[j]])
                s_a > 1e-10 && (ζ[inner_ranges[j]] ./= s_a)
            end
            (obj0 - obj1) < tol * (abs(obj0) + 0.1) && break
        end
    end

    # ── Final quantities (expected-information covariance) ─────────────────
    grad_f, H_f, _ = _gn_parts(ζ, log_sp)
    F = _ridge_chol(H_f)
    F === nothing && error("gam_nl: final Hessian could not be stabilized")
    Ainv = inv(F)
    fill!(S_λ, 0.0)
    for g in 1:nsp
        S_λ .+= exp(log_sp[g]) .* penalties[g]
    end
    edf_total = p_tot - tr(Ainv * S_λ)
    dev_final = _dev(ζ)
    φ = family isa Union{Poisson, Bernoulli, Binomial} ? 1.0 :
        max(dev_final / max(n_eff_obs - edf_total, 1.0), 1e-10)
    Vp = Symmetric(Ainv .* φ) |> Matrix

    # standardization constants at the optimum, for prediction
    standardize = Tuple{Float64, Float64}[]
    for j in 1:n_eff
        st = _train_st(j, ζ[inner_ranges[j]])
        μs = sum(st) / n
        σs = sqrt(sum(abs2, st .- μs) / n + 1e-12)
        push!(standardize, (μs, σs))
    end

    null_dev = _nested_null_deviance(family, link_eff, y, wt, off)

    coef_ranges = Dict{Symbol, UnitRange{Int}}(
        :parametric => 1:p_para, :smooth => (p_para + 1):(p_para + p_std))

    return NestedGamModel(gf, family, link_eff, y, wt, off, ζ, coef_ranges,
        para_terms, smooths, X_fixed, nested_specs, nested_data, nested_aux,
        outer_bases, inner_ranges, outer_ranges, standardize, log_sp,
        penalties, edf_total, φ, dev_final, null_dev, Vp, converged,
        iterations)
end

"""Parametric design matrix from schema-applied terms (intercept first)."""
function _nested_parametric_matrix(
    para_terms::AbstractVector{<:StatsModels.AbstractTerm},
    has_intercept::Bool, tbl, n::Int)
    X = has_intercept ? ones(n, 1) : Matrix{Float64}(undef, n, 0)
    for pt in para_terms
        X = hcat(X, _term_matrix(pt, tbl))
    end
    return X
end

"""Intercept of the intercept-only model with the given offset and weights,
by 1-D Fisher scoring (used for initialization and the null deviance)."""
function _nested_null_intercept(family, link, y, wt, off)
    μ0 = sum(wt .* y) / sum(wt)
    β0 = link isa IdentityLink ? μ0 :
         link isa LogLink ? log(max(μ0, 1e-8)) :
         GLM.linkfun(link, clamp(μ0, 1e-8, 1.0 - 1e-8))
    all(iszero, off) && link isa IdentityLink && return β0
    for _ in 1:100
        g = 0.0
        h = 0.0
        for i in eachindex(y)
            η = β0 + off[i]
            μ = _nested_linkinv(link, η)
            dμ = _nested_mueta(link, η)
            V = _nested_varfun(family, μ)
            g -= wt[i] * (y[i] - μ) * dμ / V
            h += wt[i] * dμ * dμ / V
        end
        step = g / max(h, 1e-12)
        β0 -= step
        abs(step) < 1e-10 && break
    end
    return β0
end

"""Null deviance: intercept-only model, honoring weights and (via 1-D Fisher
scoring) any offset — matching the intercept-plus-offset null convention of
mgcv/R's glm."""
function _nested_null_deviance(family, link, y, wt, off)
    if all(iszero, off)
        μ̄ = sum(wt .* y) / sum(wt)
        return sum(wt[i] * _nested_unit_dev(family, y[i], μ̄)
                   for i in eachindex(y))
    end
    β0 = _nested_null_intercept(family, link, y, wt, off)
    return sum(wt[i] * _nested_unit_dev(family, y[i],
        _nested_linkinv(link, β0 + off[i])) for i in eachindex(y))
end

"""Convert a plain StatsModels formula (possibly containing `s_nest`/`s`
FunctionTerms) into a `GamFormula` without constructing any smooth."""
function _nested_formula_to_gamformula(f::StatsModels.FormulaTerm)
    response = f.lhs isa Term ? f.lhs.sym :
               throw(ArgumentError("gam_nl requires a single response variable"))
    rhs = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    parametric = Symbol[]
    specs = SmoothSpec[]
    has_intercept = true
    for t in rhs
        if t isa StatsModels.InterceptTerm{false} ||
           (t isa ConstantTerm && iszero(t.n))
            has_intercept = false
        elseif t isa StatsModels.InterceptTerm || t isa ConstantTerm
            has_intercept = true
        elseif t isa Term
            push!(parametric, t.sym)
        elseif t isa StatsModels.FunctionTerm && _is_smooth_function(t.f)
            push!(specs, _functionterm_to_smoothspec(t))
        else
            throw(ArgumentError("gam_nl cannot handle formula term $t; use @formulak"))
        end
    end
    return GamFormula(response, parametric, has_intercept, specs)
end

gam_nl(f::StatsModels.FormulaTerm, data; kwargs...) =
    gam_nl(_nested_formula_to_gamformula(f), data; kwargs...)

"""Whether a plain-formula RHS contains an `s_nest` FunctionTerm.

Only top-level RHS terms are scanned: an `s_nest` nested inside an
interaction or other composite term is not routed here and falls through to
the standard pipeline, which rejects it with a clear error."""
function _formula_has_nested(f::StatsModels.FormulaTerm)
    rhs = f.rhs isa Tuple ? f.rhs : (f.rhs,)
    return any(t -> t isa StatsModels.FunctionTerm && t.f === s_nest, rhs)
end

# ============================================================================
# Prediction and accessors
# ============================================================================

"""Training-style inner-transform evaluation for effect `j` at parameters
`a`, using the stored data and (for mgks) the fixed training neighborhoods."""
function _model_train_st(m::NestedGamModel, j::Int, a::AbstractVector)
    t = m.nested_specs[j].basis.trans
    if t isa TransMGKS
        return _mgks_smooth(a, m.nested_data[j],
            view(m.nested_data[j], :, 2:size(m.nested_data[j], 2)),
            view(m.nested_data[j], :, 1);
            exclude_self = true, neigh = m.nested_aux[j])
    end
    return _trans_eval(t, a, m.nested_data[j])
end

# η at new data as a function of the coefficient vector ζ (AD-generic in ζ,
# for delta-method standard errors). Design pieces that do not depend on ζ
# (parametric/standard-smooth matrices, mgks prediction neighborhoods) are
# passed in precomputed.
function _nested_eta_newdata_ζ(m::NestedGamModel, ζ::AbstractVector,
    X_lin::Matrix{Float64}, Xins::Vector{Matrix{Float64}},
    pred_neighs::Vector{Union{Nothing, Matrix{Int}}}, n_new::Int)
    η = X_lin * ζ[1:(m.coef_ranges[:smooth].stop)]
    for (j, sp) in enumerate(m.nested_specs)
        a = ζ[m.inner_ranges[j]]
        t = sp.basis.trans
        st = t isa TransMGKS ?
             _mgks_smooth(a, Xins[j],
                view(m.nested_data[j], :, 2:size(m.nested_data[j], 2)),
                view(m.nested_data[j], :, 1);
                exclude_self = false, neigh = pred_neighs[j]) :
             _trans_eval(t, a, Xins[j])
        # Standardization constants recomputed from the training data as a
        # function of `a` (matching the fitting objective): at ζ = ζ̂ they
        # equal the stored values, and differentiating through them keeps the
        # delta-method Jacobian invariant to the standardization-unidentified
        # directions of the inner parameters.
        st_tr = _model_train_st(m, j, a)
        ntr = length(st_tr)
        μs = sum(st_tr) / ntr
        σs = sqrt(sum(abs2, st_tr .- μs) / ntr + 1e-12)
        b = ζ[m.outer_ranges[j]]
        ob = m.outer_bases[j]
        row0 = _bspline_row_interior(0.0, ob.knots, 4, sp.k)
        for i in 1:n_new
            u = (st[i] - μs) / σs
            η[i] += dot(_nested_design_row(ob, u, row0, sp.k), b)
        end
    end
    return η
end

# Precompute the ζ-independent prediction pieces for a new table
function _nested_predict_pieces(m::NestedGamModel, tbl, n_new::Int)
    X_para = _nested_parametric_matrix(m.para_terms,
        m.formula.has_intercept, tbl, n_new)
    Xs = Matrix{Float64}[predict_matrix(sm, tbl) for sm in m.smooths]
    X_lin = isempty(Xs) ? X_para : hcat(X_para, Xs...)
    Xins = Matrix{Float64}[]
    pred_neighs = Union{Nothing, Matrix{Int}}[]
    for sp in m.nested_specs
        cols = [Float64.(Tables.getcolumn(tbl, v)) for v in sp.term_vars]
        Xin = reduce(hcat, cols)
        push!(Xins, Xin)
        t = sp.basis.trans
        if t isa TransMGKS && t.nn > 0
            j = length(Xins)
            Ctr = m.nested_data[j][:, 2:end]
            push!(pred_neighs, _mgks_neighbors(Xin[:, 2:end], Ctr, t.nn;
                exclude_self = false))
        else
            push!(pred_neighs, nothing)
        end
    end
    return X_lin, Xins, pred_neighs
end

"""
    predict(m::NestedGamModel, newdata; type = :link, se = false, offset = nothing)

Predictions from a nested-effect GAM. `type = :link` (default) or
`:response`. `offset` supplies the known additive link-scale term for the
new observations when the model was fitted with one. With `se = true`,
returns `(predictions, standard_errors)` where the standard errors propagate
the joint uncertainty of all coefficients — inner transformation parameters
included — through the composition by the delta method: `se² = diag(J Vp J')`
with `J = ∂η/∂ζ` computed by ForwardDiff and `Vp` the expected-information
covariance (response-scale errors are scaled by `|dμ/dη|`). The offset does
not affect the standard errors.

Note (`trans_mgks` models): every row of `newdata` is treated as a new
location and smoothed over *all* stored training points, so `predict` on the
training table includes each point's own value and differs from `fitted`,
which uses the leave-one-out smoother of the fitting objective.
"""
function StatsAPI.predict(m::NestedGamModel, newdata; type::Symbol = :link,
    se::Bool = false, offset::Union{AbstractVector{<:Real}, Nothing} = nothing)
    type in (:link, :response) ||
        throw(ArgumentError("type must be :link or :response, got :$type"))
    tbl = Tables.columntable(newdata)
    n_new = length(Tables.getcolumn(tbl, first(Tables.columnnames(tbl))))
    off_new = offset === nothing ? nothing : Float64.(offset)
    off_new === nothing || length(off_new) == n_new || throw(ArgumentError(
        "offset length $(length(off_new)) does not match number of rows $n_new"))
    X_lin, Xins, pred_neighs = _nested_predict_pieces(m, tbl, n_new)
    η = _nested_eta_newdata_ζ(m, m.ζ, X_lin, Xins, pred_neighs, n_new)
    off_new === nothing || (η .+= off_new)
    μ = [_nested_linkinv(m.link, e) for e in η]
    pred = type == :link ? η : μ
    se || return pred
    J = ForwardDiff.jacobian(
        z -> _nested_eta_newdata_ζ(m, z, X_lin, Xins, pred_neighs, n_new), m.ζ)
    se_eta = [sqrt(max(dot(view(J, i, :), m.Vp * view(J, i, :)), 0.0))
              for i in 1:n_new]
    if type == :response
        dμdη = [abs(GLM.mueta(m.link, e)) for e in η]
        return pred, se_eta .* dμdη
    end
    return pred, se_eta
end

StatsAPI.coef(m::NestedGamModel) = m.ζ
StatsAPI.deviance(m::NestedGamModel) = m.deviance
StatsAPI.nulldeviance(m::NestedGamModel) = m.null_deviance
StatsAPI.nobs(m::NestedGamModel) = length(m.y)

function StatsAPI.fitted(m::NestedGamModel)
    η = m.X_fixed * m.ζ[1:(m.coef_ranges[:smooth].stop)]
    η .+= m.offset
    n = length(m.y)
    for (j, sp) in enumerate(m.nested_specs)
        a = m.ζ[m.inner_ranges[j]]
        st = _model_train_st(m, j, a)
        μs, σs = m.standardize[j]
        b = m.ζ[m.outer_ranges[j]]
        ob = m.outer_bases[j]
        row0 = _bspline_row_interior(0.0, ob.knots, 4, sp.k)
        for i in 1:n
            u = (st[i] - μs) / σs
            η[i] += dot(_nested_design_row(ob, u, row0, sp.k), b)
        end
    end
    return [_nested_linkinv(m.link, e) for e in η]
end

"""
    inner_coef(m::NestedGamModel, j::Int = 1)

The estimated inner-transformation parameters `a` of the `j`-th nested
effect. For `trans_linear` the index direction is identified only up to
scale and sign (the inner output is standardized); the fitter already
reparameterizes the stored coefficients to unit norm post-fit, and this
accessor (defensively) re-normalizes and fixes the sign so the
largest-magnitude entry is positive.
"""
function inner_coef(m::NestedGamModel, j::Int = 1)
    a = copy(m.ζ[m.inner_ranges[j]])
    if m.nested_specs[j].basis.trans isa TransLinear
        a ./= (norm(a) + 1e-300)
        if a[findmax(abs.(a))[2]] < 0
            a .= -a
        end
    end
    return a
end

deviance_explained(m::NestedGamModel) = 1.0 - m.deviance / m.null_deviance

function Base.show(io::IO, m::NestedGamModel)
    println(io, "NestedGamModel ($(nameof(typeof(m.family))), $(nameof(typeof(m.link))))")
    println(io, "  Formula: ", m.formula)
    println(io, "  Standard smooths: $(length(m.smooths))   Nested effects: $(length(m.nested_specs))")
    for sp in m.nested_specs
        t = sp.basis.trans
        tname = t isa TransLinear ? "trans_linear" :
                t isa TransExpSmooth ? "trans_nexpsm" : "trans_mgks"
        println(io, "    ", sp.label, "  [", tname, ", k=", sp.k, "]")
    end
    println(io, "  Deviance: ", round(m.deviance; digits = 4),
        "   Deviance explained: ", round(100 * deviance_explained(m); digits = 1), "%")
    println(io, "  Total EDF: ", round(m.edf_total; digits = 2),
        "   Scale: ", round(m.scale; digits = 5))
    print(io, "  Converged: ", m.converged, " (", m.iterations, " outer iterations)")
end
