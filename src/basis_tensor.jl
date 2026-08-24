# Tensor product smooths — te() and ti()
#
# Implements tensor product basis construction following Wood (2006, §4.1.8).
# The model matrix is the row-wise Kronecker product of marginal bases.
# Penalties are Sⱼ ⊗ (⊗_{i≠j} Iₖᵢ), one per marginal dimension.
#
# For te(), marginal bases are constructed WITHOUT absorbing identifiability
# constraints; a single overall sum-to-zero constraint is applied to the full
# tensor product. For ti(), a sum-to-zero constraint is absorbed into EACH
# marginal before forming the tensor product (mgcv's mc=TRUE convention), so
# the ti() span excludes the constant and all marginal main effects.

"""
    RawMarginalBasis

Unconstrained marginal basis for use in tensor product construction.
Contains the raw basis matrix, penalty, knot locations, and null space info.
"""
struct RawMarginalBasis
    X::Matrix{Float64}
    S::Vector{Matrix{Float64}}
    null_dim::Int
    knots::Vector{Float64}
    spec::SmoothSpec
    Ain::Union{Matrix{Float64}, Nothing}
    bin::Union{Vector{Float64}, Nothing}
    Aeq::Union{Matrix{Float64}, Nothing}
    beq::Union{Vector{Float64}, Nothing}
    template::ConstructedSmooth
end

function RawMarginalBasis(X::Matrix{Float64}, S::Vector{Matrix{Float64}},
                          null_dim::Int, knots::Vector{Float64}, spec::SmoothSpec;
                          Ain = nothing,
                          bin = nothing,
                          Aeq = nothing,
                          beq = nothing,
                          constraint = nothing,
                          Sigma = nothing,
                          cmX = nothing,
                          p_ident = nothing,
                          rank::Union{Int, Nothing} = nothing,
                          template::Union{ConstructedSmooth, Nothing} = nothing)
    if template === nothing
        rank_val = rank === nothing ? max(size(X, 2) - null_dim, 0) : rank
        template = ConstructedSmooth(
            spec, X, S, knots, null_dim, rank_val,
            constraint, nothing, 0, 0,
            Sigma, cmX, p_ident,
            Int[],
            Ain, bin, Aeq, beq,
        )
    end
    return RawMarginalBasis(X, S, null_dim, knots, spec, Ain, bin, Aeq, beq, template)
end

function _embed_tensor_constraint(A::Matrix{Float64}, pos::Int, marginal_dims::Vector{Int})
    P = Matrix{Float64}(I, 1, 1)
    for i in 1:length(marginal_dims)
        if i == pos
            P = kron(P, A)
        else
            P = kron(P, Matrix{Float64}(I, marginal_dims[i], marginal_dims[i]))
        end
    end
    return P
end

function _repeat_tensor_rhs(b::Vector{Float64}, pos::Int, marginal_dims::Vector{Int})
    inner = pos < length(marginal_dims) ? prod(marginal_dims[(pos + 1):end]) : 1
    outer = pos > 1 ? prod(marginal_dims[1:(pos - 1)]) : 1
    return repeat(b; inner = inner, outer = outer)
end

function _merge_tensor_constraint_blocks(raw_marginals::Vector{RawMarginalBasis},
                                         marginal_dims::Vector{Int},
                                         which::Symbol)
    A_merged = nothing
    b_merged = nothing
    rhs_field = which === :Ain ? :bin : :beq
    for (i, rm) in enumerate(raw_marginals)
        A = getfield(rm, which)
        b = getfield(rm, rhs_field)
        if A !== nothing && b !== nothing && size(A, 1) > 0
            A_full = _embed_tensor_constraint(A, i, marginal_dims)
            b_full = _repeat_tensor_rhs(b, i, marginal_dims)
            A_merged, b_merged = _append_constraint_block(A_merged, b_merged, A_full, b_full)
        end
    end
    return A_merged, b_merged
end

"""
    _row_kronecker(matrices::Vector{Matrix{Float64}}) -> Matrix{Float64}

Row-wise Kronecker product. Given X₁ (n×k₁), X₂ (n×k₂), ..., returns
n × (k₁*k₂*...) where row i = kron(X₁[i,:], X₂[i,:], ...).
"""
function _row_kronecker(matrices::Vector{Matrix{Float64}})
    length(matrices) >= 1 || throw(ArgumentError("Need at least one matrix"))
    n = size(matrices[1], 1)
    all(m -> size(m, 1) == n, matrices) ||
        throw(ArgumentError("All matrices must have the same number of rows"))

    result = matrices[1]
    for j in 2:length(matrices)
        Xj = matrices[j]
        k1 = size(result, 2)
        k2 = size(Xj, 2)
        new_result = zeros(n, k1 * k2)
        for i in 1:n
            new_result[i, :] .= kron(result[i, :], Xj[i, :])
        end
        result = new_result
    end
    return result
end

"""
    _build_raw_marginal(spec::SmoothSpec, data, user_knots) -> RawMarginalBasis

Build an unconstrained marginal basis. Dispatches on the basis type to call
the appropriate low-level basis constructor without applying absorb_constraints!.
"""
function _build_raw_marginal(spec::SmoothSpec, data, user_knots)
    return _build_raw_marginal(spec.basis, spec, data, user_knots)
end

# CR spline marginal (most common case for tensor products)
function _build_raw_marginal(::CubicSpline, spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    knots = user_knots !== nothing ? Float64.(user_knots) : place_knots(x, k)
    k = length(knots)
    X, S = _cr_basis(x, knots)
    return RawMarginalBasis(X, [S], 2, knots, spec)
end

function _build_raw_marginal(::CubicShrink, spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    knots = user_knots !== nothing ? Float64.(user_knots) : place_knots(x, k)
    k = length(knots)
    X, S = _cr_basis(x, knots)
    # Single modified penalty (mgcv cs), consistent with _construct_cr
    S = _shrink_penalty(S)
    return RawMarginalBasis(X, [S], 0, knots, spec)
end

function _build_raw_marginal(::CyclicCubic, spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    knots = user_knots !== nothing ? Float64.(user_knots) : place_knots(x, k)
    X, S = _cc_basis(x, knots)
    return RawMarginalBasis(X, [S], 1, knots, spec)
end

# P-spline marginal — same knot construction as s(x, bs=:ps), no constraint
function _build_raw_marginal(::PSpline, spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    spline_order = m_order + 2
    m2 = spline_order - 1

    knot_vec = _bspline_knot_vector(x, k, m2; user_knots = user_knots)

    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)
    S = _diff_penalty(actual_k, m_order)
    return RawMarginalBasis(X, [S], m_order, knot_vec, spec)
end

# TPRS marginal — the real TPRS construction, without constraint absorption.
# The returned template carries a working TPRSPredictCache so tensor smooths
# with tp margins can be predicted at new data.
function _build_raw_marginal(::Union{ThinPlateSpline, ThinPlateShrink},
                             spec::SmoothSpec, data, user_knots)
    sm = _construct_tprs(spec, data, user_knots;
        shrink = spec.basis isa ThinPlateShrink, absorb_cons = false)
    return RawMarginalBasis(sm.X, sm.S, sm.null_dim, sm.knots, spec;
        rank = sm.rank, template = sm)
end

# Fallback: build via the normal path (uses constraint absorption, less ideal)
function _build_raw_marginal(::AbstractBasisType, spec::SmoothSpec, data, user_knots)
    sm = _smooth_construct(spec.basis, spec, data, user_knots)
    return RawMarginalBasis(sm.X, sm.S, sm.null_dim, sm.knots, spec;
        Ain = sm.Ain, bin = sm.bin, Aeq = sm.Aeq, beq = sm.beq,
        constraint = sm.constraint,
        Sigma = sm.Sigma, cmX = sm.cmX, p_ident = sm.p_ident,
        rank = sm.rank, template = sm)
end

"""
    _raw_predict_marginal(raw::RawMarginalBasis, newdata) -> Matrix{Float64}

Build the unconstrained prediction matrix for a raw marginal at new data.
"""
function _raw_predict_marginal(raw::RawMarginalBasis, newdata)
    return predict_matrix(raw.template, newdata)
end

"""
    _penalty_nullity(S::Vector{Matrix{Float64}}, k::Int) -> Int

Numerically compute the dimension of the joint null space of a set of
penalty matrices acting on a k-dimensional coefficient space.
"""
function _penalty_nullity(S::Vector{Matrix{Float64}}, k::Int)
    isempty(S) && return k
    St = zeros(k, k)
    for Si in S
        nrm = opnorm(Si)
        if nrm > 0
            St .+= Si ./ nrm
        end
    end
    eigs = eigvals(Symmetric(St))
    mx = maximum(eigs)
    mx <= 0 && return k
    return count(e -> e < mx * eps()^0.75, eigs)
end

"""
    _merge_ti_constraint_blocks(raw_marginals, marginal_Zs, cons_dims, which)

Like `_merge_tensor_constraint_blocks`, but for ti(): each marginal's
linear constraint matrix is first mapped into the constrained marginal
coordinates via A_j Z_j before being embedded in the tensor product.
"""
function _merge_ti_constraint_blocks(raw_marginals::Vector{RawMarginalBasis},
                                     marginal_Zs::Vector{Matrix{Float64}},
                                     cons_dims::Vector{Int},
                                     which::Symbol)
    A_merged = nothing
    b_merged = nothing
    rhs_field = which === :Ain ? :bin : :beq
    for (i, rm) in enumerate(raw_marginals)
        A = getfield(rm, which)
        b = getfield(rm, rhs_field)
        if A !== nothing && b !== nothing && size(A, 1) > 0
            A_cons = A * marginal_Zs[i]
            A_full = _embed_tensor_constraint(A_cons, i, cons_dims)
            b_full = _repeat_tensor_rhs(b, i, cons_dims)
            A_merged, b_merged = _append_constraint_block(A_merged, b_merged, A_full, b_full)
        end
    end
    return A_merged, b_merged
end

function _smooth_construct(::TensorProduct, spec::SmoothSpec, data, user_knots)
    return _construct_tensor(spec, data, user_knots, interaction_only=false)
end

function _smooth_construct(::TensorInteraction, spec::SmoothSpec, data, user_knots)
    return _construct_tensor(spec, data, user_knots, interaction_only=true)
end

function _smooth_construct(::T2TensorProduct, spec::SmoothSpec, data, user_knots)
    return _construct_t2(spec, data, user_knots)
end

"""
    _construct_tensor(spec, data, user_knots; interaction_only)

!!! note
    mgcv's default `np=TRUE` reparameterization of each te marginal to the
    function-value parameterization (for smoothing-parameter scale
    invariance) is NOT applied here; the model space is the same but
    estimated smoothing parameters are on a different scale than mgcv's.

Core tensor product smooth construction:
1. Build unconstrained marginal bases
2. For ti(): absorb a sum-to-zero constraint into each marginal (X̃ⱼ = Xⱼ Zⱼ,
   S̃ⱼ = Zⱼ' Sⱼ Zⱼ), then form the row-wise Kronecker product of the
   constrained marginals — no further constraint is needed
3. For te(): form the row-wise Kronecker product of the raw marginals,
   assemble tensor product penalties, and absorb a single overall
   sum-to-zero constraint on the full product
"""
function _construct_tensor(spec::SmoothSpec, data, user_knots;
                           interaction_only::Bool=false)
    marginal_specs = _get_marginals(spec)
    marginal_specs !== nothing ||
        throw(ArgumentError("No marginal specs registered. Use te() or ti()."))

    # A single knot vector cannot be unambiguously assigned to multiple
    # marginals; per-margin knots are not currently supported.
    user_knots === nothing || throw(ArgumentError(
        "user-supplied knots are not supported for tensor product smooths " *
        "(they cannot be assigned unambiguously to the marginals); " *
        "control the marginals via k= instead"))

    d = length(marginal_specs)

    # 1. Build unconstrained marginal bases
    raw_marginals = RawMarginalBasis[]
    for mspec in marginal_specs
        push!(raw_marginals, _build_raw_marginal(mspec, data, user_knots))
    end

    marginal_Xs = [rm.X for rm in raw_marginals]
    marginal_dims = [size(X, 2) for X in marginal_Xs]
    marginal_null_dims = [rm.null_dim for rm in raw_marginals]

    # For ti(): absorb a sum-to-zero constraint into EACH marginal BEFORE
    # forming the tensor product (mgcv's mc=TRUE convention). The constrained
    # marginal X̃_j = X_j Z_j contains neither the constant nor any function
    # with non-zero data mean, so the row-wise Kronecker product of the X̃_j
    # contains no marginal main effects and no constant by construction.
    if interaction_only
        marginal_Zs = Matrix{Float64}[]
        cons_Xs = Matrix{Float64}[]
        cons_Ss = Vector{Matrix{Float64}}[]
        for rm in raw_marginals
            kj = size(rm.X, 2)
            # Reuse the deterministic QR-based absorption used for ordinary
            # smooths so fit and predict transforms match exactly.
            S_work = [copy(Si) for Si in rm.S]
            Xc, Sc, Cj, _ = absorb_constraints!(copy(rm.X), S_work)
            Zj = _constraint_basis(Cj, kj)
            push!(marginal_Zs, Zj)
            push!(cons_Xs, Xc)
            push!(cons_Ss, Sc)
        end

        cons_dims = [size(Xc, 2) for Xc in cons_Xs]

        # Row-wise Kronecker product of CONSTRAINED marginals
        X_tensor = _row_kronecker(cons_Xs)

        # Tensor product penalties from constrained marginal penalties:
        # S_i = I ⊗ … ⊗ S̃_i ⊗ … ⊗ I
        penalties = Matrix{Float64}[]
        for j in 1:d
            for Sj in cons_Ss[j]
                P = ones(1, 1)
                for i in 1:d
                    if i == j
                        P = kron(P, Sj)
                    else
                        P = kron(P, Matrix{Float64}(I, cons_dims[i], cons_dims[i]))
                    end
                end
                push!(penalties, P)
            end
        end

        # mgcv-style penalty rescaling relative to the tensor model matrix
        # (mirrors the scale_penalty block in absorb_constraints!, which is
        # not called on the full ti product).
        maXX = opnorm(X_tensor, Inf)^2
        if maXX > 0
            for i in eachindex(penalties)
                nS = opnorm(penalties[i], 1)
                if nS > 0
                    penalties[i] = penalties[i] * (maXX / nS)
                end
            end
        end

        Ain, bin = _merge_ti_constraint_blocks(raw_marginals, marginal_Zs, cons_dims, :Ain)
        Aeq, beq = _merge_ti_constraint_blocks(raw_marginals, marginal_Zs, cons_dims, :Aeq)

        total_k = size(X_tensor, 2)

        # Null space of the ti block = tensor product of the constrained
        # marginal penalty null spaces. Compute each nullity numerically.
        nullities = [_penalty_nullity(cons_Ss[j], cons_dims[j]) for j in 1:d]
        null_dim = prod(nullities)
        pen_rank = max(total_k - null_dim, 0)

        # Identifiability constraints are already absorbed in the marginals;
        # no further overall constraint is applied (constraint = nothing).
        sm = ConstructedSmooth(
            spec, X_tensor, penalties,
            Float64[],
            null_dim, pen_rank,
            nothing, nothing, 0, 0,
            nothing, nothing, nothing,
            Int[],
            Ain, bin, Aeq, beq,
            predict_cache = TensorPredictCache(raw_marginals, marginal_Zs),
        )
        return sm
    end

    # 2. Row-wise Kronecker product
    X_tensor = _row_kronecker(marginal_Xs)

    # 3. Tensor product penalties
    penalties = Matrix{Float64}[]
    for j in 1:d
        for Sj in raw_marginals[j].S
            P = ones(1, 1)
            for i in 1:d
                if i == j
                    P = kron(P, Sj)
                else
                    P = kron(P, Matrix{Float64}(I, marginal_dims[i], marginal_dims[i]))
                end
            end
            push!(penalties, P)
        end
    end

    Ain, bin = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Ain)
    Aeq, beq = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Aeq)

    total_k = size(X_tensor, 2)

    # Null space dimension
    null_dim = prod(marginal_null_dims)
    pen_rank = max(total_k - null_dim, 0)

    # 4. Absorb identifiability constraints
    X_cons, S_cons, C, _ = absorb_constraints!(X_tensor, penalties)
    Z = _constraint_basis(C, size(X_tensor, 2))
    Ain_cons = Ain === nothing ? nothing : Ain * Z
    Aeq_cons = Aeq === nothing ? nothing : Aeq * Z

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        Ain_cons, bin, Aeq_cons, beq,
        predict_cache = TensorPredictCache(raw_marginals, Matrix{Float64}[]),
    )
    return sm
end

function _predict_matrix(::Union{TensorProduct, TensorInteraction},
                         smooth::ConstructedSmooth, newdata)
    cache = smooth.predict_cache
    cache isa TensorPredictCache ||
        throw(ArgumentError("Cannot find marginal info for tensor product prediction"))
    raw_marginals = cache.raw_marginals

    interaction_only = smooth.spec.basis isa TensorInteraction

    marginal_Xs = [_raw_predict_marginal(rm, newdata) for rm in raw_marginals]

    if interaction_only
        # Apply the SAME marginal constraint transforms Z_j that were absorbed
        # at construction time, then form the row-wise Kronecker product.
        # No further constraint applies (smooth.constraint === nothing).
        marginal_Zs = cache.marginal_Zs
        !isempty(marginal_Zs) ||
            throw(ArgumentError("Cannot find marginal constraint transforms for ti() prediction"))
        cons_Xs = [marginal_Xs[j] * marginal_Zs[j] for j in eachindex(marginal_Zs)]
        return _row_kronecker(cons_Xs)
    end

    X_tensor = _row_kronecker(marginal_Xs)

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X_tensor, 2))
        return X_tensor * Z
    end
    return X_tensor
end

# ============================================================================
# t2() — alternative tensor product smooth (mgcv t2())
# ============================================================================

"""
Prediction cache for t2 tensor products.

`reparam` is the single matrix `M` mapping the raw row-Kronecker basis to the
final constrained basis: `X_final = rowkron(marginals) * M`. It folds the
marginal null/range reparameterization, the block-grouping permutation, and
the identifiability constraint into one operator, so prediction reproduces
the training basis exactly (no QR is recomputed at predict time).
"""
struct T2PredictCache <: AbstractSmoothPredictCache
    raw_marginals::Vector{RawMarginalBasis}
    reparam::Matrix{Float64}
end

"""
    _t2_marginal_reparam(rm) -> (R, n_range)

Reparameterize one marginal into `[range | null]` form (Wood, Scheipl &
Faraway 2013). Returns `R` such that `X_i * R` has its first `n_range`
columns carrying an IDENTITY penalty and the remaining columns unpenalized.

The split uses the eigendecomposition of the marginal's summed penalty:
range directions are rescaled by `1/sqrt(eigenvalue)` so their penalty
becomes the identity; null directions are passed through unchanged.
"""
function _t2_marginal_reparam(rm::RawMarginalBasis)
    k = size(rm.X, 2)
    if isempty(rm.S)
        return Matrix{Float64}(I, k, k), 0
    end

    S_sum = zeros(k, k)
    for Si in rm.S
        S_sum .+= Si
    end
    S_sum .= (S_sum .+ S_sum') ./ 2

    eig = eigen(Symmetric(S_sum))
    idx = k:-1:1                      # descending eigenvalues
    vals = eig.values[idx]
    vecs = eig.vectors[:, idx]

    # Use the basis's ANALYTIC null-space dimension rather than an eigenvalue
    # threshold: marginal penalties can carry numerically tiny positive
    # eigenvalues, and a threshold rule makes the split data-dependent (one
    # marginal can be classified differently from an identical sibling).
    # mgcv likewise splits on the known null space.
    tol = eps(Float64) * max(maximum(abs, vals), 1.0) * k
    n_range = clamp(k - rm.null_dim, 0, k)

    R = Matrix{Float64}(undef, k, k)
    for j in 1:n_range
        R[:, j] = vecs[:, j] ./ sqrt(max(vals[j], tol))
    end
    for j in (n_range + 1):k
        R[:, j] = vecs[:, j]
    end
    return R, n_range
end

"""
    _t2_blocks(range_dims, marginal_dims) -> (blocks, null_block)

Group the tensor-product columns by which marginals contribute a *range*
(penalized) factor. Column `c` of the row-Kronecker product corresponds to a
multi-index with the LAST marginal varying fastest; a column belongs to the
block labelled by the tuple of per-marginal range/null flags.

Returns the penalized blocks (one per non-all-null combination that has
columns) and the all-null block, which carries the unpenalized columns.
"""
function _t2_blocks(range_dims::Vector{Int}, marginal_dims::Vector{Int})
    d = length(marginal_dims)
    total = prod(marginal_dims)

    # strides: last marginal varies fastest
    strides = ones(Int, d)
    for i in (d - 1):-1:1
        strides[i] = strides[i + 1] * marginal_dims[i + 1]
    end

    labels = Vector{Int}(undef, total)   # bitmask: bit i set => marginal i is range
    for c in 0:(total - 1)
        rem = c
        mask = 0
        for i in 1:d
            j = div(rem, strides[i]) + 1
            rem = mod(rem, strides[i])
            if j <= range_dims[i]
                mask |= (1 << (i - 1))
            end
        end
        labels[c + 1] = mask
    end

    blocks = Vector{Vector{Int}}()
    for mask in 1:((1 << d) - 1)
        cols = findall(==(mask), labels)
        isempty(cols) || push!(blocks, cols)
    end
    null_block = findall(==(0), labels)
    return blocks, null_block
end

"""
    _construct_t2(spec, data, user_knots)

Construct a t2() tensor product smooth following mgcv's construction
(Wood, Scheipl & Faraway 2013).

Each marginal is reparameterized into orthogonal null and range parts, with
the range part rescaled to carry an identity penalty. The tensor-product
columns then partition into 2^d blocks by which marginals contribute a range
factor. Every block except the all-null one receives its own penalty — an
identity on that block's columns and zero elsewhere — so the penalties are
diagonal with NON-OVERLAPPING support. The all-null block is unpenalized and
carries the identifiability constraint.

This diagonal, non-overlapping structure is what makes t2 usable as
independent random-effect blocks in mixed-model software (lme4/gamm4), in
contrast to te(), whose overlapping penalties require mgcv's `pdTens`.
"""
function _construct_t2(spec::SmoothSpec, data, user_knots)
    marginal_specs = _get_marginals(spec)
    marginal_specs !== nothing ||
        throw(ArgumentError("No marginal specs registered. Use t2()."))

    user_knots === nothing || throw(ArgumentError(
        "user-supplied knots are not supported for tensor product smooths " *
        "(they cannot be assigned unambiguously to the marginals); " *
        "control the marginals via k= instead"))

    d = length(marginal_specs)

    # 1. Unconstrained marginal bases
    raw_marginals = RawMarginalBasis[]
    for mspec in marginal_specs
        push!(raw_marginals, _build_raw_marginal(mspec, data, user_knots))
    end

    marginal_Xs = [rm.X for rm in raw_marginals]
    marginal_dims = [size(X, 2) for X in marginal_Xs]

    # 2. Per-marginal null/range reparameterization
    reparams = Matrix{Float64}[]
    range_dims = Int[]
    for rm in raw_marginals
        R, nr = _t2_marginal_reparam(rm)
        push!(reparams, R)
        push!(range_dims, nr)
    end

    # 3. Raw row-Kronecker basis and the folded reparameterization operator.
    #    (A*Ra) ⊙ (B*Rb) == (A ⊙ B) * (Ra ⊗ Rb), so one matrix suffices.
    X_raw = _row_kronecker(marginal_Xs)
    T = reparams[1]
    for i in 2:d
        T = kron(T, reparams[i])
    end

    # 4. Block structure, ordered penalized-blocks-first, null block last
    blocks, null_block = _t2_blocks(range_dims, marginal_dims)
    perm = vcat(blocks..., null_block)
    T = T[:, perm]
    X_repar = X_raw * T

    total_k = size(X_repar, 2)
    n_pen_cols = total_k - length(null_block)

    # 5. One identity penalty per penalized block (non-overlapping supports)
    penalties = Matrix{Float64}[]
    col = 0
    for blk in blocks
        P = zeros(total_k, total_k)
        for _ in 1:length(blk)
            col += 1
            P[col, col] = 1.0
        end
        push!(penalties, P)
    end

    # mgcv-style penalty rescaling (as in absorb_constraints!), applied
    # before the constraint so the scaling matches other smooths
    maXX = opnorm(X_repar, Inf)^2
    if maXX > 0
        for i in eachindex(penalties)
            nS = opnorm(penalties[i], 1)
            nS > 0 && (penalties[i] .*= maXX / nS)
        end
    end

    # 6. Identifiability constraint. The constant function lies entirely in
    #    the all-null block, so — as in mgcv — the constraint acts ONLY on
    #    that block, leaving the penalized blocks (and hence the diagonal,
    #    non-overlapping penalty structure) untouched.
    n_null = length(null_block)
    if n_null >= 1
        c_null = vec(sum(X_repar[:, (n_pen_cols + 1):total_k]; dims = 1))
        Z_null = _constraint_basis(reshape(c_null, 1, n_null), n_null)
        Z = zeros(total_k, n_pen_cols + size(Z_null, 2))
        Z[1:n_pen_cols, 1:n_pen_cols] = Matrix{Float64}(I, n_pen_cols, n_pen_cols)
        Z[(n_pen_cols + 1):total_k, (n_pen_cols + 1):end] = Z_null
        C = zeros(1, total_k)
        C[1, (n_pen_cols + 1):total_k] = c_null
    else
        # Every marginal is fully penalized (e.g. shrinkage marginals): there
        # is no null block to constrain, so fall back to a whole-basis
        # sum-to-zero constraint. This mixes columns and the resulting
        # penalties are no longer exactly block-diagonal.
        C = reshape(vec(sum(X_repar; dims = 1)), 1, total_k)
        Z = _constraint_basis(C, total_k)
    end

    X_cons = X_repar * Z
    S_cons = [Symmetric(Z' * P * Z) |> Matrix for P in penalties]

    M = T * Z   # raw row-Kronecker basis -> final constrained basis

    Ain, bin = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Ain)
    Aeq, beq = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Aeq)
    Ain_cons = Ain === nothing ? nothing : Ain * M
    Aeq_cons = Aeq === nothing ? nothing : Aeq * M

    null_dim = size(X_cons, 2) - n_pen_cols
    pen_rank = n_pen_cols

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        Ain_cons, bin, Aeq_cons, beq,
        predict_cache = T2PredictCache(raw_marginals, M),
    )
    return sm
end

function _predict_matrix(::T2TensorProduct, smooth::ConstructedSmooth, newdata)
    cache = smooth.predict_cache
    cache isa T2PredictCache ||
        throw(ArgumentError("Cannot find marginal info for t2 tensor product prediction"))

    marginal_Xs = [_raw_predict_marginal(rm, newdata) for rm in cache.raw_marginals]
    # The stored operator folds the marginal reparameterization, the block
    # permutation, and the constraint, so this reproduces the training basis.
    return _row_kronecker(marginal_Xs) * cache.reparam
end
