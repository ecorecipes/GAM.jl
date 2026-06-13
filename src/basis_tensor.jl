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
    S_shrink = Matrix{Float64}(I, k, k) .* (1e-2 * tr(S) / k)
    return RawMarginalBasis(X, [S, S_shrink], 2, knots, spec)
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

# P-spline marginal
function _build_raw_marginal(::PSpline, spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    spline_order = m_order + 2

    n_interior = k - spline_order
    n_interior >= 1 || throw(ArgumentError(
        "k=$k too small for P-spline of order $spline_order"))

    lo, hi = minimum(x), maximum(x)
    dx = (hi - lo) * 0.001
    interior = user_knots !== nothing ? Float64.(user_knots) : knot_quantiles(x, n_interior)
    knot_vec = vcat(fill(lo - dx, spline_order), interior, fill(hi + dx, spline_order))

    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)
    S = _diff_penalty(actual_k, m_order)
    return RawMarginalBasis(X, [S], m_order, knot_vec, spec)
end

# TPRS marginal — simplified 1d version
function _build_raw_marginal(::Union{ThinPlateSpline, ThinPlateShrink},
                             spec::SmoothSpec, data, user_knots)
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    M = m_order  # null space dim for 1d

    xk = if user_knots !== nothing && length(user_knots) >= k
        Float64.(user_knots[1:k])
    elseif n > max(k * 3, 200)
        place_knots(x, k)
    else
        x
    end

    E = _tps_penalty_matrix(xk, m_order)
    T_null = _tps_null_space_basis(xk, m_order)

    eig = eigen(Symmetric(E))
    idx = sortperm(eig.values; rev=true)
    n_basis = k - M
    Uk = eig.vectors[:, idx[1:n_basis]]
    Dk = eig.values[idx[1:n_basis]]

    if length(xk) < n
        E_nk = zeros(n, length(xk))
        for j in eachindex(xk), i in 1:n
            E_nk[i, j] = _tps_eta(abs(x[i] - xk[j]), m_order, 1)
        end
        X_eigbasis = E_nk * Uk * Diagonal(1.0 ./ Dk)
        T_data = _tps_null_space_basis(x, m_order)
    else
        X_eigbasis = Uk
        T_data = T_null
    end

    X_full = hcat(X_eigbasis, T_data)
    S_diag = zeros(k)
    S_diag[1:n_basis] .= 1.0 ./ max.(abs.(Dk), eps())
    S_mat = Matrix(Diagonal(S_diag))

    penalties = Matrix{Float64}[S_mat]
    if spec.basis isa ThinPlateShrink
        S_shrink = zeros(k, k)
        for i in (n_basis + 1):k
            S_shrink[i, i] = 1.0
        end
        push!(penalties, S_shrink)
    end

    knots_out = length(xk) > 0 ? Float64.(xk) : Float64[]
    return RawMarginalBasis(X_full, penalties, M, knots_out, spec)
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
    _construct_t2(spec, data, user_knots)

Construct a t2() tensor product smooth. The basis matrix is the same as te()
(row-wise Kronecker product of marginals), but the penalties differ:

For d marginals, each with penalties S_j^(m):
- For each marginal m and each penalty j of that marginal:
  P = I_1 ⊗ ... ⊗ S_j^(m) ⊗ ... ⊗ I_d  (penalty in position m, identity elsewhere)
- Plus a "full interaction" penalty: S_1^(1) ⊗ S_1^(2) ⊗ ... ⊗ S_1^(d)

For 2 marginals each with 1 penalty, this gives 3 penalties:
  S^(1) ⊗ I_2, I_1 ⊗ S^(2), S^(1) ⊗ S^(2)
"""
function _construct_t2(spec::SmoothSpec, data, user_knots)
    marginal_specs = _get_marginals(spec)
    marginal_specs !== nothing ||
        throw(ArgumentError("No marginal specs registered. Use t2()."))

    d = length(marginal_specs)

    # 1. Build unconstrained marginal bases
    raw_marginals = RawMarginalBasis[]
    for mspec in marginal_specs
        push!(raw_marginals, _build_raw_marginal(mspec, data, user_knots))
    end

    marginal_Xs = [rm.X for rm in raw_marginals]
    marginal_dims = [size(X, 2) for X in marginal_Xs]
    marginal_null_dims = [rm.null_dim for rm in raw_marginals]

    # 2. Row-wise Kronecker product (same as te())
    X_tensor = _row_kronecker(marginal_Xs)

    # 3. t2-style penalties:
    #    For each marginal m and each penalty S_j of that marginal:
    #      I_1 ⊗ ... ⊗ S_j ⊗ ... ⊗ I_d
    #    Plus the full interaction: S_1^(1) ⊗ S_1^(2) ⊗ ...
    penalties = Matrix{Float64}[]

    for m in 1:d
        for Sj in raw_marginals[m].S
            P = _t2_single_penalty(Sj, m, marginal_dims, d)
            push!(penalties, P)
        end
    end

    Ain, bin = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Ain)
    Aeq, beq = _merge_tensor_constraint_blocks(raw_marginals, marginal_dims, :Aeq)

    # Full interaction penalty: kronecker of first penalty from each marginal
    P_full = Matrix{Float64}(I, 1, 1)
    for m in 1:d
        Sm = raw_marginals[m].S[1]
        P_full = kron(P_full, Sm)
    end
    push!(penalties, P_full)

    total_k = size(X_tensor, 2)
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

"""
    _t2_single_penalty(Sj, pos, marginal_dims, d)

Build a single t2 penalty: I_1 ⊗ ... ⊗ S_j ⊗ ... ⊗ I_d,
where S_j is placed at position `pos`.
"""
function _t2_single_penalty(Sj::Matrix{Float64}, pos::Int,
                            marginal_dims::Vector{Int}, d::Int)
    P = Matrix{Float64}(I, 1, 1)
    for i in 1:d
        if i == pos
            P = kron(P, Sj)
        else
            P = kron(P, Matrix{Float64}(I, marginal_dims[i], marginal_dims[i]))
        end
    end
    return P
end

function _predict_matrix(::T2TensorProduct, smooth::ConstructedSmooth, newdata)
    cache = smooth.predict_cache
    cache isa TensorPredictCache ||
        throw(ArgumentError("Cannot find marginal info for t2 tensor product prediction"))
    raw_marginals = cache.raw_marginals

    marginal_Xs = [_raw_predict_marginal(rm, newdata) for rm in raw_marginals]
    X_tensor = _row_kronecker(marginal_Xs)

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X_tensor, 2))
        return X_tensor * Z
    end
    return X_tensor
end
