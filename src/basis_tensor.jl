# Tensor product smooths — te() and ti()
#
# Implements tensor product basis construction following Wood (2006, §4.1.8).
# The model matrix is the row-wise Kronecker product of marginal bases.
# Penalties are Sⱼ ⊗ (⊗_{i≠j} Iₖᵢ), one per marginal dimension.
#
# Marginal bases are constructed WITHOUT absorbing identifiability constraints.
# Constraints are applied only to the full tensor product.

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
end

# Module-level storage for marginal info (keyed by objectid of ConstructedSmooth)
const _TENSOR_MARGINALS = Dict{UInt, Vector{RawMarginalBasis}}()

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
    return RawMarginalBasis(sm.X, sm.S, sm.null_dim, sm.knots, spec)
end

"""
    _raw_predict_marginal(raw::RawMarginalBasis, newdata) -> Matrix{Float64}

Build the unconstrained prediction matrix for a raw marginal at new data.
"""
function _raw_predict_marginal(raw::RawMarginalBasis, newdata)
    return _build_raw_marginal(raw.spec, newdata, nothing).X
end

"""
    _ti_select_columns(marginal_dims, marginal_null_dims) -> Vector{Int}

For ti(), select only interaction columns from the tensor product.
A column indexed by (j₁, j₂, ..., jd) is an interaction term if
ALL margins contribute a range-space basis function (index > null_dim).

In the unconstrained basis, the first null_dim columns span the penalty null
space (polynomials). Interaction terms require every margin to contribute
at least one "wiggle" (range-space) function.
"""
function _ti_select_columns(marginal_dims::Vector{Int},
                            marginal_null_dims::Vector{Int})
    d = length(marginal_dims)
    total_k = prod(marginal_dims)
    keep = Int[]

    for col in 1:total_k
        idx = col - 1
        margin_indices = zeros(Int, d)
        for j in d:-1:1
            margin_indices[j] = (idx % marginal_dims[j]) + 1
            idx = div(idx, marginal_dims[j])
        end

        # Keep only if ALL margins contribute a range-space function
        # Range-space functions have index > null_dim
        all_range = all(j -> margin_indices[j] > marginal_null_dims[j], 1:d)
        if all_range
            push!(keep, col)
        end
    end

    return keep
end

function _smooth_construct(::TensorProduct, spec::SmoothSpec, data, user_knots)
    return _construct_tensor(spec, data, user_knots, interaction_only=false)
end

function _smooth_construct(::TensorInteraction, spec::SmoothSpec, data, user_knots)
    return _construct_tensor(spec, data, user_knots, interaction_only=true)
end

"""
    _construct_tensor(spec, data, user_knots; interaction_only)

Core tensor product smooth construction:
1. Build unconstrained marginal bases
2. Form row-wise Kronecker product
3. Assemble tensor product penalties
4. For ti(): remove main-effect columns
5. Absorb identifiability constraints on the full product
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

    # 4. For ti(): keep only interaction columns
    if interaction_only
        keep_cols = _ti_select_columns(marginal_dims, marginal_null_dims)
        if !isempty(keep_cols)
            X_tensor = X_tensor[:, keep_cols]
            penalties = [S[keep_cols, keep_cols] for S in penalties]
        end
    end

    total_k = size(X_tensor, 2)

    # Null space dimension
    null_dim = interaction_only ? 0 : prod(marginal_null_dims)
    pen_rank = max(total_k - null_dim, 0)

    # 5. Absorb identifiability constraints
    X_cons, S_cons, C, _ = absorb_constraints!(X_tensor, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
    )

    _TENSOR_MARGINALS[objectid(sm)] = raw_marginals
    return sm
end

function _predict_matrix(::Union{TensorProduct, TensorInteraction},
                         smooth::ConstructedSmooth, newdata)
    raw_marginals = get(_TENSOR_MARGINALS, objectid(smooth), nothing)
    raw_marginals !== nothing ||
        throw(ArgumentError("Cannot find marginal info for tensor product prediction"))

    interaction_only = smooth.spec.basis isa TensorInteraction

    marginal_Xs = [_raw_predict_marginal(rm, newdata) for rm in raw_marginals]
    marginal_dims = [size(X, 2) for X in marginal_Xs]
    marginal_null_dims = [rm.null_dim for rm in raw_marginals]

    X_tensor = _row_kronecker(marginal_Xs)

    if interaction_only
        # Use original training dimensions for consistency
        orig_dims = [size(rm.X, 2) for rm in raw_marginals]
        keep_cols = _ti_select_columns(orig_dims, marginal_null_dims)
        if !isempty(keep_cols) && length(keep_cols) <= size(X_tensor, 2)
            X_tensor = X_tensor[:, keep_cols]
        end
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = nullspace(C)
        return X_tensor * Z
    end
    return X_tensor
end
