# mgcv::scasm support: bs="sc" / bs="scad" plus general `pc` constraints.

function _append_constraint_block(A_old, b_old, A_new, b_new)
    if A_new === nothing || b_new === nothing || size(A_new, 1) == 0
        return A_old, b_old
    elseif A_old === nothing || b_old === nothing || size(A_old, 1) == 0
        return Matrix{Float64}(A_new), Vector{Float64}(b_new)
    end
    return vcat(A_old, A_new), vcat(b_old, b_new)
end

function _append_linear_constraints!(sm::ConstructedSmooth, Ain, bin, Aeq, beq)
    sm.Ain, sm.bin = _append_constraint_block(sm.Ain, sm.bin, Ain, bin)
    sm.Aeq, sm.beq = _append_constraint_block(sm.Aeq, sm.beq, Aeq, beq)
    return sm
end

has_linear_constraints(sm::ConstructedSmooth) =
    (sm.Ain !== nothing && size(sm.Ain, 1) > 0) ||
    (sm.Aeq !== nothing && size(sm.Aeq, 1) > 0)

has_linear_constraints(smooths::Vector{<:ConstructedSmooth}) = any(has_linear_constraints, smooths)

function _xt_constraints(spec::SmoothSpec)
    constraints = get(spec.xt, :constraints, String[])
    if constraints isa AbstractString
        return String[String(constraints)]
    elseif constraints isa AbstractVector
        return String[string(c) for c in constraints]
    end
    return String[]
end

function _raw_bspline_basis(spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Shape-constrained B-splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    spline_order = m_order + 2

    m2 = spline_order - 1
    nk = k - m2 + 1
    nk >= 2 || throw(ArgumentError(
        "k=$k too small for B-spline of order $spline_order (need k ≥ $(m2 + 2))"))

    lo, hi = minimum(x), maximum(x)

    knot_vec = if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] : (hi - lo)
        vcat(
            [interior[1] - dk * i for i in m2:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:m2],
        )
    else
        k_new = collect(range(lo, hi; length = nk))
        dk = k_new[2] - k_new[1]
        vcat(
            [k_new[1] - dk * i for i in m2:-1:1],
            k_new,
            [k_new[end] + dk * i for i in 1:m2],
        )
    end

    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)
    S = _derivative_penalty(knot_vec, spline_order, m_order, actual_k)
    penalties = Matrix{Float64}[S]
    null_dim = m_order
    pen_rank = actual_k - null_dim

    return X, penalties, knot_vec, null_dim, pen_rank, spline_order
end

function _raw_adaptive_basis(spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Shape-constrained adaptive smooths only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m
    spline_order = m_order + 2
    n_penalties = Int(get(spec.xt, :n_penalties, 5))

    m2 = spline_order - 1
    nk = k - m2 + 1
    nk >= 2 || throw(ArgumentError(
        "k=$k too small for adaptive smooth of order $spline_order (need k ≥ $(m2 + 2))"))

    lo, hi = minimum(x), maximum(x)
    knot_vec = if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] : (hi - lo)
        vcat(
            [interior[1] - dk * i for i in m2:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:m2],
        )
    else
        k_new = collect(range(lo, hi; length = nk))
        dk = k_new[2] - k_new[1]
        vcat(
            [k_new[1] - dk * i for i in m2:-1:1],
            k_new,
            [k_new[end] + dk * i for i in 1:m2],
        )
    end

    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)
    D = _ad_diff_matrix(actual_k, m_order)
    n_rows = size(D, 1)
    n_pen = min(n_penalties, n_rows)
    pou_weights = _partition_of_unity_weights(n_rows, n_pen)
    penalties = Matrix{Float64}[]
    for wj in pou_weights
        Sj = D' * Diagonal(wj) * D
        push!(penalties, (Sj + Sj') / 2)
    end

    null_dim = m_order
    pen_rank = actual_k - null_dim

    return X, penalties, knot_vec, null_dim, pen_rank, spline_order
end

function _bspline_derivative_row(knots::Vector{Float64}, order::Int, x::Float64)
    h = 1e-7
    B_plus = vec(_bspline_basis([x + h], knots, order))
    B_minus = vec(_bspline_basis([x - h], knots, order))
    return (B_plus .- B_minus) ./ (2h)
end

function _build_sc_constraints(knots::Vector{Float64}, spline_order::Int, p::Int, constraints::Vector{String})
    isempty(constraints) && return nothing, nothing

    Ip = Matrix{Float64}(I, p, p)
    Ain = Matrix{Float64}(undef, 0, p)
    ll = knots[spline_order]
    ul = knots[end - spline_order + 1]
    value_start = reshape(vec(_bspline_basis([ll], knots, spline_order)), 1, :)
    value_end = reshape(vec(_bspline_basis([ul], knots, spline_order)), 1, :)
    deriv_start = reshape(_bspline_derivative_row(knots, spline_order, ll), 1, :)
    deriv_end = reshape(_bspline_derivative_row(knots, spline_order, ul), 1, :)

    if "c+" in constraints
        Ain = vcat(Ain, diff(diff(Ip; dims = 1); dims = 1))
        if "m+" in constraints
            Ain = vcat(Ain, deriv_start)
            if "+" in constraints
                Ain = vcat(Ain, value_start)
            end
        elseif "m-" in constraints
            Ain = vcat(Ain, -deriv_end)
            if "+" in constraints
                Ain = vcat(Ain, value_end)
            end
        elseif "+" in constraints
            Ain = vcat(Ain, Ip)
        end
    elseif "c-" in constraints
        Ain = vcat(Ain, -diff(diff(Ip; dims = 1); dims = 1))
        if "m+" in constraints
            Ain = vcat(Ain, deriv_end)
            if "+" in constraints
                Ain = vcat(Ain, value_start)
            end
        elseif "m-" in constraints
            Ain = vcat(Ain, -deriv_start)
            if "+" in constraints
                Ain = vcat(Ain, value_end)
            end
        elseif "+" in constraints
            Ain = vcat(Ain, value_start, value_end)
        end
    elseif "m+" in constraints
        Ain = vcat(Ain, diff(Ip; dims = 1))
        if "+" in constraints
            Ain = vcat(Ain, value_start)
        end
    elseif "m-" in constraints
        Ain = vcat(Ain, -diff(Ip; dims = 1))
        if "+" in constraints
            Ain = vcat(Ain, value_end)
        end
    elseif "+" in constraints
        Ain = Ip
    end

    return size(Ain, 1) == 0 ? nothing : Ain, size(Ain, 1) == 0 ? nothing : zeros(size(Ain, 1))
end

function _build_scad_constraints(p::Int, constraints::Vector{String})
    isempty(constraints) && return nothing, nothing
    Ip = Matrix{Float64}(I, p, p)
    Ain = Matrix{Float64}(undef, 0, p)
    if "+" in constraints
        Ain = vcat(Ain, Ip)
    end
    if "m+" in constraints
        Ain = vcat(Ain, diff(Ip; dims = 1))
    elseif "m-" in constraints
        Ain = vcat(Ain, -diff(Ip; dims = 1))
    end
    if "c+" in constraints
        Ain = vcat(Ain, diff(diff(Ip; dims = 1); dims = 1))
    elseif "c-" in constraints
        Ain = vcat(Ain, -diff(diff(Ip; dims = 1); dims = 1))
    end
    return size(Ain, 1) == 0 ? nothing : Ain, size(Ain, 1) == 0 ? nothing : zeros(size(Ain, 1))
end

function _finalize_scasm_smooth(spec::SmoothSpec, X_raw, penalties, knots, null_dim, pen_rank, Ain_raw, bin_raw; scale_penalty=true)
    plus_only = "+" in _xt_constraints(spec)

    if plus_only
        X = copy(X_raw)
        S = copy(penalties)
        C = nothing
        Z = Matrix{Float64}(I, size(X_raw, 2), size(X_raw, 2))
    else
        X, S, C, _ = absorb_constraints!(copy(X_raw), copy(penalties); scale_penalty = scale_penalty)
        Z = _constraint_basis(C, size(X_raw, 2))
    end

    Ain = Ain_raw === nothing ? nothing : Ain_raw * Z
    bin = bin_raw === nothing ? nothing : copy(bin_raw)

    return ConstructedSmooth(
        spec, X, S, knots, null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        Ain, bin, nothing, nothing,
    )
end

function _smooth_construct(::ShapeConstrainedBSpline, spec::SmoothSpec, data, user_knots)
    X_raw, penalties, knots, null_dim, pen_rank, spline_order =
        _raw_bspline_basis(spec, data, user_knots)
    Ain_raw, bin_raw = _build_sc_constraints(knots, spline_order, size(X_raw, 2), _xt_constraints(spec))
    return _finalize_scasm_smooth(spec, X_raw, penalties, knots, null_dim, pen_rank, Ain_raw, bin_raw)
end

function _smooth_construct(::ShapeConstrainedAdaptive, spec::SmoothSpec, data, user_knots)
    X_raw, penalties, knots, null_dim, pen_rank, _ =
        _raw_adaptive_basis(spec, data, user_knots)
    Ain_raw, bin_raw = _build_scad_constraints(size(X_raw, 2), _xt_constraints(spec))
    return _finalize_scasm_smooth(spec, X_raw, penalties, knots, null_dim, pen_rank, Ain_raw, bin_raw; scale_penalty = false)
end

function _predict_matrix(::ShapeConstrainedBSpline, smooth::ConstructedSmooth, newdata)
    return _predict_matrix(BSplineBasis(), smooth, newdata)
end

function _predict_matrix(::ShapeConstrainedAdaptive, smooth::ConstructedSmooth, newdata)
    return _predict_matrix(PSpline(), smooth, newdata)
end

function _raw_shape_constrained_marginal(spec::SmoothSpec, data, user_knots, Ain_raw, bin_raw, builder)
    X_raw, penalties, knots, null_dim, _, _ = builder(spec, data, user_knots)
    plus_only = "+" in _xt_constraints(spec)
    if plus_only
        X = copy(X_raw)
        S = copy(penalties)
        C = nothing
        Z = Matrix{Float64}(I, size(X_raw, 2), size(X_raw, 2))
    else
        X, S, C, _ = absorb_constraints!(copy(X_raw), copy(penalties))
        Z = _constraint_basis(C, size(X_raw, 2))
    end
    Ain = Ain_raw === nothing ? nothing : Ain_raw * Z
    Aeq = nothing
    beq = nothing
    return RawMarginalBasis(X, S, null_dim, knots, spec;
        Ain = Ain, bin = bin_raw, Aeq = Aeq, beq = beq,
        constraint = C)
end

function _build_raw_marginal(::ShapeConstrainedBSpline, spec::SmoothSpec, data, user_knots)
    X_raw, _, knots, _, _, spline_order = _raw_bspline_basis(spec, data, user_knots)
    Ain_raw, bin_raw = _build_sc_constraints(knots, spline_order, size(X_raw, 2), _xt_constraints(spec))
    return _raw_shape_constrained_marginal(spec, data, user_knots, Ain_raw, bin_raw, _raw_bspline_basis)
end

function _build_raw_marginal(::ShapeConstrainedAdaptive, spec::SmoothSpec, data, user_knots)
    X_raw, _, _, _, _, _ = _raw_adaptive_basis(spec, data, user_knots)
    Ain_raw, bin_raw = _build_scad_constraints(size(X_raw, 2), _xt_constraints(spec))
    return _raw_shape_constrained_marginal(spec, data, user_knots, Ain_raw, bin_raw, _raw_adaptive_basis)
end

function _pc_block_data(block, term_vars::Vector{Symbol})
    block_dict = block isa AbstractDict ? Dict(Symbol(k) => v for (k, v) in pairs(block)) :
        block isa NamedTuple ? Dict(Symbol(k) => v for (k, v) in pairs(block)) :
        throw(ArgumentError("pc block must be a Dict or NamedTuple, got $(typeof(block))"))

    weights = get(block_dict, :weights, get(block_dict, :w, nothing))
    rhs = get(block_dict, :rhs, get(block_dict, :b, nothing))
    weights !== nothing || throw(ArgumentError("pc block must contain :weights"))
    rhs !== nothing || throw(ArgumentError("pc block must contain :rhs"))

    coord_mats = Matrix{Float64}[]
    for v in term_vars
        mat = get(block_dict, v, nothing)
        mat === nothing && throw(ArgumentError("pc block missing coordinate matrix for :$v"))
        push!(coord_mats, Matrix{Float64}(mat))
    end

    weights_mat = Matrix{Float64}(weights)
    rhs_vec = vec(Float64.(rhs))
    size(weights_mat) == size(coord_mats[1]) ||
        throw(ArgumentError("pc :weights dimensions must match coordinate matrices"))
    all(size(mat) == size(weights_mat) for mat in coord_mats) ||
        throw(ArgumentError("All pc coordinate matrices must have the same dimensions"))
    length(rhs_vec) == size(weights_mat, 1) ||
        throw(ArgumentError("pc :rhs length must match number of constraint rows"))

    return coord_mats, weights_mat, rhs_vec
end

function _pc_block_matrix(smooth::ConstructedSmooth, block)
    coord_mats, weights_mat, rhs_vec = _pc_block_data(block, smooth.spec.term_vars)
    names = Tuple(smooth.spec.term_vars)
    cols = map(vec, coord_mats)
    newdata = NamedTuple{names}(Tuple(cols))
    P = predict_matrix(smooth, newdata)

    n_rows, n_cols = size(weights_mat)
    A = zeros(n_rows, size(P, 2))
    for j in 1:n_cols
        rows = ((j - 1) * n_rows + 1):(j * n_rows)
        A .+= P[rows, :] .* weights_mat[:, j]
    end
    return A, rhs_vec
end

function _append_pc_constraints!(sm::ConstructedSmooth, data)
    pc = get(sm.spec.xt, :pc, nothing)
    pc === nothing && return sm

    pc_dict = pc isa AbstractDict ? Dict(Symbol(k) => v for (k, v) in pairs(pc)) :
        pc isa NamedTuple ? Dict(Symbol(k) => v for (k, v) in pairs(pc)) :
        throw(ArgumentError("pc must be a Dict or NamedTuple, got $(typeof(pc))"))

    ineq = get(pc_dict, :ineq, get(pc_dict, :inequality, nothing))
    eq = get(pc_dict, :eq, get(pc_dict, :equality, nothing))

    if ineq !== nothing
        Ain, bin = _pc_block_matrix(sm, ineq)
        _append_linear_constraints!(sm, Ain, bin, nothing, nothing)
    end
    if eq !== nothing
        Aeq, beq = _pc_block_matrix(sm, eq)
        _append_linear_constraints!(sm, nothing, nothing, Aeq, beq)
    end
    return sm
end
