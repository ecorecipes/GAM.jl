# P-splines: B-spline basis with difference penalty — bs="ps"
#
# Based on Eilers & Marx (1996). Flexible smoothing with B-splines and penalties.
# The basis is a B-spline of given order, and the penalty is the squared
# d-th order difference of the coefficients.

"""
    _bspline_basis(x, knots, order) -> Matrix{Float64}

Evaluate B-spline basis functions of given `order` at points `x` using
the Cox-de Boor recursion.

`knots` should include the boundary knots repeated `order` times.
"""
function _bspline_basis(x::AbstractVector{<:Real}, knots::Vector{Float64}, order::Int)
    n = length(x)
    nk = length(knots)
    n_basis = nk - order
    n_basis >= 1 || throw(ArgumentError("Not enough knots for B-spline of order $order"))

    B = zeros(n, n_basis)

    for i in 1:n
        xi = x[i]
        # Order 1 (piecewise constant)
        b_prev = zeros(nk - 1)
        for j in 1:(nk - 1)
            if j == nk - 1
                b_prev[j] = (knots[j] <= xi <= knots[j + 1]) ? 1.0 : 0.0
            else
                b_prev[j] = (knots[j] <= xi < knots[j + 1]) ? 1.0 : 0.0
            end
        end

        # Recursion for higher orders
        for p in 2:order
            b_curr = zeros(nk - p)
            for j in 1:(nk - p)
                denom1 = knots[j + p - 1] - knots[j]
                denom2 = knots[j + p] - knots[j + 1]
                t1 = denom1 > 0 ? (xi - knots[j]) / denom1 * b_prev[j] : 0.0
                t2 = denom2 > 0 ? (knots[j + p] - xi) / denom2 * b_prev[j + 1] : 0.0
                b_curr[j] = t1 + t2
            end
            b_prev = b_curr
        end

        B[i, :] .= b_prev[1:n_basis]
    end
    return B
end

"""
    _diff_penalty(k, d) -> Matrix{Float64}

Construct the d-th order difference penalty matrix D'D of size k × k.
D is the (k-d) × k finite difference matrix.
"""
function _diff_penalty(k::Int, d::Int)
    d >= 0 || throw(ArgumentError("penalty order d must be ≥ 0"))
    d < k || throw(ArgumentError("penalty order d=$d must be < k=$k"))

    # Start with identity
    D = Matrix{Float64}(I, k, k)

    # Apply differencing d times
    for _ in 1:d
        m = size(D, 1)
        D_new = zeros(m - 1, size(D, 2))
        for i in 1:(m - 1)
            D_new[i, :] .= D[i + 1, :] .- D[i, :]
        end
        D = D_new
    end

    return D' * D
end

function _smooth_construct(::PSpline, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("P-splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m  # difference penalty order
    spline_order = m_order + 2  # B-spline order (degree + 1), default = cubic

    # R's mgcv P-spline knot placement (smooth.construct.pspline.smooth.spec):
    # nk = k - m[2] + 1 evenly-spaced knots from min(x) to max(x)
    # Then extend by m[2] = (spline_order-1) knots on each side
    m2 = spline_order - 1  # = degree
    nk = k - m2 + 1
    nk >= 2 || throw(ArgumentError(
        "k=$k too small for P-spline of order $spline_order (need k ≥ $(m2 + 2))"))

    lo, hi = minimum(x), maximum(x)

    if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] : (hi - lo)
        knot_vec = vcat(
            [interior[1] - dk * i for i in m2:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:m2],
        )
    else
        k_new = range(lo, hi, length=nk) |> collect
        dk = k_new[2] - k_new[1]
        knot_vec = vcat(
            [k_new[1] - dk * i for i in m2:-1:1],
            k_new,
            [k_new[end] + dk * i for i in 1:m2],
        )
    end

    # B-spline basis
    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)

    # Difference penalty
    S = _diff_penalty(actual_k, m_order)
    penalties = Matrix{Float64}[S]

    null_dim = m_order  # polynomials of degree < m_order are in null space
    pen_rank = actual_k - null_dim

    # Absorb constraints
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knot_vec,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

function _smooth_construct(::BSplineBasis, spec::SmoothSpec, data, user_knots)
    # B-spline basis with integrated squared derivative penalty
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("B-splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m  # derivative penalty order
    spline_order = m_order + 2

    m2 = spline_order - 1
    nk = k - m2 + 1
    nk >= 2 || throw(ArgumentError(
        "k=$k too small for B-spline of order $spline_order (need k ≥ $(m2 + 2))"))

    lo, hi = minimum(x), maximum(x)

    if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] : (hi - lo)
        knot_vec = vcat(
            [interior[1] - dk * i for i in m2:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:m2],
        )
    else
        k_new = range(lo, hi, length=nk) |> collect
        dk = k_new[2] - k_new[1]
        knot_vec = vcat(
            [k_new[1] - dk * i for i in m2:-1:1],
            k_new,
            [k_new[end] + dk * i for i in 1:m2],
        )
    end

    X = _bspline_basis(x, knot_vec, spline_order)
    actual_k = size(X, 2)

    # Integrated squared derivative penalty via Gauss-Legendre quadrature
    S = _derivative_penalty(knot_vec, spline_order, m_order, actual_k)
    penalties = Matrix{Float64}[S]

    null_dim = m_order
    pen_rank = actual_k - null_dim

    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knot_vec,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

"""
    _derivative_penalty(knots, order, deriv_order, n_basis) -> Matrix{Float64}

Compute integrated squared derivative penalty for B-splines using
Gauss-Legendre quadrature: S[i,j] = ∫ B_i^(d)(x) B_j^(d)(x) dx
"""
function _derivative_penalty(knots::Vector{Float64}, order::Int,
    deriv_order::Int, n_basis::Int)
    # Use a fine grid for numerical integration
    lo = knots[order]
    hi = knots[end - order + 1]
    n_quad = max(n_basis * 5, 100)
    h = (hi - lo) / n_quad

    S = zeros(n_basis, n_basis)

    # Evaluate d-th derivative of each B-spline at quadrature points
    # B-spline derivative: B_i^(d)(x) = (order-1) * [B_{i,order-1}(x)/(knots[i+order-1]-knots[i])
    #                                   - B_{i+1,order-1}(x)/(knots[i+order]-knots[i+1])]
    # Recursively apply d times to get d-th derivative, then use midpoint rule

    for q in 0:(n_quad - 1)
        x_mid = lo + (q + 0.5) * h
        # Compute d-th derivative basis at x_mid
        bd = _bspline_deriv_at(x_mid, knots, order, deriv_order, n_basis)
        S .+= h .* (bd * bd')
    end

    return Symmetric(S) |> Matrix
end

"""
Evaluate the deriv_order-th derivative of all n_basis B-splines at point x.
Uses the recursive derivative formula for B-splines.
"""
function _bspline_deriv_at(x::Float64, knots::Vector{Float64},
    order::Int, deriv_order::Int, n_basis::Int)
    if deriv_order == 0
        # Just evaluate the basis
        row = zeros(1, n_basis)
        B = _bspline_basis([x], knots, order)
        return vec(B)
    end

    if deriv_order >= order
        return zeros(n_basis)
    end

    # Derivative of order-p B-spline: (p-1) * [B_{i,p-1}/(knots[i+p-1]-knots[i]) - ...]
    # Use finite differences on B-spline basis of lower order
    reduced_order = order - deriv_order
    if reduced_order < 1
        return zeros(n_basis)
    end

    # For the d-th derivative, we need B-splines of order (order-d)
    # evaluated at x, then scaled by factorial terms
    nk = length(knots)
    n_reduced = nk - reduced_order

    B_low = _bspline_basis([x], knots, reduced_order)
    b_low = vec(B_low)

    # d-th derivative involves a linear combination of lower-order splines
    # Use the standard recurrence: B_i^(d) = (p-1)/(t_{i+p-1} - t_i) B_{i,p-1}^(d-1)
    #                                       - (p-1)/(t_{i+p} - t_{i+1}) B_{i+1,p-1}^(d-1)
    # Applying d times gives a scaled differencing of order-d splines
    coeffs = ones(n_reduced)
    col_indices = collect(1:n_reduced)

    for dd in 1:deriv_order
        p = order - dd + 1  # current spline order
        new_coeffs = Float64[]
        new_indices = Int[]
        for (c, j) in zip(coeffs, col_indices)
            denom = knots[j + p - 1] - knots[j]
            if denom > 0
                push!(new_coeffs, c * (p - 1) / denom)
                push!(new_indices, j)
            end
            if j + 1 <= nk - (p - 1)
                denom2 = knots[j + p] - knots[j + 1]
                if denom2 > 0
                    push!(new_coeffs, -c * (p - 1) / denom2)
                    push!(new_indices, j + 1)
                end
            end
        end
        coeffs = new_coeffs
        col_indices = new_indices
    end

    result = zeros(n_basis)
    for (c, j) in zip(coeffs, col_indices)
        if 1 <= j <= length(b_low)
            result[min(j, n_basis)] += c * b_low[j]
        end
    end

    return result
end

function _predict_matrix(::Union{PSpline, BSplineBasis},
    smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    knots = smooth.knots
    m_order = smooth.spec.m === nothing ? 2 : smooth.spec.m
    spline_order = m_order + 2

    X_new = _bspline_basis(x_new, knots, spline_order)

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = nullspace(C)
        return X_new * Z
    end
    return X_new
end

# ============================================================================
# Cyclic P-splines: periodic B-spline basis with cyclic difference penalty
# ============================================================================

"""
    _cyclic_diff_penalty(k, d) -> Matrix{Float64}

Construct the d-th order cyclic difference penalty matrix D'D of size k × k.
D is the k × k cyclic finite difference matrix where rows wrap around.
"""
function _cyclic_diff_penalty(k::Int, d::Int)
    d >= 0 || throw(ArgumentError("penalty order d must be ≥ 0"))
    d < k || throw(ArgumentError("penalty order d=$d must be < k=$k"))

    D = zeros(k, k)
    @inbounds for i in 1:k
        for j in 0:d
            col = mod1(i + j, k)
            D[i, col] += (-1)^(d - j) * binomial(d, j)
        end
    end
    return D' * D
end

function _smooth_construct(::CyclicPSpline, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Cyclic P-splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    k = min(spec.k, n)
    m_order = spec.m === nothing ? 2 : spec.m  # difference penalty order
    spline_order = m_order + 2  # B-spline order (degree + 1)
    degree = spline_order - 1

    ndx = k  # number of intervals = number of basis functions after wrapping
    ndx >= 2 || throw(ArgumentError(
        "k=$k too small for cyclic P-spline (need k ≥ 2)"))

    lo, hi = minimum(x), maximum(x)
    period = hi - lo
    period > 0 || throw(ArgumentError("Data range is zero — cannot build cyclic basis"))

    if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] : period
    else
        # Place ndx+1 evenly spaced knots spanning [lo, hi]
        interior = collect(range(lo, hi; length = ndx + 1))
        dk = interior[2] - interior[1]
    end

    # Extend by degree knots on each side for B-spline evaluation
    knot_vec = vcat(
        [interior[1] - dk * i for i in degree:-1:1],
        interior,
        [interior[end] + dk * i for i in 1:degree],
    )

    # B-spline basis (n × (ndx + degree))
    X_full = _bspline_basis(x, knot_vec, spline_order)

    # Wrap: add the last `degree` columns onto the first `degree` columns
    X = X_full[:, 1:ndx]
    @inbounds for j in 1:degree
        X[:, j] .+= X_full[:, ndx + j]
    end

    # Cyclic difference penalty
    S = _cyclic_diff_penalty(ndx, m_order)
    penalties = Matrix{Float64}[S]

    # For cyclic d-th order differences, only constants are in the null space
    null_dim = 1
    pen_rank = ndx - null_dim

    # Absorb constraints
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        knot_vec,
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

function _predict_matrix(::CyclicPSpline, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    knots = smooth.knots
    m_order = smooth.spec.m === nothing ? 2 : smooth.spec.m
    spline_order = m_order + 2
    degree = spline_order - 1

    # Recover period from knot vector
    lo = knots[degree + 1]
    hi = knots[end - degree]
    period = hi - lo

    # Map to [lo, hi) via modular arithmetic
    x_mod = lo .+ mod.(x_new .- lo, period)

    # Build B-spline basis at mapped points
    X_full = _bspline_basis(x_mod, knots, spline_order)
    ndx = size(X_full, 2) - degree

    # Wrap columns
    X = X_full[:, 1:ndx]
    @inbounds for j in 1:degree
        X[:, j] .+= X_full[:, ndx + j]
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = nullspace(C)
        return X * Z
    end
    return X
end
