# Cubic Regression Splines — bs="cr", bs="cs", bs="cc"
#
# Natural cubic spline basis with knots at data quantiles.
# Based on mgcv smooth.construct.cr.smooth.spec

"""
    _cr_basis(x, knots) -> (X, S)

Construct the natural cubic regression spline basis matrix and penalty.

The basis functions are the set of natural cubic splines with values 1 at
one knot and 0 at all others. The penalty is the integrated squared second
derivative: S_ij = ∫ B_i''(x) B_j''(x) dx.

Uses the band representation: the penalty for a natural cubic spline
with knots at k_1 < k_2 < ... < k_q is tridiagonal in the space of
second derivatives.
"""
function _cr_basis(x::AbstractVector{<:Real}, knots::Vector{Float64})
    n = length(x)
    q = length(knots)
    q >= 3 || throw(ArgumentError("Need ≥ 3 knots for cubic spline, got $q"))

    # Intervals
    h = diff(knots)

    # Build the tridiagonal system for natural cubic spline second derivatives
    # From Green & Silverman (1994), the matrices B and Q:
    # Q is (q-2) × q: Q_ij encodes the relationship between spline values and 2nd derivs
    # B is (q-2) × (q-2) symmetric tridiagonal

    # Q matrix (q-2) × q — encodes finite differences
    Q = zeros(q - 2, q)
    for i in 1:(q - 2)
        Q[i, i] = 1.0 / h[i]
        Q[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
        Q[i, i + 2] = 1.0 / h[i + 1]
    end

    # B matrix (q-2) × (q-2) symmetric tridiagonal — integrated products of linear basis
    B = zeros(q - 2, q - 2)
    for i in 1:(q - 2)
        B[i, i] = (h[i] + h[i + 1]) / 3.0
    end
    for i in 1:(q - 3)
        B[i, i + 1] = h[i + 1] / 6.0
        B[i + 1, i] = h[i + 1] / 6.0
    end

    # Penalty: S = Q' * B^{-1} * Q  (integrated squared second derivative)
    B_chol = cholesky(Symmetric(B))
    BinvQ = B_chol \ Q
    S = Q' * BinvQ

    # Compute full basis matrix using natural cubic spline interpolation
    X = _cr_basis_eval(x, knots, B_chol, Q, h)

    return X, Matrix(Symmetric(S))
end

"""
    _cr_basis_eval(x, knots, B_chol, Q, h) -> Matrix{Float64}

Evaluate all q natural cubic spline basis functions at points x.
Precomputes D = B⁻¹Q (all second derivatives) as a single matrix solve.
"""
function _cr_basis_eval(x::AbstractVector{<:Real}, knots::Vector{Float64},
    B_chol, Q::Matrix{Float64}, h::Vector{Float64})
    n = length(x)
    q = length(knots)
    X = zeros(n, q)

    # D_interior = B⁻¹ Q  is (q-2) × q — second derivatives at interior knots for each basis
    D_int = B_chol \ Q  # single matrix solve

    # Full second derivatives: (q × q), with zeros at endpoints (natural spline)
    DD = zeros(q, q)
    DD[2:(q - 1), :] .= D_int

    # Precompute scaled second derivatives: a[j,l] = h[j]² * DD[j,l] / 6
    # We only need DD[j,l] and DD[j+1,l] for interval j

    @inbounds for i in 1:n
        xi = clamp(x[i], knots[1], knots[end])
        j = searchsortedlast(knots, xi)
        j = clamp(j, 1, q - 1)
        t = (xi - knots[j]) / h[j]

        # Cubic spline: s_l(x) = (1-t)*δ_{j,l} + t*δ_{j+1,l}
        #   + h²/6 * [((1-t)³-(1-t))*DD[j,l] + (t³-t)*DD[j+1,l]]
        t1 = 1.0 - t
        c_left = (t1 * t1 * t1 - t1) * h[j]^2 / 6.0
        c_right = (t * t * t - t) * h[j]^2 / 6.0

        # Only two columns get the linear part
        X[i, j] += t1
        X[i, j + 1] += t

        # Cubic correction for all basis functions (vectorized over l)
        for l in 1:q
            X[i, l] += c_left * DD[j, l] + c_right * DD[j + 1, l]
        end
    end
    return X
end

function _smooth_construct(::CubicSpline, spec::SmoothSpec, data, user_knots)
    return _construct_cr(spec, data, user_knots; shrink = false, cyclic = false)
end

function _smooth_construct(::CubicShrink, spec::SmoothSpec, data, user_knots)
    return _construct_cr(spec, data, user_knots; shrink = true, cyclic = false)
end

function _smooth_construct(::CyclicCubic, spec::SmoothSpec, data, user_knots)
    return _construct_cr(spec, data, user_knots; shrink = false, cyclic = true)
end

function _construct_cr(spec::SmoothSpec, data, user_knots;
    shrink::Bool = false, cyclic::Bool = false)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Cubic splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)
    k = min(spec.k, n)

    # Place knots at quantiles of x
    knots = if user_knots !== nothing
        Float64.(user_knots)
    else
        place_knots(x, k)
    end
    k = length(knots)

    if cyclic
        X, S = _cc_basis(x, knots)
        null_dim = 1  # only intercept in null space for cyclic
    else
        X, S = _cr_basis(x, knots)
        null_dim = 2  # constant + linear in null space
    end

    penalties = Matrix{Float64}[S]

    # Shrinkage: add penalty on null space
    if shrink && !cyclic
        # Small penalty on full space to shrink toward zero
        S_shrink = I(k) |> Matrix{Float64}
        S_shrink .*= 1e-2 * tr(S) / k
        push!(penalties, S_shrink)
    end

    pen_rank = k - null_dim

    # Absorb identifiability constraints
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons, knots, null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

"""
    _cc_basis(x, knots) -> (X, S)

Cyclic cubic regression spline basis. Wraps around so that
the function value and first two derivatives match at the boundaries.
"""
function _cc_basis(x::AbstractVector{<:Real}, knots::Vector{Float64})
    q = length(knots)
    q >= 4 || throw(ArgumentError("Need ≥ 4 knots for cyclic spline"))
    n = length(x)

    lo, hi = knots[1], knots[end]
    period = hi - lo

    # Map x to [lo, hi) via modular arithmetic
    x_mod = lo .+ mod.(x .- lo, period)

    # For cyclic spline, wrap the knots — merge first and last
    h = diff(knots)

    # Build cyclic versions of B and Q (periodic boundary conditions)
    # B is now q-1 × q-1, wrapping around
    q_int = q - 1  # effective number of basis functions (last = first)
    B = zeros(q_int, q_int)
    Q = zeros(q_int, q_int)

    # Cyclic finite difference matrix
    for i in 1:q_int
        ip = mod1(i + 1, q_int)
        im = mod1(i - 1, q_int)
        hi_cur = i <= length(h) ? h[i] : h[1]
        hi_prev = i > 1 ? h[i - 1] : h[end - 1]

        B[i, i] = (hi_prev + hi_cur) / 3.0
        B[i, ip] = hi_cur / 6.0
        B[ip, i] = hi_cur / 6.0

        Q[i, i] = -(1.0 / hi_prev + 1.0 / hi_cur)
        Q[i, ip] = 1.0 / hi_cur
        Q[i, im] = 1.0 / hi_prev
    end

    # Penalty
    B_sym = Symmetric(B + B') / 2
    B_chol = cholesky(B_sym)
    S = Q' * (B_chol \ Q)

    # Basis evaluation (simplified — use cardinal spline approach)
    X = zeros(n, q_int)
    for l in 1:q_int
        e_l = zeros(q_int)
        e_l[l] = 1.0
        dd = B_chol \ (Q * e_l)

        for i in 1:n
            xi = x_mod[i]
            j = searchsortedlast(knots[1:(end - 1)], xi)
            j = clamp(j, 1, q_int)
            j_next = mod1(j + 1, q_int)
            hj = j <= length(h) ? h[j] : period - knots[end - 1] + knots[1]
            t = (xi - knots[j]) / hj

            X[i, l] = (1 - t) * e_l[j] + t * e_l[j_next] +
                       hj^2 * (
                ((1 - t)^3 - (1 - t)) * dd[j] +
                (t^3 - t) * dd[j_next]
            ) / 6.0
        end
    end

    return X, Matrix(Symmetric(S))
end

function _predict_matrix(::Union{CubicSpline, CubicShrink, CyclicCubic},
    smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    x_new = Float64.(Tables.getcolumn(newdata, var))
    knots = smooth.knots
    cyclic = smooth.spec.basis isa CyclicCubic

    if cyclic
        X_new, _ = _cc_basis(x_new, knots)
    else
        h = diff(knots)
        q = length(knots)
        Q = zeros(q - 2, q)
        for i in 1:(q - 2)
            Q[i, i] = 1.0 / h[i]
            Q[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
            Q[i, i + 2] = 1.0 / h[i + 1]
        end
        B = zeros(q - 2, q - 2)
        for i in 1:(q - 2)
            B[i, i] = (h[i] + h[i + 1]) / 3.0
        end
        for i in 1:(q - 3)
            B[i, i + 1] = h[i + 1] / 6.0
            B[i + 1, i] = h[i + 1] / 6.0
        end
        B_chol = cholesky(Symmetric(B))
        X_new = _cr_basis_eval(x_new, knots, B_chol, Q, h)
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = nullspace(C)
        return X_new * Z
    end
    return X_new
end
