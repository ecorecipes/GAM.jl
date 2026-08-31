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
        xi = x[i]

        # Linear extrapolation beyond the boundary knots (natural-spline
        # boundary condition f'' = 0 implies linear tails), matching mgcv's
        # Predict.matrix.cr.smooth rather than clamping to a constant.
        if xi < knots[1] || xi > knots[end]
            at_lo = xi < knots[1]
            j = at_lo ? 1 : q - 1
            xb = at_lo ? knots[1] : knots[end]
            hj = h[j]
            for l in 1:q
                # Basis value at the boundary knot
                vb = (at_lo ? (l == 1 ? 1.0 : 0.0) : (l == q ? 1.0 : 0.0))
                # First derivative at the boundary (from the boundary
                # interval's cubic; DD is zero at the end knots)
                if at_lo
                    db = ((l == 2 ? 1.0 : 0.0) - (l == 1 ? 1.0 : 0.0)) / hj -
                         hj * DD[2, l] / 6.0
                else
                    db = ((l == q ? 1.0 : 0.0) - (l == q - 1 ? 1.0 : 0.0)) / hj +
                         hj * DD[q - 1, l] / 6.0
                end
                X[i, l] = vb + (xi - xb) * db
            end
            continue
        end

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
    shrink::Bool = false, cyclic::Bool = false, absorb_cons::Bool = true)
    length(spec.term_vars) == 1 ||
        throw(ArgumentError("Cubic splines only support 1d smooths"))
    var = spec.term_vars[1]
    x = Float64.(Tables.getcolumn(data, var))
    n = length(x)

    # A knot-based basis cannot carry more basis functions than there are
    # distinct covariate values; mgcv raises "x has insufficient unique values
    # to support k knots" rather than silently shrinking the basis.
    if user_knots === nothing
        n_unique = length(unique(x))
        n_unique >= spec.k || throw(ArgumentError(
            "s($var) has fewer unique covariate combinations ($n_unique) " *
            "than the basis dimension k=$(spec.k); reduce k (mgcv raises the " *
            "same error)"))
    end

    k = min(spec.k, n)

    # Place knots at quantiles of x
    knots = if user_knots !== nothing
        uk = Float64.(user_knots)
        # mgcv's convention for a cyclic basis: exactly two knots specify the
        # PERIOD endpoints (`knots = list(week = c(0, 52))`) and the interior
        # knots are filled in evenly; any other count is the full knot vector.
        # Without this, a cyclic smooth always took its period from the
        # observed data range, so a series covering weeks 0-51 wrapped over a
        # 51-week year and f(0) differed from f(52) by 0.135 on an otherwise
        # exactly periodic signal.
        if cyclic && length(uk) == 2
            collect(range(uk[1], uk[2]; length = k))
        else
            uk
        end
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

    # Shrinkage (cs): as in mgcv, modify the SINGLE penalty by raising the
    # null-space eigenvalues (one smoothing parameter, full-rank penalty),
    # rather than appending a second penalty.
    if shrink && !cyclic
        # mgcv's `cs` CASCADES the two null eigenvalues (R/smooth.r:1495-1496),
        # unlike `ts`, which gives them all the same value. Use mgcv's rule.
        #
        # Measured against mgcv 1.9-4 (n=200), holding sp FIXED at mgcv's own
        # selected value so the smoothing-parameter optimizer cannot contaminate
        # the comparison:
        #
        #     k       cascade Δedf      flat Δedf
        #     5       4.44e-15          2.73e-05
        #     10      0.00e+00          5.12e-05
        #     20      5.33e-15          3.47e-05
        #
        # Cascade reproduces mgcv's edf to machine precision; the flat rule is
        # ~10 orders of magnitude worse. Do not "simplify" this back to flat.
        #
        # On the eigenbasis question, since it is easy to get wrong: the two
        # null eigenvalues ARE numerically degenerate — their gap (1.4e-13 at
        # k=5, 5.0e-11 at k=20) sits below eps*‖S‖ (4.0e-13, 7.1e-11) — so the
        # basis LAPACK returns within that 2-D subspace is arbitrary, and it
        # genuinely differs from R's (exactly 2 columns disagree, |cos| ≈ 0.89
        # to 0.94; syevr!/syevd!/syev! all disagree with R and with each other,
        # and R links reference LAPACK 3.12.1 while Julia links OpenBLAS).
        # Reconstructing the penalty from mgcv's own eigenvectors reproduces
        # mgcv exactly (rel 1e-16); from ours it is off by rel 1e-3 (k=5) to
        # 1e-6 (k=20). But that perturbation lives in the null block and does
        # NOT propagate to the fit: at fixed sp the edf still matches to 1e-15
        # above. The residual free-fit gap (~0.15 edf at k=15) is the EFS vs
        # outer-Newton smoothing-parameter optimizer, not the eigenbasis.
        S = _shrink_penalty(S; null_dim = null_dim, cascade = true)
        null_dim = 0
    end

    penalties = Matrix{Float64}[S]

    # Penalty rank = (number of basis columns before constraint absorption)
    # minus the penalty null-space dimension. The cyclic basis has k-1
    # columns (last knot ≡ first) with only the constant in the null space,
    # so its rank is k-2; the non-cyclic basis has k columns and rank k-2
    # (k for the full-rank shrinkage variant).
    n_col = cyclic ? k - 1 : k
    pen_rank = n_col - null_dim

    # `absorb_cons = false` returns the RAW basis, matching what mgcv's
    # `smooth.construct` hands back before `smoothCon` applies identifiability
    # constraints. Only `bs=:sz` uses it (as a marginal, where the per-level
    # constants must stay in the span); the default leaves this path exactly
    # as it was.
    if !absorb_cons
        return ConstructedSmooth(
            spec, X, penalties, knots, null_dim, pen_rank,
            nothing, nothing, 0, 0,
            nothing, nothing, nothing,
            Int[],
        )
    end

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
        # Cyclic indexing: h_0 ≡ h_{q-1} (Wood 2017 §5.4.2). For i = 1 the
        # wrap-around (previous) interval is the last one, h[end] =
        # knots[end] - knots[end-1], since knots[end] ≡ knots[1].
        hi_prev = i > 1 ? h[i - 1] : h[end]

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

    # Basis evaluation via cardinal splines. DD[:, l] holds the second
    # derivatives at the knots for basis function l (all l at once via a
    # single matrix solve), so the point loop is O(n·q) total.
    DD = B_chol \ Q   # q_int × q_int
    interior_knots = @view knots[1:(end - 1)]
    X = zeros(n, q_int)
    @inbounds for i in 1:n
        xi = x_mod[i]
        j = searchsortedlast(interior_knots, xi)
        j = clamp(j, 1, q_int)
        j_next = mod1(j + 1, q_int)
        hj = h[j]
        t = (xi - knots[j]) / hj
        t1 = 1.0 - t
        c_left = (t1 * t1 * t1 - t1) * hj^2 / 6.0
        c_right = (t * t * t - t) * hj^2 / 6.0

        X[i, j] += t1
        X[i, j_next] += t
        for l in 1:q_int
            X[i, l] += c_left * DD[j, l] + c_right * DD[j_next, l]
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
        Z = _constraint_basis(C, size(X_new, 2))
        return X_new * Z
    end
    return X_new
end
