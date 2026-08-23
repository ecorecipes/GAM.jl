# Thin Plate Regression Splines (TPRS) — bs="tp" and bs="ts"
#
# Implements the truncated eigen-decomposition approach from:
# Wood, S.N. (2003). Thin plate regression splines. JRSSB 65(1), 95-114.
#
# Matches mgcv's tprs_setup() in tprs.c exactly:
# 1. Compute E (semi-kernel) with normalization constant eta_const(m,d)
# 2. Eigendecompose E to get top k eigenpairs (U, v)
# 3. Apply constraint: form TU = T'U, QR factorize → null space Z
# 4. Basis X = U·diag(v)·Z ∪ T  (eigenvalues absorbed into basis)
# 5. Penalty S = Z'·diag(v)·Z  (penalty has eigenvalues, not inverses)
# 6. Column-wise RMS rescaling of X, S, UZ

struct TPRSPredictCache <: AbstractSmoothPredictCache
    centers::Matrix{Float64}
    UZ::Matrix{Float64}
    col_scales::Vector{Float64}
end

"""
    _eta_const(m::Int, d::Int) -> Float64

Normalization constant for TPS semi-kernel η_{m,d}.
Matches mgcv's eta_const() in tprs.c exactly.
"""
function _eta_const(m::Int, d::Int)
    d2 = d ÷ 2
    if iseven(d)
        # d even
        f = iseven(m + 1 + d2) ? 1.0 : -1.0
        for _ in 1:(2m - 1)
            f /= 2.0
        end
        for _ in 1:d2
            f /= π
        end
        for i in 2:(m - 1)
            f /= i
        end
        for i in 2:(m - d2)
            f /= i
        end
    else
        # d odd
        Ghalf = sqrt(π)
        f = Ghalf
        k = m - (d - 1) ÷ 2
        for i in 0:(k - 1)
            f /= (-0.5 - i)
        end
        for _ in 1:m
            f /= 4.0
        end
        for _ in 1:d2
            f /= π
        end
        f /= Ghalf  # dividing by pi^(d/2) when d odd
        for i in 2:(m - 1)
            f /= i
        end
    end
    return f
end

"""
    _tps_eta(r::Real, m::Int, d::Int)

Evaluate the TPS radial basis function η_md(r) with correct normalization.
Matches mgcv's fast_eta() in tprs.c. Note: r is the DISTANCE (not r²).
For d odd:  η(r) = C * r^(2m-d)
For d even: η(r) = C * r^(2m-d) * log(r)  (with η(0) = 0)
"""
function _tps_eta(r::Real, m::Int, d::Int)
    if r ≤ 0
        return 0.0
    end
    f = _eta_const(m, d)
    r2 = r * r  # r² (matching R's convention where fast_eta receives r²)
    power = 2m - d
    if iseven(d)
        f *= log(r2) * 0.5  # log(r) = log(r²)/2
        d2 = d ÷ 2
        for _ in 1:(m - d2)
            f *= r2
        end
    else
        d2 = d ÷ 2
        for _ in 1:(m - d2 - 1)
            f *= r2
        end
        f *= r  # sqrt(r²) = r
    end
    return f
end

"""
    _tps_penalty_matrix(x::AbstractVector, m::Int) -> Matrix{Float64}

Compute the TPS penalty (semi-kernel) matrix E for 1d data.
E_ij = η_md(|x_i - x_j|) where d=1.
"""
function _tps_penalty_matrix(x::AbstractVector{<:Real}, m::Int)
    n = length(x)
    E = zeros(n, n)
    d = 1  # 1d smooth
    for j in 1:n, i in j:n
        r = abs(x[i] - x[j])
        E[i, j] = _tps_eta(r, m, d)
        E[j, i] = E[i, j]
    end
    return E
end

"""
    _tps_null_space_basis(x::AbstractVector, m::Int) -> Matrix{Float64}

Polynomial null space basis T for 1d TPS. Columns are [1, x, x², ..., x^(m-1)].
The null space dimension M = m for 1d.
"""
function _tps_null_space_basis(x::AbstractVector{<:Real}, m::Int)
    n = length(x)
    M = m  # null space dimension for 1d
    T = zeros(n, M)
    for j in 1:M
        @. T[:, j] = x^(j - 1)
    end
    return T
end

"""
    _tps_multi_penalty_matrix(X_data::Matrix{Float64}, m::Int) -> Matrix{Float64}

Compute the TPS penalty matrix for multi-dimensional data.
"""
function _tps_multi_penalty_matrix(X_data::Matrix{Float64}, m::Int)
    n, d = size(X_data)
    E = zeros(n, n)
    for j in 1:n, i in j:n
        r = norm(view(X_data, i, :) .- view(X_data, j, :))
        E[i, j] = _tps_eta(r, m, d)
        E[j, i] = E[i, j]
    end
    return E
end

"""
    _tps_monomial_exponents(d::Int, m::Int) -> Vector{Vector{Int}}

All monomial exponent vectors of total degree < m in d variables, in
graded order. Their count is binomial(m + d - 1, d) — the TPS null-space
dimension.
"""
function _tps_monomial_exponents(d::Int, m::Int)
    exps = Vector{Vector{Int}}()
    for deg in 0:(m - 1)
        # Enumerate all d-vectors of nonnegative integers summing to deg
        stack = [(Int[], deg)]
        while !isempty(stack)
            prefix, rem = pop!(stack)
            if length(prefix) == d - 1
                push!(exps, vcat(prefix, rem))
            else
                for e in rem:-1:0
                    push!(stack, (vcat(prefix, e), rem - e))
                end
            end
        end
    end
    return exps
end

"""
    _tps_multi_null_basis(X_data::Matrix{Float64}, m::Int) -> Matrix{Float64}

Polynomial null space for d-dimensional TPS: ALL monomials of total
degree < m. Dimension M = binomial(m + d - 1, d).
"""
function _tps_multi_null_basis(X_data::Matrix{Float64}, m::Int)
    n, d = size(X_data)
    exps = _tps_monomial_exponents(d, m)
    M = binomial(m + d - 1, d)
    length(exps) == M || error("TPS null basis enumeration bug: got " *
        "$(length(exps)) monomials, expected $M")
    T = ones(n, M)
    for (col, e) in enumerate(exps)
        for j in 1:d
            if e[j] > 0
                T[:, col] .*= X_data[:, j] .^ e[j]
            end
        end
    end
    return T
end

"""
    _tps_default_m(d::Int) -> Int

mgcv's default TPS penalty order: the smallest m with 2m > d + 1
(m=2 for d ≤ 2, m=3 for d = 3, 4, ...).
"""
function _tps_default_m(d::Int)
    m = 2
    while 2m <= d + 1
        m += 1
    end
    return m
end

function _tps_cross_matrix(X_new::Matrix{Float64}, centers::Matrix{Float64}, m::Int)
    n_new, d = size(X_new)
    nk = size(centers, 1)
    E = zeros(n_new, nk)
    if d == 1
        @inbounds for j in 1:nk, i in 1:n_new
            E[i, j] = _tps_eta(abs(X_new[i, 1] - centers[j, 1]), m, 1)
        end
    else
        @inbounds for j in 1:nk, i in 1:n_new
            E[i, j] = _tps_eta(norm(view(X_new, i, :) .- view(centers, j, :)), m, d)
        end
    end
    return E
end

function _scale_columns!(X::Matrix{Float64}, scales::Vector{Float64})
    @inbounds for j in 1:size(X, 2)
        scale = scales[j]
        if scale > 0
            X[:, j] ./= scale
        end
    end
    return X
end

function _smooth_construct(::ThinPlateSpline, spec::SmoothSpec, data, knots)
    return _construct_tprs(spec, data, knots; shrink = false)
end

function _smooth_construct(::ThinPlateShrink, spec::SmoothSpec, data, knots)
    return _construct_tprs(spec, data, knots; shrink = true)
end

"""
    _construct_tprs(spec, data, knots; shrink=false)

Core TPRS construction matching mgcv's tprs_setup() in tprs.c.

Steps (matching R exactly):
1. Compute E (semi-kernel matrix with normalization constant) and T (polynomial null space)
2. Eigendecompose E: top k eigenpairs (U, v) using largest ABSOLUTE values
3. Constraint handling: TU = T'U, QR → null space Z
4. Build X = U·diag(v)·Z ∪ T (eigenvalues absorbed into basis columns)
5. Build S = Z'·diag(v)·Z (penalty has eigenvalues on diagonal before Z rotation)
6. Column-wise RMS rescaling of X, S (R's lines 493-498)
"""
function _construct_tprs(spec::SmoothSpec, data, knots; shrink::Bool = false,
    absorb_cons::Bool = true)
    vars = spec.term_vars
    d = length(vars)
    # mgcv default-order rule: smallest m with 2m > d + 1. A user-supplied m
    # must satisfy 2m > d for the thin-plate kernel to exist.
    m_order = spec.m === nothing ? _tps_default_m(d) : spec.m
    2 * m_order > d || throw(ArgumentError(
        "TPS penalty order m=$m_order invalid for d=$d covariates: " *
        "need 2m > d (mgcv default is m=$(_tps_default_m(d)))"))
    k = spec.k

    # Extract data and choose knots (subsampling data for large n, in any
    # dimension, to avoid an O(n³) eigendecomposition)
    if d == 1
        x = Float64.(Tables.getcolumn(data, vars[1]))
        Xd = reshape(x, :, 1)
    else
        Xd = hcat([Float64.(Tables.getcolumn(data, v)) for v in vars]...)
    end
    n = size(Xd, 1)
    k = min(k, n)

    # mgcv-style guard: the basis needs at least k distinct covariate
    # combinations; with fewer, knot selection either fails outright or
    # produces a silently rank-deficient (edf ≈ 0) basis.
    if knots === nothing
        n_unique = d == 1 ? length(unique(vec(Xd))) :
                   length(unique(collect(eachrow(Xd))))
        k <= n_unique || throw(ArgumentError(
            "s($(join(vars, ","))) has fewer unique covariate combinations " *
            "($n_unique) than the basis dimension k=$k; reduce k (mgcv " *
            "raises the same error)"))
    end

    if knots !== nothing
        d == 1 || throw(ArgumentError(
            "user-supplied knots are not supported for multi-dimensional " *
            "TPS smooths"))
        length(knots) >= k || throw(ArgumentError(
            "user-supplied knots for a TPS smooth must have length ≥ k=$k, " *
            "got $(length(knots))"))
        XK = reshape(Float64.(knots[1:k]), :, 1)
    elseif d == 1 && n > max(k * 3, 200)
        XK = reshape(place_knots(vec(Xd), k), :, 1)
    elseif d > 1 && n > max(k * 3, 200)
        # Deterministic subsample of unique rows (evenly spaced in the
        # lexicographic ordering) — caps the eigenproblem like mgcv's
        # max.knots default
        rows = unique(collect(eachrow(Xd)))
        nk_target = min(length(rows), max(4 * k, 200))
        sel = round.(Int, range(1, length(rows); length = nk_target))
        XK = Matrix(reduce(hcat, rows[unique(sel)])')
    else
        XK = Xd
    end
    knots_are_data = XK === Xd || (size(XK) == size(Xd) && XK == Xd)

    if d == 1
        E = _tps_penalty_matrix(vec(XK), m_order)
        T_null = _tps_null_space_basis(vec(XK), m_order)
    else
        E = _tps_multi_penalty_matrix(Matrix(XK), m_order)
        T_null = _tps_multi_null_basis(Matrix(XK), m_order)
    end
    M = size(T_null, 2)

    k >= M + 1 || throw(ArgumentError(
        "basis dimension k=$k too small for penalty order m=$m_order " *
        "(need k ≥ $(M + 1) = null_dim + 1)"))

    nk = size(XK, 1)

    # --- Step 2: Eigendecomposition of E ---
    # R uses Lanczos for top k eigenpairs sorted by ABSOLUTE value (tprs.c line 408).
    # R's Rlanczos with minus=-1 returns k eigenvectors sorted by decreasing |eigenvalue|.
    eig = eigen(Symmetric(E))
    # Sort by absolute value (descending) to match R's convention
    idx = sortperm(abs.(eig.values); rev = true)
    U = eig.vectors[:, idx[1:k]]   # nk × k eigenvectors
    v = eig.values[idx[1:k]]       # k eigenvalues (may include negative ones)

    # --- Step 3: Constraint handling via T'U null space ---
    # Form TU = T'U (M × k)
    TU = T_null' * U   # M × k

    # QR factorize TU' to find null space Z of TU
    # TU·Z = 0 means Z spans the null space of TU (k × (k-M))
    # R uses QT factorization: TU Q = [0, B] → Q = [Z, Y]
    # In Julia: QR of TU' gives us the null space
    qr_TU = qr(TU')
    # Need FULL Q (k × k), not thin Q. Multiply Q by identity to expand.
    Q_full = qr_TU.Q * Matrix(I, k, k)  # k × k orthogonal matrix
    Z = Q_full[:, (M + 1):k]   # k × (k-M)

    n_basis = k - M  # number of non-null basis functions

    # --- Step 4: Build design matrix ---
    # R: X = U·diag(v)·Z ∪ T  (eigenvalues absorbed into eigenvector columns)
    # For data-as-knots case: X_eig = U·diag(v)·Z directly
    # For knot-based case: Nystrom extension X_eig = E_nk·U·Z (the Nystrom
    # factor diag(1/v) cancels against the absorbed diag(v))

    if !knots_are_data
        E_nk = _tps_cross_matrix(Xd, Matrix(XK), m_order)
        X_eig = E_nk * (U * Z)
        T_data = d == 1 ? _tps_null_space_basis(vec(Xd), m_order) :
                          _tps_multi_null_basis(Xd, m_order)
    else
        # Data-as-knots: X_eig = U·diag(v)·Z
        X_eig = U * Diagonal(v) * Z   # nk × (k-M)
        T_data = T_null
    end

    # Full basis: [constrained eigenbasis | polynomial null space]
    X_full = hcat(X_eig, T_data)  # n × k

    # --- Step 5: Build penalty matrix ---
    # S = Z'·diag(v)·Z with null space zeroed
    S_eigpart = Z' * Diagonal(v) * Z   # (k-M) × (k-M)
    S_full = zeros(k, k)
    S_full[1:n_basis, 1:n_basis] .= S_eigpart
    # Null space block (last M cols/rows) stays zero

    # --- Step 6: Column-wise RMS rescaling (R tprs.c lines 493-498) ---
    # Each column of X is rescaled to have RMS = 1.
    # S is rescaled accordingly: S[i,j] /= (w_i * w_j)
    col_scales = zeros(k)
    for j in 1:k
        ss = 0.0
        for i in 1:size(X_full, 1)
            ss += X_full[i, j]^2
        end
        col_scales[j] = sqrt(ss / size(X_full, 1))
        if col_scales[j] > 0
            X_full[:, j] ./= col_scales[j]
        end
    end
    for j in 1:k, i in 1:k
        denom = col_scales[i] * col_scales[j]
        if denom > 0
            S_full[i, j] /= denom
        end
    end

    # Shrinkage (ts): as in mgcv, modify the SINGLE penalty by raising the
    # null-space eigenvalues (one smoothing parameter, full-rank penalty),
    # rather than appending a second null-space penalty.
    null_dim = M
    pen_rank = n_basis
    if shrink
        S_full = _shrink_penalty(S_full)
        null_dim = 0
        pen_rank = k
    end
    penalties = Matrix{Float64}[S_full]

    centers = Matrix(XK)
    predict_cache = TPRSPredictCache(centers, U * Z, copy(col_scales))
    knots_out = d == 1 ? vec(copy(XK)) : Float64[]

    if !absorb_cons
        # Raw (unconstrained) smooth for tensor-product marginals etc.
        # The predict path applies no constraint (constraint === nothing),
        # so predict_matrix reproduces this X exactly.
        return ConstructedSmooth(
            spec,
            X_full,
            penalties,
            knots_out,
            null_dim,
            pen_rank,
            nothing,
            nothing,
            0, 0,
            nothing, nothing, nothing,
            Int[],
            predict_cache = predict_cache,
        )
    end

    # --- Step 7: Penalty rescaling (R's smoothCon, smooth.r lines 3879-3886),
    # applied BEFORE constraint absorption using the pre-absorption X and S,
    # consistent with absorb_constraints! and mgcv ---
    maXX = opnorm(X_full, Inf)^2
    if maXX > 0
        for i in eachindex(penalties)
            nS = opnorm(penalties[i], 1)  # R's default norm() for matrices = "O" = 1-norm
            if nS > 0
                penalties[i] = penalties[i] * (maXX / nS)
            end
        end
    end

    # Sum-to-zero constraint (C = column sums of X, matching R's smoothCon)
    C = sum(X_full; dims = 1)  # 1 × k
    C_mat = Matrix(C)

    # --- Step 8: Absorb sum-to-zero constraint (R's absorb.cons in smooth.r) ---
    # R uses: qrc = qr(t(C)), Z = qr.Q(qrc, complete=TRUE)[, -(1:nrow(C))]
    # This gives a specific rotation that we must match exactly.
    qr_C = qr(C_mat')  # QR of C' (k × 1 matrix)
    Z_cons = (qr_C.Q * Matrix(I, k, k))[:, 2:k]  # k × (k-1), skip first column
    X_cons = X_full * Z_cons
    S_cons = [Z_cons' * Si * Z_cons for Si in penalties]

    return ConstructedSmooth(
        spec,
        X_cons,
        S_cons,
        knots_out,
        null_dim,
        pen_rank,
        C_mat,
        nothing,
        0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = predict_cache,
    )
end

function _predict_matrix(::Union{ThinPlateSpline, ThinPlateShrink},
    smooth::ConstructedSmooth, newdata)
    spec = smooth.spec
    vars = spec.term_vars
    d = length(vars)
    m_order = spec.m === nothing ? _tps_default_m(d) : spec.m

    cache = smooth.predict_cache
    cache isa TPRSPredictCache || throw(ArgumentError("Missing fitted TPRS prediction cache"))

    X_new = hcat([Float64.(Tables.getcolumn(newdata, v)) for v in vars]...)
    E_new = _tps_cross_matrix(X_new, cache.centers, m_order)
    X_eig = E_new * cache.UZ
    T_new = d == 1 ? _tps_null_space_basis(view(X_new, :, 1), m_order) :
                     _tps_multi_null_basis(X_new, m_order)
    X_full = hcat(X_eig, T_new)
    _scale_columns!(X_full, cache.col_scales)

    if smooth.constraint !== nothing
        return X_full * _constraint_basis(smooth.constraint, size(X_full, 2))
    end
    return X_full
end
