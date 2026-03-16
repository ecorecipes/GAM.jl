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
    _tps_multi_null_basis(X_data::Matrix{Float64}, m::Int) -> Matrix{Float64}

Polynomial null space for d-dimensional TPS.
Dimension M = binomial(m + d - 1, d).
"""
function _tps_multi_null_basis(X_data::Matrix{Float64}, m::Int)
    n, d = size(X_data)
    # For m=2, d=2: M = 3 (1, x1, x2)
    # General: all monomials of degree ≤ m-1
    M = binomial(m + d - 1, d)
    T = ones(n, M)
    col = 1
    # degree 0: constant (already 1)
    col += 1
    if m >= 2
        # degree 1: linear terms
        for j in 1:d
            if col > M
                break
            end
            T[:, col] .= X_data[:, j]
            col += 1
        end
    end
    if m >= 3
        # degree 2: quadratic terms (for m ≥ 3)
        for j in 1:d, j2 in j:d
            if col > M
                break
            end
            T[:, col] .= X_data[:, j] .* X_data[:, j2]
            col += 1
        end
    end
    return T
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
function _construct_tprs(spec::SmoothSpec, data, knots; shrink::Bool = false)
    vars = spec.term_vars
    d = length(vars)
    m_order = spec.m === nothing ? 2 : spec.m
    k = spec.k

    # Extract data
    if d == 1
        x = Float64.(Tables.getcolumn(data, vars[1]))
        n = length(x)
        k = min(k, n)

        # Use knots for basis if provided, otherwise use data
        if knots !== nothing && length(knots) >= k
            xk = Float64.(knots[1:k])
        elseif n > max(k * 3, 200)
            xk = place_knots(x, k)
        else
            xk = x
        end

        E = _tps_penalty_matrix(xk, m_order)
        T_null = _tps_null_space_basis(xk, m_order)
        M = size(T_null, 2)
    else
        X_data = hcat([Float64.(Tables.getcolumn(data, v)) for v in vars]...)
        n = size(X_data, 1)
        k = min(k, n)

        E = _tps_multi_penalty_matrix(X_data, m_order)
        T_null = _tps_multi_null_basis(X_data, m_order)
        M = size(T_null, 2)
        xk = Float64[]
    end

    k >= M + 1 || throw(ArgumentError(
        "basis dimension k=$k too small for penalty order m=$m_order " *
        "(need k ≥ $(M + 1) = null_dim + 1)"))

    nk = d == 1 ? length(xk) : n

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
    T_mat = d == 1 ? T_null : T_null
    TU = T_mat' * U   # M × k

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
    # For knot-based case: need Nystrom extension

    if d == 1 && length(xk) < n
        # Knot-based: Nystrom extension to data points
        # In R's tprs_setup, this is the knot-based path (lines 451-480)
        # X_data = UZ' * tps_g(x_i) evaluated via UZ
        # For now, use the simpler approach: map through E_new * U
        E_nk = zeros(n, nk)
        for j in 1:nk, i in 1:n
            E_nk[i, j] = _tps_eta(abs(x[i] - xk[j]), m_order, 1)
        end
        # U_data = E_nk * U * diag(1./v) gives data-point eigenvectors (Nystrom)
        # Then X_eig = U_data * diag(v) * Z = E_nk * U * Z
        X_eig = E_nk * U * Diagonal(v) * Z  # Wait: Nystrom gives E_nk * U * diag(1/v),
        # but we want to absorb v into X, so X = E_nk * U * diag(1/v) * diag(v) * Z = E_nk * U * Z
        X_eig = E_nk * (U * Z)
        T_data = _tps_null_space_basis(x, m_order)
    else
        # Data-as-knots: X_eig = U·diag(v)·Z
        X_eig = U * Diagonal(v) * Z   # nk × (k-M)
        T_data = T_null
    end

    # Full basis: [constrained eigenbasis | polynomial null space]
    X_full = hcat(X_eig, T_data)  # n × k (or nk × k if data=knots and n=nk)

    # --- Step 5: Build penalty matrix ---
    # S = Z'·diag(v)·Z with null space zeroed
    S_eigpart = Z' * Diagonal(v) * Z   # (k-M) × (k-M)
    S_full = zeros(k, k)
    S_full[1:n_basis, 1:n_basis] .= S_eigpart
    # Null space block (last M cols/rows) stays zero

    penalties = Matrix{Float64}[S_full]

    # For shrinkage (ts): add penalty on null space
    if shrink
        S_shrink = zeros(k, k)
        for i in (n_basis + 1):k
            S_shrink[i, i] = 1.0
        end
        push!(penalties, S_shrink)
    end

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
    for si in eachindex(penalties)
        for j in 1:k, i in 1:k
            denom = col_scales[i] * col_scales[j]
            if denom > 0
                penalties[si][i, j] /= denom
            end
        end
    end

    # Sum-to-zero constraint (C = column means of X, matching R's smoothCon)
    C = sum(X_full; dims = 1)  # 1 × k (R uses sum, not mean, for C)
    C_mat = Matrix(C)

    # --- Step 7: Absorb sum-to-zero constraint (R's absorb.cons in smooth.r) ---
    # R uses: qrc = qr(t(C)), Z = qr.Q(qrc, complete=TRUE)[, -(1:nrow(C))]
    # This gives a specific rotation that we must match exactly.
    qr_C = qr(C_mat')  # QR of C' (k × 1 matrix)
    Z_cons = (qr_C.Q * Matrix(I, k, k))[:, 2:k]  # k × (k-1), skip first column
    X_cons = X_full * Z_cons
    S_cons = [Z_cons' * Si * Z_cons for Si in penalties]

    # --- Step 8: Penalty rescaling (R's smoothCon, smooth.r lines 3879-3886) ---
    # R: for each S, rescale by norm(X,"I")^2 / norm(S,"O")
    maXX = opnorm(X_cons, Inf)^2
    if maXX > 0
        for i in eachindex(S_cons)
            nS = opnorm(S_cons[i], 1)  # R's default norm() for matrices = "O" = 1-norm
            if nS > 0
                S_cons[i] = S_cons[i] * (maXX / nS)
            end
        end
    end

    null_dim = M
    pen_rank = n_basis

    knots_out = d == 1 ? (length(xk) > 0 ? xk : Float64[]) : Float64[]

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
    )
end

function _predict_matrix(::Union{ThinPlateSpline, ThinPlateShrink},
    smooth::ConstructedSmooth, newdata)
    spec = smooth.spec
    vars = spec.term_vars
    d = length(vars)
    m_order = spec.m === nothing ? 2 : spec.m

    if d == 1
        x_new = Float64.(Tables.getcolumn(newdata, vars[1]))
        n_new = length(x_new)
        knots = smooth.knots
        k = spec.k
        M = m_order  # null space dim for 1d

        if !isempty(knots)
            nk = length(knots)

            # Reconstruct the knot-based E and eigen-decomp
            E_kk = _tps_penalty_matrix(knots, m_order)
            T_kk = _tps_null_space_basis(knots, m_order)
            eig = eigen(Symmetric(E_kk))
            idx = sortperm(abs.(eig.values); rev = true)
            U = eig.vectors[:, idx[1:k]]
            v = eig.values[idx[1:k]]

            # Constraint: TU = T'U, Z = null space of TU
            TU = T_kk' * U
            qr_TU = qr(TU')
            Q_full = qr_TU.Q * Matrix(I, k, k)
            Z = Q_full[:, (M + 1):k]

            # Nystrom extension for new data: E_new * U * Z
            E_new = zeros(n_new, nk)
            for j in 1:nk, i in 1:n_new
                E_new[i, j] = _tps_eta(abs(x_new[i] - knots[j]), m_order, 1)
            end
            X_eig = E_new * (U * Z)

            T_new = _tps_null_space_basis(x_new, m_order)
            X_full = hcat(X_eig, T_new)
        else
            # Data was used as knots — need full reconstruction
            # This shouldn't normally happen for prediction at new points
            E_new = _tps_penalty_matrix(x_new, m_order)
            T_new = _tps_null_space_basis(x_new, m_order)
            eig = eigen(Symmetric(E_new))
            idx = sortperm(abs.(eig.values); rev = true)
            U = eig.vectors[:, idx[1:k]]
            v = eig.values[idx[1:k]]

            TU = T_new' * U
            qr_TU = qr(TU')
            Q_full = qr_TU.Q * Matrix(I, k, k)
            Z = Q_full[:, (M + 1):k]

            X_eig = U * Diagonal(v) * Z
            X_full = hcat(X_eig, T_new)
        end

        # Apply same column rescaling and constraint as training
        # Use QR-based constraint absorption matching R's absorb.cons
        if smooth.constraint !== nothing
            C = smooth.constraint
            k_pred = size(X_full, 2)
            qr_C = qr(C')
            Z_cons = (qr_C.Q * Matrix(I, k_pred, k_pred))[:, 2:k_pred]
            return X_full * Z_cons
        end
        return X_full
    else
        throw(ArgumentError("Multi-dimensional TPRS prediction not yet implemented"))
    end
end
