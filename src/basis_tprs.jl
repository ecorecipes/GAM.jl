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
    # Per-covariate mean shift subtracted before building E and T, matching
    # mgcv's `shift` (smooth.construct.tp.smooth.spec, lines 12-16). Stored so
    # prediction re-applies exactly the same shift; without it the TPS kernel
    # is not translation invariant and `sp` depends on the covariate origin.
    shift::Vector{Float64}
end

# Backward-compatible constructor for caches deserialized from models fitted
# before `shift` existed (those were built on unshifted covariates).
function TPRSPredictCache(centers::Matrix{Float64}, UZ::Matrix{Float64},
    col_scales::Vector{Float64})
    return TPRSPredictCache(centers, UZ, col_scales, zeros(size(centers, 2)))
end

"""
    _lanczos_eigen(E::Symmetric{Float64}, k::Int; tol, maxiter) -> (values, vectors)

The `k` eigenpairs of `E` with the largest ABSOLUTE eigenvalues, by Lanczos
iteration with full reorthogonalization.

This mirrors mgcv's `Rlanczos`/`slanczos`, which `tprs.c` uses instead of a
full eigendecomposition: only the leading `k + M` eigenpairs of the `n × n`
semi-kernel are needed, so an `O(n³)` factorization is wasted work. Lanczos
converges quickly at BOTH ends of the spectrum, which is what "largest
absolute value" requires — the TPS semi-kernel has large negative eigenvalues
as well as large positive ones.

The starting vector is deterministic (no RNG), so repeated construction on the
same data is bit-reproducible.
"""
function _lanczos_eigen(E::Symmetric{Float64}, k::Int;
    tol::Float64 = 1e-12, maxiter::Int = 0)
    n = size(E, 1)
    k = min(k, n)
    # Deterministic dense-in-the-eigenbasis start vector.
    q = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        q[i] = sin(0.7 * i) + 0.3 * cos(0.11 * i)
    end
    q ./= norm(q)

    maxit = maxiter > 0 ? min(maxiter, n) : min(n, max(4k + 20, 60))
    Q = Matrix{Float64}(undef, n, maxit)
    Q[:, 1] .= q
    # α/β are preallocated with explicit lengths rather than grown by `push!`,
    # and `αc`/`βc` receive the copies `eigen` needs (LAPACK's `stegr!`
    # overwrites its tridiagonal input, so the live arrays cannot be passed).
    α = Vector{Float64}(undef, maxit)
    β = Vector{Float64}(undef, maxit)
    αc = Vector{Float64}(undef, maxit)
    βc = Vector{Float64}(undef, maxit)
    na = 0
    nb = 0
    w = Vector{Float64}(undef, n)
    proj = Vector{Float64}(undef, maxit)   # Qj' * w
    tmp = Vector{Float64}(undef, n)        # Qj * proj
    perm = Vector{Int}(undef, maxit)
    j_final = maxit
    for j in 1:maxit
        qj = view(Q, :, j)
        mul!(w, E, qj)
        if j > 1
            @inbounds @simd for i in 1:n
                w[i] -= β[j - 1] * Q[i, j - 1]
            end
        end
        a = dot(qj, w)
        na += 1
        α[na] = a
        @inbounds @simd for i in 1:n
            w[i] -= a * qj[i]
        end
        # Full reorthogonalization, applied twice (classical Gram-Schmidt
        # loses orthogonality in one pass; twice is the standard remedy).
        # Split into two `mul!`s against preallocated buffers: the same pair of
        # BLAS gemv calls `Qj * (Qj' * w)` makes, without the two temporaries.
        Qj = view(Q, :, 1:j)
        pv = view(proj, 1:j)
        for _ in 1:2
            mul!(pv, Qj', w)
            mul!(tmp, Qj, pv)
            w .-= tmp
        end

        # Ritz values of the j-step tridiagonal, and their residual bounds.
        if j >= k
            copyto!(αc, 1, α, 1, na)
            copyto!(βc, 1, β, 1, nb)
            Tj = SymTridiagonal(view(αc, 1:na), view(βc, 1:nb))
            F = eigen(Tj)
            pj = view(perm, 1:length(F.values))
            sortperm!(pj, F.values; by = abs, rev = true)
            sel = view(pj, 1:min(k, length(pj)))
            bnorm = norm(w)
            scale = maximum(abs, F.values)
            # An explicit loop, not `all(sel) do i ... end`: the closure
            # captures `j`/`bnorm`/`scale`/`tol`/`F`, which box and dominated
            # this function's allocations (621 of 1425 at n=1000, k=10).
            conv = true
            if scale != 0
                @inbounds for t in eachindex(sel)
                    if abs(bnorm * F.vectors[j, sel[t]]) > tol * scale
                        conv = false
                        break
                    end
                end
            end
            if conv || j == maxit
                j_final = j
                V = view(Q, :, 1:j) * F.vectors[:, sel]
                return (F.values[sel], Matrix(V))
            end
        end

        b = norm(w)
        if b <= tol * max(1.0, abs(a))
            # Invariant subspace found: no further directions available.
            j_final = j
            copyto!(αc, 1, α, 1, na)
            copyto!(βc, 1, β, 1, nb)
            Tj = SymTridiagonal(view(αc, 1:na), view(βc, 1:nb))
            F = eigen(Tj)
            pj = view(perm, 1:length(F.values))
            sortperm!(pj, F.values; by = abs, rev = true)
            sel = view(pj, 1:min(k, length(pj)))
            V = view(Q, :, 1:j) * F.vectors[:, sel]
            return (F.values[sel], Matrix(V))
        end
        if j < maxit
            nb += 1
            β[nb] = b
            @inbounds @simd for i in 1:n
                Q[i, j + 1] = w[i] / b
            end
        end
    end
    error("TPRS Lanczos failed to converge in $j_final iterations")
end

"""
    _mgcv_eigen_order(v, U)

Reorder `k` selected eigenpairs into mgcv's `Rlanczos` output layout:
**descending signed eigenvalue** — positives large-to-small, then negatives
least-to-most negative.

`tprs_setup` calls `Rlanczos` with `lm = -1` (`tprs.c:406-408`), which selects
the `k` largest-*magnitude* eigenpairs. The order they are written out in is a
separate matter: `mat.c:3852-3862` fills output columns `0..m-1` from
`d[0..m-1]` and the remaining columns from `d[j-lm..j-1]`, where `d` holds the
tridiagonal eigenvalues in **descending** order. That yields the selected
positives descending followed by the selected negatives descending, i.e. plain
descending signed order.

Ordering by `|λ|` descending instead (the natural Julia choice) spans the same
space and gives the same fit, but produces a different `T'U`, so mgcv's `QT`
returns a different null-space basis `Z`. Because the column-RMS rescaling and
the `‖S‖₁/‖X‖∞²` penalty normalisation are both applied *after* that rotation,
the resulting smoothing parameter lives on a different scale. Matching this
order is what makes `sp` transferable between GAM.jl and mgcv: it takes
`‖S‖₁/‖X‖∞²` from 14.67 to mgcv's 19.017082287 (agreeing to ~1e-10) on the
reference `s(x, k = 10)`, n = 200 problem.
"""
function _mgcv_eigen_order(v::Vector{Float64}, U::Matrix{Float64})
    ord = sortperm(v; rev = true)   # descending signed value
    return (v[ord], U[:, ord])
end

"""
    _tprs_top_eigen(E, k) -> (values, vectors)

Top-`k`-by-absolute-value eigenpairs of the symmetric semi-kernel `E`, using a
full eigendecomposition for small problems (where it is faster and exact) and
Lanczos above that. The crossover is set where the partial solver starts to
win: a full `eigen` costs ~1.35 s at n = 2000 versus ~0.03 s for the matvecs
Lanczos needs. Results are emitted in mgcv's ascending-signed order
(see [`_mgcv_eigen_order`](@ref)).
"""
function _tprs_top_eigen(E::Symmetric{Float64}, k::Int)
    n = size(E, 1)
    n <= max(400, 4k) && return _dense_top_eigen(E, k)
    v, U = _lanczos_eigen(E, k)
    length(v) == k && return _mgcv_eigen_order(v, U)
    # Lanczos stops early when the Krylov space generated by its start vector is
    # E-invariant with dimension < k — E is then effectively rank-deficient from
    # that vector (a covariate with almost no spread does it). Every caller
    # requires exactly k eigenpairs, so redo the solve densely rather than
    # silently hand back a short basis: `_construct_tprs` would otherwise build
    # `TU = T_null' * U` with too few columns and fail in the QR that follows.
    return _dense_top_eigen(E, k)
end

function _dense_top_eigen(E::Symmetric{Float64}, k::Int)
    eig = eigen(E)
    # Select the k largest-magnitude eigenpairs (mgcv's `lm = -1`), then emit
    # them in mgcv's ascending-signed output order — see _mgcv_eigen_order.
    idx = sortperm(abs.(eig.values); rev = true)[1:k]
    return _mgcv_eigen_order(eig.values[idx], eig.vectors[:, idx])
end

"""
    _mgcv_qt(A::Matrix{Float64}) -> Matrix{Float64}

Faithful port of mgcv's `QT(Q, A, fullQ = 0)` (`src/matrix.c:394`).

For `A` of size `Ar × Ac` with `Ar ≤ Ac`, computes the orthogonal `Q` with

    A·Q = [0, T],   T reverse lower triangular (Tᵢⱼ = 0 if i+j < Ar),

so the **first** `Ac − Ar` columns of `Q` span the null space of `A`. Returns the
`Ar × Ac` matrix of Householder vectors `uᵢ` (row `i`, zero-padded past
`Ac − i`), defining `Hᵢ = I − uᵢuᵢᵀ` and `Q = H₀H₁⋯H_{Ar−1}`.

This is deliberately *not* `qr(A')`: mgcv reflects a **decreasing prefix** of
each row (element `Ac−i−1` is the pivot, `matrix.c:416-427`), which yields a
different orthonormal basis of the same null space than LAPACK's Householder
complement. Both are valid null-space bases, but the choice is observable: the
column-RMS rescaling at the end of `tprs_setup` is applied *after* this
rotation, so a different `Z` gives different column scales, hence a different
`S`, a different `‖S‖₁/‖X‖∞²` penalty-normalisation constant, and ultimately a
smoothing parameter on a different scale from mgcv's. Matching `QT` is what
makes `sp` transferable between the two packages.
"""
function _mgcv_qt(A::AbstractMatrix{Float64})
    Ar, Ac = size(A)
    Ar <= Ac || throw(ArgumentError("_mgcv_qt requires Ar ≤ Ac, got $Ar × $Ac"))
    W = Matrix{Float64}(A)
    U = zeros(Float64, Ar, Ac)
    @inbounds for i in 0:(Ar - 1)
        ri = i + 1
        len = Ac - i                     # active prefix: columns 1:len
        p = view(W, ri, 1:len)
        # Scale by the row max to avoid over/underflow (matrix.c:418-419).
        mx = 0.0
        for j in 1:len
            x = abs(p[j])
            x > mx && (mx = x)
        end
        mx != 0.0 && (p ./= mx)
        lsq = 0.0
        for j in 1:len
            lsq += p[j]^2
        end
        lsq = sqrt(lsq)
        p[len] < 0.0 && (lsq = -lsq)     # sign choice (matrix.c:422)
        p[len] += lsq
        g = lsq != 0.0 ? 1.0 / (lsq * p[len]) : 0.0
        lsq *= mx                        # value landing on A[i, Ac-i-1]
        # Apply the reflector down the remaining rows of A (matrix.c:428-433).
        for j in (ri + 1):Ar
            x = 0.0
            for kk in 1:len
                x += p[kk] * W[j, kk]
            end
            x *= g
            for kk in 1:len
                W[j, kk] -= x * p[kk]
            end
        end
        # Store uᵢ = p·√g, zero-padded (matrix.c:441-444).
        sg = sqrt(g)
        for j in 1:len
            U[ri, j] = p[j] * sg
        end
        W[ri, len] = -lsq
        for j in 1:(len - 1)
            W[ri, j] = 0.0
        end
    end
    return U
end

"""
    _hq_mult_right!(C, U) -> C

Port of mgcv's `HQmult(C, U, p = 0, t = 0)` (`src/matrix.c:325`): overwrites
`C` with `C·Q`, where `Q = H₀H₁⋯` and `Hₖ = I − uₖuₖᵀ` for `uₖ` the `k`-th row
of `U`. Reflectors are applied in increasing order (`matrix.c:379-386`).
"""
function _hq_mult_right!(C::AbstractMatrix{Float64}, U::AbstractMatrix{Float64})
    Ur = size(U, 1)
    Cr, Cc = size(C)
    Cc == size(U, 2) ||
        throw(DimensionMismatch("C has $Cc columns, U has $(size(U, 2))"))
    Cu = zeros(Float64, Cr)
    @inbounds for k in 1:Ur
        u = view(U, k, :)
        for i in 1:Cr
            s = 0.0
            for j in 1:Cc
                s += C[i, j] * u[j]
            end
            Cu[i] = s
        end
        for j in 1:Cc
            uj = u[j]
            uj == 0.0 && continue
            for i in 1:Cr
                C[i, j] -= Cu[i] * uj
            end
        end
    end
    return C
end

"""
    _mgcv_null_space(TU::AbstractMatrix{Float64}, k::Int, M::Int)

Null-space basis `Z` (`k × (k−M)`) of the constraint matrix `TU` (`M × k`),
using mgcv's `QT` convention: the first `k−M` columns of `Q = H₀⋯H_{M−1}`.
"""
function _mgcv_null_space(TU::AbstractMatrix{Float64}, k::Int, M::Int)
    U_hh = _mgcv_qt(TU)
    Q = Matrix{Float64}(I, k, k)
    _hq_mult_right!(Q, U_hh)
    return Q[:, 1:(k - M)]
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

"""Euclidean distance between rows `i` of `A` and `j` of `B`, without
allocating the difference vector (`norm(view(A,i,:) .- view(B,j,:))` builds a
temporary per call, which dominates multi-dimensional TPS construction)."""
@inline function _row_distance(A::AbstractMatrix{Float64}, i::Int,
    B::AbstractMatrix{Float64}, j::Int, d::Int)
    ss = 0.0
    @inbounds @simd for l in 1:d
        δ = A[i, l] - B[j, l]
        ss += δ * δ
    end
    return sqrt(ss)
end

"""
    _tps_multi_penalty_matrix(X_data::Matrix{Float64}, m::Int) -> Matrix{Float64}

Compute the TPS penalty matrix for multi-dimensional data.
"""
function _tps_multi_penalty_matrix(X_data::Matrix{Float64}, m::Int)
    n, d = size(X_data)
    E = zeros(n, n)
    @inbounds for j in 1:n, i in j:n
        r = _row_distance(X_data, i, X_data, j, d)
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
    # Faithful port of mgcv's gen_tps_poly_powers (tprs.c:100-129): an odometer
    # over exponent vectors that increments the LOWEST index while the total
    # degree is below m-1, then carries upward.
    #
    # The order matters, and is not merely cosmetic. These exponents index the
    # columns of the polynomial null-space basis T; a different column order
    # permutes the rows of TU = T'U, so mgcv's QT factorization returns a
    # different null-space basis Z, and the resulting smoothing parameter sits
    # on a different scale (see _mgcv_eigen_order). Enumerating by total degree
    # — the natural choice — gives [1, z, x] for d = 2, m = 2 where mgcv gives
    # [1, x, z], which alone put 2-D `sp` transfer out by ~1e-3.
    M = 1
    for i in 0:(d - 1)
        M *= d + m - 1 - i
    end
    for i in 2:d
        M ÷= i
    end
    exps = Vector{Vector{Int}}(undef, M)
    index = zeros(Int, d)
    for i in 1:M
        exps[i] = copy(index)
        s = sum(index)
        if s < m - 1
            index[1] += 1
        else
            s -= index[1]
            index[1] = 0
            for j in 2:d
                index[j] += 1
                s += 1
                if s == m
                    s -= index[j]
                    index[j] = 0
                else
                    break
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
            E[i, j] = _tps_eta(_row_distance(X_new, i, centers, j, d), m, d)
        end
    end
    return E
end

function _scale_columns!(X::Matrix{Float64}, scales::Vector{Float64})
    @inbounds for j in 1:size(X, 2)
        scale = scales[j]
        scale > 0 || continue
        col = view(X, :, j)   # a view, so no per-column copy is allocated
        @simd for i in eachindex(col)
            col[i] /= scale
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

    # --- Mean-centre each covariate (mgcv's `shift`, tp constructor lines
    # 12-16) --- The thin-plate kernel η(r) depends only on distances, but the
    # polynomial null space T = [1, x, x², …] and the column-RMS rescaling that
    # follows do NOT: without centring, translating a covariate changes the
    # basis parameterization and hence the reported `sp`. mgcv subtracts the
    # column mean before building E and T, and stores the shift for prediction.
    shift = vec(mean(Xd; dims = 1))
    Xd = Xd .- shift'

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

    # Knot rule, matching mgcv (tp constructor lines 5-7 and 47-57): the
    # eigenproblem is built on ALL the (unique) data unless n exceeds
    # `max.knots`, whose default is 2000 — NOT on k knots. GAM.jl previously
    # dropped to a rank-k Nyström approximation as soon as n > max(3k, 200),
    # which made every fit with n > 200 a different (and measurably worse)
    # model than mgcv's: at n = 500, k = 20 the edf was 11.35 against mgcv's
    # 11.44, and the reported `sp` was off by a factor of 38.
    max_knots = Int(get(spec.xt, :max_knots, 2000))::Int
    max_knots > 0 || throw(ArgumentError(
        "max_knots must be positive, got $max_knots"))
    if knots !== nothing
        d == 1 || throw(ArgumentError(
            "user-supplied knots are not supported for multi-dimensional " *
            "TPS smooths"))
        length(knots) >= k || throw(ArgumentError(
            "user-supplied knots for a TPS smooth must have length ≥ k=$k, " *
            "got $(length(knots))"))
        # mgcv shifts user knots by the same per-covariate mean (line 30).
        XK = reshape(Float64.(knots[1:k]) .- shift[1], :, 1)
    else
        # mgcv builds the eigenproblem on the UNIQUE covariate combinations
        # (`uniquecombs`, line 48), not on all n rows. This matters: duplicated
        # covariate values make E rank-deficient and needlessly large, and the
        # Lanczos iteration converges slowly on the resulting degenerate
        # spectrum. Measured on 1480 rows over 40 distinct x values, building E
        # at full size took 12.8 s against 0.02 s deduplicated — for a fit
        # identical to 5 decimal places.
        rows = d == 1 ? reshape(sort!(unique(vec(Xd))), :, 1) :
               Matrix(reduce(hcat, unique(collect(eachrow(Xd))))')
        nu = size(rows, 1)
        if nu > max_knots
            # Deterministic evenly-spaced subsample. mgcv draws `max.knots` of
            # them at random under a fixed seed (`xt$seed`, default 1); an
            # evenly-spaced draw is reproducible without depending on R's RNG
            # stream and covers the range at least as well. Both are
            # approximations above this threshold, so exact parity with mgcv is
            # not attainable here in either case.
            sel = unique(round.(Int, range(1, nu; length = max_knots)))
            XK = rows[sel, :]
        elseif nu == n
            # No duplicates: keep the data ordering so the exact
            # data-as-knots branch is taken rather than the (equivalent but
            # slightly more expensive) Nyström extension.
            XK = Xd
        else
            XK = rows
        end
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
    v, U = _tprs_top_eigen(Symmetric(E), k)  # k eigenpairs, |λ| descending

    # --- Step 3: Constraint handling via T'U null space ---
    # Form TU = T'U (M × k)
    TU = T_null' * U   # M × k

    # Null space Z of the constraint TU·δ = 0, following mgcv's QT factorization
    # (tprs.c:419-420 → matrix.c:394): TU·Q = [0, B], so the null space is the
    # FIRST k−M columns of Q. LAPACK's `qr(TU')` complement spans the same
    # space but in a different orthonormal basis, and the choice is observable
    # downstream: the column-RMS rescaling below happens *after* this rotation,
    # so a different Z yields different column scales, a different S, and a
    # smoothing parameter on a scale incompatible with mgcv's. See _mgcv_qt.
    Z = _mgcv_null_space(TU, k, M)   # k × (k-M)

    n_basis = k - M  # number of non-null basis functions

    # --- Step 4: Build design matrix ---
    # R: X = U·diag(v)·Z ∪ T  (eigenvalues absorbed into eigenvector columns)
    # For data-as-knots case: X_eig = U·diag(v)·Z directly
    # For knot-based case: Nystrom extension X_eig = E_nk·U·Z (the Nystrom
    # factor diag(1/v) cancels against the absorbed diag(v))

    # Both branches produce Matrix{Float64}, but the annotations keep the
    # join concretely inferred: without them `X_full` widens and the scalar
    # loops below box every element (~6 allocations per entry).
    local X_eig::Matrix{Float64}
    local T_data::Matrix{Float64}
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

    # Full basis: [constrained eigenbasis | polynomial null space].
    # Built by copy rather than `hcat`: with SparseArrays loaded, `hcat`
    # dispatches to its generic method and the result loses concrete
    # inference here.
    X_full = Matrix{Float64}(undef, size(X_eig, 1), k)
    copyto!(view(X_full, :, 1:n_basis), X_eig)
    copyto!(view(X_full, :, (n_basis + 1):k), T_data)

    # --- Step 5: Build penalty matrix ---
    # S = Z'·diag(v)·Z with null space zeroed
    S_eigpart = Z' * Diagonal(v) * Z   # (k-M) × (k-M)
    S_full = zeros(k, k)
    S_full[1:n_basis, 1:n_basis] .= S_eigpart
    # Null space block (last M cols/rows) stays zero

    # --- Step 6: Column-wise RMS rescaling (R tprs.c lines 493-498) ---
    # Each column of X is rescaled to have RMS = 1.
    # S is rescaled accordingly: S[i,j] /= (w_i * w_j)
    # Column views avoid the copy that `X_full[:, j] ./= s` allocates; the
    # division (rather than multiplication by a reciprocal) keeps results
    # bit-identical to the previous implementation.
    col_scales = zeros(k)
    nrow = size(X_full, 1)
    @inbounds for j in 1:k
        col = view(X_full, :, j)
        ss = 0.0
        @simd for i in eachindex(col)
            ss += col[i]^2
        end
        cs = sqrt(ss / nrow)
        col_scales[j] = cs
        if cs > 0
            @simd for i in eachindex(col)
                col[i] /= cs
            end
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
        S_full = _shrink_penalty(S_full; null_dim = M)
        null_dim = 0
        pen_rank = k
    end
    penalties = Matrix{Float64}[S_full]

    centers = Matrix(XK)
    predict_cache = TPRSPredictCache(centers, U * Z, copy(col_scales), copy(shift))
    # Knots are reported on the ORIGINAL covariate scale (the shift is an
    # internal parameterization detail), matching what a user supplied.
    knots_out = d == 1 ? (vec(copy(XK)) .+ shift[1]) : Float64[]

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
    # Apply the construction-time mean shift, so the null-space polynomials and
    # kernel distances are evaluated in the same frame the basis was built in.
    X_new = X_new .- cache.shift'
    E_new = _tps_cross_matrix(X_new, cache.centers, m_order)
    X_eig::Matrix{Float64} = E_new * cache.UZ
    T_new::Matrix{Float64} = d == 1 ?
        _tps_null_space_basis(view(X_new, :, 1), m_order) :
        _tps_multi_null_basis(X_new, m_order)
    # Concrete assembly rather than `hcat` — see the note in `_construct_tprs`
    n_eig = size(X_eig, 2)
    X_full = Matrix{Float64}(undef, size(X_eig, 1), n_eig + size(T_new, 2))
    copyto!(view(X_full, :, 1:n_eig), X_eig)
    copyto!(view(X_full, :, (n_eig + 1):size(X_full, 2)), T_new)
    _scale_columns!(X_full, cache.col_scales)

    if smooth.constraint !== nothing
        return X_full * _constraint_basis(smooth.constraint, size(X_full, 2))
    end
    return X_full
end
