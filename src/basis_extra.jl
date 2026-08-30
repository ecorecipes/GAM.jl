# Duchon spline smooth — bs="ds"
#
# A port of mgcv's `smooth.construct.ds.smooth.spec` (R/smooth.r), together
# with its two helpers `DuchonE` and `DuchonT`. Duchon (1977) splines
# generalize thin-plate splines: the kernel exponent is 2m + 2s − d rather
# than 2m − d, so a second order `s` (which may be a half-integer) shifts the
# kernel independently of the penalty order `m`.
#
# The construction differs from TPRS in three ways that matter, and each is
# marked at the point where it bites:
#   1. the kernel (`_duchon_eta`) carries mgcv's own sign convention and NO
#      normalizing constant, where `_tps_eta` carries `_eta_const`;
#   2. the null-space rotation comes from a plain QR of `U'T`, not from the
#      QT factorization `tprs.c` uses (`_mgcv_qt`);
#   3. there is NO column-RMS rescaling of `X`. mgcv's `tp` constructor
#      rescales in C (tprs.c:493-498); its `ds` constructor does not.

"""
    _duchon_orders(spec, d) -> (m::Int, s::Float64)

Resolve and validate the Duchon orders `(m, s)`, following mgcv's
`smooth.construct.ds.smooth.spec` (R/smooth.r): `m` defaults to 2 and `s` to
0; `m` is rounded to an integer and `s` to the nearest half-integer; then `s`
is clamped to (−d/2, d/2) and, if `m + s <= d/2`, raised to the smallest value
giving a continuous function.

`m` is `SmoothSpec.m` (a scalar in GAM.jl). `s` is supplied through `xt`,
because mgcv spells the pair as `m = c(m, s)` while GAM.jl's `m` is
deliberately scalar (see `_normalize_m` in `smoothspec.jl`):

```julia
s(:x, bs = :ds, m = 2, xt = Dict(:s => 0.5))   # mgcv's m = c(2, 0.5)
```
"""
function _duchon_orders(spec::SmoothSpec, d::Int)
    m = spec.m === nothing ? 2 : round(Int, spec.m)
    s_raw = Float64(get(spec.xt, :s, 0.0))
    # mgcv: p.order[2] <- round(p.order[2]*2)/2 — s lives on a half-integer
    # grid, because 2s must be an integer for the kernel exponent to be one.
    s = round(s_raw * 2) / 2
    m < 1 && (m = 1)
    if s >= d / 2
        s = (d - 1) / 2
        @warn "bs=:ds: s reduced to $s (mgcv requires s < d/2 = $(d/2))" maxlog = 1
    end
    if s <= -d / 2
        s = -(d - 1) / 2
        @warn "bs=:ds: s increased to $s (mgcv requires s > -d/2)" maxlog = 1
    end
    if m + s <= d / 2
        s = 1 / 2 + d / 2 - m
        s >= d / 2 && throw(ArgumentError(
            "bs=:ds: no suitable s for m=$m, d=$d — increase m " *
            "(mgcv raises the same error)"))
        @warn "bs=:ds: s modified to $s to give a continuous function" maxlog = 1
    end
    return m, s
end

"""
    _duchon_kernel_exponent(m, s, d) -> (kint::Int, log_term::Bool, sign::Int)

The Duchon kernel is `sign * r^k` (or `sign * r^k * log r` when `k` is even),
with `k = 2m + 2s − d`. Ported from mgcv's `DuchonE` (R/smooth.r). `2s` is an
integer by construction, so `k` always is too.
"""
function _duchon_kernel_exponent(m::Int, s::Float64, d::Int)
    kk = 2m + 2s - d
    kint = round(Int, kk)
    isapprox(kk, kint; atol = 1e-9) || throw(ArgumentError(
        "bs=:ds: kernel exponent 2m+2s-d = $kk is not an integer"))
    # mgcv: signE <- 1 - 2*((floor(k/2)+1) %% 2)
    sgn = 1 - 2 * mod(fld(kint, 2) + 1, 2)
    return kint, iseven(kint), sgn
end

"""
    _duchon_E!(E, X1, X2, m, s, d)

Fill `E[i, j]` with the Duchon kernel evaluated at the distance between row
`i` of `X1` and row `j` of `X2`. Port of mgcv's `DuchonE` (R/smooth.r).
Note there is no normalizing constant here — unlike the thin-plate kernel,
which carries `_eta_const`.
"""
function _duchon_E!(E::AbstractMatrix{Float64}, X1::AbstractMatrix{Float64},
    X2::AbstractMatrix{Float64}, m::Int, s::Float64, d::Int)
    kint, log_term, sgn = _duchon_kernel_exponent(m, s, d)
    n1, n2 = size(X1, 1), size(X2, 1)
    @inbounds for j in 1:n2, i in 1:n1
        r2 = 0.0
        for c in 1:d
            δ = X1[i, c] - X2[j, c]
            r2 += δ * δ
        end
        r = sqrt(r2)
        val = if log_term
            r == 0.0 ? 0.0 : r^kint * log(r)
        else
            r^kint
        end
        E[i, j] = sgn * val
    end
    return E
end

"""
    _duchon_E(X1, X2, m, s, d) -> Matrix{Float64}

Allocating form of [`_duchon_E!`](@ref).
"""
function _duchon_E(X1::AbstractMatrix{Float64}, X2::AbstractMatrix{Float64},
    m::Int, s::Float64, d::Int)
    return _duchon_E!(Matrix{Float64}(undef, size(X1, 1), size(X2, 1)),
                      X1, X2, m, s, d)
end

"""
    _duchon_T(X, m, d) -> Matrix{Float64}

Polynomial null-space basis: all monomials of total degree `< m`, in mgcv's
`poly.pow` column order. Port of mgcv's `DuchonT` (R/smooth.r); it is the same
monomial enumeration the thin-plate null space uses, so this delegates to
[`_tps_multi_null_basis`](@ref) rather than duplicating the odometer.
"""
_duchon_T(X::AbstractMatrix{Float64}, m::Int, d::Int) =
    _tps_multi_null_basis(Matrix{Float64}(X), m)

"""
Cached quantities for predicting from a fitted Duchon smooth: the (shifted)
knots, the `nk × (k−M)` matrix `UZ = U·Q₂` that maps kernel evaluations onto
the penalized basis, the per-covariate `shift`, and the orders.
"""
struct DuchonPredictCache <: AbstractSmoothPredictCache
    knots::Matrix{Float64}
    UZ::Matrix{Float64}
    shift::Vector{Float64}
    m::Int
    s::Float64
    d::Int
end

"""
Duchon spline basis (registered as `bs=:ds`), a port of mgcv's `bs="ds"`.

Duchon (1977) splines generalize thin-plate regression splines. The kernel is
`r^(2m+2s-d)` (times `log r` when that exponent is even), so a second order
`s` shifts the kernel independently of the penalty order `m`; `m = 2, s = 0`
is the default. `s` may be a half-integer and must satisfy `-d/2 < s < d/2`
and `m + s > d/2`; values outside those ranges are adjusted as mgcv adjusts
them, with a warning.

Because `SmoothSpec.m` is a scalar in GAM.jl where mgcv writes `m = c(m, s)`,
the second order is supplied through `xt`:

```julia
s(:x, bs = :ds)                              # mgcv's s(x, bs="ds")
s(:x, bs = :ds, m = 2, xt = Dict(:s => 0.5)) # mgcv's s(x, bs="ds", m=c(2, 0.5))
```

The basis matches mgcv's up to the signs of individual basis columns, which
are not observable: fitted values, EDF and smoothing parameters agree (see
`test/test_duchon_rcall.jl`). Unlike `bs=:tp`, the Duchon basis is not
column-RMS rescaled, matching mgcv.
"""
struct DuchonSpline <: AbstractBasisType end

BASIS_TYPES[:ds] = DuchonSpline()

"""
    _construct_duchon(spec, data, user_knots; absorb_cons = true)

Build a Duchon spline smooth. Port of mgcv's
`smooth.construct.ds.smooth.spec` (R/smooth.r) and
`Predict.matrix.duchon.spline`.
"""
function _construct_duchon(spec::SmoothSpec, data, user_knots;
    absorb_cons::Bool = true)
    vars = spec.term_vars
    d = length(vars)
    m, s = _duchon_orders(spec, d)

    Xd = d == 1 ?
         reshape(Float64.(Tables.getcolumn(data, vars[1])), :, 1) :
         hcat([Float64.(Tables.getcolumn(data, v)) for v in vars]...)
    n = size(Xd, 1)
    k = min(spec.k, n)

    # Row multiplicities, as in `_construct_tprs`: `shift` and the sum-to-zero
    # constraint `C` are the two quantities that depend on how often each row
    # occurs. `:ds` is not on the reduced-construction whitelist, so `_rw` is
    # normally `nothing`; handling it keeps the two constructors consistent.
    _rw = _ROW_WEIGHTS[]
    if _rw !== nothing && length(_rw) != n
        throw(DimensionMismatch(
            "row_weights has $(length(_rw)) entries but the Duchon basis is " *
            "being built at $n covariate values"))
    end
    _wsum = _rw === nothing ? Float64(n) : sum(_rw)

    # mgcv centres each covariate (ds constructor: object$shift <- colMeans(x)).
    shift = _rw === nothing ? vec(mean(Xd; dims = 1)) : vec((_rw' * Xd) ./ _wsum)
    Xd = Xd .- shift'

    # Knots: mgcv uses the unique covariate combinations, subsampling to
    # `max.knots` (default 2000) above that. As in `_construct_tprs`, the
    # subsample is deterministic and evenly spaced rather than mgcv's seeded
    # random draw — both are approximations in that regime.
    max_knots = Int(get(spec.xt, :max_knots, 2000))::Int
    max_knots > 0 || throw(ArgumentError("max_knots must be positive, got $max_knots"))
    if user_knots !== nothing
        d == 1 || throw(ArgumentError(
            "user-supplied knots are not supported for multi-dimensional " *
            "Duchon smooths"))
        length(user_knots) >= k || throw(ArgumentError(
            "user-supplied knots for a ds smooth must have length ≥ k=$k, " *
            "got $(length(user_knots))"))
        XK = reshape(Float64.(user_knots) .- shift[1], :, 1)
    else
        rows = d == 1 ? reshape(sort!(unique(vec(Xd))), :, 1) :
               Matrix(reduce(hcat, unique(collect(eachrow(Xd))))')
        nu = size(rows, 1)
        if nu > max_knots
            sel = unique(round.(Int, range(1, nu; length = max_knots)))
            XK = rows[sel, :]
        else
            XK = rows
        end
    end
    nk = size(XK, 1)

    T_knot = _duchon_T(XK, m, d)
    M = size(T_knot, 2)
    k >= M + 1 || throw(ArgumentError(
        "basis dimension k=$k too small for Duchon orders m=$m, s=$s with " *
        "d=$d covariates (need k ≥ $(M + 1) = null_dim + 1)"))
    k <= nk || throw(ArgumentError(
        "basis dimension k=$k exceeds the $nk unique covariate " *
        "combination(s) available for a ds smooth; reduce k"))

    E = _duchon_E(XK, XK, m, s, d)

    # mgcv: er <- slanczos(E, k, -1) — SELECT the k largest-magnitude
    # eigenpairs, but RETURN them ordered by decreasing signed eigenvalue.
    # Selecting and then sorting by |λ| instead gives a different Q₂ below and
    # a basis that does not correspond to mgcv's at all (measured: relative
    # difference ~1.0 in X, i.e. no agreement whatsoever).
    F = eigen(Symmetric(E))
    sel = sortperm(abs.(F.values); rev = true)[1:k]
    sel = sel[sortperm(F.values[sel]; rev = true)]
    v = F.values[sel]
    U = F.vectors[:, sel]                       # nk × k

    # mgcv: qru <- qr(U1) with U1 = t(t(T) %*% er$vectors) = U'T, then the
    # null-space rotation is columns (M+1):k of Q. Note this is a plain QR,
    # NOT the QT factorization `tprs.c` uses for thin-plate splines — the two
    # give different orthonormal bases for the same subspace.
    U1 = U' * T_knot                            # k × M
    Qfull = Matrix(qr(U1).Q * Matrix{Float64}(I, k, k))
    Q2 = Qfull[:, (M + 1):k]                    # k × (k−M)
    UZ = U * Q2                                 # nk × (k−M)

    n_basis = k - M
    X_full = Matrix{Float64}(undef, n, k)
    # mgcv: X <- cbind(DuchonE(x, knt) %*% UZ, DuchonT(x))
    mul!(view(X_full, :, 1:n_basis), _duchon_E(Xd, XK, m, s, d), UZ)
    copyto!(view(X_full, :, (n_basis + 1):k), _duchon_T(Xd, m, d))

    # mgcv: S = Q₂' diag(λ) Q₂, padded with a zero null-space block.
    S_full = zeros(k, k)
    S_full[1:n_basis, 1:n_basis] .= Q2' * Diagonal(v) * Q2
    penalties = Matrix{Float64}[S_full]

    # NO column-RMS rescaling here: mgcv's ds constructor does not rescale,
    # where its tp constructor does (in C). Adding it would put the smoothing
    # parameter on a scale incompatible with mgcv's.

    predict_cache = DuchonPredictCache(Matrix(XK), UZ, copy(shift), m, s, d)
    knots_out = d == 1 ? (vec(copy(XK)) .+ shift[1]) : Float64[]

    if !absorb_cons
        return ConstructedSmooth(
            spec, X_full, penalties, knots_out, M, n_basis,
            nothing, nothing, 0, 0, nothing, nothing, nothing, Int[],
            predict_cache = predict_cache)
    end

    # Generic smoothCon steps, shared with every other basis: penalty
    # rescaling by maXX/‖S‖₁, then absorption of the sum-to-zero constraint.
    maXX = opnorm(X_full, Inf)^2
    if maXX > 0
        for i in eachindex(penalties)
            nS = opnorm(penalties[i], 1)
            nS > 0 && (penalties[i] = penalties[i] * (maXX / nS))
        end
    end

    C = _rw === nothing ? sum(X_full; dims = 1) : reshape(_rw' * X_full, 1, :)
    C_mat = Matrix(C)
    qr_C = qr(C_mat')
    Z_cons = (qr_C.Q * Matrix(I, k, k))[:, 2:k]
    X_cons = X_full * Z_cons
    S_cons = [Z_cons' * Si * Z_cons for Si in penalties]

    return ConstructedSmooth(
        spec, X_cons, S_cons, knots_out, M, n_basis,
        C_mat, nothing, 0, 0, nothing, nothing, nothing, Int[],
        predict_cache = predict_cache)
end

function _smooth_construct(::DuchonSpline, spec::SmoothSpec, data, user_knots)
    return _construct_duchon(spec, data, user_knots)
end

function _predict_matrix(::DuchonSpline, smooth::ConstructedSmooth, newdata)
    cache = smooth.predict_cache::DuchonPredictCache
    vars = smooth.spec.term_vars
    d = cache.d
    Xn = d == 1 ?
         reshape(Float64.(Tables.getcolumn(newdata, vars[1])), :, 1) :
         hcat([Float64.(Tables.getcolumn(newdata, v)) for v in vars]...)
    Xn = Xn .- cache.shift'

    n = size(Xn, 1)
    n_basis = size(cache.UZ, 2)
    T_new = _duchon_T(Xn, cache.m, d)
    k = n_basis + size(T_new, 2)
    X = Matrix{Float64}(undef, n, k)
    mul!(view(X, :, 1:n_basis),
         _duchon_E(Xn, cache.knots, cache.m, cache.s, d), cache.UZ)
    copyto!(view(X, :, (n_basis + 1):k), T_new)

    if smooth.constraint !== nothing
        k_full = size(X, 2)
        qr_C = qr(Matrix(smooth.constraint)')
        Z_cons = (qr_C.Q * Matrix(I, k_full, k_full))[:, 2:k_full]
        X = X * Z_cons
    end
    return X
end

# ─── Markov Random Field smooth — bs="mrf" ───────────────────────────────

"""Markov random field smooth (mgcv `bs="mrf"`)."""
struct MarkovRandomField <: AbstractBasisType end

BASIS_TYPES[:mrf] = MarkovRandomField()

"""
    _nb_to_adjacency(nb, n_regions) -> Matrix{Float64}

Convert a neighbourhood specification to an adjacency matrix.
Accepts either a symmetric matrix or a `Vector{Vector{Int}}` of neighbour lists.
"""
function _nb_to_adjacency(nb::AbstractMatrix, n_regions::Int)
    size(nb) == (n_regions, n_regions) ||
        throw(ArgumentError("Neighbourhood matrix must be $n_regions × $n_regions, " *
            "got $(size(nb))"))
    A = Float64.(nb)
    # Ensure symmetric
    if !issymmetric(A)
        A = (A .+ A') ./ 2.0
    end
    # Zero diagonal
    for i in 1:n_regions
        A[i, i] = 0.0
    end
    return A
end

function _nb_to_adjacency(nb::AbstractVector{<:AbstractVector{<:Integer}}, n_regions::Int)
    length(nb) == n_regions ||
        throw(ArgumentError("Neighbour list must have $n_regions entries, got $(length(nb))"))
    A = zeros(n_regions, n_regions)
    for (i, neighbours) in enumerate(nb)
        for j in neighbours
            1 <= j <= n_regions ||
                throw(ArgumentError("Neighbour index $j out of range [1, $n_regions]"))
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end
    return A
end

"""
Prediction cache for rank-reduced MRF smooths: stores the region labels and
the Laplacian eigenvector truncation `U_k` applied at construction (nothing
when the MRF was built at full rank).
"""
struct MRFTruncPredictCache <: AbstractSmoothPredictCache
    levels::Vector
    U_k::Union{Matrix{Float64}, Nothing}
end

"""
    _graph_components(A::Matrix{Float64}) -> Int

Number of connected components of the undirected graph with adjacency A.
"""
function _graph_components(A::Matrix{Float64})
    nr = size(A, 1)
    seen = falses(nr)
    ncomp = 0
    stack = Int[]
    for s in 1:nr
        seen[s] && continue
        ncomp += 1
        push!(stack, s)
        seen[s] = true
        while !isempty(stack)
            u = pop!(stack)
            for v in 1:nr
                if A[u, v] != 0 && !seen[v]
                    seen[v] = true
                    push!(stack, v)
                end
            end
        end
    end
    return ncomp
end

function _smooth_construct(::MarkovRandomField, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("MRF smooth requires at least one variable"))

    haskey(spec.xt, :nb) ||
        throw(ArgumentError("MRF smooth requires a neighbourhood matrix. " *
            "Pass xt=Dict(:nb => adjacency_matrix) to s()."))

    var = spec.term_vars[1]
    col = Tables.getcolumn(data, var)
    levels = sort(unique(col))
    n_regions = length(levels)
    n = length(col)

    # Build level → index mapping
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Convert neighbourhood to adjacency matrix
    nb = spec.xt[:nb]
    A = _nb_to_adjacency(nb, n_regions)

    # k = n_regions unless user specified smaller (rank-reduced MRF below)
    k = spec.k > 0 ? min(spec.k, n_regions) : n_regions

    # Build indicator/dummy matrix (n × n_regions)
    X = zeros(n, n_regions)
    for i in 1:n
        j = level_map[col[i]]
        X[i, j] = 1.0
    end

    # Build penalty: graph Laplacian S = D - A. Its null space has one
    # dimension per connected component of the neighbourhood graph.
    D = Diagonal(vec(sum(A; dims = 2)))
    S_pen = Matrix{Float64}(D - A)
    n_components = _graph_components(A)

    # Rank reduction (mgcv-style): if k < n_regions, project onto the k
    # lowest-frequency Laplacian eigenvectors.
    U_k = nothing
    if k < n_regions
        es = eigen(Symmetric(S_pen))
        # eigenvalues ascending: keep the k smoothest (incl. constants)
        U_k = es.vectors[:, 1:k]
        X = X * U_k
        S_pen = Matrix(Symmetric(U_k' * S_pen * U_k))
        null_dim = min(n_components, k)
    else
        null_dim = n_components
    end
    pen_rank = size(X, 2) - null_dim
    penalties = Matrix{Float64}[S_pen]

    # Apply sum-to-zero constraint (absorb like other smooths)
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64.(1:n_regions),  # dummy knots (region indices)
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = MRFTruncPredictCache(collect(levels), U_k),
    )

    return sm
end

function _predict_matrix(::MarkovRandomField, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    col = Tables.getcolumn(newdata, var)
    n_new = length(col)

    # Retrieve stored region labels (and eigen-truncation, if any)
    cache = smooth.predict_cache
    n_regions = length(smooth.knots)

    levels, U_k = if cache isa MRFTruncPredictCache
        cache.levels, cache.U_k
    elseif cache isa MRFPredictCache
        cache.levels, nothing
    else
        collect(1:n_regions), nothing
    end

    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Build indicator matrix for new data
    X = zeros(n_new, n_regions)
    for i in 1:n_new
        j = get(level_map, col[i], nothing)
        if j !== nothing
            X[i, j] = 1.0
        end
        # Unknown regions get zero rows (no contribution)
    end

    if U_k !== nothing
        X = X * U_k
    end

    # Apply same constraint as training
    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X, 2))
        return X * Z
    end
    return X
end

# ─── Factor-smooth interaction — bs="fs" ─────────────────────────────────
#
# A factor-smooth interaction creates a separate copy of a smooth basis for
# each level of a factor variable, with the smoothing parameter(s) shared
# across all levels.  This allows group-specific smooth curves regularized
# to a common degree of smoothness.
#
# Convention: s(x, group, bs=:fs, k=10)
#   - term_vars = [:x, :group]  — last variable is the factor
#   - The marginal smooth is built (with constraint) for the continuous variable(s)
#   - Model matrix is block-diagonal: one block per factor level
#   - Penalty matrices are block-diagonal replications of the marginal penalty

"""Factor-smooth interaction basis (mgcv `bs="fs"`)."""
struct FactorSmooth <: AbstractBasisType end

BASIS_TYPES[:fs] = FactorSmooth()

"""
Prediction cache for factor-smooth interactions (`bs="fs"`).

Extends the marginal-smooth/level bookkeeping with `P`, the `nat.param`
reparameterization matrix. mgcv's `Predict.matrix.fs.interaction`
(R/smooth.r:2166) forms the base prediction matrix and then post-multiplies by
the smooth's stored `P`, so `P` has to survive to prediction time or the
fitted and predicted bases silently disagree.
"""
struct FactorSmoothReparamCache <: AbstractSmoothPredictCache
    levels::Vector
    marginal_smooth  # ConstructedSmooth
    factor_var::Symbol
    P::Matrix{Float64}   # X_reparam = X_base * P
end

"""
    _nat_param_type1(X, S, rank) -> (X=..., D=..., P=..., rank=...)

Port of mgcv's `nat.param(X, S, rank, type = 1, unit.fnorm = TRUE)`
(R/smooth.r:15-129), the QR branch.

Reparameterizes so the penalty becomes the identity on the range space and
zero on the null space, with the range and null blocks each rescaled to unit
mean square. mgcv uses `type = 1` for `fs` specifically — the comment at
R/smooth.r:2063-2066 explains why: the range and null spaces should be at
least approximately orthogonal, because their variance components are treated
as independent.

Returns the transformed model matrix, the positive diagonal `D` of the
penalty, and `P` with `X_new == X * P`.
"""
function _nat_param_type1(X::Matrix{Float64}, S::Matrix{Float64}, rank::Int;
                          tol::Float64 = eps()^0.8)
    n, p = size(X)
    F = qr(X)
    R = Matrix(F.R)                                  # p × p upper triangular
    Q = Matrix(F.Q * Matrix{Float64}(I, n, p))       # thin Q, n × p

    # RSR = R^{-T} S R^{-1}; mgcv writes this as a pair of forwardsolves.
    Rt = transpose(R)
    RSR = Rt \ Matrix(transpose(Rt \ S))

    es = eigen(Symmetric(RSR))
    # LAPACK returns ascending eigenvalues; mgcv indexes R's DEscending order,
    # so reverse before applying its `1:rank` / `(rank+1):p` slicing.
    ord = sortperm(es.values; rev = true)
    vals = es.values[ord]
    vecs = es.vectors[:, ord]

    r = rank
    if r < 1 || r > p
        r = count(>(maximum(vals) * tol), vals)
    end
    null_exists = r < p

    D = vals[1:r]
    Xn = Q * vecs
    P = R \ vecs

    # type = 1: divide through by sqrt(D) so the penalty becomes the identity.
    E = ones(p)
    @inbounds for j in 1:r
        E[j] = sqrt(max(D[j], 0.0))
    end
    Xn = Xn ./ transpose(E)
    P = P ./ transpose(E)
    D = ones(r)

    # unit.fnorm: rescale range and null blocks to unit mean square. Note the
    # QR branch scales P's COLUMNS (mgcv's type 2/3 branch scales rows), which
    # is what preserves `X * P == X_new`.
    rng = 1:r
    scale = 1.0 / sqrt(mean(abs2, view(Xn, :, rng)))
    Xn[:, rng] .*= scale
    P[:, rng] .*= scale
    D .*= scale^2
    if null_exists
        nrng = (r + 1):p
        scalef = 1.0 / sqrt(mean(abs2, view(Xn, :, nrng)))
        Xn[:, nrng] .*= scalef
        P[:, nrng] .*= scalef
    end

    return (X = Xn, D = D, P = P, rank = r)
end

function _smooth_construct(::FactorSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 2 ||
        throw(ArgumentError("Factor-smooth interactions require at least 2 variables: " *
            "continuous variable(s) and a grouping factor. Got: $(spec.term_vars)"))

    # Last variable is the factor, rest are continuous
    factor_var = spec.term_vars[end]
    cont_vars = spec.term_vars[1:end-1]

    factor_col = Tables.getcolumn(data, factor_var)
    levels = sort(unique(factor_col))
    L = length(levels)
    n = length(factor_col)

    # Marginal basis: mgcv's `fs` takes it from `xt=list(bs=...)` and defaults
    # to "tp" (R/smooth.r:2052-2053, which calls the base constructor via
    # `class(object) <- object$base.bs`). Honour the same option — it was
    # previously hardcoded to TPRS, so `xt=Dict(:bs => :cc)` was silently
    # ignored and the user got a thin-plate marginal instead.
    marginal_bs = get(spec.xt, :bs, :tp)
    marginal_bs isa Symbol || throw(ArgumentError(
        "fs smooth: xt[:bs] must be a Symbol naming the marginal basis, " *
        "got $(typeof(marginal_bs))"))
    marginal_basis = resolve_basis_type(marginal_bs)

    # `fs` needs the UNCONSTRAINED marginal (see below), which is what the
    # specialised `_build_raw_marginal` methods produce. Bases without one
    # fall through to a method that absorbs constraints, silently dropping a
    # column and giving a centred basis mgcv would never build here — so
    # reject those rather than return a wrong answer.
    marginal_basis isa Union{ThinPlateSpline, ThinPlateShrink, CubicSpline,
                             CubicShrink, CyclicCubic, PSpline} ||
        throw(ArgumentError(
            "fs smooth: marginal basis :$marginal_bs is not supported. " *
            "Use one of :tp, :ts, :cr, :cs, :cc, :ps."))

    marginal_spec = SmoothSpec(
        cont_vars, marginal_basis, spec.k,
        nothing, spec.id, spec.sp, spec.fx, spec.m,
        "s($(join(cont_vars, ",")),bs=$marginal_bs)",
    )

    # Construct the marginal smooth WITHOUT constraint absorption, as mgcv's
    # fs does: the uncentered basis keeps per-level constants (the
    # random-intercept components) in the span. Identifiability with the
    # model intercept comes from FULL penalization below, which is also what
    # makes the gam.side exemption for fs safe.
    # `_build_raw_marginal` is the shared unconstrained-marginal builder that
    # tensor smooths already use; its per-basis methods construct X and S
    # directly, without constraint absorption. Its `template` is a
    # ConstructedSmooth carrying knots and rank, and with `constraint ===
    # nothing` it also predicts unconstrained at new data, which is what the
    # prediction path below relies on.
    marginal_raw = _build_raw_marginal(marginal_basis, marginal_spec, data, user_knots)
    marginal_sm = marginal_raw.template
    X_base = marginal_sm.X        # n × k (uncentered)
    k_eff = size(X_base, 2)

    # mgcv errors rather than guessing when the base basis is multiply
    # penalized (R/smooth.r:2053).
    length(marginal_sm.S) == 1 || throw(ArgumentError(
        "\"fs\" smooth cannot use a multiply penalized basis " *
        "(got $(length(marginal_sm.S)) penalties from the marginal smooth)"))

    # Reparameterize to separate range from null space, exactly as mgcv does
    # at R/smooth.r:2065 — `nat.param(X, S[[1]], rank, type = 1)`.
    rp = _nat_param_type1(X_base, marginal_sm.S[1], marginal_sm.rank)
    X_marginal = rp.X
    r = rp.rank
    null_d = k_eff - r

    # Block-diagonal model matrix: one block per factor level
    total_cols = L * k_eff
    X = zeros(n, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))
    for i in 1:n
        l = level_map[factor_col[i]]
        col_offset = (l - 1) * k_eff
        @inbounds for j in 1:k_eff
            X[i, col_offset + j] = X_marginal[i, j]
        end
    end

    # Penalties, following mgcv R/smooth.r:2110-2114 exactly.
    #
    #   S[[1]]   <- diag(rep(c(rp$D, rep(0, null.d)), nf))   # range space
    #   S[[i+1]] <- diag(rep(um, nf))                        # one per null dim
    #
    # The second line is the substantive point: mgcv gives EVERY null-space
    # direction its own penalty, hence its own smoothing parameter and its own
    # variance component. GAM.jl previously lumped them into a single shared
    # ridge, which is a strictly smaller model class (the per-level constant
    # and per-level slope were forced to share one variance component).
    # Like sz (and unlike factor-`by`), these penalties are deliberately NOT
    # stored narrow via `S_offsets`: each spans all L levels under ONE shared
    # smoothing parameter — that sharing is fs's model (mgcv's bs="fs" exists
    # precisely so every level shares a smoothness). Offset sub-penalties each
    # get their own λ, so narrowing would change the model. The dense storage
    # of these diagonal matrices is a separate (Diagonal-storage) question.
    penalties = Matrix{Float64}[]

    S_range = zeros(total_cols, total_cols)
    for l in 1:L
        off = (l - 1) * k_eff
        @inbounds for j in 1:r
            S_range[off + j, off + j] = rp.D[j]
        end
    end
    push!(penalties, S_range)

    for i in 1:null_d
        S_i = zeros(total_cols, total_cols)
        for l in 1:L
            idx = (l - 1) * k_eff + r + i
            S_i[idx, idx] = 1.0
        end
        push!(penalties, S_i)
    end

    # mgcv-style per-penalty rescaling. `smoothCon` (R/smooth.r:3879-3886)
    # applies this to EVERY penalty of EVERY smooth:
    #
    #   maXX <- norm(sm$X, type = "I")^2
    #   maS  <- norm(sm$S[[i]]) / maXX          # norm() default is the 1-norm
    #   sm$S[[i]] <- sm$S[[i]] / maS
    #
    # `fs` reached this point without it, because it is the one basis that
    # builds no constraint (mgcv sets `C <- matrix(0, 0, ncol(X))`) and so
    # never calls `absorb_constraints!`, where GAM.jl otherwise applies the
    # rescale. The omission is invisible in a free fit — the optimizer simply
    # selects differently scaled `sp` — but it puts our `sp` vector on a
    # different scale from mgcv's, so smoothing parameters cannot transfer
    # between the packages. Verified against mgcv 1.9-4: before this, our
    # per-penalty 1-norms were exactly `sm$S.scale` times mgcv's
    # ([17.8994, 0.00967411, 0.00967411] for a 4-level k=6 fs), i.e. our raw
    # penalties equal mgcv's pre-rescale penalties and only this step was
    # missing.
    maXX = opnorm(X, Inf)^2
    if maXX > 0
        for i in eachindex(penalties)
            nS = opnorm(penalties[i], 1)
            if nS > 0
                penalties[i] = penalties[i] * (maXX / nS)
            end
        end
    end

    # Fully penalized: no unpenalized directions remain (mgcv sets
    # null.space.dim = 0). The summed penalty is full rank, which is what the
    # single per-block `rank` field feeds into `Mp = p - sum(rank)`.
    pen_rank = total_cols
    null_dim = 0

    sm = ConstructedSmooth(
        spec, X, penalties,
        marginal_sm.knots,
        null_dim, pen_rank,
        nothing, nothing, 0, 0,   # no additional constraint on the full fs smooth
        nothing, nothing, nothing,
        Int[],
        predict_cache = FactorSmoothReparamCache(
            collect(levels), marginal_sm, factor_var, rp.P),
    )
    return sm
end

function _predict_matrix(::FactorSmooth, smooth::ConstructedSmooth, newdata)
    info = smooth.predict_cache
    info isa FactorSmoothReparamCache ||
        throw(ArgumentError("Cannot find factor smooth metadata for prediction"))

    factor_col = Tables.getcolumn(newdata, info.factor_var)
    n_new = length(factor_col)

    # Predict marginal at new data (handles constraint absorption automatically),
    # then apply the same nat.param reparameterization used at fit time —
    # mgcv's `Xb <- Predict.matrix(object,data) %*% object$P` (R/smooth.r:2166).
    marginal_sm = info.marginal_smooth
    X_marginal = _predict_matrix(marginal_sm.spec.basis, marginal_sm, newdata) * info.P
    k_eff = size(X_marginal, 2)
    L = length(info.levels)
    total_cols = L * k_eff

    X = zeros(n_new, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(info.levels))
    unseen = Set{eltype(factor_col)}()
    for i in 1:n_new
        l = get(level_map, factor_col[i], 0)
        if l > 0
            col_offset = (l - 1) * k_eff
            @inbounds for j in 1:k_eff
                X[i, col_offset + j] = X_marginal[i, j]
            end
        else
            push!(unseen, factor_col[i])
        end
        # Unknown levels get zero rows → no contribution to prediction
    end
    if !isempty(unseen)
        # Same warn-and-zero convention as `by=` and `bs=:re`; this path used
        # to zero SILENTLY, unlike its siblings (mgcv errors here).
        @warn "Factor smooth $(smooth.spec.label): level(s) not seen during " *
              "fitting get zero contribution." unseen_levels =
            sort!(collect(unseen); by = string)
    end

    return X
end

# ─── Soap Film smooth — bs="so" ──────────────────────────────────────────
#
# A 2D smooth that respects an irregular boundary, following Wood et al.
# (2008). The basis decomposes into:
#   1. Boundary film: cyclic spline along the boundary, extended into the
#      interior by solving the Laplace equation (∇²f = 0).
#   2. Interior wiggly: Green's-function-like basis from point sources at
#      interior knots, also via the discrete Laplacian.
#
# Reference: Wood, Bravington & Hedley (2008) "Soap film smoothing",
#            Journal of the Royal Statistical Society B, 70(5), 931-955.

"""Soap film smooth basis (mgcv `bs="so"`)."""
struct SoapFilm <: AbstractBasisType end

BASIS_TYPES[:so] = SoapFilm()

# ── Point-in-polygon (ray casting) ───────────────────────────────────────

"""
    _point_in_polygon(px, py, poly_x, poly_y) -> Bool

Ray-casting point-in-polygon test.
"""
function _point_in_polygon(px::Real, py::Real,
                           poly_x::AbstractVector, poly_y::AbstractVector)
    n = length(poly_x)
    inside = false
    j = n
    @inbounds for i in 1:n
        if ((poly_y[i] > py) != (poly_y[j] > py)) &&
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) /
                 (poly_y[j] - poly_y[i]) + poly_x[i])
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    _in_soap_domain(px, py, bnd) -> Bool

Check whether `(px, py)` is inside the domain defined by boundary loops.
First loop is the outer boundary; subsequent loops are holes.
"""
function _in_soap_domain(px::Real, py::Real, bnd::Vector{Matrix{Float64}})
    _point_in_polygon(px, py, bnd[1][:, 1], bnd[1][:, 2]) || return false
    for i in 2:length(bnd)
        _point_in_polygon(px, py, bnd[i][:, 1], bnd[i][:, 2]) && return false
    end
    return true
end

# ── Closest point on polygon ─────────────────────────────────────────────

"""
    _closest_on_polygon(px, py, poly_x, poly_y) -> (dist, arc_length, total_length)

Find the closest point on a closed polygon to `(px, py)`.
Returns the distance, the arc-length parameter at the closest point,
and the total arc-length of the polygon.
"""
function _closest_on_polygon(px::Real, py::Real,
                             poly_x::AbstractVector, poly_y::AbstractVector)
    nv = length(poly_x)
    cum_len = zeros(nv)
    for i in 2:nv
        cum_len[i] = cum_len[i - 1] +
            sqrt((poly_x[i] - poly_x[i - 1])^2 + (poly_y[i] - poly_y[i - 1])^2)
    end
    close_edge = sqrt((poly_x[1] - poly_x[nv])^2 + (poly_y[1] - poly_y[nv])^2)
    total_len = cum_len[end] + close_edge

    best_dist = Inf
    best_arc  = 0.0

    @inbounds for i in 1:nv
        next_i = i < nv ? i + 1 : 1
        edx = poly_x[next_i] - poly_x[i]
        edy = poly_y[next_i] - poly_y[i]
        elen = sqrt(edx^2 + edy^2)
        elen < 1e-15 && continue

        t = clamp(((px - poly_x[i]) * edx + (py - poly_y[i]) * edy) / (elen^2),
                  0.0, 1.0)
        cx = poly_x[i] + t * edx
        cy = poly_y[i] + t * edy
        dist = sqrt((px - cx)^2 + (py - cy)^2)

        if dist < best_dist
            best_dist = dist
            best_arc  = cum_len[i] + t * elen
        end
    end
    return best_dist, best_arc, total_len
end

# ── Bilinear interpolation on a grid ─────────────────────────────────────

"""
    _soap_bilinear(grid, inside, px, py, x0, y0, dx, dy, nx, ny) -> Float64

Bilinear interpolation of `grid` (ny × nx) at `(px, py)`.
Returns `NaN` when the point is outside the domain.
"""
function _soap_bilinear(grid::AbstractMatrix{Float64}, inside::BitMatrix,
                        px::Real, py::Real,
                        x0::Float64, y0::Float64,
                        dx::Float64, dy::Float64,
                        nx::Int, ny::Int)
    fi = (px - x0) / dx + 1.0   # 1-based column
    fj = (py - y0) / dy + 1.0   # 1-based row

    i_lo = floor(Int, fi);  i_hi = i_lo + 1
    j_lo = floor(Int, fj);  j_hi = j_lo + 1

    i_lo = clamp(i_lo, 1, nx);  i_hi = clamp(i_hi, 1, nx)
    j_lo = clamp(j_lo, 1, ny);  j_hi = clamp(j_hi, 1, ny)

    s = clamp(fi - floor(fi), 0.0, 1.0)
    t = clamp(fj - floor(fj), 0.0, 1.0)

    # Fast path: all four corners inside
    if inside[j_lo, i_lo] && inside[j_hi, i_lo] &&
       inside[j_lo, i_hi] && inside[j_hi, i_hi]
        return (1 - s) * (1 - t) * grid[j_lo, i_lo] +
                    s  * (1 - t) * grid[j_lo, i_hi] +
               (1 - s) *      t  * grid[j_hi, i_lo] +
                    s  *      t  * grid[j_hi, i_hi]
    end

    # Slow path: weighted average over inside corners only
    val = 0.0;  w = 0.0
    for (jj, wj) in ((j_lo, 1.0 - t), (j_hi, t))
        for (ii, wi) in ((i_lo, 1.0 - s), (i_hi, s))
            if inside[jj, ii]
                wt = wi * wj
                val += wt * grid[jj, ii]
                w   += wt
            end
        end
    end
    return w > 0 ? val / w : NaN
end

# ── Main soap film constructor ───────────────────────────────────────────

function _smooth_construct(::SoapFilm, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 2 ||
        throw(ArgumentError("Soap film smooth requires exactly 2 variables"))
    haskey(spec.xt, :bnd) ||
        throw(ArgumentError("Soap film smooth requires boundary via " *
            "xt=Dict(:bnd => [boundary_matrix])"))

    # ── Unpack inputs ────────────────────────────────────────────────────
    bnd = Matrix{Float64}[Float64.(b) for b in spec.xt[:bnd]]
    nmax = Int(get(spec.xt, :nmax, 200))::Int

    x = Float64.(Tables.getcolumn(data, spec.term_vars[1]))
    y = Float64.(Tables.getcolumn(data, spec.term_vars[2]))
    n = length(x)

    # ── Grid setup ───────────────────────────────────────────────────────
    all_bx = vcat([b[:, 1] for b in bnd]...)
    all_by = vcat([b[:, 2] for b in bnd]...)
    x_lo, x_hi = extrema(all_bx)
    y_lo, y_hi = extrema(all_by)
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    if x_range >= y_range
        dx = x_range / max(nmax - 1, 1)
        nx = nmax
        ny = max(2, ceil(Int, y_range / dx) + 1)
    else
        dx = y_range / max(nmax - 1, 1)
        ny = nmax
        nx = max(2, ceil(Int, x_range / dx) + 1)
    end
    dy = dx                   # square cells
    x0 = x_lo - dx           # one-cell padding
    y0 = y_lo - dy
    nx += 2;  ny += 2

    # ── Classify grid points ─────────────────────────────────────────────
    inside = falses(ny, nx)
    for j in 1:ny, i in 1:nx
        gx = x0 + (i - 1) * dx
        gy = y0 + (j - 1) * dy
        inside[j, i] = _in_soap_domain(gx, gy, bnd)
    end

    is_boundary = falses(ny, nx)
    is_interior = falses(ny, nx)
    for j in 1:ny, i in 1:nx
        inside[j, i] || continue
        has_ext = false
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ni, nj = i + di, j + dj
            if ni < 1 || ni > nx || nj < 1 || nj > ny || !inside[nj, ni]
                has_ext = true; break
            end
        end
        if has_ext
            is_boundary[j, i] = true
        else
            is_interior[j, i] = true
        end
    end

    # Index interior grid points 1..ng
    G = zeros(Int, ny, nx)
    ng = 0
    for j in 1:ny, i in 1:nx
        if is_interior[j, i]
            ng += 1
            G[j, i] = ng
        end
    end
    ng > 0 || throw(ArgumentError(
        "No interior grid points. Boundary may be too small or nmax too low."))

    # ── Sparse 5-point Laplacian on interior ─────────────────────────────
    II = Int[];  JJ = Int[];  VV = Float64[]
    sizehint!(II, 5 * ng);  sizehint!(JJ, 5 * ng);  sizehint!(VV, 5 * ng)

    for j in 1:ny, i in 1:nx
        idx = G[j, i]
        idx == 0 && continue
        push!(II, idx); push!(JJ, idx); push!(VV, -4.0)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ni, nj = i + di, j + dj
            (1 <= ni <= nx && 1 <= nj <= ny) || continue
            nidx = G[nj, ni]
            if nidx > 0
                push!(II, idx); push!(JJ, nidx); push!(VV, 1.0)
            end
        end
    end
    L = sparse(II, JJ, VV, ng, ng)
    L_lu = lu(L)

    # ── Boundary arc-length parameterisation ─────────────────────────────
    outer_x = bnd[1][:, 1]
    outer_y = bnd[1][:, 2]
    nv_bnd  = size(bnd[1], 1)

    cum_arc = zeros(nv_bnd)
    for i in 2:nv_bnd
        cum_arc[i] = cum_arc[i - 1] +
            sqrt((outer_x[i] - outer_x[i - 1])^2 +
                 (outer_y[i] - outer_y[i - 1])^2)
    end
    total_arc = cum_arc[end] +
        sqrt((outer_x[1] - outer_x[end])^2 + (outer_y[1] - outer_y[end])^2)

    # Assign arc-lengths to boundary grid points
    bnd_grid_ij  = Tuple{Int,Int}[]
    bnd_arc_vals = Float64[]
    for j in 1:ny, i in 1:nx
        is_boundary[j, i] || continue
        gx = x0 + (i - 1) * dx
        gy = y0 + (j - 1) * dy
        _, arc, _ = _closest_on_polygon(gx, gy, outer_x, outer_y)
        push!(bnd_grid_ij, (i, j))
        push!(bnd_arc_vals, arc)
    end
    n_bnd_grid = length(bnd_grid_ij)

    # Boundary basis dimension
    k_bnd = min(max(spec.k ÷ 3, 6), n_bnd_grid - 1, nv_bnd)
    k_int = max(spec.k - k_bnd, 1)

    # Cyclic spline on [0, total_arc]
    bnd_knots = collect(range(0.0, total_arc; length = k_bnd + 1))
    X_bnd_grid, S_bnd = _cc_basis(bnd_arc_vals, bnd_knots)
    k_bnd_cc = size(X_bnd_grid, 2)          # == k_bnd

    # Fast lookup: (i,j) → index in bnd_grid_ij
    bnd_ij_map = Dict{Tuple{Int,Int}, Int}()
    for (bi, ij) in enumerate(bnd_grid_ij)
        bnd_ij_map[ij] = bi
    end

    # ── Solve PDE for each boundary basis function ───────────────────────
    grid_bnd = zeros(ny, nx, k_bnd_cc)

    for col in 1:k_bnd_cc
        # Prescribe boundary values
        bval = zeros(ny, nx)
        for (bi, (gi, gj)) in enumerate(bnd_grid_ij)
            bval[gj, gi] = X_bnd_grid[bi, col]
        end
        # RHS from boundary contributions
        rhs = zeros(ng)
        for j in 1:ny, i in 1:nx
            idx = G[j, i]; idx == 0 && continue
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ni, nj = i + di, j + dj
                (1 <= ni <= nx && 1 <= nj <= ny) || continue
                if is_boundary[nj, ni]
                    rhs[idx] -= bval[nj, ni]
                end
            end
        end
        u = L_lu \ rhs
        # Fill grid with PDE solution
        for j in 1:ny, i in 1:nx
            if is_interior[j, i]
                grid_bnd[j, i, col] = u[G[j, i]]
            elseif is_boundary[j, i]
                bi = get(bnd_ij_map, (i, j), 0)
                if bi > 0
                    grid_bnd[j, i, col] = X_bnd_grid[bi, col]
                end
            end
        end
    end

    # ── Interior knot placement ──────────────────────────────────────────
    interior_ij = [(i, j) for j in 1:ny for i in 1:nx if is_interior[j, i]]

    if length(interior_ij) <= k_int
        knot_ij = interior_ij
    else
        step = length(interior_ij) / k_int
        idxs = [clamp(round(Int, (i - 0.5) * step) + 1, 1, length(interior_ij))
                for i in 1:k_int]
        knot_ij = interior_ij[unique(idxs)]
    end
    k_int_actual = length(knot_ij)

    # ── Solve PDE for interior (wiggly) basis ────────────────────────────
    # Each basis function is b = u2 / mx2 with L·u2 = u1 (normalized), so the
    # discrete Laplacian of b is ∇²b = (L·b)/dx² = u1 / (mx2·dx²). The soap
    # penalty ∫(∇²f)² therefore needs the SOURCES u1 (suitably scaled), not
    # the solutions u2. (This whole construction is a grid-PDE approximation
    # of Wood, Bravington & Hedley's exact soap-film basis.)
    grid_int = zeros(ny, nx, k_int_actual)
    g_mat    = zeros(ng, k_int_actual)        # ∇² of each basis fn, for penalty

    for ki in 1:k_int_actual
        gi, gj = knot_ij[ki]
        knot_idx = G[gj, gi]

        # Delta forcing → first solve
        rhs1 = zeros(ng);  rhs1[knot_idx] = 1.0
        u1 = L_lu \ rhs1
        mx1 = maximum(abs, u1)
        mx1 > 0 && (u1 ./= mx1)

        # Second solve for smoother basis
        u2 = L_lu \ u1
        mx2 = maximum(abs, u2)
        mx2 > 0 && (u2 ./= mx2)

        # ∇²(u2/mx2) = u1/(mx2·dx²) — consistent with the basis normalization
        g_mat[:, ki] .= u1 ./ (max(mx2, eps()) * dx^2)

        for j in 1:ny, i in 1:nx
            is_interior[j, i] || continue
            grid_int[j, i, ki] = u2[G[j, i]]
        end
    end

    # ── Evaluate basis at data points (bilinear interpolation) ───────────
    p = k_bnd_cc + k_int_actual
    X = zeros(n, p)

    for col in 1:k_bnd_cc
        g = @view grid_bnd[:, :, col]
        for i in 1:n
            X[i, col] = _soap_bilinear(g, inside, x[i], y[i],
                                       x0, y0, dx, dy, nx, ny)
        end
    end
    for col in 1:k_int_actual
        g = @view grid_int[:, :, col]
        for i in 1:n
            X[i, k_bnd_cc + col] = _soap_bilinear(g, inside, x[i], y[i],
                                                   x0, y0, dx, dy, nx, ny)
        end
    end
    replace!(X, NaN => 0.0)

    # ── Column scaling for conditioning ──────────────────────────────────
    irng = zeros(p)
    for j in 1:p
        lo, hi = extrema(@view X[:, j])
        rng = hi - lo
        irng[j] = rng > 0 ? 1.0 / rng : 1.0
    end
    X .= X .* irng'

    # ── Penalty matrices ─────────────────────────────────────────────────
    # 1. Boundary: cyclic-spline wiggliness penalty
    S_bnd_full = zeros(p, p)
    for a in 1:k_bnd_cc, b in 1:k_bnd_cc
        S_bnd_full[a, b] = S_bnd[a, b] * irng[a] * irng[b]
    end

    # 2. Interior: Gram matrix of the basis LAPLACIANS (∫∫ ∇²b_a ∇²b_b),
    # approximated by a grid sum with cell area dx·dy. Boundary-film basis
    # functions are harmonic (∇² = 0) so they carry no interior penalty.
    S_int = g_mat' * g_mat * (dx * dy)
    S_int_full = zeros(p, p)
    for a in 1:k_int_actual, b in 1:k_int_actual
        S_int_full[k_bnd_cc + a, k_bnd_cc + b] =
            S_int[a, b] * irng[k_bnd_cc + a] * irng[k_bnd_cc + b]
    end

    # Symmetrize to fix floating-point round-off
    S_bnd_full .= (S_bnd_full .+ S_bnd_full') ./ 2
    S_int_full .= (S_int_full .+ S_int_full') ./ 2
    penalties = Matrix{Float64}[S_bnd_full, S_int_full]

    # ── Cache for prediction ─────────────────────────────────────────────
    grid_basis = cat(grid_bnd, grid_int; dims = 3)
    soap_cache = SoapPredictCache(Dict{Symbol, Any}(
        :bnd    => bnd,
        :x0     => x0,  :y0  => y0,
        :dx     => dx,  :dy  => dy,
        :nx     => nx,  :ny  => ny,
        :inside => inside,
        :grid_basis => grid_basis,
        :irng   => irng,
    ))

    # ── Absorb identifiability constraints ───────────────────────────────
    null_dim = 1
    pen_rank = p - null_dim
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],          # no 1-D knot vector
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = soap_cache,
    )
end

# ── Prediction ───────────────────────────────────────────────────────────

function _predict_matrix(::SoapFilm, smooth::ConstructedSmooth, newdata)
    cache = smooth.predict_cache
    cache isa SoapPredictCache || throw(ArgumentError(
        "No cached soap-film data for '$(smooth.spec.label)'. " *
        "Was the smooth constructed correctly?"))
    sd = cache.data

    xv = Float64.(Tables.getcolumn(newdata, smooth.spec.term_vars[1]))
    yv = Float64.(Tables.getcolumn(newdata, smooth.spec.term_vars[2]))
    nn = length(xv)

    bnd    = sd[:bnd]::Vector{Matrix{Float64}}
    x0     = sd[:x0]::Float64;   y0  = sd[:y0]::Float64
    dxg    = sd[:dx]::Float64;   dyg = sd[:dy]::Float64
    nxg    = sd[:nx]::Int;       nyg = sd[:ny]::Int
    ins    = sd[:inside]::BitMatrix
    gbasis = sd[:grid_basis]::Array{Float64, 3}   # ny × nx × p
    irng   = sd[:irng]::Vector{Float64}
    p      = size(gbasis, 3)

    X = zeros(nn, p)
    for col in 1:p
        g = @view gbasis[:, :, col]
        for i in 1:nn
            if _in_soap_domain(xv[i], yv[i], bnd)
                X[i, col] = _soap_bilinear(g, ins, xv[i], yv[i],
                                           x0, y0, dxg, dyg, nxg, nyg)
            else
                X[i, col] = NaN
            end
        end
    end
    replace!(X, NaN => 0.0)

    # Apply column scaling
    X .= X .* irng'

    # Apply constraint
    if smooth.constraint !== nothing
        Z = _constraint_basis(smooth.constraint, size(X, 2))
        return X * Z
    end
    return X
end
