# ============================================================================
# Design-matrix abstraction for `bam`
# ============================================================================
#
# `bam` fits by accumulating the normal equations rather than factorising the
# model matrix, and it touches `X` through only a handful of operations. This
# file names those operations so the storage behind them can change.
#
# The motivation is mgcv's `discrete = TRUE`: when a covariate takes few
# distinct values, `X` need never be materialised at all — the marginal basis
# is evaluated once per unique value and combined through an index vector, so
# `XᵀWX` costs O(m·p) rather than O(n·p²) in the row dimension. A profile of
# GAM.jl's `bam` puts the ceiling for that at 1.10–1.55x (Gaussian) and
# 1.68–4.80x (non-Gaussian) end to end — Amdahl against basis construction and
# `pirls_finalize` — but ~70x on peak memory, which is the real prize:
# `te(15,15)` at n = 1e6 peaks at 6825 MB today.
#
# GAM.jl is unusually well placed for this. `outer_iteration_bam`'s EFS loop
# consumes only `XᵀWX`, `XᵀWz`, `β`, `S` and `tr(A⁻¹Sⱼ)` — it never touches
# `X` — so unlike mgcv, which had to write a second fitting function
# (`bgam.fitd`) because its non-discrete path is QR-based, no optimiser work is
# required here. The design contact is exactly the operations below.
#
# THIS FILE IS THE INTERFACE ONLY. `DenseDesign` wraps today's `Matrix` and
# must be bit-identical to the code it replaced; no discretisation lives here.

"""
    BamDesign

Storage-agnostic model matrix for [`bam`](@ref).

Implementations provide [`ncols`](@ref), [`nrows`](@ref), [`intercept_col`](@ref),
[`mul_eta!`](@ref), [`accumulate_XtWX_XtWz!`](@ref), [`accumulate_XtWX!`](@ref)
and [`design_finalize`](@ref). Nothing in `bam`'s fitting path assumes more
than that, so an implementation is free to store marginal bases plus index
vectors instead of an `n x p` matrix.

Currently implemented by [`DenseDesign`](@ref).
"""
abstract type BamDesign end

"""
    DenseDesign(X::Matrix{Float64})

The conventional dense model matrix, wrapping an `n x p` array.

Every operation forwards to the same chunked kernels `bam` used before the
[`BamDesign`](@ref) interface existed, so results are bit-identical.

The intercept column is located lazily and memoised: `pirls_bam` needs it only
when starting from `mustart` (not when warm-started from an outer iteration's
coefficients), and the scan is `O(n*p)`, so it is neither done eagerly nor
repeated across the outer loop's inner solves.
"""
struct DenseDesign <: BamDesign
    # Concrete: the chunked `syrk`/`gemv` kernels require a strided
    # `Matrix{Float64}`, and `setup_gam` always produces one. Keeping it
    # concrete also leaves every accessor below trivially inferrable.
    X::Matrix{Float64}
    # -2 = not yet scanned, 0 = no intercept column, j > 0 = column j.
    # A concrete `Int` rather than `Union{Int,Nothing}` to keep `intercept_col`
    # inferrable.
    icpt::Base.RefValue{Int}
end

DenseDesign(X::Matrix{Float64}) = DenseDesign(X, Ref(-2))

"""
    bam_design(X::Matrix{Float64}) -> DenseDesign

Wrap a materialised model matrix as a [`BamDesign`](@ref).

This is the construction seam. A discrete design is built from the smooth
specifications and the data — it never receives an `X` — so it will enter
through a sibling method rather than by wrapping the output of `setup_gam`.
Keeping construction behind a factory is what lets that land without touching
`pirls_bam` or `outer_iteration_bam` again.
"""
bam_design(X::Matrix{Float64}) = DenseDesign(X)

"""
    ncols(D::BamDesign) -> Int

Number of model-matrix columns (`p`), i.e. the coefficient-vector length.
"""
function ncols end

"""
    nrows(D::BamDesign) -> Int

Number of observations (`n`).
"""
function nrows end

ncols(D::DenseDesign) = size(D.X, 2)
nrows(D::DenseDesign) = size(D.X, 1)
Base.size(D::BamDesign) = (nrows(D), ncols(D))

"""
    intercept_col(D::BamDesign) -> Int

Index of the column that is identically 1, or `0` if there is none.

`pirls_bam` uses this to start from the constant fit on the link scale, rather
than assuming the intercept is column 1.
"""
function intercept_col end

function intercept_col(D::DenseDesign)
    if D.icpt[] == -2
        X = D.X
        found = findfirst(j -> all(==(1.0), view(X, :, j)), 1:size(X, 2))
        D.icpt[] = found === nothing ? 0 : found
    end
    return D.icpt[]
end

"""
    mul_eta!(eta, D::BamDesign, beta) -> eta

Write the linear predictor `eta = X * beta` (the offset is added by the caller).
"""
function mul_eta! end

mul_eta!(eta::Vector{Float64}, D::DenseDesign, beta::AbstractVector{Float64}) =
    mul!(eta, D.X, beta)

"""
    accumulate_XtWX_XtWz!(XtWX, XtWz, D::BamDesign, w, z; chunk_size)

Overwrite `XtWX` with `X'WX` and `XtWz` with `X'Wz`, for diagonal `W = Diagonal(w)`.
"""
function accumulate_XtWX_XtWz! end

function accumulate_XtWX_XtWz!(XtWX::Matrix{Float64}, XtWz::Vector{Float64},
    D::DenseDesign, w::Vector{Float64}, z::Vector{Float64};
    chunk_size::Int = 10000)
    _accumulate_XtWX_XtWz_chunked!(XtWX, XtWz, D.X, w, z, chunk_size)
    return XtWX, XtWz
end

"""
    accumulate_XtWX!(XtWX, D::BamDesign, w; chunk_size)

Overwrite `XtWX` with `X'WX`, without the right-hand side.
"""
function accumulate_XtWX! end

function accumulate_XtWX!(XtWX::Matrix{Float64}, D::DenseDesign,
    w::Vector{Float64}; chunk_size::Int = 10000)
    _accumulate_XtWX_chunked!(XtWX, D.X, w, chunk_size)
    return XtWX
end

"""
    design_finalize(D::BamDesign, w, XtWX, A_chol; chunk_size, compute_hat_diag)
        -> (edf_vec, hat_diag, R)

Effective degrees of freedom, weighted leverages and the Cholesky factor.

`edf_vec = diag(A^-1 X'WX)` is `O(p^3)` and always computed; `hat_diag` is the
`O(n*p^2)` leverage sweep and is diagnostics-only, so a design that cannot form
it cheaply may decline by returning `Float64[]` — `bam`'s Gaussian path already
tolerates an empty `hat_diag`, and the outer loop's inner solves never read it.
"""
function design_finalize end

function design_finalize(D::DenseDesign, w::Vector{Float64},
    XtWX::Matrix{Float64}, A_chol::Cholesky;
    chunk_size::Int = 1024, compute_hat_diag::Bool = true)
    return pirls_finalize(D.X, w, XtWX, A_chol;
        chunk_size = chunk_size, compute_hat_diag = compute_hat_diag)
end

# ============================================================================
# Discrete design — mgcv's `discrete = TRUE`
# ============================================================================
#
# A 1-D smooth evaluated at a covariate taking `m << n` distinct values needs
# only its `m x p` basis plus an `n`-vector of row indices. The full model
# matrix is then `X[i, cols] = Xd[k[i], :]`, and the normal equations can be
# formed without ever writing that down:
#
#   XᵀWz  block  :  bin `w.*z` into `m` cells (O(n)), then one `m x p` gemv
#   XᵀWX  diag   :  bin `w` into `m` cells (O(n)), then `Xdᵀ diag(wb) Xd`
#   XᵀWX  cross  :  accumulate the `m_a x m_b` cross-weight table, then 2 gemms
#
# so the n-proportional work falls from `O(n p²)` to `O(n)` per block. See
# `discrete.c:1742-1792` (mgcv's `i == j` shortcut) and `:1801-2006` (its
# cross-block strategies), and `discrete_plan.md` §1.4 for the derivation.
#
# M1 SCOPE: 1-D smooths with no `by=`. Tensors, `by=`, random effects and
# factor smooths stay dense — that is M2/M3. Anything not discretised is
# carried in `Xdense` and keeps the existing chunked BLAS-3 kernels, so a
# partially-discretised model is exact wherever it is dense.

"""
    DiscreteBlock

One discretised 1-D smooth: its basis at the unique covariate values (`Xd`,
`m x p`), the per-observation row index into it (`k`), and the columns of the
coefficient vector it owns.

`exact` records whether binning was lossless (the covariate had at most `m`
distinct values, so `Xd[k, :]` reproduces the dense block bit-for-bit) or
involved rounding onto mgcv's equally spaced grid, which is where
`discrete = true` becomes an approximation.
"""
struct DiscreteBlock
    Xd::Matrix{Float64}
    k::Vector{Int32}
    cols::UnitRange{Int}
    m::Int
    exact::Bool
    label::String
    # Scratch, sized once at construction so the hot loops never allocate.
    wb::Vector{Float64}
    wzb::Vector{Float64}
    sw::Vector{Float64}
    Xw::Matrix{Float64}
    work::Vector{Float64}
end

function DiscreteBlock(Xd::Matrix{Float64}, k::Vector{Int32},
    cols::UnitRange{Int}, exact::Bool, label::String)
    m, pb = size(Xd)
    return DiscreteBlock(Xd, k, cols, m, exact, label,
        zeros(m), zeros(m), zeros(m), zeros(m, pb), zeros(m))
end

"""
    DiscreteDesign

Model matrix stored as discretised 1-D smooth blocks plus a dense remainder.

Implements the full [`BamDesign`](@ref) interface, so `bam` fits against it
unchanged. `hat_diag` is still available — it is `O(n p²)` either way, and
discretisation buys nothing there (mgcv says the same of `diagXVXt`,
`discrete.c:641`) — but it is computed by gathering rows a chunk at a time, so
the `n x p` matrix is never materialised.
"""
struct DiscreteDesign <: BamDesign
    blocks::Vector{DiscreteBlock}
    Xdense::Matrix{Float64}
    dense_cols::Vector{Int}
    n::Int
    p::Int
    icpt::Base.RefValue{Int}
    bdense::Vector{Float64}
    Wdense::Matrix{Float64}
    wzdense::Vector{Float64}
    # True when the dense remainder is exactly the intercept column. Then its
    # diagonal entry is `sum(w)` and its cross-term with block `a` is
    # `Xd_a' * wb_a` -- both already available from the binning pass -- so the
    # two extra O(n) sweeps the general path needs disappear. This is the
    # common shape (intercept + smooths) and it is what makes the kernel hit
    # its one-O(n)-pass target.
    icpt_only::Bool
    Tcross::Vector{Matrix{Float64}}
end

function DiscreteDesign(blocks::Vector{DiscreteBlock}, Xdense::Matrix{Float64},
    dense_cols::Vector{Int}, n::Int, p::Int, icpt::Int)
    pd = length(dense_cols)
    icpt_only = pd == 1 && icpt != 0 && dense_cols[1] == icpt
    Tcross = [zeros(blk.m, pd) for blk in blocks]
    return DiscreteDesign(blocks, Xdense, dense_cols, n, p, Ref(icpt),
        zeros(pd), zeros(pd, pd), zeros(pd), icpt_only, Tcross)
end

ncols(D::DiscreteDesign) = D.p
nrows(D::DiscreteDesign) = D.n
intercept_col(D::DiscreteDesign) = D.icpt[]

"""
    _bin_covariate(x, m; shuffle=true, seed=8547) -> (xu, k, exact)

Port of mgcv's `compress.df` (`R/bam.r:126-183`) for a single metric variable.

Two branches, as in mgcv. If `x` has at most `m` distinct values they are kept
exactly and the representation is lossless. Otherwise each value is rounded
onto an equally spaced `m`-point grid spanning the observed range
(`kx <- round((x-xl)/dx)+1`, `bam.r:153-159`) — this, and only this, is where
`discrete = true` stops being exact.

`shuffle` permutes the unique values under a fixed seed, as mgcv does at
`bam.r:173-176`. mgcv needs it because it packs every marginal into one padded
model frame, where a shared row ordering would induce spurious dependence
between separately discretised covariates and confuse `gam.side`. Here each
block carries its own index vector, so no such coupling exists and the
permutation is numerically inert — `Xd[k, :]` is unchanged by it. It is kept
on by default anyway: it costs nothing, it matches mgcv, and a later milestone
that adopts the padded shared-frame representation would need it.
"""
function _bin_covariate(x::AbstractVector, m::Int;
    shuffle::Bool = true, seed::UInt64 = 0x000000000000216b)

    n = length(x)
    xf = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        xf[i] = Float64(x[i])
    end

    ux = unique(xf)
    local k::Vector{Int32}
    local exact::Bool

    if length(ux) <= m
        sort!(ux)
        pos = Dict{Float64, Int32}()
        sizehint!(pos, length(ux))
        @inbounds for (i, v) in enumerate(ux)
            pos[v] = Int32(i)
        end
        k = Vector{Int32}(undef, n)
        @inbounds for i in 1:n
            k[i] = pos[xf[i]]
        end
        exact = true
    else
        xl, xh = extrema(xf)
        dx = (xh - xl) / (m - 1)
        g = Vector{Int32}(undef, n)
        @inbounds for i in 1:n
            g[i] = Int32(clamp(round(Int, (xf[i] - xl) / dx) + 1, 1, m))
        end
        used = sort!(unique(g))
        remap = Dict{Int32, Int32}()
        sizehint!(remap, length(used))
        @inbounds for (i, v) in enumerate(used)
            remap[v] = Int32(i)
        end
        ux = [xl + (Float64(v) - 1) * dx for v in used]
        k = Vector{Int32}(undef, n)
        @inbounds for i in 1:n
            k[i] = remap[g[i]]
        end
        exact = false
    end

    if shuffle && length(ux) > 1
        perm = _lcg_permutation(length(ux), seed)
        inv = Vector{Int32}(undef, length(ux))
        @inbounds for (newpos, old) in enumerate(perm)
            inv[old] = Int32(newpos)
        end
        ux = ux[perm]
        @inbounds for i in 1:n
            k[i] = inv[k[i]]
        end
    end

    return ux, k, exact
end

# Fisher-Yates under a self-contained 64-bit LCG. Written out rather than
# taken from `Random` so the binning is reproducible independently of Julia's
# RNG stream, which is not stable across versions.
function _lcg_permutation(n::Int, seed::UInt64)
    p = collect(1:n)
    s = seed == 0 ? 0x9e3779b97f4a7c15 : seed
    @inbounds for i in n:-1:2
        s = 6364136223846793005 * s + 1442695040888963407
        j = Int(rem(s >> 33, UInt64(i))) + 1
        p[i], p[j] = p[j], p[i]
    end
    return p
end

"""
    _reduced_knots(spec, col) -> Union{Vector{Float64}, Nothing}

Knots for reduced construction, taken from the FULL covariate `col` rather
than from the grid — otherwise the basis itself changes and every parity
result against the dense path is void.

`nothing` is returned where automatic placement is already exact on the grid,
and passing knots back would be wrong:

  - `ThinPlateSpline` selects the unique covariate values (capped at
    `max_knots`), and on the reduced grid those *are* the unique values, so
    the automatic branch reproduces the full knot vector elementwise.
  - `PSpline`/`CyclicPSpline`/`BSplineBasis` store a *padded* B-spline knot
    vector which re-pads if supplied back (16 → 22 knots, 19 → 17 columns).
    They place knots from the covariate range alone, which the grid spans.
"""
_reduced_knots(spec::SmoothSpec{ThinPlateSpline}, col) = nothing
_reduced_knots(spec::SmoothSpec{PSpline}, col) = nothing
_reduced_knots(spec::SmoothSpec{CyclicPSpline}, col) = nothing
_reduced_knots(spec::SmoothSpec{BSplineBasis}, col) = nothing
_reduced_knots(spec::SmoothSpec, col) = place_knots(col, spec.k)

"""
    _reduced_smooth(spec, t, m_grid, n) -> (sm, k, counts) | nothing

Construct `spec` at the `m` unique covariate values instead of all `n` rows,
so the `n × p_b` block is never formed. Returns `nothing` when the smooth does
not qualify, leaving the caller to construct it densely.

The count vector is supplied as `row_weights`, which is what makes this
faithful rather than approximate: the sum-to-zero constraint `C = colSums(X)`
is multiplicity-dependent, so `Σ_b count_b · Xd[b,:]` reproduces the full-data
column sums. TPRS additionally weights its mean-centring `shift` and its
column-RMS `col_scales` from the same scoped value.
"""
function _reduced_smooth(spec::SmoothSpec, t, m_grid::Int, n::Int)
    length(spec.term_vars) == 1 || return nothing
    spec.by === nothing || return nothing
    v = spec.term_vars[1]
    v in Tables.columnnames(t) || return nothing
    col = Tables.getcolumn(t, v)
    eltype(col) <: Real || return nothing

    # `shuffle = false` is required here, unlike in `bam_design`. The shuffle
    # is inert when the basis is reconstructed as `Xd[k, :]`, but TPRS takes
    # the unique covariate values AS its knots, and knot *order* changes the
    # parameterization: with the shuffle on, the knot sets match the dense path
    # exactly (setdiff 0/0) while the basis differs by a relative 1.97.
    xu, k, _ = _bin_covariate(col, m_grid; shuffle = false)
    m = length(xu)
    # Reducing buys nothing once the grid is as large as the sample.
    m < n || return nothing

    counts = zeros(Float64, m)
    @inbounds for i in 1:n
        counts[k[i]] += 1.0
    end

    knots = _reduced_knots(spec, col)
    nt = NamedTuple{(v,)}((xu,))
    sm = try
        with_row_weights(counts) do
            smooth_construct(spec, nt, knots)
        end
    catch
        return nothing
    end
    size(sm.X, 1) == m || return nothing
    return (sm, k, counts)
end

"""
    bam_design(X, smooths, data, discrete) -> BamDesign

Build the design for a `bam` fit, discretising where `discrete` allows.

`discrete` is `false` (always dense), `true` (mgcv's default grid, `m = 1000`
for a 1-D marginal), or an integer giving the grid resolution directly.

A smooth is discretised only if it is 1-D, has no `by=` variable, and its
basis reproduces the dense block from the unique covariate values. Everything
else stays in the dense remainder, so an unsupported term costs correctness
nothing. Returns a [`DenseDesign`](@ref) unchanged if nothing qualifies.
"""
function bam_design(X::Matrix{Float64}, smooths, data, discrete)
    discrete === false && return DenseDesign(X)
    m_grid = discrete === true ? 1000 : Int(discrete)
    m_grid >= 2 || throw(ArgumentError(
        "discrete grid resolution must be at least 2, got $m_grid"))

    n, p = size(X)
    t = Tables.columntable(data)
    blocks = DiscreteBlock[]
    taken = falses(p)

    for sm in smooths
        spec = sm.spec
        length(spec.term_vars) == 1 || continue
        spec.by === nothing || continue
        v = spec.term_vars[1]
        Tables.columnnames(t) isa Tuple && !(v in Tables.columnnames(t)) && continue
        col = Tables.getcolumn(t, v)
        eltype(col) <: Real || continue

        cols = sm.first_para:sm.last_para
        (first(cols) >= 1 && last(cols) <= p) || continue
        pb = length(cols)
        pb >= 1 || continue

        xu, k, exact = _bin_covariate(col, m_grid)
        mm = length(xu)
        # Discretising buys nothing once the grid is as large as the sample.
        mm < n || continue

        Xd = if exact
            # Lossless: every row in a bin has the same covariate value, so
            # taking one representative row of the dense block reproduces it
            # BIT-for-bit rather than merely to basis-evaluation precision.
            rep = zeros(Int, mm)
            nfilled = 0
            @inbounds for i in 1:n
                j = k[i]
                if rep[j] == 0
                    rep[j] = i
                    nfilled += 1
                    nfilled == mm && break
                end
            end
            X[rep, cols]
        else
            # Rounded: bin members differ, so evaluate the basis at the grid
            # value, as mgcv does.
            nt = NamedTuple{(v,)}((xu,))
            Xp = try
                predict_matrix(sm, nt)
            catch
                nothing
            end
            (Xp === nothing || size(Xp) != (mm, pb)) ? nothing : Matrix{Float64}(Xp)
        end

        Xd === nothing && continue
        push!(blocks, DiscreteBlock(Xd, k, cols, exact, spec.label))
        taken[cols] .= true
    end

    isempty(blocks) && return DenseDesign(X)

    dense_cols = findall(!, taken)
    Xdense = X[:, dense_cols]
    icpt = 0
    for j in 1:p
        if all(==(1.0), view(X, :, j))
            icpt = j
            break
        end
    end
    return DiscreteDesign(blocks, Xdense, dense_cols, n, p, icpt)
end

function mul_eta!(eta::Vector{Float64}, D::DiscreteDesign,
    beta::AbstractVector{Float64})
    fill!(eta, 0.0)
    if !isempty(D.dense_cols)
        bd = D.bdense
        @inbounds for (c, j) in enumerate(D.dense_cols)
            bd[c] = beta[j]
        end
        mul!(eta, D.Xdense, bd, 1.0, 1.0)
    end
    for blk in D.blocks
        work = blk.work
        mul!(work, blk.Xd, view(beta, blk.cols))
        k = blk.k
        @inbounds for i in 1:D.n
            eta[i] += work[k[i]]
        end
    end
    return eta
end

# Bin `w` (and optionally `w .* z`) into each block's cells: the single O(n)
# pass that replaces the dense O(n·p²) sweep.
function _bin_weights!(D::DiscreteDesign, w::Vector{Float64},
    z::Union{Vector{Float64}, Nothing})
    for blk in D.blocks
        wb = blk.wb
        fill!(wb, 0.0)
        k = blk.k
        if z === nothing
            @inbounds for i in 1:D.n
                wb[k[i]] += w[i]
            end
        else
            wzb = blk.wzb
            fill!(wzb, 0.0)
            @inbounds for i in 1:D.n
                j = k[i]
                wi = w[i]
                wb[j] += wi
                wzb[j] += wi * z[i]
            end
        end
        sw = blk.sw
        @inbounds for j in 1:blk.m
            sw[j] = sqrt(max(wb[j], 0.0))
        end
    end
    return nothing
end

function _accumulate_discrete!(XtWX::Matrix{Float64},
    XtWz::Union{Vector{Float64}, Nothing}, D::DiscreteDesign,
    w::Vector{Float64}, z::Union{Vector{Float64}, Nothing}, chunk_size::Int)

    fill!(XtWX, 0.0)
    XtWz === nothing || fill!(XtWz, 0.0)
    _bin_weights!(D, w, z)
    n = D.n
    nb = length(D.blocks)
    pd = length(D.dense_cols)

    # --- diagonal blocks, and the discrete part of XᵀWz -------------------
    for blk in D.blocks
        Xw = blk.Xw
        Xd = blk.Xd
        sw = blk.sw
        @inbounds for c in 1:size(Xd, 2), r in 1:blk.m
            Xw[r, c] = Xd[r, c] * sw[r]
        end
        BLAS.syrk!('U', 'T', 1.0, Xw, 0.0, view(XtWX, blk.cols, blk.cols))
        Cb = view(XtWX, blk.cols, blk.cols)
        @inbounds for j in 1:size(Cb, 1), i in (j + 1):size(Cb, 1)
            Cb[i, j] = Cb[j, i]
        end
        if XtWz !== nothing
            mul!(view(XtWz, blk.cols), transpose(Xd), blk.wzb)
        end
    end

    # --- discrete x discrete cross blocks ---------------------------------
    for a in 1:nb, b in (a + 1):nb
        ba, bb = D.blocks[a], D.blocks[b]
        pa, pbc = size(ba.Xd, 2), size(bb.Xd, 2)
        ka, kb = ba.k, bb.k
        # Strategy (c) vs (d), as `discrete.c:1801` chooses: accumulate the
        # full cross-weight table when the sample is large relative to it,
        # otherwise build only the thin `m_a x p_b` factor.
        Mab = if n > ba.m * bb.m
            Wt = zeros(ba.m, bb.m)
            @inbounds for i in 1:n
                Wt[ka[i], kb[i]] += w[i]
            end
            Wt * bb.Xd
        else
            T = zeros(ba.m, pbc)
            Xb = bb.Xd
            @inbounds for i in 1:n
                ja = ka[i]
                jb = kb[i]
                wi = w[i]
                for c in 1:pbc
                    T[ja, c] += wi * Xb[jb, c]
                end
            end
            T
        end
        Blk = transpose(ba.Xd) * Mab
        @inbounds for j in 1:pbc, i in 1:pa
            XtWX[ba.cols[i], bb.cols[j]] = Blk[i, j]
            XtWX[bb.cols[j], ba.cols[i]] = Blk[i, j]
        end
    end

    # --- dense remainder, and discrete x dense ----------------------------
    if D.icpt_only
        # Intercept-only remainder: everything below is already binned.
        j0 = D.dense_cols[1]
        b1 = D.blocks[1]
        XtWX[j0, j0] = sum(b1.wb)
        XtWz === nothing || (XtWz[j0] = sum(b1.wzb))
        for blk in D.blocks
            v = transpose(blk.Xd) * blk.wb
            @inbounds for (ci, c) in enumerate(blk.cols)
                XtWX[c, j0] = v[ci]
                XtWX[j0, c] = v[ci]
            end
        end
    elseif pd > 0
        Xden = D.Xdense
        Wd = D.Wdense
        if XtWz === nothing
            _accumulate_XtWX_chunked!(Wd, Xden, w, chunk_size)
        else
            _accumulate_XtWX_XtWz_chunked!(Wd, D.wzdense, Xden, w, z, chunk_size)
            @inbounds for (c, j) in enumerate(D.dense_cols)
                XtWz[j] = D.wzdense[c]
            end
        end
        @inbounds for cj in 1:pd, ci in 1:pd
            XtWX[D.dense_cols[ci], D.dense_cols[cj]] = Wd[ci, cj]
        end

        for (ai, blk) in enumerate(D.blocks)
            pa = size(blk.Xd, 2)
            k = blk.k
            T = D.Tcross[ai]
            fill!(T, 0.0)
            @inbounds for i in 1:n
                j = k[i]
                wi = w[i]
                for c in 1:pd
                    T[j, c] += wi * Xden[i, c]
                end
            end
            Blk = transpose(blk.Xd) * T
            @inbounds for cj in 1:pd, ci in 1:pa
                XtWX[blk.cols[ci], D.dense_cols[cj]] = Blk[ci, cj]
                XtWX[D.dense_cols[cj], blk.cols[ci]] = Blk[ci, cj]
            end
        end
    end

    return nothing
end

function accumulate_XtWX_XtWz!(XtWX::Matrix{Float64}, XtWz::Vector{Float64},
    D::DiscreteDesign, w::Vector{Float64}, z::Vector{Float64};
    chunk_size::Int = 10000)
    _accumulate_discrete!(XtWX, XtWz, D, w, z, chunk_size)
    return XtWX, XtWz
end

function accumulate_XtWX!(XtWX::Matrix{Float64}, D::DiscreteDesign,
    w::Vector{Float64}; chunk_size::Int = 10000)
    _accumulate_discrete!(XtWX, nothing, D, w, nothing, chunk_size)
    return XtWX
end

"""
    _gather_rows!(H, D::DiscreteDesign, start, stop)

Materialise rows `start:stop` of the implied model matrix into `H`.

Used only by the leverage sweep, which is `O(n p²)` for any representation.
Working a chunk at a time keeps the `n x p` matrix from ever existing.
"""
function _gather_rows!(H::AbstractMatrix{Float64}, D::DiscreteDesign,
    start::Int, stop::Int)
    nr = stop - start + 1
    if !isempty(D.dense_cols)
        @inbounds for (c, j) in enumerate(D.dense_cols), i in 1:nr
            H[i, j] = D.Xdense[start + i - 1, c]
        end
    end
    for blk in D.blocks
        Xd = blk.Xd
        k = blk.k
        cols = blk.cols
        @inbounds for i in 1:nr
            r = k[start + i - 1]
            for (c, j) in enumerate(cols)
                H[i, j] = Xd[r, c]
            end
        end
    end
    return H
end

function design_finalize(D::DiscreteDesign, w::Vector{Float64},
    XtWX::Matrix{Float64}, A_chol::Cholesky;
    chunk_size::Int = 1024, compute_hat_diag::Bool = true)

    F = A_chol \ XtWX
    edf_vec = diag(F)
    U = A_chol.U
    compute_hat_diag || return edf_vec, Float64[], Matrix(U)

    n, p = D.n, D.p
    hat_diag = zeros(n)
    cs = clamp(chunk_size, 1, max(n, 1))
    H = Matrix{Float64}(undef, cs, p)

    for start in 1:cs:n
        stop = min(start + cs - 1, n)
        nr = stop - start + 1
        Hv = view(H, 1:nr, :)
        fill!(Hv, 0.0)
        _gather_rows!(Hv, D, start, stop)
        rdiv!(Hv, U)
        @inbounds for i in 1:nr
            s = 0.0
            for j in 1:p
                s += Hv[i, j]^2
            end
            hat_diag[start + i - 1] = w[start + i - 1] * s
        end
    end

    return edf_vec, hat_diag, Matrix(U)
end
