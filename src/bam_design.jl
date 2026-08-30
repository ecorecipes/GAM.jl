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
    # Chunk scratch, allocated on first use and reused for the life of the
    # design. The accumulators previously allocated a fresh `chunk_size x p`
    # buffer per call, and on the non-Gaussian path they are called once per
    # outer EFS iteration: measured at 320 MiB for a Poisson 4 x s(cr,k=20)
    # fit at n=1e5, which is 4.1x the six hoisted n-vectors in `pirls_bam`.
    # The design outlives the whole fit, so one buffer suffices.
    Xw::Base.RefValue{Matrix{Float64}}
    wz::Base.RefValue{Vector{Float64}}
end

DenseDesign(X::Matrix{Float64}) =
    DenseDesign(X, Ref(-2), Ref(Matrix{Float64}(undef, 0, 0)), Ref(Float64[]))

"""
    _chunk_scratch!(D::DenseDesign, rows, p) -> (Xw, wz)

Return chunk scratch of at least `rows x p`, growing the cached buffers only
when a larger chunk is requested. Views are taken by the caller, so a buffer
sized for an earlier, larger chunk is reused rather than reallocated.
"""
function _chunk_scratch!(D::DenseDesign, rows::Int, p::Int)
    Xw = D.Xw[]
    if size(Xw, 1) < rows || size(Xw, 2) != p
        Xw = Matrix{Float64}(undef, rows, p)
        D.Xw[] = Xw
    end
    wz = D.wz[]
    if length(wz) < rows
        wz = Vector{Float64}(undef, rows)
        D.wz[] = wz
    end
    return Xw, wz
end

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
    nc = min(chunk_size, size(D.X, 1))
    Xw, wz = _chunk_scratch!(D, nc, size(D.X, 2))
    _accumulate_XtWX_XtWz_chunked!(XtWX, XtWz, D.X, w, z, chunk_size;
        Xw_scratch = Xw, wz_scratch = wz)
    return XtWX, XtWz
end

"""
    accumulate_XtWX!(XtWX, D::BamDesign, w; chunk_size)

Overwrite `XtWX` with `X'WX`, without the right-hand side.
"""
function accumulate_XtWX! end

function accumulate_XtWX!(XtWX::Matrix{Float64}, D::DenseDesign,
    w::Vector{Float64}; chunk_size::Int = 10000)
    nc = min(chunk_size, size(D.X, 1))
    Xw, _ = _chunk_scratch!(D, nc, size(D.X, 2))
    _accumulate_XtWX_chunked!(XtWX, D.X, w, chunk_size; Xw_scratch = Xw)
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
    # Optional per-row multiplier: row `i` of the block is
    # `scale[i] * Xd[k[i], :]` rather than `Xd[k[i], :]`. `nothing` is the
    # unscaled case and costs nothing — the kernels branch once per block, not
    # per row. It is what lets a random effect with a slope, and a numeric
    # `by=`, use this representation: both are a fixed cell basis with a
    # continuously varying per-row factor, which no `Xd`/`k` pair alone can
    # express.
    scale::Union{Nothing, Vector{Float64}}
    # Scratch, sized once at construction so the hot loops never allocate.
    wb::Vector{Float64}
    wzb::Vector{Float64}
    sw::Vector{Float64}
    Xw::Matrix{Float64}
    work::Vector{Float64}
end

function DiscreteBlock(Xd::Matrix{Float64}, k::Vector{Int32},
    cols::UnitRange{Int}, exact::Bool, label::String;
    scale::Union{Nothing, Vector{Float64}} = nothing)
    m, pb = size(Xd)
    scale === nothing || length(scale) == length(k) || throw(DimensionMismatch(
        "scale has $(length(scale)) entries, k has $(length(k))"))
    return DiscreteBlock(Xd, k, cols, m, exact, label, scale,
        zeros(m), zeros(m), zeros(m), zeros(m, pb), zeros(m))
end

"""Per-row multiplier for `blk`, `1.0` when the block is unscaled."""
@inline _blk_scale(blk::DiscreteBlock, i::Int) =
    blk.scale === nothing ? 1.0 : @inbounds(blk.scale[i])

"""True when any block carries a per-row scale."""
_any_scaled(blocks::Vector{DiscreteBlock}) = any(b -> b.scale !== nothing, blocks)

"""
    ByBlock

One factor-`by` smooth, `s(x, by = f)` with `f` a factor of `L` levels, held
as ONE shared `m x kb` basis plus two per-row index vectors — the grid cell
`k[i]` and the level `lev[i]` — instead of the dense `n x (kb*L)` replication
`_apply_by_variable!` builds.

Row `i` is zero except in the `kb` columns of its own level:

    dense[i, (lev[i]-1)*kb .+ (1:kb)] == Xd[k[i], :]

The whole point is what that implies for `X'WX`. No row contributes to two
levels, so the level blocks are **orthogonal**: every off-diagonal
`(l, l')` sub-block is identically zero. Representing this as `L` separate
`DiscreteBlock`s would be correct and need no new kernel, but the generic
discrete x discrete loop would then compute `L(L-1)/2` cross-blocks that are
all zero — 28 wasted O(n) passes at `L = 8`, **1225 at `L = 50`**. A prior
attempt measured exactly that and declined to ship it. `ByBlock` instead runs
`L` `syrk!` calls on the diagonal and skips the off-diagonals entirely.

Cross-terms against other blocks keep the same one-pass shape: `T[l]` is the
`m x p_other` table for level `l`, all `L` filled in a single sweep over the
sample, so the cost is `O(n * p_other)` regardless of `L`.

`lev[i] == 0` marks a row whose `by` level was unseen at fit time; it
contributes nothing, matching `predict_matrix`'s warn-and-zero rule.
"""
struct ByBlock
    Xd::Matrix{Float64}
    k::Vector{Int32}
    lev::Vector{Int32}
    cols::UnitRange{Int}
    m::Int
    kb::Int
    L::Int
    exact::Bool
    label::String
    # Scratch, sized once at construction so the hot loops never allocate.
    # `wb`/`wzb` are L x m: the binning pass fills every level in one sweep.
    wb::Matrix{Float64}
    wzb::Matrix{Float64}
    sw::Vector{Float64}
    Xw::Matrix{Float64}
    fbuf::Matrix{Float64}
end

function ByBlock(Xd::Matrix{Float64}, k::Vector{Int32}, lev::Vector{Int32},
    cols::UnitRange{Int}, L::Int, exact::Bool, label::String)
    m, kb = size(Xd)
    length(k) == length(lev) || throw(DimensionMismatch(
        "k has $(length(k)) entries, lev has $(length(lev))"))
    length(cols) == kb * L || throw(DimensionMismatch(
        "cols spans $(length(cols)) columns, expected kb*L = $(kb * L)"))
    return ByBlock(Xd, k, lev, cols, m, kb, L, exact, label,
        zeros(L, m), zeros(L, m), zeros(m), zeros(m, kb), zeros(L, m))
end

"""Columns of `blk` belonging to level `l`."""
@inline _by_cols(blk::ByBlock, l::Int) =
    blk.cols[((l - 1) * blk.kb + 1):(l * blk.kb)]

"""
    TensorBlock

One `te` smooth held as per-marginal bases plus index vectors.

`Z` is the sum-to-zero constraint basis, applied after accumulation. `pb` is
the number of columns of the row tensor of all marginals *except* the last,
which is the unit the kernels iterate over (mgcv's `dX`, `discrete.c:301`).
"""
struct TensorBlock
    Xds::Vector{Matrix{Float64}}
    ks::Vector{Vector{Int32}}
    Z::Matrix{Float64}
    cols::UnitRange{Int}
    pdims::Vector{Int}
    pb::Int
    praw::Int
    exact::Bool
    label::String
    dcol_a::Vector{Float64}
    dcol_b::Vector{Float64}
    tvec::Vector{Float64}
    Wraw::Matrix{Float64}
    braw::Vector{Float64}
end

function TensorBlock(Xds::Vector{Matrix{Float64}}, ks::Vector{Vector{Int32}},
    Z::Matrix{Float64}, cols::UnitRange{Int}, exact::Bool, label::String)
    pdims = [size(Xd, 2) for Xd in Xds]
    praw = prod(pdims)
    pb = praw ÷ pdims[end]
    n = length(ks[1])
    md = size(Xds[end], 1)
    return TensorBlock(Xds, ks, Z, cols, pdims, pb, praw, exact, label,
        zeros(n), zeros(n), zeros(md), zeros(praw, praw), zeros(praw))
end

ndim(blk::TensorBlock) = length(blk.Xds)

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
    # `te` smooths, held as per-marginal bases plus one index vector each. The
    # row tensor is never formed; see the TensorBlock section below.
    tblocks::Vector{TensorBlock}
    # factor-`by` smooths, one shared basis plus a grid index and a level
    # index. Level sub-blocks are orthogonal, so the kernel runs `L` diagonal
    # `syrk!`s and skips every off-diagonal; see the ByBlock section above.
    byblocks::Vector{ByBlock}
end

function DiscreteDesign(blocks::Vector{DiscreteBlock}, Xdense::Matrix{Float64},
    dense_cols::Vector{Int}, n::Int, p::Int, icpt::Int,
    tblocks::Vector{TensorBlock} = TensorBlock[],
    byblocks::Vector{ByBlock} = ByBlock[])
    pd = length(dense_cols)
    # The intercept-only shortcut is only valid when every non-dense column is
    # a 1-D block, whose cross-terms fall out of the binning pass.
    # The shortcut reads `sum(blk.wb)` as Σw and `Xd' * wb` as the cross-term.
    # Both identities assume UNSCALED binning: a scaled block bins Σ w·s², so
    # neither holds. Scaled blocks therefore take the general path.
    # A ByBlock also breaks the shortcut: its rows bin per level, so
    # `sum(wb)` over one level is not Σw and the intercept cross-term needs
    # the per-level tables rather than a single `Xd' * wb`.
    icpt_only = pd == 1 && icpt != 0 && dense_cols[1] == icpt &&
        isempty(tblocks) && isempty(byblocks) && !_any_scaled(blocks)
    Tcross = [zeros(blk.m, pd) for blk in blocks]
    return DiscreteDesign(blocks, Xdense, dense_cols, n, p, Ref(icpt),
        zeros(pd), zeros(pd, pd), zeros(pd), icpt_only, Tcross, tblocks,
        byblocks)
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
# ThinPlateShrink shares `_construct_tprs` with ThinPlateSpline, so the same
# rule applies: its dense knots are the unique covariate values, which on the
# reduced grid ARE the unique values. Feeding it `place_knots` quantile knots
# instead built a different basis — fitted values 29% off dense, silently.
_reduced_knots(spec::SmoothSpec{ThinPlateShrink}, col) = nothing
_reduced_knots(spec::SmoothSpec{PSpline}, col) = nothing
_reduced_knots(spec::SmoothSpec{CyclicPSpline}, col) = nothing
_reduced_knots(spec::SmoothSpec{BSplineBasis}, col) = nothing
_reduced_knots(spec::SmoothSpec, col) = place_knots(col, spec.k)

"""
    _reduced_eligible(basis) -> Bool

Whitelist of bases proven faithful under reduced construction — building at
the `m` unique covariate values reproduces the dense basis (verified against
the dense path at fixed `sp` to ≤1e-12 per basis; pinned by the per-basis
parity testset in test/test_discrete.jl). Everything else falls back to dense
construction for that smooth: silently-correct rather than silently-wrong.

This is a WHITELIST deliberately. The previous default-allow rule fed
`place_knots` quantile knots into constructors whose dense path uses
`user_knots === nothing`, silently mis-discretising five bases (`:ts` fitted
values 29% off dense, `:gp` 4.2%, `:ad` even changing the coefficient count,
`:fp`, `:lo`) with no warning — the same defect shape as five earlier silent
representation regressions in this package. Known non-members and why:

  - `GPSmooth`: dense knots are the unique covariate values capped at 2000;
    quantile knots build a different basis.
  - `AdaptiveSmooth`: internal evenly-spaced knot builder; supplying knots
    changes the basis DIMENSION itself.
  - `FractionalPolynomial`, `LoessSmooth`: construction is not
    row-weight-faithful (loess local regression especially).
  - `DuchonSpline`: warns-and-delegates to `tp`; keep it dense rather than
    reason about the aliasing under reduction.
  - `RandomEffect`: guarded separately (its `k` is the level count).

To add a basis: prove parity first — the per-basis testset fails for any
registered basis that is neither faithful nor falling back.
"""
_reduced_eligible(::AbstractBasisType) = false
_reduced_eligible(::ThinPlateSpline) = true
_reduced_eligible(::ThinPlateShrink) = true
_reduced_eligible(::CubicSpline) = true
_reduced_eligible(::CubicShrink) = true
_reduced_eligible(::CyclicCubic) = true
_reduced_eligible(::PSpline) = true
_reduced_eligible(::CyclicPSpline) = true
_reduced_eligible(::BSplineBasis) = true

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
    # A factor `by` reduces too, on the (cell, level) pair grid; numeric `by`
    # is a per-row multiplier rather than a level replication and is rejected
    # inside `_reduced_by_smooth`.
    spec.by === nothing || return _reduced_by_smooth(spec, t, m_grid, n)
    # A random effect's `k` is its level count, so rebuilding it on a reduced
    # grid yields a different (and possibly negative) k -- it used to throw
    # `ArgumentError: k must be >= 1, got -1`. The compact form for `bs=:re`
    # exists (`re_marginal_representation`, basis_re.jl) but is not wired into
    # the design yet, so fall back to dense rather than error.
    spec.basis isa RandomEffect && return nothing
    # Whitelist gate: only bases proven faithful under reduced construction
    # reduce; the rest are constructed densely by the caller (see
    # `_reduced_eligible` for the roll of silently-wrong bases this replaced).
    _reduced_eligible(spec.basis) || return nothing
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
    _reduced_by_smooth(spec, t, m_grid, n) -> (sm, pairidx, counts) | nothing

Reduced construction for a **factor** `by=` smooth. Returns `nothing` when the
smooth does not qualify, leaving the caller to construct it densely.

The dense block is `n × (kb·L)`, and row `i` is zero except in level `lev[i]`'s
`kb` columns, where it holds the base basis at `x[i]`. So its distinct rows are
exactly the distinct observed `(cell, level)` pairs — at most `m·L`, and in
practice far fewer than `n`. That makes it the same "row `i` is row
`rowmap[i]`" contract the plain reduced path already uses.

The base is built on the plain **cell** grid, not on the pair grid, and that is
deliberate: `smooth_construct` applies `_apply_by_variable!` *after* the base is
constructed and its constraint absorbed, so the base only ever sees the
covariate. Building it on the pair grid instead would repeat each covariate
value `L` times, and `_reduced_knots` returns `nothing` for TPRS precisely
because the reduced grid's unique values *are* its knots — a repeated column
would change the knot ORDER and silently reparameterise the basis, the same
failure that made the row shuffle unsafe for TPRS elsewhere.

Row weights therefore come from the **cell** counts summed over levels, which
is what the base constraint `C = colSums(X)` sees across all `n` rows.
"""
function _reduced_by_smooth(spec::SmoothSpec, t, m_grid::Int, n::Int)
    spec.by === nothing && return nothing
    # `bs=:re` with a `by` silently dropped the multiplier once already
    # (relΔcoef 0.49, 31% of fitted range); keep it dense.
    spec.basis isa RandomEffect && return nothing
    # Same whitelist as the plain path: the base basis must itself be
    # faithful under reduced construction.
    _reduced_eligible(spec.basis) || return nothing
    v = spec.term_vars[1]
    v in Tables.columnnames(t) || return nothing
    spec.by in Tables.columnnames(t) || return nothing
    col = Tables.getcolumn(t, v)
    eltype(col) <: Real || return nothing
    by_col = Tables.getcolumn(t, spec.by)
    # Numeric `by` multiplies every row; it is not a level replication.
    eltype(by_col) <: Real && return nothing

    xu, kcell, _ = _bin_covariate(col, m_grid; shuffle = false)
    m = length(xu)
    m < n || return nothing

    levels = sort!(_unique_levels(by_col))
    L = length(levels)
    L >= 1 || return nothing
    pos = Dict(lev => l for (l, lev) in enumerate(levels))

    # Distinct observed (cell, level) pairs. Sorting the packed key orders them
    # cell-major, which keeps the expanded rows grouped by covariate value.
    key = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        l = get(pos, by_col[i], 0)
        l == 0 && return nothing      # unseen level: leave it to the dense path
        key[i] = (Int(kcell[i]) - 1) * L + l
    end
    ukeys = sort!(unique(key))
    mp = length(ukeys)
    mp < n || return nothing
    kpos = Dict(kk => Int32(p) for (p, kk) in enumerate(ukeys))
    pairidx = Int32[kpos[key[i]] for i in 1:n]

    cell_of_pair = Vector{Int}(undef, mp)
    lev_of_pair = Vector{Int}(undef, mp)
    @inbounds for (p, kk) in enumerate(ukeys)
        cell_of_pair[p] = div(kk - 1, L) + 1
        lev_of_pair[p] = mod(kk - 1, L) + 1
    end

    pair_counts = zeros(Float64, mp)
    cell_counts = zeros(Float64, m)
    @inbounds for i in 1:n
        pair_counts[pairidx[i]] += 1.0
        cell_counts[kcell[i]] += 1.0
    end

    knots = _reduced_knots(spec, col)
    nt_cells = NamedTuple{(v,)}((xu,))
    sm = try
        with_row_weights(cell_counts) do
            s = _smooth_construct(spec.basis, spec, nt_cells, knots)
            _append_pc_constraints!(s, nt_cells)
            s
        end
    catch
        return nothing
    end
    size(sm.X, 1) == m || return nothing
    # `_apply_by_variable!` rejects linear-constraint (scasm) smooths outright.
    (sm.Ain === nothing && sm.Aeq === nothing) || return nothing

    # Expand cell rows to pair rows, then let the ordinary by-expansion
    # replicate per level exactly as the dense path does.
    sm.X = sm.X[cell_of_pair, :]
    pair_t = NamedTuple{(v, spec.by)}(
        (xu[cell_of_pair], [levels[l] for l in lev_of_pair]))
    try
        _apply_by_variable!(sm, pair_t)
    catch
        return nothing
    end
    size(sm.X, 1) == mp || return nothing

    return (sm, pairidx, pair_counts)
end


"""
    _re_block(spec, t, n, cols) -> DiscreteBlock | nothing

Represent a `bs=:re` smooth compactly. Returns `nothing` when it does not
qualify, leaving the caller to keep the dense block.

The dense random-effect block is an `n x k` indicator scaled by the random
slope: row `i` is zero except at column `index[i]`, where it is `slope[i]`
(see `_re_level_index`). That is `scale[i] * Xd[k[i], :]` with `Xd = I_k`, so
it is a [`DiscreteBlock`](@ref) verbatim, and reproduces the dense block
BIT-for-bit rather than to basis-evaluation precision — the identity rows are
the dense rows.

Storage is `k^2` for the identity plus `O(n)` for the index, against `O(n*k)`
dense. The `k < n` guard keeps that a win: at `n = 5e5` with 200 levels it is
~2.3 MB against ~800 MB.
"""
function _re_block(spec::SmoothSpec, t, n::Int, cols::UnitRange{Int})
    all(v -> v in Tables.columnnames(t), spec.term_vars) || return nothing
    index, slope, k, _, _, _ = try
        _re_level_index(spec, t)
    catch
        return nothing
    end
    # The column range must be exactly the level count, or this is not the
    # plain random-effect block we think it is.
    length(cols) == k || return nothing
    length(index) == n || return nothing
    # An unseen level would give index 0, which the dense constructor cannot
    # produce on its own data; bail rather than guess.
    all(>(Int32(0)), index) || return nothing
    # `I_k` costs k^2; reducing is only a win while k stays well under n.
    k < n || return nothing

    Xd = Matrix{Float64}(I, k, k)
    scale = all(isone, slope) ? nothing : copy(slope)
    return DiscreteBlock(Xd, index, cols, true, spec.label; scale = scale)
end

"""
    _smooth_rows(sm, rows) -> Matrix{Float64}

Rows `rows` of the smooth's `n`-row model-matrix block, without ever forming
that block.

`sm.X` is bitwise `X[:, sm.first_para:sm.last_para]`, so for an ordinary
smooth this is a plain gather. When the smooth holds a REDUCED basis
(`bam(...; discrete=true)` builds it at the `m` unique covariate values),
row `i` of the dense block is `sm.X[sm.rowmap[i], :]`, so the row map has to
be composed in.

That composition is not cosmetic: `setup_gam_discrete` bins with
`shuffle = false` while `bam_design` bins with mgcv's row shuffle, so the two
grids can order their cells differently. Indexing `sm.X` directly with a row
number computed here would silently pick the wrong cell.
"""
function _smooth_rows(sm, rows::AbstractVector{Int})
    Xsm = sm.X
    if isempty(sm.rowmap)
        return Xsm[rows, :]
    end
    rm = sm.rowmap
    out = Matrix{Float64}(undef, length(rows), size(Xsm, 2))
    @inbounds for (j, i) in enumerate(rows)
        src = rm[i]
        for c in axes(Xsm, 2)
            out[j, c] = Xsm[src, c]
        end
    end
    return out
end

"""
    _by_block(sm, t, m_grid, n, cols) -> ByBlock | nothing

Build a [`ByBlock`](@ref) for a factor-`by` smooth, or `nothing` if it does
not qualify.

Only a FACTOR `by` qualifies. A numeric `by` is a per-row multiplier on a
single block, which `DiscreteBlock`'s `scale` field already expresses; it is
not routed here and keeps its existing dense fallback, because conflating the
two is what produced a 31%-of-range wrong fit on `bs=:re` earlier in this
work.

The shared basis comes from the dense block without re-evaluating anything
when binning is lossless: every row in a grid cell has the same covariate
value, and row `i` carries the base basis in its own level's columns, so one
representative row per cell reproduces `Xd` BIT-for-bit. When the covariate
had to be rounded onto a grid, the base spec is rebuilt (`by=` stripped) and
evaluated at the grid values, as mgcv does.
"""
function _by_block(sm, t, m_grid::Int, n::Int, cols::UnitRange{Int})
    spec = sm.spec
    length(spec.term_vars) == 1 || return nothing
    spec.by === nothing && return nothing
    v = spec.term_vars[1]
    v in Tables.columnnames(t) || return nothing
    spec.by in Tables.columnnames(t) || return nothing

    by_col = Tables.getcolumn(t, spec.by)
    # Numeric `by` is not a factor replication; leave it to the dense path.
    eltype(by_col) <: Real && return nothing

    col = Tables.getcolumn(t, v)
    eltype(col) <: Real || return nothing

    levels = get(spec.xt, :_by_levels, nothing)
    levels === nothing && return nothing
    L = length(levels)
    L >= 1 || return nothing
    pb = length(cols)
    (pb % L == 0) || return nothing
    kb = pb ÷ L
    kb >= 1 || return nothing

    pos = Dict(lev => Int32(l) for (l, lev) in enumerate(levels))
    lev = Int32[get(pos, x, Int32(0)) for x in by_col]

    xu, k, exact = _bin_covariate(col, m_grid)
    mm = length(xu)
    # Discretising buys nothing once the grid is as large as the sample.
    mm < n || return nothing

    Xd = if exact
        # One representative row per grid cell, read out of that row's OWN
        # level columns — the base basis is identical across levels, so this
        # is exact rather than merely close.
        rep = zeros(Int, mm)
        nfilled = 0
        @inbounds for i in 1:n
            lev[i] == 0 && continue
            j = k[i]
            if rep[j] == 0
                rep[j] = i
                nfilled += 1
                nfilled == mm && break
            end
        end
        any(==(0), rep) && return nothing
        # Read out of the smooth's OWN block rather than the assembled model
        # matrix: `sm.X` is bitwise `X[:, cols]`, so this is the same numbers
        # without needing `X` to exist.
        #
        # A `by` smooth CAN now be reduced, so observation `i` maps through
        # `sm.rowmap`. Going via the observation index is what makes this safe
        # despite the two binnings disagreeing: `setup_gam_discrete` bins with
        # `shuffle = false` while the `k` above comes from mgcv's shuffled
        # binning, so the cell numberings differ and indexing `sm.X` by `k[i]`
        # would silently read the wrong cell.
        Xsm = sm.X
        red = is_reduced(sm)
        B = Matrix{Float64}(undef, mm, kb)
        @inbounds for j in 1:mm
            i = rep[j]
            ri = red ? Int(sm.rowmap[i]) : i
            off = (lev[i] - 1) * kb
            for c in 1:kb
                B[j, c] = Xsm[ri, off + c]
            end
        end
        B
    else
        base_sm = try
            first(by_marginal_representation(spec, t))
        catch
            nothing
        end
        base_sm === nothing && return nothing
        nt = NamedTuple{(v,)}((xu,))
        Xp = try
            predict_matrix(base_sm, nt)
        catch
            nothing
        end
        (Xp === nothing || size(Xp) != (mm, kb)) ? nothing : Matrix{Float64}(Xp)
    end

    Xd === nothing && return nothing
    return ByBlock(Xd, k, lev, cols, L, exact, spec.label)
end

"""
    _tensor_block(sm, t, m_grid, n, cols) -> TensorBlock | nothing

Build a [`TensorBlock`](@ref) for a `te` smooth, or `nothing` if it does not
qualify.

Only `te` qualifies: `ti` absorbs a constraint into each marginal and `t2`
folds a per-marginal reparameterization, so neither has the single post-hoc
`Z` this representation relies on. Both are detected structurally (a non-empty
`marginal_Zs`, or a missing overall `constraint`) rather than by basis type, so
a future construction that changes shape cannot silently slip through.
"""
function _tensor_block(sm, t, m_grid::Int, n::Int, cols::UnitRange{Int})
    cache = sm.predict_cache
    cache isa TensorPredictCache || return nothing
    isempty(cache.marginal_Zs) || return nothing       # ti
    sm.constraint === nothing && return nothing        # no post-hoc Z
    raws = cache.raw_marginals
    length(raws) >= 2 || return nothing

    Xds = Matrix{Float64}[]
    ks = Vector{Int32}[]
    all_exact = true
    for rm in raws
        mspec = rm.spec
        length(mspec.term_vars) == 1 || return nothing
        v = mspec.term_vars[1]
        v in Tables.columnnames(t) || return nothing
        col = Tables.getcolumn(t, v)
        eltype(col) <: Real || return nothing
        # The exact branch below reads `rm.X[rep, :]`, so the raw marginal must
        # still carry one row per observation. A `TensorPredictCache` that has
        # dropped its n-row marginals (a memory optimisation) leaves an empty
        # `X` here; fall back to the dense design rather than throwing, so the
        # two representations compose instead of colliding.
        size(rm.X, 1) == n || return nothing
        # `shuffle = false`: several marginal bases take the unique values as
        # their knots, and knot ORDER changes the parameterization -- the same
        # trap that cost the 1-D reduced path a relative 1.97 in the basis.
        xu, k, exact = _bin_covariate(col, m_grid; shuffle = false)
        m = length(xu)
        m < n || return nothing
        Xd = if exact
            # Lossless: one representative row per bin reproduces the marginal
            # bit-for-bit, not merely to basis-evaluation precision.
            rep = zeros(Int, m)
            nf = 0
            @inbounds for i in 1:n
                j = k[i]
                if rep[j] == 0
                    rep[j] = i
                    nf += 1
                    nf == m && break
                end
            end
            rm.X[rep, :]
        else
            all_exact = false
            nt = NamedTuple{(v,)}((xu,))
            r2 = try
                _build_raw_marginal(mspec, nt, _reduced_knots(mspec, col))
            catch
                nothing
            end
            (r2 === nothing || size(r2.X) != (m, size(rm.X, 2))) ? nothing :
                Matrix{Float64}(r2.X)
        end
        Xd === nothing && return nothing
        push!(Xds, Xd)
        push!(ks, k)
    end

    praw = prod(size(Xd, 2) for Xd in Xds)
    Z = _constraint_basis(sm.constraint, praw)
    size(Z, 1) == praw && size(Z, 2) == length(cols) || return nothing

    blk = TensorBlock(Xds, ks, Z, cols, all_exact, sm.spec.label)
    # The accumulator walks pb(pb+1)/2 sub-blocks, each an O(n) pass, so cost
    # grows as pb^2. mgcv pays ~100 s for a single XᵀWX on te(6,6,6,6) at
    # n = 1e6, where pb = 216. Warn rather than refuse: it is still correct,
    # and the dense path may be worse on memory.
    if blk.pb > 50
        @warn "Discretised tensor $(blk.label) has pb = $(blk.pb) (>50): " *
              "XᵀWX walks $(blk.pb * (blk.pb + 1) ÷ 2) sub-blocks and cost " *
              "grows as pb^2. Consider fewer or smaller marginals, or " *
              "discrete = false for this term." maxlog = 1
    end
    return blk
end

"""
    _discrete_blocks(smooths, t, m_grid, n, p) -> (blocks, tblocks, byblocks, taken)

Build every compact block a discrete design can hold, reading only the
per-smooth bases — never an assembled `n x p` model matrix. `taken[j]` marks
column `j` as covered by some block; the caller supplies the dense remainder
for the rest.
"""
function _discrete_blocks(smooths, t, m_grid::Int, n::Int, p::Int)
    blocks = DiscreteBlock[]
    tblocks = TensorBlock[]
    byblocks = ByBlock[]
    taken = falses(p)

    for sm in smooths
        spec = sm.spec
        # `bs=:re` first: its dense block is an `n x k` scaled indicator, so it
        # is a DiscreteBlock with `Xd = I_k` and `k` the level index -- exactly
        # the "row i is Xd[k[i], :]" contract, hence bit-exact. A random slope
        # becomes the per-row `scale`. This must precede the tensor branch,
        # because a multi-variable RE (an interaction) has more than one term
        # variable and would otherwise be swallowed there.
        if spec.basis isa RandomEffect
            # A `by=` variable multiplies every row, and the compact block does
            # not encode it -- `_re_block` builds a pure {0,1} indicator, so
            # discretising here silently DROPS the multiplier and returns a
            # wrong fit rather than falling back. Measured on a genuine random
            # slope: relΔcoef 0.49, max|Δfitted| 2.34 on a fitted range of 7.56
            # (31% of range). This branch precedes the `by` guard below, so it
            # must repeat it.
            spec.by === nothing || continue
            cols = sm.first_para:sm.last_para
            rb = (first(cols) >= 1 && last(cols) <= p) ?
                _re_block(spec, t, n, cols) : nothing
            if rb !== nothing
                push!(blocks, rb)
                taken[cols] .= true
            end
            continue
        end
        if length(spec.term_vars) > 1
            cols = sm.first_para:sm.last_para
            if first(cols) >= 1 && last(cols) <= p
                tb = _tensor_block(sm, t, m_grid, n, cols)
                if tb !== nothing
                    push!(tblocks, tb)
                    taken[cols] .= true
                end
            end
            continue
        end
        length(spec.term_vars) == 1 || continue
        # A factor `by` replicates the basis per level, and the level blocks
        # are orthogonal — `ByBlock` exploits that. A NUMERIC `by` is a
        # different thing (a per-row multiplier) and is not covered here, so
        # it must keep falling back: dropping that distinction is what
        # silently produced a wrong fit on `bs=:re` earlier in this work.
        if spec.by !== nothing
            cols = sm.first_para:sm.last_para
            bb = (first(cols) >= 1 && last(cols) <= p) ?
                _by_block(sm, t, m_grid, n, cols) : nothing
            if bb !== nothing
                push!(byblocks, bb)
                taken[cols] .= true
            end
            continue
        end
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
            _smooth_rows(sm, rep)
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

    return blocks, tblocks, byblocks, taken
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
    m_grid = _discrete_grid(discrete)

    n, p = size(X)
    t = Tables.columntable(data)
    blocks, tblocks, byblocks, taken =
        _discrete_blocks(smooths, t, m_grid, n, p)

    (isempty(blocks) && isempty(tblocks) && isempty(byblocks)) &&
        return DenseDesign(X)

    dense_cols = findall(!, taken)
    Xdense = X[:, dense_cols]
    icpt = 0
    for j in 1:p
        if all(==(1.0), view(X, :, j))
            icpt = j
            break
        end
    end
    return DiscreteDesign(blocks, Xdense, dense_cols, n, p, icpt, tblocks,
        byblocks)
end

"""
    bam_design_reduced(X_para, smooths, data, discrete, n, p) -> BamDesign

Build a discrete design **without an assembled `n x p` model matrix**.

`X` is `[X_para | sm_1.X | sm_2.X | ...]` block-wise, so every quantity the
design needs is available from the parametric block and the individual
smooths. Avoiding the assembly is the point: `setup_gam` materialises the full
matrix before the design is built, so it sat in peak RSS whether or not it was
retained, masking design-side wins of 53.66x (factor-`by`), 421x (tensor) and
127x (random effect).

Falls back to a dense design assembled the same way when no smooth discretises,
so the caller never has to special-case that.
"""
function bam_design_reduced(X_para::Matrix{Float64}, smooths, data, discrete,
    n::Int, p::Int)
    m_grid = _discrete_grid(discrete)
    t = Tables.columntable(data)
    blocks, tblocks, byblocks, taken =
        _discrete_blocks(smooths, t, m_grid, n, p)

    npar = size(X_para, 2)
    if isempty(blocks) && isempty(tblocks) && isempty(byblocks)
        # Nothing discretised: assemble the dense matrix after all. This costs
        # exactly what the non-discrete path costs and keeps the contract that
        # a design is always returned.
        return DenseDesign(_assemble_dense(X_para, smooths, n, p))
    end

    dense_cols = findall(!, taken)
    Xdense = _gather_columns(X_para, smooths, dense_cols, npar, n)
    # The intercept can only be a parametric column -- no smooth basis is
    # identically one -- so scanning `X_para` is both sufficient and cheap.
    icpt = 0
    for j in 1:npar
        if all(==(1.0), view(X_para, :, j))
            icpt = j
            break
        end
    end
    return DiscreteDesign(blocks, Xdense, dense_cols, n, p, icpt, tblocks,
        byblocks)
end

_discrete_grid(discrete) = begin
    m_grid = discrete === true ? 1000 : Int(discrete)
    m_grid >= 2 || throw(ArgumentError(
        "discrete grid resolution must be at least 2, got $m_grid"))
    m_grid
end

"""
    _assemble_dense(X_para, smooths, n, p) -> Matrix{Float64}

Assemble the `n x p` model matrix from its blocks, filling column-block by
column-block rather than `hcat`ing (which would hold a second full copy).
"""
function _assemble_dense(X_para::Matrix{Float64}, smooths, n::Int, p::Int)
    X = Matrix{Float64}(undef, n, p)
    npar = size(X_para, 2)
    npar > 0 && copyto!(view(X, :, 1:npar), X_para)
    for sm in smooths
        _scatter_block!(view(X, :, sm.first_para:sm.last_para), sm)
    end
    return X
end

"""
    _gather_columns(X_para, smooths, cols, npar, n) -> Matrix{Float64}

The `n x length(cols)` sub-matrix of the model matrix at absolute column
indices `cols`, read from the parametric block and the smooth bases directly.
Used for the dense remainder of a discrete design, which is the parametric
columns plus any smooth that did not discretise.
"""
function _gather_columns(X_para::Matrix{Float64}, smooths,
    cols::Vector{Int}, npar::Int, n::Int)
    out = Matrix{Float64}(undef, n, length(cols))
    for (c, j) in enumerate(cols)
        if j <= npar
            @inbounds copyto!(view(out, :, c), view(X_para, :, j))
            continue
        end
        placed = false
        for sm in smooths
            if sm.first_para <= j <= sm.last_para
                _scatter_column!(view(out, :, c), sm, j - sm.first_para + 1)
                placed = true
                break
            end
        end
        placed || throw(ArgumentError(
            "bam_design_reduced: model-matrix column $j belongs to no " *
            "parametric block or smooth; the design cannot be built without " *
            "an assembled X."))
    end
    return out
end

"""
    _scatter_column!(dest, sm, c) -> dest

Column `c` of the smooth's `n`-row block, written into `dest` without forming
the block. Honours a reduced basis via `sm.rowmap`.
"""
function _scatter_column!(dest::AbstractVector{Float64}, sm, c::Int)
    Xsm = sm.X
    if isempty(sm.rowmap)
        @inbounds copyto!(dest, view(Xsm, :, c))
    else
        rm = sm.rowmap
        @inbounds for i in eachindex(dest)
            dest[i] = Xsm[rm[i], c]
        end
    end
    return dest
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
        s = blk.scale
        if s === nothing
            @inbounds for i in 1:D.n
                eta[i] += work[k[i]]
            end
        else
            @inbounds for i in 1:D.n
                eta[i] += s[i] * work[k[i]]
            end
        end
    end
    for tb in D.tblocks
        # Lift the constrained coefficients back to the raw tensor basis, then
        # contract through the marginals -- X_cons*b = X_raw*(Z*b).
        mul!(tb.braw, tb.Z, view(beta, tb.cols))
        _tensor_eta!(eta, tb, tb.braw)
    end
    for bb in D.byblocks
        # One `m`-vector per level, then a single gather over the sample:
        # `L` small matvecs plus O(n), never an n x (kb*L) product.
        fb = bb.fbuf
        @inbounds for l in 1:bb.L
            mul!(view(fb, l, :), bb.Xd, view(beta, _by_cols(bb, l)))
        end
        k, lv = bb.k, bb.lev
        @inbounds for i in 1:D.n
            l = lv[i]
            l == 0 && continue          # unseen level contributes nothing
            eta[i] += fb[l, k[i]]
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
        s = blk.scale
        # Row `i` contributes `s_i * Xd[k_i, :]`, so the cell weight for X'WX
        # is `w_i * s_i^2` and the cell right-hand side is `w_i * s_i * z_i`.
        if z === nothing
            if s === nothing
                @inbounds for i in 1:D.n
                    wb[k[i]] += w[i]
                end
            else
                @inbounds for i in 1:D.n
                    si = s[i]
                    wb[k[i]] += w[i] * si * si
                end
            end
        else
            wzb = blk.wzb
            fill!(wzb, 0.0)
            if s === nothing
                @inbounds for i in 1:D.n
                    j = k[i]
                    wi = w[i]
                    wb[j] += wi
                    wzb[j] += wi * z[i]
                end
            else
                @inbounds for i in 1:D.n
                    j = k[i]
                    wi = w[i]
                    si = s[i]
                    wb[j] += wi * si * si
                    wzb[j] += wi * si * z[i]
                end
            end
        end
        sw = blk.sw
        @inbounds for j in 1:blk.m
            sw[j] = sqrt(max(wb[j], 0.0))
        end
    end
    # Factor-`by`: one sweep fills every level's row of the L x m tables, so
    # the binning cost is O(n) regardless of L.
    for bb in D.byblocks
        wb = bb.wb
        fill!(wb, 0.0)
        k, lv = bb.k, bb.lev
        if z === nothing
            @inbounds for i in 1:D.n
                l = lv[i]
                l == 0 && continue
                wb[l, k[i]] += w[i]
            end
        else
            wzb = bb.wzb
            fill!(wzb, 0.0)
            @inbounds for i in 1:D.n
                l = lv[i]
                l == 0 && continue
                j = k[i]
                wi = w[i]
                wb[l, j] += wi
                wzb[l, j] += wi * z[i]
            end
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

    # --- factor-by diagonal blocks, and their part of XᵀWz -----------------
    # Level sub-blocks are ORTHOGONAL (no row is in two levels), so only the
    # L diagonal blocks are formed. The off-diagonals are identically zero and
    # are never computed -- that is the whole reason this is not L separate
    # DiscreteBlocks, which would cost L(L-1)/2 zero cross-blocks.
    for bb in D.byblocks
        Xd = bb.Xd
        Xw = bb.Xw
        sw = bb.sw
        for l in 1:bb.L
            @inbounds for j in 1:bb.m
                sw[j] = sqrt(max(bb.wb[l, j], 0.0))
            end
            @inbounds for c in 1:bb.kb, r in 1:bb.m
                Xw[r, c] = Xd[r, c] * sw[r]
            end
            cl = _by_cols(bb, l)
            BLAS.syrk!('U', 'T', 1.0, Xw, 0.0, view(XtWX, cl, cl))
            Cb = view(XtWX, cl, cl)
            @inbounds for j in 1:bb.kb, i in (j + 1):bb.kb
                Cb[i, j] = Cb[j, i]
            end
            if XtWz !== nothing
                mul!(view(XtWz, cl), transpose(Xd), view(bb.wzb, l, :))
            end
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
                Wt[ka[i], kb[i]] += w[i] * _blk_scale(ba, i) * _blk_scale(bb, i)
            end
            Wt * bb.Xd
        else
            T = zeros(ba.m, pbc)
            Xb = bb.Xd
            @inbounds for i in 1:n
                ja = ka[i]
                jb = kb[i]
                wi = w[i] * _blk_scale(ba, i) * _blk_scale(bb, i)
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

    # --- factor-by cross blocks -------------------------------------------
    # Each cross-term keeps the one-pass shape: `L` tables of size
    # `m x p_other` are all filled in a single sweep over the sample, so the
    # cost is O(n * p_other) whatever `L` is.
    for (ai, bb) in enumerate(D.byblocks)
        Xa, ka, lva, kba = bb.Xd, bb.k, bb.lev, bb.kb

        # by x 1-D discrete
        for ob in D.blocks
            q = size(ob.Xd, 2)
            Ts = [zeros(bb.m, q) for _ in 1:bb.L]
            ko, Xo = ob.k, ob.Xd
            @inbounds for i in 1:n
                l = lva[i]
                l == 0 && continue
                wi = w[i] * _blk_scale(ob, i)
                Tl = Ts[l]
                j = ka[i]
                r = ko[i]
                for c in 1:q
                    Tl[j, c] += wi * Xo[r, c]
                end
            end
            for l in 1:bb.L
                Blk = transpose(Xa) * Ts[l]
                cl = _by_cols(bb, l)
                @inbounds for cj in 1:q, ci in 1:kba
                    XtWX[cl[ci], ob.cols[cj]] = Blk[ci, cj]
                    XtWX[ob.cols[cj], cl[ci]] = Blk[ci, cj]
                end
            end
        end

        # by x by. Distinct factor-`by` smooths generally overlap on every
        # level pair, so this needs L_a x L_b tables rather than L.
        for bj in (ai + 1):length(D.byblocks)
            ob = D.byblocks[bj]
            Xo, ko, lvo, kbo = ob.Xd, ob.k, ob.lev, ob.kb
            Ts = [zeros(bb.m, kbo) for _ in 1:(bb.L * ob.L)]
            @inbounds for i in 1:n
                la = lva[i]
                lb = lvo[i]
                (la == 0 || lb == 0) && continue
                Tl = Ts[(la - 1) * ob.L + lb]
                wi = w[i]
                j = ka[i]
                r = ko[i]
                for c in 1:kbo
                    Tl[j, c] += wi * Xo[r, c]
                end
            end
            for la in 1:bb.L, lb in 1:ob.L
                Blk = transpose(Xa) * Ts[(la - 1) * ob.L + lb]
                cl = _by_cols(bb, la)
                cr = _by_cols(ob, lb)
                @inbounds for cj in 1:kbo, ci in 1:kba
                    XtWX[cl[ci], cr[cj]] = Blk[ci, cj]
                    XtWX[cr[cj], cl[ci]] = Blk[ci, cj]
                end
            end
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
                wi = w[i] * _blk_scale(blk, i)
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

        # factor-by x dense remainder, same one-pass shape.
        for bb in D.byblocks
            Ts = [zeros(bb.m, pd) for _ in 1:bb.L]
            @inbounds for i in 1:n
                l = bb.lev[i]
                l == 0 && continue
                Tl = Ts[l]
                j = bb.k[i]
                wi = w[i]
                for c in 1:pd
                    Tl[j, c] += wi * Xden[i, c]
                end
            end
            for l in 1:bb.L
                Blk = transpose(bb.Xd) * Ts[l]
                cl = _by_cols(bb, l)
                @inbounds for cj in 1:pd, ci in 1:bb.kb
                    XtWX[cl[ci], D.dense_cols[cj]] = Blk[ci, cj]
                    XtWX[D.dense_cols[cj], cl[ci]] = Blk[ci, cj]
                end
            end
        end
    end

    # --- tensor blocks -----------------------------------------------------
    # Accumulate each block UNCONSTRAINED (p_raw x p_raw) and apply Z after, as
    # mgcv does at `discrete.c:2229-2266`. No Z ever enters the hot loop.
    for (ti, tb) in enumerate(D.tblocks)
        Wr = _tensor_diag!(tb.Wraw, tb, w)
        Bc = transpose(tb.Z) * Wr * tb.Z
        @inbounds for (jj, cj) in enumerate(tb.cols), (ii, ci) in enumerate(tb.cols)
            XtWX[ci, cj] = Bc[ii, jj]
        end
        if XtWz !== nothing
            vraw = zeros(tb.praw)
            wz = similar(w)
            @inbounds for i in 1:n
                wz[i] = w[i] * z[i]
            end
            _tensor_Xty!(vraw, tb, wz)
            mul!(view(XtWz, tb.cols), transpose(tb.Z), vraw)
        end

        # tensor x dense remainder
        if pd > 0
            Craw = zeros(tb.praw, pd)
            _tensor_cross!(Craw, tb, D.Xdense, w)
            Cc = transpose(tb.Z) * Craw
            @inbounds for cj in 1:pd, (ii, ci) in enumerate(tb.cols)
                XtWX[ci, D.dense_cols[cj]] = Cc[ii, cj]
                XtWX[D.dense_cols[cj], ci] = Cc[ii, cj]
            end
        end

        # tensor x 1-D blocks
        for ob in D.blocks
            q = size(ob.Xd, 2)
            Craw = zeros(tb.praw, q)
            _tensor_cross_block!(Craw, tb, ob, w)
            Cc = transpose(tb.Z) * Craw
            @inbounds for (jj, cj) in enumerate(ob.cols), (ii, ci) in enumerate(tb.cols)
                XtWX[ci, cj] = Cc[ii, jj]
                XtWX[cj, ci] = Cc[ii, jj]
            end
        end

        # tensor x factor-by
        for bb in D.byblocks
            Craw = zeros(tb.praw, bb.kb * bb.L)
            _tensor_cross_byblock!(Craw, tb, bb, w)
            Cc = transpose(tb.Z) * Craw
            @inbounds for (jj, cj) in enumerate(bb.cols), (ii, ci) in enumerate(tb.cols)
                XtWX[ci, cj] = Cc[ii, jj]
                XtWX[cj, ci] = Cc[ii, jj]
            end
        end

        # tensor x tensor
        for tj in (ti + 1):length(D.tblocks)
            ob = D.tblocks[tj]
            Craw = zeros(tb.praw, ob.praw)
            _tensor_cross_tensor!(Craw, tb, ob, w)
            Cc = transpose(tb.Z) * Craw * ob.Z
            @inbounds for (jj, cj) in enumerate(ob.cols), (ii, ci) in enumerate(tb.cols)
                XtWX[ci, cj] = Cc[ii, jj]
                XtWX[cj, ci] = Cc[ii, jj]
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
    start::Int, stop::Int,
    Rbuf::Matrix{Float64} = _tensor_row_scratch(D, stop - start + 1))
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
        s = blk.scale
        @inbounds for i in 1:nr
            row = start + i - 1
            r = k[row]
            si = s === nothing ? 1.0 : s[row]
            for (c, j) in enumerate(cols)
                H[i, j] = si * Xd[r, c]
            end
        end
    end
    for bb in D.byblocks
        Xd = bb.Xd
        k, lv, kb = bb.k, bb.lev, bb.kb
        cols = bb.cols
        # Every column of the block is written: a row is nonzero only in its
        # own level's `kb` columns, and zero everywhere else. Missing the zero
        # fill would leave whatever the caller's buffer held.
        @inbounds for i in 1:nr
            row = start + i - 1
            for j in cols
                H[i, j] = 0.0
            end
            l = lv[row]
            l == 0 && continue
            r = k[row]
            off = (l - 1) * kb
            for c in 1:kb
                H[i, cols[off + c]] = Xd[r, c]
            end
        end
    end
    for tb in D.tblocks
        _tensor_rows!(H, tb, start, stop, Rbuf)
    end
    return H
end

# Scratch for the unconstrained row tensor, sized for the widest tensor block
# so one buffer serves them all. Zero-sized when there are no tensor blocks.
function _tensor_row_scratch(D::DiscreteDesign, nr::Int)
    isempty(D.tblocks) && return Matrix{Float64}(undef, 0, 0)
    return Matrix{Float64}(undef, nr, maximum(tb.praw for tb in D.tblocks))
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
    Rbuf = _tensor_row_scratch(D, cs)

    for start in 1:cs:n
        stop = min(start + cs - 1, n)
        nr = stop - start + 1
        Hv = view(H, 1:nr, :)
        fill!(Hv, 0.0)
        _gather_rows!(Hv, D, start, stop, Rbuf)
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

# ─── Tensor blocks (M2) ──────────────────────────────────────────────────
#
# A `te` smooth is stored as its per-marginal bases at the unique values of
# each covariate, plus one index vector per marginal. The row tensor is never
# formed: row `i` of the unconstrained block is
# `kron(Xd_1[k_1[i],:], ..., Xd_d[k_d[i],:])`, and every kernel works from the
# marginals directly.
#
# This is sound because `te` in GAM.jl applies a SINGLE overall sum-to-zero
# constraint to the raw row tensor (`basis_tensor.jl:446`) rather than
# reparameterising each marginal. Verified: `X_cons == X_raw * Z` exactly (max
# |Δ| = 0.0) for 2-D and 3-D `te`, with `marginal_Zs` empty. So the constraint
# can be applied AFTER accumulation, as mgcv does at `discrete.c:2229-2266`:
# accumulate the unconstrained `p_raw × p_raw` block, then form `Zᵀ · W · Z`.
#
# Note this is why the plan's falsification check does not fire. The identity
# `(A·Ra) ⊙ (B·Rb) == (A ⊙ B)·(Ra ⊗ Rb)` is an unconditional algebraic fact
# (verified to 1.8e-15), but `te` needs no per-marginal reparameterisation at
# all -- mgcv's `np=TRUE` is deliberately not applied here
# (`basis_tensor.jl:295-299`). It is `t2` that folds per-marginal reparams via
# that identity, and `t2` is not discretised.


"""
    _tensor_dcol!(out, blk, cidx)

mgcv's `tensorXj` (`discrete.c:301-327`): one column of the row tensor of all
marginals but the last, in `O(n·d)` and without forming `dX`.

`_row_kronecker` varies the LAST marginal fastest, so `cidx` decodes with
marginal `d-1` fastest among the remaining ones.
"""
function _tensor_dcol!(out::Vector{Float64}, blk::TensorBlock, cidx::Int)
    d = ndim(blk)
    rem = cidx - 1
    fill!(out, 1.0)
    @inbounds for j in (d - 1):-1:1
        pj = blk.pdims[j]
        aj = (rem % pj) + 1
        rem = rem ÷ pj
        Xd = blk.Xds[j]
        k = blk.ks[j]
        col = view(Xd, :, aj)
        for i in eachindex(out)
            out[i] *= col[k[i]]
        end
    end
    return out
end

# η += X_raw * b_raw, mgcv's `tensorXb` (`discrete.c:396-444`).
# C = Xd_d * reshape(b_raw, p_d, pb) is (m_d × pb); then one O(n) pass per
# column of dX. O(n·pb·d + m_d·p_d·pb) against the dense O(n·p_raw).
function _tensor_eta!(eta::Vector{Float64}, blk::TensorBlock,
    braw::Vector{Float64})
    pd = blk.pdims[end]
    C = blk.Xds[end] * reshape(braw, pd, blk.pb)
    kd = blk.ks[end]
    dcol = blk.dcol_a
    @inbounds for cidx in 1:blk.pb
        _tensor_dcol!(dcol, blk, cidx)
        Ccol = view(C, :, cidx)
        for i in eachindex(eta)
            eta[i] += dcol[i] * Ccol[kd[i]]
        end
    end
    return eta
end

# out_raw = X_rawᵀ v, mgcv's `tensorXty` (`discrete.c:346-373`).
function _tensor_Xty!(out::Vector{Float64}, blk::TensorBlock,
    v::Vector{Float64})
    pd = blk.pdims[end]
    Xdd = blk.Xds[end]
    kd = blk.ks[end]
    dcol = blk.dcol_a
    tv = blk.tvec
    @inbounds for cidx in 1:blk.pb
        _tensor_dcol!(dcol, blk, cidx)
        fill!(tv, 0.0)
        for i in eachindex(v)
            tv[kd[i]] += v[i] * dcol[i]
        end
        mul!(view(out, ((cidx - 1) * pd + 1):(cidx * pd)), transpose(Xdd), tv)
    end
    return out
end

# Unconstrained X_rawᵀ W X_raw, via mgcv's diagonal shortcut
# (`discrete.c:1742-1792`): for each (cidx, cidx') pair accumulate
# wb[l] = Σ_{i: k_d[i]=l} w_i·dX[i,cidx]·dX[i,cidx'] in one O(n) pass, then the
# sub-block is Xd_dᵀ diag(wb) Xd_d.
function _tensor_diag!(W::Matrix{Float64}, blk::TensorBlock, w::Vector{Float64})
    fill!(W, 0.0)
    pd = blk.pdims[end]
    Xdd = blk.Xds[end]
    kd = blk.ks[end]
    da, db = blk.dcol_a, blk.dcol_b
    tv = blk.tvec
    n = length(w)
    @inbounds for ca in 1:blk.pb
        _tensor_dcol!(da, blk, ca)
        for cb in ca:blk.pb
            if cb == ca
                copyto!(db, da)
            else
                _tensor_dcol!(db, blk, cb)
            end
            fill!(tv, 0.0)
            for i in 1:n
                tv[kd[i]] += w[i] * da[i] * db[i]
            end
            sub = transpose(Xdd) * (tv .* Xdd)
            ra = ((ca - 1) * pd + 1):(ca * pd)
            rb = ((cb - 1) * pd + 1):(cb * pd)
            for (jj, cj) in enumerate(rb), (ii, ci) in enumerate(ra)
                W[ci, cj] = sub[ii, jj]
                ca == cb || (W[cj, ci] = sub[ii, jj])
            end
        end
    end
    return W
end

# Unconstrained X_rawᵀ W M for an arbitrary n-row `M` (the dense remainder, or
# another block's rows). O(pb·n·q).
function _tensor_cross!(out::Matrix{Float64}, blk::TensorBlock,
    M::AbstractMatrix{Float64}, w::Vector{Float64})
    pd = blk.pdims[end]
    q = size(M, 2)
    Xdd = blk.Xds[end]
    kd = blk.ks[end]
    dcol = blk.dcol_a
    md = size(Xdd, 1)
    T = zeros(md, q)
    n = length(w)
    @inbounds for cidx in 1:blk.pb
        _tensor_dcol!(dcol, blk, cidx)
        fill!(T, 0.0)
        for i in 1:n
            l = kd[i]
            s = w[i] * dcol[i]
            for c in 1:q
                T[l, c] += s * M[i, c]
            end
        end
        mul!(view(out, ((cidx - 1) * pd + 1):(cidx * pd), :), transpose(Xdd), T)
    end
    return out
end


# X_a_rawᵀ W X_b for a 1-D discrete block `b`, gathering `b`'s rows on the fly.
function _tensor_cross_block!(out::Matrix{Float64}, blk::TensorBlock,
    other::DiscreteBlock, w::Vector{Float64})
    pd = blk.pdims[end]
    Xdd = blk.Xds[end]
    kd = blk.ks[end]
    Xo, ko = other.Xd, other.k
    q = size(Xo, 2)
    md = size(Xdd, 1)
    T = zeros(md, q)
    dcol = blk.dcol_a
    n = length(w)
    @inbounds for cidx in 1:blk.pb
        _tensor_dcol!(dcol, blk, cidx)
        fill!(T, 0.0)
        for i in 1:n
            l = kd[i]
            s = w[i] * dcol[i]
            r = ko[i]
            for c in 1:q
                T[l, c] += s * Xo[r, c]
            end
        end
        mul!(view(out, ((cidx - 1) * pd + 1):(cidx * pd), :), transpose(Xdd), T)
    end
    return out
end

# X_rawᵀ W X_by for a tensor block against a factor-`by` block. Same shape as
# `_tensor_cross_block!`, except the other side's row `i` lands in its own
# level's `kb` columns rather than in a single column range.
function _tensor_cross_byblock!(out::Matrix{Float64}, blk::TensorBlock,
    other::ByBlock, w::Vector{Float64})
    pd = blk.pdims[end]
    Xdd = blk.Xds[end]
    kd = blk.ks[end]
    Xo, ko, lvo, kbo = other.Xd, other.k, other.lev, other.kb
    md = size(Xdd, 1)
    T = zeros(md, kbo * other.L)
    dcol = blk.dcol_a
    n = length(w)
    @inbounds for cidx in 1:blk.pb
        _tensor_dcol!(dcol, blk, cidx)
        fill!(T, 0.0)
        for i in 1:n
            l = lvo[i]
            l == 0 && continue
            row = kd[i]
            s = w[i] * dcol[i]
            r = ko[i]
            off = (l - 1) * kbo
            for c in 1:kbo
                T[row, off + c] += s * Xo[r, c]
            end
        end
        mul!(view(out, ((cidx - 1) * pd + 1):(cidx * pd), :), transpose(Xdd), T)
    end
    return out
end

# X_a_rawᵀ W X_b_raw for two tensor blocks, via mgcv's cross-weight table
# (`discrete.c:1801`) on the last marginals of each.
function _tensor_cross_tensor!(out::Matrix{Float64}, a::TensorBlock,
    b::TensorBlock, w::Vector{Float64})
    pda, pdb = a.pdims[end], b.pdims[end]
    Xa, Xb = a.Xds[end], b.Xds[end]
    ka, kb = a.ks[end], b.ks[end]
    ma, mb = size(Xa, 1), size(Xb, 1)
    Wt = zeros(ma, mb)
    da, db = a.dcol_a, b.dcol_a
    n = length(w)
    @inbounds for ca in 1:a.pb
        _tensor_dcol!(da, a, ca)
        for cb in 1:b.pb
            _tensor_dcol!(db, b, cb)
            fill!(Wt, 0.0)
            for i in 1:n
                Wt[ka[i], kb[i]] += w[i] * da[i] * db[i]
            end
            sub = transpose(Xa) * (Wt * Xb)
            ra = ((ca - 1) * pda + 1):(ca * pda)
            rb = ((cb - 1) * pdb + 1):(cb * pdb)
            for (jj, cj) in enumerate(rb), (ii, ci) in enumerate(ra)
                out[ci, cj] = sub[ii, jj]
            end
        end
    end
    return out
end

# Materialise rows start:stop of the CONSTRAINED block, for the leverage sweep.
#
# `Rbuf` is caller-owned scratch holding the UNCONSTRAINED row tensor for this
# chunk; `design_finalize` allocates it once for the whole sweep. It used to be
# a fresh `zeros(nr, praw)` per chunk, which at n = 2e5 was 196 chunks x 1.8 MB
# = 349 MiB of garbage for one call.
#
# The constraint is then applied by `mul!`, i.e. one BLAS gemm. It used to be a
# hand-rolled scalar triple loop over (cj, i, c), which is the same arithmetic
# but ~1.0e10 unvectorised flops at that n and dominated the entire fit: the
# discrete leverage sweep measured 14.13 s against the dense path's 0.17 s,
# while the accumulation kernels this milestone was benchmarked on were already
# 2.9x FASTER than dense. Summation order differs from the old loop, so
# `hat_diag` (and the `leverage`/`cooksdistance` built from it) can move in the
# last bits; it is diagnostics-only and feeds no fitted quantity -- `edf_vec`
# comes from `diag(F)`, not from here.
function _tensor_rows!(H::AbstractMatrix{Float64}, blk::TensorBlock,
    start::Int, stop::Int, Rbuf::Matrix{Float64})
    nr = stop - start + 1
    d = ndim(blk)
    R = view(Rbuf, 1:nr, 1:blk.praw)
    fill!(R, 0.0)
    @inbounds for i in 1:nr
        row = start + i - 1
        R[i, 1] = 1.0
        width = 1
        for j in 1:d
            Xd = blk.Xds[j]
            r = blk.ks[j][row]
            pj = blk.pdims[j]
            for c in width:-1:1
                v = R[i, c]
                for a in pj:-1:1
                    R[i, (c - 1) * pj + a] = v * Xd[r, a]
                end
            end
            width *= pj
        end
    end
    mul!(view(H, 1:nr, blk.cols), R, blk.Z)
    return H
end
