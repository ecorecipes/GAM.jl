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
