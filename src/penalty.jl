# Penalty setup and reparameterization
#
# Builds the block-diagonal penalty structure from a list of smooth terms.
# Equivalent to mgcv's Sl.setup and gam.reparam.

"""
    setup_penalties(smooths::Vector{<:ConstructedSmooth}, n_parametric::Int)

Build the [`PenaltySetup`](@ref) for a list of constructed smooth terms.
Assigns parameter indices and creates block structure.

# Arguments
- `smooths`: constructed smooth terms
- `n_parametric`: number of parametric coefficients (including intercept)
"""
function setup_penalties(smooths::Vector{<:ConstructedSmooth}, n_parametric::Int)
    blocks = PenaltyBlock[]
    sp_all = Float64[]
    p_start = n_parametric + 1  # smooth params start after parametric

    for sm in smooths
        k = size(sm.X, 2)
        sm.first_para = p_start
        sm.last_para = p_start + k - 1

        # Square root penalties
        rS = Matrix{Float64}[]
        for Si in sm.S
            # Compute matrix square root via eigen decomposition
            eig = eigen(Symmetric(Si))
            pos = eig.values .> eps() * maximum(abs.(eig.values))
            if any(pos)
                rSi = eig.vectors[:, pos] * Diagonal(sqrt.(eig.values[pos]))
                push!(rS, rSi)
            else
                push!(rS, zeros(k, 0))
            end
        end

        rank = sm.rank
        block = PenaltyBlock(sm.S, rS, rank, p_start, p_start + k - 1, true)
        push!(blocks, block)

        # Initial smoothing parameters (log scale) — one per penalty matrix
        for _ in sm.S
            push!(sp_all, 0.0)  # log(1.0) = 0, will be optimized
        end

        p_start += k
    end

    # Total penalty square root for rank detection
    p_total = p_start - 1
    E = zeros(p_total, p_total)
    for block in blocks
        idx = block.start:block.stop
        for (i, Si) in enumerate(block.S)
            E[idx, idx] .+= Si
        end
    end

    return PenaltySetup(blocks, sp_all, E)
end

"""
    _initial_sp(X, penalty)

Compute initial smoothing parameters using mgcv's heuristic (R's initial.sp):
adjust sp so that mean(diag(X'X) / (diag(X'X) + diag(Σ sp·S))) ≈ 0.4.
"""
function _initial_sp(X::Matrix{Float64}, penalty::PenaltySetup)
    n_sp = length(penalty.sp)
    if n_sp == 0
        return
    end
    p = size(X, 2)
    ldxx = vec(sum(X .^ 2; dims = 1))  # diag(X'X)

    sp_init = zeros(n_sp)
    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            ss = diag(Si)
            xx = ldxx[idx]
            # Use only truly penalized columns
            thresh = eps()^0.8 * maximum(abs.(Si))
            rs = vec(sum(abs.(Si); dims = 2))
            cs = vec(sum(abs.(Si); dims = 1))
            ds = abs.(diag(Si))
            ind = (rs .> thresh) .& (cs .> thresh) .& (ds .> thresh)
            if any(ind)
                sizeXX = mean(xx[ind])
                sizeS = mean(ss[ind])
                if sizeS > 0
                    sp_init[sp_idx] = sizeXX / sizeS
                else
                    sp_init[sp_idx] = 1.0
                end
            else
                sp_init[sp_idx] = 1.0
            end
            sp_idx += 1
        end
    end

    # Adjust so mean EDF ratio ≈ 0.4
    ldss = zeros(p)
    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            ldss[idx] .+= sp_init[sp_idx] .* diag(Si)
            sp_idx += 1
        end
    end

    pen = ldss .> 0
    if any(pen .& (ldxx .> 0))
        xx_pen = ldxx[pen .& (ldxx .> 0)]
        ss_pen = ldss[pen .& (ldxx .> 0)]
        while mean(xx_pen ./ (xx_pen .+ ss_pen)) > 0.4
            sp_init .*= 10
            ss_pen .*= 10
        end
        while mean(xx_pen ./ (xx_pen .+ ss_pen)) < 0.4
            sp_init ./= 10
            ss_pen ./= 10
        end
    end

    # Store as log(sp)
    for i in 1:n_sp
        penalty.sp[i] = log(max(sp_init[i], 1e-15))
    end
end

"""
    total_penalty(penalty::PenaltySetup, log_sp, p::Int) -> Matrix

Compute the total penalty matrix Σ λ_j S_j given log smoothing parameters.
Accepts any numeric vector for `log_sp` (including ForwardDiff Dual types).
"""
function total_penalty(penalty::PenaltySetup, log_sp::AbstractVector, p::Int)
    T = promote_type(Float64, eltype(log_sp))
    S_total = zeros(T, p, p)
    sp_idx = 1

    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            λ = exp(log_sp[sp_idx])
            @inbounds for j in eachindex(idx), k in eachindex(idx)
                S_total[idx[j], idx[k]] += λ * Si[j, k]
            end
            sp_idx += 1
        end
    end
    return S_total
end

"""
    total_penalty!(S_total, penalty, log_sp, p) -> S_total

In-place version of [`total_penalty`](@ref) for Float64 smoothing parameters.
Zeroes `S_total` and accumulates Σ λ_j S_j into it, avoiding allocation.
"""
function total_penalty!(S_total::Matrix{Float64}, penalty::PenaltySetup,
    log_sp::Vector{Float64}, p::Int)
    fill!(S_total, 0.0)
    sp_idx = 1

    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            λ = exp(log_sp[sp_idx])
            @inbounds for j in eachindex(idx), k in eachindex(idx)
                S_total[idx[j], idx[k]] += λ * Si[j, k]
            end
            sp_idx += 1
        end
    end
    return S_total
end

"""
    penalty_edf(X, W, S_total; XtWX=nothing, A_chol=nothing) -> (edf_vec, hat_diag)

Compute effective degrees of freedom and hat matrix diagonal.
edf = trace(F) where F = (X'WX + S)^{-1} X'WX is the hat-like matrix.

Optionally accepts pre-computed `XtWX` and Cholesky of A=X'WX+S to avoid
redundant computation.
"""
function penalty_edf(X::Matrix{Float64}, W::Vector{Float64},
    S_total::Matrix{Float64};
    XtWX::Union{Matrix{Float64}, Nothing} = nothing,
    A_chol::Union{Cholesky, Nothing} = nothing)
    p = size(X, 2)
    n = size(X, 1)

    if XtWX === nothing
        Xw_tmp = similar(X)
        @inbounds for i in 1:n
            sw = sqrt(W[i])
            for j in 1:p
                Xw_tmp[i, j] = X[i, j] * sw
            end
        end
        XtWX_local = zeros(p, p)
        BLAS.syrk!('U', 'T', 1.0, Xw_tmp, 0.0, XtWX_local)
        @inbounds for j in 1:p
            for k in (j + 1):p
                XtWX_local[k, j] = XtWX_local[j, k]
            end
        end
    else
        XtWX_local = XtWX
    end

    if A_chol === nothing
        A = XtWX_local + S_total
        A_chol_local = cholesky(Symmetric(A))
    else
        A_chol_local = A_chol
    end

    # F = A^{-1} * X'WX — the influence/hat matrix in coef space
    F = A_chol_local \ XtWX_local

    # EDF = trace(F) per parameter, hat diagonal = diag(X * F * X^{-})
    edf_vec = diag(F)
    hat_diag = vec(sum((X / A_chol_local.U) .^ 2; dims = 2))

    return edf_vec, hat_diag
end

"""
    smooth_edf(edf_vec, smooths) -> Vector{Float64}

Sum per-parameter EDF into per-smooth EDF.
"""
function smooth_edf(edf_vec::Vector{Float64}, smooths::Vector{<:ConstructedSmooth})
    edf_smooth = Float64[]
    for sm in smooths
        idx = sm.first_para:sm.last_para
        push!(edf_smooth, sum(edf_vec[idx]))
    end
    return edf_smooth
end
