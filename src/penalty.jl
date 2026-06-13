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
function setup_penalties(smooths::Vector{<:ConstructedSmooth}, n_parametric::Int;
    select::Bool = false)
    blocks = PenaltyBlock[]
    sp_all = Float64[]
    p_start = n_parametric + 1  # smooth params start after parametric

    for sm in smooths
        k = size(sm.X, 2)
        sm.first_para = p_start
        sm.last_para = p_start + k - 1

        S_list = copy(sm.S)
        block_rank = sm.rank

        # select=TRUE (Marra & Wood 2011): append a penalty on the null space
        # of the existing penalties so the unpenalized part of each smooth
        # (the linear/polynomial null space) can be shrunk to zero, giving
        # automatic term selection. Skip if the smooth is already full-rank
        # penalized (e.g. random effects, or ts/cs which carry their own
        # shrinkage penalty).
        if select
            S_null = _null_space_penalty(S_list, k)
            if S_null !== nothing
                push!(S_list, S_null)
                block_rank = k  # combined penalty is now full rank
            end
        end

        # Square root penalties
        rS = Matrix{Float64}[]
        for Si in S_list
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

        block = PenaltyBlock(S_list, rS, block_rank, p_start, p_start + k - 1, true)
        push!(blocks, block)

        # Initial smoothing parameters (log scale) — one per penalty matrix
        for _ in S_list
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
    _null_space_penalty(S_list, k) -> Union{Matrix{Float64}, Nothing}

Build the null-space penalty for `select=TRUE` (Marra & Wood 2011). Given a
smooth's existing penalty matrices, find the null space of their sum (the
directions left unpenalized — typically the linear/polynomial part of the
smooth) and return the projection onto that null space, scaled to the same
order of magnitude as the existing penalty so the two smoothing parameters
are comparable. Returns `nothing` if there is no null space to penalize.
"""
function _null_space_penalty(S_list::Vector{Matrix{Float64}}, k::Int)
    isempty(S_list) && return nothing
    S_sum = zeros(k, k)
    for Si in S_list
        S_sum .+= Si
    end
    eg = eigen(Symmetric(S_sum))
    maxev = maximum(abs, eg.values)
    maxev <= 0 && return nothing
    tol = maxev * sqrt(eps())
    null_idx = findall(<=(tol), eg.values)
    isempty(null_idx) && return nothing
    length(null_idx) == k && return nothing  # nothing is penalized — skip

    U0 = eg.vectors[:, null_idx]
    S_null = U0 * U0'                       # rank = length(null_idx), eigvals in {0,1}
    # Scale to the mean nonzero eigenvalue of the existing penalty so the
    # initial smoothing parameters are on a comparable footing.
    pos = eg.values[eg.values .> tol]
    scale = isempty(pos) ? 1.0 : sum(pos) / length(pos)
    S_null .*= scale
    return Symmetric(S_null) |> Matrix
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

    # EDF = trace(F) per parameter; leverage h_i = w_i * x_i' A^{-1} x_i,
    # so that sum(hat_diag) == total EDF.
    edf_vec = diag(F)
    hat_diag = W .* vec(sum((X / A_chol_local.U) .^ 2; dims = 2))

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
