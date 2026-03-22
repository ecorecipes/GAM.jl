# GAMLSS RS (Rigby-Stasinopoulos) and CG (Cole-Green) solvers
#
# Implements the backfitting-style algorithms from R's gamlss package.
# RS: cyclic updates of each parameter using penalized IRLS in η-space.
# CG: like RS but with cross-derivative adjustments for faster convergence.
#
# Both solvers work in η-space (linear predictor), using diagonal Hessian
# entries as working weights and Newton-direction working responses.

using LinearAlgebra

# ============================================================================
# Helper: global deviance from η
# ============================================================================

"""Compute global deviance = 2 * NLL from linear predictors."""
function _global_deviance(family::MultiParameterFamily, y::AbstractVector,
                          η_list::Vector{<:AbstractVector})
    return 2.0 * nll_total(family, y, η_list)
end

# ============================================================================
# Helper: per-parameter η-space derivatives
# ============================================================================

"""
    _param_derivs_eta(family, y, η_list, K, derivs) → derivs

Fill the full derivative matrix (gradient + Hessian in η-space).
Returns the same `derivs` matrix for reuse.
"""
function _param_derivs_eta!(family::MultiParameterFamily, y::AbstractVector,
                            η_list::Vector{<:AbstractVector},
                            derivs::Matrix{Float64})
    nll_derivs!(family, derivs, y, η_list)
    return derivs
end

# ============================================================================
# Helper: per-parameter penalized WLS solve
# ============================================================================

"""
    _penalized_wls(Xk, wk, zk, Sk) → β_new

Solve penalized weighted least squares: (X'WX + S)β = X'Wz
where W = diag(wk).
"""
function _penalized_wls(Xk::Matrix{Float64}, wk::Vector{Float64},
                        zk::Vector{Float64}, Sk::Matrix{Float64})
    pk = size(Xk, 2)

    # X'WX: form (sqrt(w) .* X)' * (sqrt(w) .* X) + S
    XtWX = Matrix{Float64}(undef, pk, pk)
    XtWz = Vector{Float64}(undef, pk)

    # Compute X' * diag(w) * X
    @inbounds for j in 1:pk
        for i in 1:j
            s = 0.0
            for row in 1:size(Xk, 1)
                s += Xk[row, i] * wk[row] * Xk[row, j]
            end
            XtWX[i, j] = s
            XtWX[j, i] = s
        end
    end
    XtWX .+= Sk

    # X' * (w .* z)
    @inbounds for j in 1:pk
        s = 0.0
        for row in 1:size(Xk, 1)
            s += Xk[row, j] * wk[row] * zk[row]
        end
        XtWz[j] = s
    end

    # Solve
    F = _safe_cholesky(Symmetric(XtWX))
    if F !== nothing
        return F \ XtWz
    else
        # Fallback with diagonal perturbation
        for i in 1:pk
            XtWX[i, i] += 1e-6
        end
        return Symmetric(XtWX) \ XtWz
    end
end

"""
    _penalized_wls_with_edf(Xk, wk, zk, Sk) → (β_new, edf, rss)

Like `_penalized_wls` but also returns effective degrees of freedom
and weighted RSS = Σw(z - Xβ)².
Uses SVD of augmented system [√W·X; √λ·D] like R gamlss pb().
"""
function _penalized_wls_with_edf(Xk::Matrix{Float64}, wk::Vector{Float64},
                                  zk::Vector{Float64}, Sk::Matrix{Float64})
    n = size(Xk, 1)
    pk = size(Xk, 2)

    # QR of √W·X
    sqw = sqrt.(wk)
    WX = sqw .* Xk
    Wz = sqw .* zk
    qrWX = qr(WX)
    R = Matrix(qrWX.R)
    Qy_full = qrWX.Q' * Wz
    Qy = Qy_full[1:pk]  # only first pk elements (thin Q projection)

    # Augmented system: [R; √S] via SVD
    # Need square-root of Sk: use Cholesky or eigendecomposition
    eig = eigen(Symmetric(Sk))
    pos = eig.values .> 1e-10 * max(maximum(abs, eig.values), 1e-20)
    if any(pos)
        sqrtS = eig.vectors[:, pos] * Diagonal(sqrt.(eig.values[pos]))
        RD = vcat(R, Matrix(sqrtS'))
    else
        RD = R
    end

    svdRD = svd(RD)
    rank = count(s -> s > max(maximum(svdRD.S), 1e-20) * sqrt(eps()), svdRD.S)
    U1 = svdRD.U[1:pk, 1:rank]
    y1 = U1' * Qy
    β_new = svdRD.V[:, 1:rank] * (y1 ./ svdRD.S[1:rank])

    # edf = tr(H) where H is the hat matrix in the augmented system
    HH = U1 * U1'
    edf = tr(HH)

    # Weighted RSS
    fv = Xk * β_new
    rss = 0.0
    @inbounds for i in 1:n
        r = zk[i] - fv[i]
        rss += wk[i] * r * r
    end

    return β_new, edf, rss
end

# ============================================================================
# Build per-parameter penalty matrices
# ============================================================================

"""
Build per-parameter penalty matrices from the global penalty list Sl and log_sp.
Returns S_k for each parameter k, sized p_k × p_k.
"""
function _per_param_penalties(Sl::Vector{Matrix{Float64}}, log_sp::Vector{Float64},
                              param_offsets::Vector{Int}, K::Int)
    S_list = Matrix{Float64}[]
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        pk = e - s + 1
        Sk = zeros(pk, pk)
        for (j, Sj) in enumerate(Sl)
            Sj_block = @view Sj[s:e, s:e]
            if any(!iszero, Sj_block)
                Sk .+= exp(log_sp[j]) .* Sj_block
            end
        end
        push!(S_list, Sk)
    end
    return S_list
end

# ============================================================================
# Per-parameter EFS smoothing parameter update
# ============================================================================

"""
Update smoothing parameters for a single parameter k using EFS,
after a penalized WLS solve.
"""
function _efs_update_param!(log_sp::Vector{Float64}, Sl::Vector{Matrix{Float64}},
                            β_k::Vector{Float64}, Xk::Matrix{Float64},
                            wk::Vector{Float64}, Sk::Matrix{Float64},
                            param_offsets::Vector{Int}, k::Int)
    s = param_offsets[k] + 1
    e = param_offsets[k + 1]
    pk = e - s + 1

    # Compute penalized information: XtWX + Sk
    XtWX_Sk = Matrix{Float64}(undef, pk, pk)
    @inbounds for j in 1:pk
        for i in 1:j
            val = 0.0
            for row in 1:size(Xk, 1)
                val += Xk[row, i] * wk[row] * Xk[row, j]
            end
            XtWX_Sk[i, j] = val + Sk[i, j]
            XtWX_Sk[j, i] = val + Sk[j, i]
        end
    end

    F = _safe_cholesky(Symmetric(XtWX_Sk))
    if F === nothing
        return
    end
    Ainv = inv(F)

    for (j, Sj) in enumerate(Sl)
        Sj_block = @view Sj[s:e, s:e]
        if !any(!iszero, Sj_block)
            continue
        end
        λ = exp(log_sp[j])
        Sj_local = Matrix(Sj_block)

        eigs = eigvals(Symmetric(Sj_local))
        rank_j = Float64(count(e -> e > 1e-10 * maximum(abs, eigs), eigs))

        bSb = dot(β_k, Sj_local * β_k)
        trAS = tr(Ainv * Sj_local)

        a = max(0.0, rank_j / λ - trAS)
        if a > 0 && bSb > eps()
            r = a / bSb
            log_sp[j] = clamp(log_sp[j] + log(max(r, 1e-15)), -15.0, 15.0)
        end
    end
end

# ============================================================================
# Local ML/GAIC/GCV smoothing parameter selection
# ============================================================================

"""
    _local_ml_update_param!(log_sp, Sl, β_k, Xk, wk, zk, Sk, param_offsets, k, order)

Local ML smoothing parameter update (R gamlss pb() style):
  σ² = RSS / (n - edf)
  τ² = β'D'Dβ / (edf - order)
  λ_new = σ² / τ²
"""
function _local_ml_update_param!(log_sp::Vector{Float64}, Sl::Vector{Matrix{Float64}},
                                  β_k::Vector{Float64}, Xk::Matrix{Float64},
                                  wk::Vector{Float64}, zk::Vector{Float64},
                                  param_offsets::Vector{Int}, k::Int;
                                  max_iter::Int=50, tol::Float64=1e-7)
    s = param_offsets[k] + 1
    e = param_offsets[k + 1]
    pk = e - s + 1
    n = length(wk)

    # Find smooth penalty indices for this parameter
    sp_indices = Int[]
    for (j, Sj) in enumerate(Sl)
        Sj_block = @view Sj[s:e, s:e]
        if any(!iszero, Sj_block)
            push!(sp_indices, j)
        end
    end

    isempty(sp_indices) && return

    for idx in sp_indices
        old_log_sp = log_sp[idx]

        for _ in 1:max_iter
            # Build current penalty for this parameter
            Sk = zeros(pk, pk)
            for (j, Sj) in enumerate(Sl)
                Sj_block = @view Sj[s:e, s:e]
                if any(!iszero, Sj_block)
                    Sk .+= exp(log_sp[j]) .* Sj_block
                end
            end

            # Solve with edf
            β_new, edf, rss = _penalized_wls_with_edf(Xk, wk, zk, Sk)

            # Penalty-specific D'D
            Sj_block = Matrix(@view Sl[idx][s:e, s:e])
            bDb = dot(β_new, Sj_block * β_new)

            # Estimate penalty order from null space
            eigs = eigvals(Symmetric(Sj_block))
            order = count(ev -> ev < 1e-10 * maximum(abs, eigs), eigs)

            # ML update: λ = σ²/τ²
            sigma2 = rss / max(n - edf, 1.0)
            tau2 = bDb / max(edf - order, 0.01)

            if tau2 > eps() && sigma2 > eps()
                new_log_sp = clamp(log(sigma2 / tau2), -15.0, 15.0)
                if abs(new_log_sp - log_sp[idx]) < tol
                    log_sp[idx] = new_log_sp
                    break
                end
                log_sp[idx] = new_log_sp
            else
                break
            end
        end
    end
end

"""
    _local_gaic_update_param!(log_sp, Sl, Xk, wk, zk, param_offsets, k, gaic_k)

Local GAIC smoothing parameter update: minimize RSS + gaic_k * edf
over λ using golden section search (R gamlss pb() style).
"""
function _local_gaic_update_param!(log_sp::Vector{Float64}, Sl::Vector{Matrix{Float64}},
                                    Xk::Matrix{Float64}, wk::Vector{Float64},
                                    zk::Vector{Float64},
                                    param_offsets::Vector{Int}, k::Int,
                                    gaic_k::Float64)
    s = param_offsets[k] + 1
    e = param_offsets[k + 1]
    pk = e - s + 1

    sp_indices = Int[]
    for (j, Sj) in enumerate(Sl)
        Sj_block = @view Sj[s:e, s:e]
        if any(!iszero, Sj_block)
            push!(sp_indices, j)
        end
    end

    isempty(sp_indices) && return

    for idx in sp_indices
        # GAIC objective as a function of log_lambda
        function gaic_obj(log_lam)
            Sk = zeros(pk, pk)
            for (j, Sj) in enumerate(Sl)
                Sj_block = @view Sj[s:e, s:e]
                if any(!iszero, Sj_block)
                    lsp_j = (j == idx) ? log_lam : log_sp[j]
                    Sk .+= exp(lsp_j) .* Sj_block
                end
            end
            _, edf, rss = _penalized_wls_with_edf(Xk, wk, zk, Sk)
            return rss + gaic_k * edf
        end

        # Golden section search on [-7, 7] (log scale)
        log_sp[idx] = _golden_section_min(gaic_obj, -7.0, 7.0, 1e-4)
    end
end

"""
    _local_gcv_update_param!(log_sp, Sl, Xk, wk, zk, param_offsets, k, gaic_k)

Local GCV: minimize n*RSS/(n - gaic_k*edf)^2.
"""
function _local_gcv_update_param!(log_sp::Vector{Float64}, Sl::Vector{Matrix{Float64}},
                                   Xk::Matrix{Float64}, wk::Vector{Float64},
                                   zk::Vector{Float64},
                                   param_offsets::Vector{Int}, k::Int,
                                   gaic_k::Float64)
    s = param_offsets[k] + 1
    e = param_offsets[k + 1]
    pk = e - s + 1
    n = length(wk)

    sp_indices = Int[]
    for (j, Sj) in enumerate(Sl)
        Sj_block = @view Sj[s:e, s:e]
        if any(!iszero, Sj_block)
            push!(sp_indices, j)
        end
    end

    isempty(sp_indices) && return

    for idx in sp_indices
        function gcv_obj(log_lam)
            Sk = zeros(pk, pk)
            for (j, Sj) in enumerate(Sl)
                Sj_block = @view Sj[s:e, s:e]
                if any(!iszero, Sj_block)
                    lsp_j = (j == idx) ? log_lam : log_sp[j]
                    Sk .+= exp(lsp_j) .* Sj_block
                end
            end
            _, edf, rss = _penalized_wls_with_edf(Xk, wk, zk, Sk)
            denom = max(n - gaic_k * edf, 1.0)
            return n * rss / (denom * denom)
        end

        log_sp[idx] = _golden_section_min(gcv_obj, -7.0, 7.0, 1e-4)
    end
end

"""Golden section minimization on [a,b] with tolerance tol."""
function _golden_section_min(f, a::Float64, b::Float64, tol::Float64)
    gr = (sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    for _ in 1:100
        if abs(b - a) < tol
            break
        end
        if f(c) < f(d)
            b = d
        else
            a = c
        end
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    end
    return (a + b) / 2.0
end

"""
Dispatch SP update based on method in GamlssControl.
"""
function _sp_update_param!(gamlss_ctrl::GamlssControl,
                            log_sp::Vector{Float64}, Sl::Vector{Matrix{Float64}},
                            β_k::Vector{Float64}, Xk::Matrix{Float64},
                            wk::Vector{Float64}, zk::Vector{Float64},
                            Sk::Matrix{Float64},
                            param_offsets::Vector{Int}, k::Int)
    m = gamlss_ctrl.sp_method
    if m == :efs
        _efs_update_param!(log_sp, Sl, β_k, Xk, wk, Sk, param_offsets, k)
    elseif m == :local_ml
        _local_ml_update_param!(log_sp, Sl, β_k, Xk, wk, zk, param_offsets, k)
    elseif m == :local_gaic
        _local_gaic_update_param!(log_sp, Sl, Xk, wk, zk, param_offsets, k, gamlss_ctrl.gaic_k)
    elseif m == :local_gcv
        _local_gcv_update_param!(log_sp, Sl, Xk, wk, zk, param_offsets, k, gamlss_ctrl.gaic_k)
    end
end

# ============================================================================
# RS (Rigby-Stasinopoulos) solver
# ============================================================================

"""
    gamlss_rs!(family, y, X_list, smooths_list, Sl, β_init, log_sp, param_offsets,
               ctrl, gamlss_ctrl, nsp, Mp, p, n) → (β, log_sp, η_list, dev, converged)

RS algorithm for GAMLSS: cyclic penalized IRLS in η-space.

For each outer iteration, cycles through parameters k=1..K:
1. Compute diagonal working weights w_k = d²NLL/dη_k² (clamped positive)
2. Compute working response z_k = η_k - (dNLL/dη_k) / w_k
3. Solve penalized WLS: (X_k'W_kX_k + S_k)β_k = X_k'W_kz_k
4. Update η_k = X_k β_k with step halving if deviance increases
5. Optionally update smoothing parameters via EFS per-parameter
"""
function gamlss_rs!(family::MultiParameterFamily, y::AbstractVector,
                    X_list::Vector{Matrix{Float64}},
                    smooths_list::Vector{Vector{ConstructedSmooth}},
                    Sl::Vector{Matrix{Float64}},
                    β_init::Vector{Float64}, log_sp::Vector{Float64},
                    param_offsets::Vector{Int},
                    ctrl::MPFitControl, gamlss_ctrl::GamlssControl,
                    nsp::Int, Mp::Int, p::Int, n::Int;
                    sp_fixed::Bool=false)
    K = nparams(family)
    ncols = deriv_ncols(K)
    derivs = Matrix{Float64}(undef, n, ncols)

    # Initialize coefficients and linear predictors
    β = copy(β_init)
    η_list = _compute_eta(X_list, β, param_offsets, K)

    # Per-parameter coefficient views
    β_list = [Vector{Float64}(undef, size(X_list[k], 2)) for k in 1:K]
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        β_list[k] .= @view(β[s:e])
    end

    # Working weight/response buffers
    wk = Vector{Float64}(undef, n)
    zk = Vector{Float64}(undef, n)

    dev = _global_deviance(family, y, η_list)
    converged = false

    for outer in 1:gamlss_ctrl.n_cyc
        dev_old = dev

        # Build per-parameter penalties from current log_sp
        S_list = _per_param_penalties(Sl, log_sp, param_offsets, K)

        for k in 1:K
            # Compute derivatives at current η
            _param_derivs_eta!(family, y, η_list, derivs)

            # Extract per-obs gradient and diagonal Hessian for param k
            gk_col = grad_col(k)
            hk_col = hess_col(K, k, k)

            @inbounds for i in 1:n
                h = derivs[i, hk_col]
                g = derivs[i, gk_col]
                # Clamp diagonal Hessian to be positive (working weight)
                w = clamp(h, 1e-10, 1e10)
                wk[i] = w
                # Working response: Newton step in η-space
                zk[i] = η_list[k][i] - g / w
            end

            # Penalized WLS solve
            β_new = _penalized_wls(X_list[k], wk, zk, S_list[k])
            η_new = X_list[k] * β_new

            # Step halving
            step = _get_step(gamlss_ctrl, k)
            η_candidate = step .* η_new .+ (1.0 - step) .* η_list[k]
            η_old_k = copy(η_list[k])
            η_list[k] = η_candidate

            if gamlss_ctrl.autostep
                dev_cand = _global_deviance(family, y, η_list)
                for _ in 1:5
                    if isfinite(dev_cand) && dev_cand <= dev + gamlss_ctrl.gd_tol
                        break
                    end
                    step *= 0.5
                    if step < 1e-10
                        η_list[k] = η_old_k
                        break
                    end
                    η_candidate = step .* η_new .+ (1.0 - step) .* η_old_k
                    η_list[k] = η_candidate
                    dev_cand = _global_deviance(family, y, η_list)
                end
            end

            # Update β for this parameter
            # Recover β from η: β = (X'X)^{-1} X'η  (or just use β_new scaled by step)
            β_list[k] .= step .* β_new .+ (1.0 - step) .* β_list[k]

            # EFS/local SP update for this parameter's smoothing parameters
            if !sp_fixed && nsp > 0
                _sp_update_param!(gamlss_ctrl, log_sp, Sl, β_list[k], X_list[k],
                                   wk, zk, S_list[k], param_offsets, k)
                S_list = _per_param_penalties(Sl, log_sp, param_offsets, K)
            end
        end

        dev = _global_deviance(family, y, η_list)

        if gamlss_ctrl.trace
            @info "RS outer $outer: deviance = $(round(dev, digits=4)), Δdev = $(round(dev_old - dev, sigdigits=3))"
        end

        if abs(dev_old - dev) < gamlss_ctrl.c_crit
            converged = true
            break
        end
    end

    # Reassemble full β
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        β[s:e] .= β_list[k]
    end

    return β, log_sp, η_list, dev, converged
end

# ============================================================================
# CG (Cole-Green) solver
# ============================================================================

"""
    gamlss_cg!(family, y, X_list, smooths_list, Sl, β_init, log_sp, param_offsets,
               ctrl, gamlss_ctrl, nsp, Mp, p, n) → (β, log_sp, η_list, dev, converged)

CG algorithm for GAMLSS: like RS but with cross-derivative corrections.

Uses 2 initial RS-style iterations to stabilize, then switches to full CG
with cross-term adjustments from the Hessian off-diagonals.
"""
function gamlss_cg!(family::MultiParameterFamily, y::AbstractVector,
                    X_list::Vector{Matrix{Float64}},
                    smooths_list::Vector{Vector{ConstructedSmooth}},
                    Sl::Vector{Matrix{Float64}},
                    β_init::Vector{Float64}, log_sp::Vector{Float64},
                    param_offsets::Vector{Int},
                    ctrl::MPFitControl, gamlss_ctrl::GamlssControl,
                    nsp::Int, Mp::Int, p::Int, n::Int;
                    sp_fixed::Bool=false)
    K = nparams(family)
    ncols = deriv_ncols(K)
    derivs = Matrix{Float64}(undef, n, ncols)

    # Initialize
    β = copy(β_init)
    η_list = _compute_eta(X_list, β, param_offsets, K)

    β_list = [Vector{Float64}(undef, size(X_list[k], 2)) for k in 1:K]
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        β_list[k] .= @view(β[s:e])
    end

    wk = Vector{Float64}(undef, n)
    zk = Vector{Float64}(undef, n)

    dev = _global_deviance(family, y, η_list)
    converged = false

    # Number of initial RS-only warm-up iterations
    n_rs_warmup = 2

    for outer in 1:gamlss_ctrl.n_cyc
        dev_old = dev
        use_cross = outer > n_rs_warmup && K > 1

        # Build per-parameter penalties
        S_list = _per_param_penalties(Sl, log_sp, param_offsets, K)

        if use_cross
            # CG mode: compute derivatives once, use cross-terms in inner loop
            _param_derivs_eta!(family, y, η_list, derivs)
            η_start = [copy(η_list[k]) for k in 1:K]

            for inner in 1:gamlss_ctrl.i_cyc
                dev_inner_old = _global_deviance(family, y, η_list)

                for k in 1:K
                    gk_col = grad_col(k)
                    hk_col = hess_col(K, k, k)

                    @inbounds for i in 1:n
                        h = clamp(derivs[i, hk_col], 1e-10, 1e10)
                        g = derivs[i, gk_col]
                        w = h
                        z = η_start[k][i] - g / w

                        # Cross-derivative adjustment
                        for j in 1:K
                            j == k && continue
                            hkj_col = hess_col(K, min(k, j), max(k, j))
                            cross = derivs[i, hkj_col]
                            delta_j = η_list[j][i] - η_start[j][i]
                            z -= cross * delta_j / w
                        end

                        wk[i] = w
                        zk[i] = z
                    end

                    # Check for NaN in working response
                    if any(!isfinite, zk) || any(!isfinite, wk)
                        continue
                    end

                    β_new = _penalized_wls(X_list[k], wk, zk, S_list[k])
                    if any(!isfinite, β_new)
                        continue
                    end
                    η_new = X_list[k] * β_new

                    # Step with step-halving
                    step = _get_step(gamlss_ctrl, k)
                    η_old_k = copy(η_list[k])
                    η_list[k] = step .* η_new .+ (1.0 - step) .* η_list[k]

                    if gamlss_ctrl.autostep
                        dev_cand = _global_deviance(family, y, η_list)
                        for _ in 1:5
                            if isfinite(dev_cand) && dev_cand <= dev_inner_old + gamlss_ctrl.gd_tol
                                break
                            end
                            step *= 0.5
                            if step < 1e-10
                                η_list[k] = η_old_k
                                step = 0.0
                                break
                            end
                            η_list[k] = step .* η_new .+ (1.0 - step) .* η_old_k
                            dev_cand = _global_deviance(family, y, η_list)
                        end
                    end

                    β_list[k] .= step .* β_new .+ (1.0 - step) .* β_list[k]
                end

                dev_inner = _global_deviance(family, y, η_list)
                if !isfinite(dev_inner) || abs(dev_inner_old - dev_inner) < gamlss_ctrl.i_cc
                    break
                end
            end
        else
            # RS mode (warm-up or K==1)
            for k in 1:K
                _param_derivs_eta!(family, y, η_list, derivs)
                gk_col = grad_col(k)
                hk_col = hess_col(K, k, k)

                @inbounds for i in 1:n
                    h = clamp(derivs[i, hk_col], 1e-10, 1e10)
                    g = derivs[i, gk_col]
                    wk[i] = h
                    zk[i] = η_list[k][i] - g / h
                end

                β_new = _penalized_wls(X_list[k], wk, zk, S_list[k])
                η_new = X_list[k] * β_new

                step = _get_step(gamlss_ctrl, k)
                η_old_k = copy(η_list[k])
                η_list[k] = step .* η_new .+ (1.0 - step) .* η_list[k]

                if gamlss_ctrl.autostep
                    dev_cand = _global_deviance(family, y, η_list)
                    for _ in 1:5
                        if isfinite(dev_cand) && dev_cand <= dev + gamlss_ctrl.gd_tol
                            break
                        end
                        step *= 0.5
                        if step < 1e-10
                            η_list[k] = η_old_k
                            step = 0.0
                            break
                        end
                        η_list[k] = step .* η_new .+ (1.0 - step) .* η_old_k
                        dev_cand = _global_deviance(family, y, η_list)
                    end
                end

                β_list[k] .= step .* β_new .+ (1.0 - step) .* β_list[k]
            end
        end

        # SP update for smoothing parameters
        if !sp_fixed && nsp > 0
            _param_derivs_eta!(family, y, η_list, derivs)
            for k in 1:K
                gk_col = grad_col(k)
                hk_col = hess_col(K, k, k)
                @inbounds for i in 1:n
                    h = clamp(derivs[i, hk_col], 1e-10, 1e10)
                    wk[i] = h
                    zk[i] = η_list[k][i] - derivs[i, gk_col] / h
                end
                _sp_update_param!(gamlss_ctrl, log_sp, Sl, β_list[k], X_list[k],
                                   wk, zk, S_list[k], param_offsets, k)
            end
            S_list = _per_param_penalties(Sl, log_sp, param_offsets, K)
        end

        dev = _global_deviance(family, y, η_list)

        if gamlss_ctrl.trace
            mode_str = use_cross ? "CG" : "RS"
            @info "$mode_str outer $outer: deviance = $(round(dev, digits=4)), Δdev = $(round(dev_old - dev, sigdigits=3))"
        end

        if !isfinite(dev)
            break
        end

        if abs(dev_old - dev) < gamlss_ctrl.c_crit
            converged = true
            break
        end
    end

    # Reassemble full β
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        β[s:e] .= β_list[k]
    end

    return β, log_sp, η_list, dev, converged
end

# ============================================================================
# Shared RS/CG dispatch from _gamlss_fit
# ============================================================================

"""
Internal function to fit GAMLSS using RS or CG solver, called from _gamlss_fit.
Handles EFS initialization, solver dispatch, and result packaging.
"""
function _gamlss_fit_rscg(method::Symbol, family::MultiParameterFamily,
                          y::AbstractVector, X_list::Vector{Matrix{Float64}},
                          smooths_list::Vector{Vector{ConstructedSmooth}},
                          Sl::Vector{Matrix{Float64}},
                          β_init::Vector{Float64}, log_sp::Vector{Float64},
                          param_offsets::Vector{Int},
                          ctrl::MPFitControl, gamlss_ctrl::GamlssControl,
                          nsp::Int, Mp::Int, p::Int, n::Int,
                          sp_fixed_input)
    K = nparams(family)
    sp_fixed = sp_fixed_input !== nothing || nsp == 0

    # Run the chosen solver
    if method == :rs
        β_opt, log_sp, η_fit, dev, conv = gamlss_rs!(
            family, y, X_list, smooths_list, Sl, β_init, log_sp,
            param_offsets, ctrl, gamlss_ctrl, nsp, Mp, p, n;
            sp_fixed=sp_fixed)
    else  # :cg
        β_opt, log_sp, η_fit, dev, conv = gamlss_cg!(
            family, y, X_list, smooths_list, Sl, β_init, log_sp,
            param_offsets, ctrl, gamlss_ctrl, nsp, Mp, p, n;
            sp_fixed=sp_fixed)
    end

    # Build final penalty for covariance computation
    S = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        if j <= length(log_sp)
            S .+= exp(log_sp[j]) .* Sj
        end
    end

    Vp, Vc, H0 = mp_covariance(family, y, X_list, β_opt, S, param_offsets)
    edf = diag(Vp * H0)
    nll_val = nll_total(family, y, η_fit)

    # REML / LAML
    reml_val = dev / 2.0  # approximate REML from deviance
    laml = mp_laml(family, y, X_list, β_opt, S, Sl, log_sp, param_offsets; Mp=Mp)

    idpars = Vector{Int}(undef, p)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        idpars[s:e] .= k
    end

    return MultiParameterModel(
        family, β_opt, η_fit, X_list, smooths_list, log_sp,
        edf, Vp, Vc, nll_val, reml_val, laml, y, n, conv, idpars, param_offsets)
end
