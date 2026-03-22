# Outer iteration — smoothing parameter optimization
#
# Two optimizers available (selected via control.sp_optimizer):
#   :efs    — Extended Fellner-Schall (Wood & Fasiolo 2017). Default.
#             Fast, monotonically convergent, 1 PIRLS per outer iteration.
#   :newton — Newton's method with autodiff Hessian via ForwardDiff.
#             Computes exact Hessian of conditional REML w.r.t. log(sp).
#             More expensive per step but fewer iterations for difficult problems.

using ForwardDiff

"""
    outer_iteration(X, y, smooths, penalty, family, link;
                    method, weights, control)

Run the outer iteration to optimize smoothing parameters.
Uses the Extended Fellner-Schall (EFS) method for robust convergence.
Returns updated `log_sp` and final `PirlsResult`.
"""
function outer_iteration(X::Matrix{Float64}, y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::UnivariateDistribution, link::GLM.Link;
    method::Symbol = :REML,
    weights::Vector{Float64} = ones(length(y)),
    control::GamControl = gam_control())

    n, p = size(X)
    n_sp = length(penalty.sp)

    if n_sp == 0
        S_total = zeros(p, p)
        result = pirls(X, y, S_total, family, link;
            weights = weights, control = control)
        return penalty.sp, result
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing

    is_gaussian_identity = family isa Normal && link isa GLM.IdentityLink
    XtWX_cached = nothing
    Xty_cached = nothing
    Xw_buf = similar(X)  # pre-allocate buffer

    if is_gaussian_identity
        # Precompute X'WX and X'Wy once — both are constant
        XtWX_cached = zeros(p, p)
        @inbounds for i in 1:n
            sw = sqrt(weights[i])
            for j in 1:p
                Xw_buf[i, j] = X[i, j] * sw
            end
        end
        BLAS.syrk!('U', 'T', 1.0, Xw_buf, 0.0, XtWX_cached)
        @inbounds for j in 1:p
            for k in (j + 1):p
                XtWX_cached[k, j] = XtWX_cached[j, k]
            end
        end
        Xty_cached = X' * (weights .* y)
    end

    A_buf = zeros(p, p)
    efs_mult = 1.0  # Step multiplier for EFS (reduced on failed steps)

    for outer_iter in 1:(control.outer_maxit)
        # Inner: P-IRLS for current smoothing parameters
        S_total = total_penalty(penalty, log_sp, p)

        if is_gaussian_identity
            # Direct solve — no IRLS iteration, no O(np²) recompute
            result = pirls_gaussian(X, y, S_total, XtWX_cached, Xty_cached;
                weights = weights)
        else
            start = prev_result === nothing ? nothing : prev_result.coefficients
            result = pirls(X, y, S_total, family, link;
                weights = weights, start = start, control = control)
        end

        if !result.converged && control.trace
            @warn "P-IRLS did not converge at outer iteration $outer_iter"
        end

        beta = result.coefficients
        w = result.working_weights
        dev = result.deviance

        # Estimate scale with EDoF correction (matching mgcv's efsudr)
        edf_total = sum(result.edf_vec)
        if _needs_scale_estimate(family)
            # EDoF-corrected scale: φ̂ = pearson × n / (n − edf)²
            # Equivalent to: (pearson / n) × n / (n − edf) = pearson / (n − edf)
            scale_est = max(result.pearson / max(n - edf_total, 1.0), 1e-10)
        else
            scale_est = 1.0
        end

        # Smoothing parameter update
        if XtWX_cached !== nothing
            copyto!(A_buf, XtWX_cached)
            @inbounds for j in 1:p, k in 1:p
                A_buf[j, k] += S_total[j, k]
            end
        else
            _build_XtWX_plus_S!(A_buf, X, w, S_total, p, n, Xw_buf)
        end
        A_chol = cholesky(Symmetric(copy(A_buf)))
        Ainv = inv(A_chol)

        log_sp_new = copy(log_sp)
        max_change = 0.0

        if control.sp_optimizer == :newton
            # Newton with autodiff Hessian on conditional REML.
            # Differentiates REML score w.r.t. log_sp, holding β and w fixed.
            log_sp_new, max_change = _newton_sp_update(
                log_sp, X, beta, w, dev, penalty, family, method,
                scale_est, n, p, edf_total, y, weights, control)
        else
            # EFS update (default) — Wood & Fasiolo (2017)
            log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty,
                scale_est, efs_mult)
            max_change = maximum(abs.(log_sp_new .- log_sp))

            # Step halving: verify REML improves, halve mult if not.
            # This matches mgcv's efsudr which reduces mult on failed steps.
            if outer_iter > 1 && max_change > control.epsilon
                # Reuse A_buf which already contains XtWX + S_total;
                # extract XtWX = A_buf - S_total
                XtWX_cur = A_buf .- S_total

                ls = _log_saturated_likelihood(family, y, weights, scale_est)
                reml_old = _conditional_reml(log_sp, XtWX_cur, beta, dev,
                    penalty, scale_est, n, p, edf_total, method,
                    control.gamma, ls)
                reml_new = _conditional_reml(log_sp_new, XtWX_cur, beta, dev,
                    penalty, scale_est, n, p, edf_total, method,
                    control.gamma, ls)

                if reml_new > reml_old + control.epsilon * abs(reml_old)
                    # REML worsened — reduce step via mult
                    for _halve in 1:4
                        efs_mult *= 0.5
                        log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty,
                            scale_est, efs_mult)
                        reml_new = _conditional_reml(log_sp_new, XtWX_cur,
                            beta, dev, penalty, scale_est, n, p, edf_total,
                            method, control.gamma, ls)
                        reml_new <= reml_old + control.epsilon * abs(reml_old) &&
                            break
                    end
                    max_change = maximum(abs.(log_sp_new .- log_sp))
                else
                    # Step succeeded — gradually restore mult toward 1
                    efs_mult = min(1.0, efs_mult * 2.0)
                end
            end
        end

        if control.trace
            println("Outer iter $outer_iter: " *
                    "sp=[$(join([@sprintf("%.4f", exp(s)) for s in log_sp_new], ", "))]" *
                    ", edf=$(round(edf_total; digits=2))" *
                    ", max_change=$(@sprintf("%.6f", max_change))")
        end

        log_sp .= log_sp_new
        prev_result = result

        # Convergence check: smoothing parameter changes small
        if max_change < control.epsilon * 10
            if control.trace
                println("Outer iteration converged at iteration $outer_iter")
            end
            break
        end
    end

    # Final solve with converged parameters
    penalty.sp .= log_sp
    S_total = total_penalty(penalty, log_sp, p)
    if is_gaussian_identity
        final_result = pirls_gaussian(X, y, S_total, XtWX_cached, Xty_cached;
            weights = weights)
    else
        final_result = pirls(X, y, S_total, family, link;
            weights = weights, start = prev_result.coefficients,
            control = control)
    end

    return log_sp, final_result
end

# ============================================================================
# EFS helper functions
# ============================================================================

"""
    _efs_sp_update(log_sp, beta, Ainv, penalty, scale_est, mult)

Compute the EFS smoothing parameter update (Wood & Fasiolo 2017, eq. 5).
Returns new `log_sp` vector. `mult` scales the step in log-space (1.0 = full).
"""
function _efs_sp_update(log_sp::Vector{Float64}, beta::Vector{Float64},
    Ainv::Matrix{Float64}, penalty::PenaltySetup,
    scale_est::Float64, mult::Float64)

    log_sp_new = copy(log_sp)
    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        beta_block = beta[idx]

        for Si in block.S
            λ = exp(log_sp[sp_idx])
            rank_j = Float64(block.rank)

            bSb = dot(beta_block, Si * beta_block)
            Ainv_block = Ainv[idx, idx]
            trVS = tr(Ainv_block * Si)

            a = max(0.0, rank_j / λ - trVS)

            if a > 0 && bSb > eps()
                r = scale_est * a / bSb
                log_sp_new[sp_idx] = clamp(
                    log_sp[sp_idx] + log(max(r, 1e-15)) * mult, -15.0, 15.0)
            end

            sp_idx += 1
        end
    end
    return log_sp_new
end

"""
    _efs_reml_score(X, y, log_sp, penalty, family, link, weights,
                    pirls_result, method, scale_est, gamma, n, p)

Compute the REML score at given `log_sp`, using current PIRLS result for β, w.
Used for EFS step halving when XtWX is not cached.
"""
function _efs_reml_score(X::Matrix{Float64}, y::Vector{Float64},
    log_sp::Vector{Float64}, penalty::PenaltySetup,
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64}, pirls_result::PirlsResult,
    method::Symbol, scale_est::Float64, gamma::Float64,
    n::Int, p::Int)

    beta = pirls_result.coefficients
    w = pirls_result.working_weights
    dev = pirls_result.deviance
    edf_total = sum(pirls_result.edf_vec)

    XtWX = X' * Diagonal(w) * X
    ls = _log_saturated_likelihood(family, y, weights, scale_est)

    return _conditional_reml(log_sp, XtWX, beta, dev, penalty,
        scale_est, n, p, edf_total, method, gamma, ls)
end

# ============================================================================
# Extended family outer iteration
# ============================================================================

"""
    outer_iteration(X, y, smooths, penalty, family::ExtendedFamily;
                    method, weights, control)

Outer iteration for extended families. Uses EFS updates with
`pirls_extended` and periodic extra-parameter estimation.
"""
function outer_iteration(X::Matrix{Float64}, y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::ExtendedFamily, link::GLM.Link;
    method::Symbol = :REML,
    weights::Vector{Float64} = ones(length(y)),
    control::GamControl = gam_control(),
    start::Union{Vector{Float64}, Nothing} = nothing)

    n, p = size(X)
    n_sp = length(penalty.sp)

    if n_sp == 0
        S_total = zeros(p, p)
        result = pirls_extended(X, y, S_total, family, link;
            weights = weights, start = start, control = control)
        return penalty.sp, result
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing
    prev_dev = Inf
    Xw_buf = similar(X)
    A_buf = zeros(p, p)
    efs_mult = 1.0

    for outer_iter in 1:(control.outer_maxit)
        S_total = total_penalty(penalty, log_sp, p)

        pirls_start = prev_result === nothing ? start : prev_result.coefficients
        result = pirls_extended(X, y, S_total, family, link;
            weights = weights, start = pirls_start, control = control)

        if !result.converged && control.trace
            @warn "P-IRLS did not converge at outer iteration $outer_iter"
        end

        beta = result.coefficients
        w = result.working_weights
        dev = result.deviance
        edf_total = sum(result.edf_vec)
        if _needs_scale_estimate(family)
            scale_est = max(result.pearson / max(n - edf_total, 1.0), 1e-10)
        else
            scale_est = 1.0
        end

        # Build A = X'WX + S and compute Ainv for EFS
        _build_XtWX_plus_S!(A_buf, X, w, S_total, p, n, Xw_buf)
        A_chol = cholesky(Symmetric(copy(A_buf)))
        Ainv = inv(A_chol)

        # EFS update
        log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty,
            scale_est, efs_mult)
        max_change = maximum(abs.(log_sp_new .- log_sp))

        # Step halving: use penalized deviance as criterion for extended families
        if outer_iter > 1 && max_change > control.epsilon
            bSb = 0.0
            sp_idx = 1
            for block in penalty.blocks
                idx = block.start:block.stop
                beta_block = beta[idx]
                for Si in block.S
                    bSb += exp(log_sp[sp_idx]) * dot(beta_block, Si * beta_block)
                    sp_idx += 1
                end
            end
            pdev_old = prev_dev + dot(prev_result.coefficients,
                total_penalty(penalty, log_sp, p) * prev_result.coefficients)
            pdev_new = dev + bSb

            if pdev_new > pdev_old + control.epsilon * abs(pdev_old)
                for _halve in 1:4
                    efs_mult *= 0.5
                    log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty,
                        scale_est, efs_mult)
                end
                max_change = maximum(abs.(log_sp_new .- log_sp))
            else
                efs_mult = min(1.0, efs_mult * 2.0)
            end
        end

        if control.trace
            println("Outer iter $outer_iter: " *
                    "sp=[$(join([@sprintf("%.4f", exp(s)) for s in log_sp_new], ", "))]" *
                    ", edf=$(round(edf_total; digits=2))" *
                    ", max_change=$(@sprintf("%.6f", max_change))")
        end

        log_sp .= log_sp_new
        prev_dev_change = abs(dev - prev_dev)
        prev_result = result
        prev_dev = dev

        # Update extra parameter after each outer iteration
        if _has_extra_param(family)
            estimate_theta!(family, y, result.fitted_values, weights, scale_est)
        end

        # Convergence: SP change small OR deviance change negligible (stable fit)
        sp_converged = max_change < control.epsilon * 10
        dev_converged = outer_iter > 3 && max_change < 1e-4 &&
            prev_dev_change < control.epsilon * abs(dev)
        if sp_converged || dev_converged
            if control.trace
                println("Outer iteration converged at iteration $outer_iter")
            end
            break
        end
    end

    # Final P-IRLS with converged parameters
    penalty.sp .= log_sp
    S_total = total_penalty(penalty, log_sp, p)
    final_result = pirls_extended(X, y, S_total, family, link;
        weights = weights, start = prev_result.coefficients,
        control = control)

    return log_sp, final_result
end

# ============================================================================
# Newton + autodiff smoothing parameter update
# ============================================================================

"""
    _conditional_reml(log_sp, XtWX, beta, dev, penalty, scale_est, n, p,
                      edf_total, method, gamma, ls)

Compute the conditional REML score as a function of `log_sp` only, holding
β, w, and deviance fixed at their current PIRLS values. This makes the
function differentiable via ForwardDiff w.r.t. `log_sp`.

`ls` is the precomputed log saturated likelihood (constant w.r.t. log_sp).
"""
function _conditional_reml(log_sp::AbstractVector, XtWX::Matrix{Float64},
    beta::Vector{Float64}, dev::Float64,
    penalty::PenaltySetup, scale_est::Float64,
    n::Int, p::Int, edf_total::Float64,
    method::Symbol, gamma::Float64, ls::Float64)

    T = eltype(log_sp)
    S_total = total_penalty(penalty, log_sp, p)

    # A = X'WX + S (XtWX is Float64, S_total may be Dual)
    A = zeros(T, p, p)
    @inbounds for j in 1:p, k in 1:p
        A[j, k] = XtWX[j, k] + S_total[j, k]
    end

    A_chol = cholesky(Symmetric(A))
    log_det_A = logdet(A_chol)

    # Log pseudo-determinant of penalty
    log_det_S = _log_penalty_det(penalty, log_sp)

    # Penalty null space dimension
    Mp = sum(b.stop - b.start + 1 - b.rank for b in penalty.blocks; init = 0)

    # Penalized deviance
    penalty_contrib = zero(T)
    @inbounds for i in eachindex(beta)
        for j in eachindex(beta)
            penalty_contrib += beta[i] * S_total[i, j] * beta[j]
        end
    end
    Dp = dev + penalty_contrib

    if method == :GCV
        denom = n - gamma * edf_total
        return T(n * dev / denom^2)
    end

    # REML/ML score (ls is constant w.r.t. log_sp, precomputed by caller)
    if method == :REML
        return (Dp / (2 * scale_est) - ls) / gamma +
               T(0.5) * log_det_A - T(0.5) * log_det_S -
               T(0.5) * Mp * (log(T(2π) * scale_est) - log(T(gamma)))
    else  # :ML
        return (Dp / (2 * scale_est) - ls) / gamma +
               T(0.5) * log_det_A - T(0.5) * log_det_S
    end
end

"""
    _newton_sp_update(log_sp, X, beta, w, dev, penalty, family, method,
                      scale_est, n, p, edf_total, weights, control)

Newton step on log smoothing parameters using ForwardDiff for the Hessian.
Returns `(log_sp_new, max_change)`.
"""
function _newton_sp_update(log_sp::Vector{Float64},
    X::Matrix{Float64}, beta::Vector{Float64},
    w::Vector{Float64}, dev::Float64,
    penalty::PenaltySetup,
    family, method::Symbol, scale_est::Float64,
    n::Int, p::Int, edf_total::Float64,
    y::Vector{Float64},
    weights::Vector{Float64}, control::GamControl)

    n_sp = length(log_sp)
    gamma = control.gamma

    # Precompute XtWX (constant w.r.t. log_sp since w is fixed)
    XtWX = X' * Diagonal(w) * X

    # Precompute log saturated likelihood (constant w.r.t. log_sp)
    # For ExtendedFamily: deviance already = -2(ll - ls), so ls contribution
    # is implicit. Set ls=0 and use dev/2 as the likelihood contribution.
    ls = family isa ExtendedFamily ? 0.0 :
        _log_saturated_likelihood(family, y, weights, scale_est)

    # Compute gradient and Hessian via ForwardDiff
    reml_fn = lsp -> _conditional_reml(lsp, XtWX, beta, dev, penalty,
        scale_est, n, p, edf_total, method, gamma, ls)

    grad = ForwardDiff.gradient(reml_fn, log_sp)
    hess = ForwardDiff.hessian(reml_fn, log_sp)

    # Stabilize Hessian: eigendecompose and flip negative eigenvalues
    # (same approach as mgcv's fast.REML.fit Newton step)
    eh = eigen(Symmetric(hess))
    ev = copy(eh.values)
    min_ev = maximum(abs.(ev)) * 1e-6
    @inbounds for i in eachindex(ev)
        ev[i] = max(abs(ev[i]), min_ev)
    end

    # Newton step: Δρ = -H⁻¹ g
    step = -(eh.vectors * Diagonal(1.0 ./ ev) * eh.vectors') * grad
    step .= clamp.(step, -5.0, 5.0)

    log_sp_new = clamp.(log_sp .+ step, -15.0, 15.0)

    # Step halving if score increases
    cur_score = reml_fn(log_sp)
    trial_score = reml_fn(log_sp_new)
    for _ in 1:30
        trial_score <= cur_score && break
        step .*= 0.5
        log_sp_new .= clamp.(log_sp .+ step, -15.0, 15.0)
        trial_score = reml_fn(log_sp_new)
    end

    max_change = maximum(abs.(log_sp_new .- log_sp))
    return log_sp_new, max_change
end
