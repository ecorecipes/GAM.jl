# Multi-parameter model fitting
#
# Inner loop: penalized Newton for β given λ
# Outer loop: EFS (default) or BFGS (fallback) on REML criterion for log(λ)
# User API: evgam() function

using LinearAlgebra
using Statistics: mean, std
using Printf: @sprintf

# ============================================================================
# Inner Newton solver
# ============================================================================

"""
    MPFitControl

Control parameters for multi-parameter model fitting.
"""
struct MPFitControl
    inner_maxit::Int
    inner_tol::Float64
    outer_maxit::Int
    outer_tol::Float64
    step_max::Float64
    trace::Bool
end

function mp_control(; inner_maxit::Int=100, inner_tol::Real=1e-6,
                      outer_maxit::Int=100, outer_tol::Real=1e-5,
                      step_max::Real=1.0, trace::Bool=false)
    MPFitControl(inner_maxit, Float64(inner_tol), outer_maxit, Float64(outer_tol),
                 Float64(step_max), trace)
end

"""
    mp_newton_inner(family, y, X_list, β, S, control) → (β, nll_pen, g, H, converged)

Penalized Newton optimization for coefficients β given penalty S.
Minimizes NLL(β) + 0.5 β'Sβ.
"""
function mp_newton_inner(family::MultiParameterFamily, y::AbstractVector,
                         X_list::Vector{Matrix{Float64}}, β::Vector{Float64},
                         S::Matrix{Float64}, control::MPFitControl;
                         Ain = nothing,
                         bin = nothing,
                         Aeq = nothing,
                         beq = nothing)
    K = nparams(family)
    n = length(y)
    p = length(β)
    ncols = deriv_ncols(K)
    derivs = Matrix{Float64}(undef, n, ncols)
    H = Matrix{Float64}(undef, p, p)

    param_offsets = cumsum([0; [size(X, 2) for X in X_list]])

    # Pre-allocate η buffers
    η_list = [Vector{Float64}(undef, n) for _ in 1:K]
    Sβ = Vector{Float64}(undef, p)

    converged = false
    nll_pen = Inf
    nll_pen_prev = Inf

    for iter in 1:control.inner_maxit
        # Compute linear predictors (in-place)
        _compute_eta!(η_list, X_list, β, param_offsets, K)

        # Compute NLL and derivatives
        nll_val = nll_total(family, y, η_list)
        nll_derivs!(family, derivs, y, η_list)

        # Assemble gradient and Hessian
        g = assemble_gradient(derivs, X_list)
        assemble_hessian!(H, derivs, X_list)

        # Add penalty
        mul!(Sβ, S, β)
        pen = 0.5 * dot(β, Sβ)
        nll_pen_new = nll_val + pen
        g .+= Sβ
        H .+= S

        has_constraints = (Ain !== nothing && size(Ain, 1) > 0) || (Aeq !== nothing && size(Aeq, 1) > 0)

        # Check convergence: gradient norm
        grad_max = maximum(abs, g)
        if grad_max < control.inner_tol &&
           (!has_constraints || _is_feasible(β, Ain, bin, Aeq, beq))
            nll_pen = nll_pen_new
            converged = true
            break
        end

        # Check convergence: objective change (relative)
        if iter > 1 && isfinite(nll_pen_prev)
            obj_rel = abs(nll_pen_new - nll_pen_prev) / (abs(nll_pen_new) + 1.0)
            if obj_rel < 1e-8 &&
               (!has_constraints || _is_feasible(β, Ain, bin, Aeq, beq))
                nll_pen = nll_pen_new
                converged = true
                break
            end
        end

        if has_constraints
            β_target = _solve_constrained_qp(H, -g + H * β, Ain, bin, Aeq, beq;
                warm_start = β,
                eps_abs = max(control.inner_tol, 1e-8),
                eps_rel = max(control.inner_tol, 1e-8))
            δ = β - β_target
        else
            # Newton step with Cholesky
            H_sym = Symmetric(H)
            F = _safe_cholesky(H_sym)
            if F === nothing
                # Escalating diagonal perturbation
                diag_base = max(1e-6 * maximum(abs, diag(H)), 1e-8)
                for attempt in 0:5
                    diag_add = diag_base * 10.0^attempt
                    H_pert = copy(H)
                    for i in 1:p
                        H_pert[i, i] += diag_add
                    end
                    F = _safe_cholesky(Symmetric(H_pert))
                    F !== nothing && break
                end
                if F === nothing
                    # Last resort: use identity-scaled step
                    δ = g ./ max(1.0, maximum(abs, g))
                else
                    δ = F \ g
                end
            else
                δ = F \ g
            end
        end

        # Step halving with simple decrease (matching evgam's approach)
        step = min(1.0, control.step_max)
        β_new = β .- step .* δ
        η_new = _compute_eta(X_list, β_new, param_offsets, K)
        nll_new = nll_total(family, y, η_new) + 0.5 * dot(β_new, S * β_new)

        for _ in 1:15
            if isfinite(nll_new) && nll_new < nll_pen_new
                break
            end
            step *= 0.5
            if step < 1e-12
                break
            end
            β_new = β .- step .* δ
            η_new = _compute_eta(X_list, β_new, param_offsets, K)
            nll_new = nll_total(family, y, η_new) + 0.5 * dot(β_new, S * β_new)
        end

        # Check step size convergence (matching evgam)
        step_mean = step * sum(abs, δ) / length(δ)
        if step_mean < 1e-12
            nll_pen = isfinite(nll_new) ? nll_new : nll_pen_new
            converged = true
            break
        end

        # Accept step even if no decrease, as long as finite
        if !isfinite(nll_new)
            nll_pen = nll_pen_new
            converged = grad_max < 1e-2
            break
        end

        nll_pen_prev = nll_pen_new
        β .= β_new
        nll_pen = nll_new

        if control.trace && iter % 10 == 0
            @info "  Inner iteration $iter: nll_pen = $(round(nll_pen, digits=4)), max|g| = $(round(grad_max, sigdigits=3))"
        end
    end

    # Final gradient/Hessian at converged β
    η_list = _compute_eta(X_list, β, param_offsets, K)
    nll_derivs!(family, derivs, y, η_list)
    g = assemble_gradient(derivs, X_list)
    assemble_hessian!(H, derivs, X_list)
    g .+= S * β
    H .+= S

    return β, nll_pen, g, H, converged
end

"""Compute linear predictors, reusing pre-allocated η_list if provided."""
function _compute_eta(X_list, β, param_offsets, K)
    η_list = Vector{Vector{Float64}}(undef, K)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        η_list[k] = X_list[k] * @view(β[s:e])
    end
    return η_list
end

function _compute_eta!(η_list::Vector{Vector{Float64}}, X_list, β, param_offsets, K)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        mul!(η_list[k], X_list[k], @view(β[s:e]))
    end
    return η_list
end

function _safe_cholesky(A::Symmetric)
    try
        return cholesky(A)
    catch
        return nothing
    end
end

# ============================================================================
# Hessian-based smoothing parameter initialization (matching R evgam's .guess)
# ============================================================================

"""
Initialize log smoothing parameters using the unpenalized Hessian diagonal,
following R evgam's approach: λ_j ≈ H_diag / S_diag × scaling_factor.
This avoids under-/over-smoothing parameters with different curvature scales.
"""
function _init_log_sp_hessian(family::MultiParameterFamily, y::AbstractVector,
                               X_list::Vector{Matrix{Float64}},
                               Sl::Vector{Matrix{Float64}},
                               β_init::Vector{Float64},
                               param_offsets::Vector{Int}, nsp::Int)
    if nsp == 0
        return Float64[]
    end

    K = nparams(family)
    n = length(y)
    p = length(β_init)
    ncols = deriv_ncols(K)
    derivs = Matrix{Float64}(undef, n, ncols)
    H = Matrix{Float64}(undef, p, p)

    # Evaluate unpenalized Hessian at initial β
    η_list = _compute_eta(X_list, β_init, param_offsets, K)
    nll_derivs!(family, derivs, y, η_list)
    assemble_hessian!(H, derivs, X_list)

    log_sp = zeros(nsp)
    for j in 1:nsp
        Sj = Sl[j]
        # Ratio of mean Hessian diagonal to mean penalty diagonal for non-zero entries
        s_diag = diag(Sj)
        h_diag = diag(H)

        # Find overlap: indices where penalty has non-zero entries
        nz = findall(d -> abs(d) > 1e-12, s_diag)
        if isempty(nz)
            log_sp[j] = 0.0
            continue
        end

        h_mean = mean(abs.(h_diag[nz]))
        s_mean = mean(abs.(s_diag[nz]))

        if s_mean > 1e-12 && h_mean > 1e-12
            # Scale factor of 1.5 matches R evgam's .guess heuristic
            log_sp[j] = log(h_mean * 1.5 / s_mean)
        end
    end

    return log_sp
end

# ============================================================================
# REML criterion
# ============================================================================

"""
    mp_reml(log_sp, family, y, X_list, Sl, β_init, param_offsets, control)
    → (reml_val, β_opt, gradient)

Compute REML criterion for given log smoothing parameters.
REML = NLL_pen(β*) + 0.5 log|H*| - 0.5 log|S| + const
"""
function mp_reml(log_sp::Vector{Float64}, family::MultiParameterFamily,
                 y::AbstractVector, X_list::Vector{Matrix{Float64}},
                 Sl::Vector{Matrix{Float64}}, β_init::Vector{Float64},
                 param_offsets::Vector{Int}, control::MPFitControl;
                 Mp::Int=0,
                 Ain = nothing,
                 bin = nothing,
                 Aeq = nothing,
                 beq = nothing)
    p = length(β_init)
    K = nparams(family)

    # Build penalty from current smoothing parameters
    S = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        S .+= exp(log_sp[j]) .* Sj
    end

    # Inner Newton
    β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_init, S, control;
        Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)

    # log|H| — penalized Hessian determinant
    F_H = _safe_cholesky(Symmetric(H))
    if F_H === nothing
        return 1e20, β_opt, zeros(length(log_sp))
    end
    logdetH = 2.0 * sum(log.(diag(F_H.L)))

    # log|S+| — penalty determinant (only non-zero eigenvalues)
    logdetS = _logdet_penalty(Sl, log_sp, p)

    # REML = NLL_pen + 0.5 log|H| - 0.5 log|S+| + 0.5 Mp log(2π)
    reml_val = nll_pen + 0.5 * logdetH - 0.5 * logdetS + 0.5 * Mp * log(2π)

    if !isfinite(reml_val)
        reml_val = 1e20
    end

    return reml_val, β_opt, g
end

"""Log determinant of penalty (sum of log of non-zero eigenvalues)."""
function _logdet_penalty(Sl::Vector{Matrix{Float64}}, log_sp::Vector{Float64}, p::Int)
    S = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        S .+= exp(log_sp[j]) .* Sj
    end
    if any(!isfinite, S)
        return 0.0
    end
    eigs = eigvals(Symmetric(S))
    pos = filter(e -> e > 1e-10, eigs)
    return isempty(pos) ? 0.0 : sum(log, pos)
end

"""
    mp_laml(family, y, X_list, β, S, Sl, log_sp, param_offsets; Mp=0) → Float64

Compute the Laplace Approximate Marginal Likelihood (LAML):
    LAML = -NLL(β*) - 0.5*β*'Sβ* - 0.5*log|H| + 0.5*log|S+| - 0.5*Mp*log(2π)

LAML = -REML, so maximizing LAML is equivalent to minimizing REML.
Useful for model comparison (higher LAML = better fit).
"""
function mp_laml(family::MultiParameterFamily, y::AbstractVector,
                 X_list::Vector{Matrix{Float64}}, β::Vector{Float64},
                 S::Matrix{Float64}, Sl::Vector{Matrix{Float64}},
                 log_sp::Vector{Float64}, param_offsets::Vector{Int};
                 Mp::Int=0)
    K = nparams(family)
    n = length(y)
    p = length(β)
    ncols = deriv_ncols(K)

    # Compute NLL at β*
    η_list = _compute_eta(X_list, β, param_offsets, K)
    nll_val = nll_total(family, y, η_list)

    # Penalty at β*
    pen = 0.5 * dot(β, S * β)

    # Penalized Hessian: H = H0 + S
    derivs = Matrix{Float64}(undef, n, ncols)
    nll_derivs!(family, derivs, y, η_list)
    H0 = Matrix{Float64}(undef, p, p)
    assemble_hessian!(H0, derivs, X_list)
    H = H0 .+ S

    F = _safe_cholesky(Symmetric(H))
    if F === nothing
        return -Inf
    end
    logdetH = 2.0 * sum(log.(diag(F.L)))

    logdetS = _logdet_penalty(Sl, log_sp, p)

    laml = -nll_val - pen - 0.5 * logdetH + 0.5 * logdetS - 0.5 * Mp * log(2π)
    return isfinite(laml) ? laml : -Inf
end

# ============================================================================
# EFS outer optimization (Extended Fellner-Schall, Wood & Fasiolo 2017)
# ============================================================================

"""
    mp_efs_outer(family, y, X_list, Sl, β_init, log_sp_init, param_offsets, control)
    → (log_sp_opt, β_opt, reml_val, iterations)

EFS optimization of smoothing parameters for multi-parameter models.
Each outer iteration: 1 inner Newton solve + closed-form SP update.
Much faster than BFGS+FD which requires 2×nsp inner solves per outer step.
"""
function mp_efs_outer(family::MultiParameterFamily, y::AbstractVector,
                      X_list::Vector{Matrix{Float64}},
                      Sl::Vector{Matrix{Float64}},
                      β_init::Vector{Float64},
                      log_sp_init::Vector{Float64},
                      param_offsets::Vector{Int},
                      control::MPFitControl;
                      Mp::Int=0,
                      Ain = nothing,
                      bin = nothing,
                      Aeq = nothing,
                      beq = nothing)
    nsp = length(log_sp_init)
    if nsp == 0
        p = length(β_init)
        S = zeros(p, p)
        β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_init, S, control;
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)
        return Float64[], β_opt, nll_pen, 0
    end

    p = length(β_init)
    log_sp = copy(log_sp_init)
    β_current = copy(β_init)
    iterations = 0

    # Precompute penalty ranks
    pen_ranks = Float64[]
    for Sj in Sl
        eigs = eigvals(Symmetric(Sj))
        push!(pen_ranks, Float64(count(e -> e > 1e-10 * maximum(abs, eigs), eigs)))
    end

    for outer_iter in 1:control.outer_maxit
        iterations = outer_iter
        # Build total penalty
        S = zeros(p, p)
        for (j, Sj) in enumerate(Sl)
            S .+= exp(log_sp[j]) .* Sj
        end

        # Inner Newton solve
        β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_current, S, control;
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)

        # H is the penalized Hessian = H0 + S
        # For EFS we need A⁻¹ where A = H (the penalized Hessian)
        F = _safe_cholesky(Symmetric(H))
        if F === nothing
            if control.trace
                @info "Outer iteration $outer_iter: Cholesky failed, stopping"
            end
            break
        end
        Ainv = inv(F)

        # EFS update for each smoothing parameter
        log_sp_new = copy(log_sp)
        max_change = 0.0

        for j in 1:nsp
            λ = exp(log_sp[j])
            Sj = Sl[j]
            rank_j = pen_ranks[j]

            bSb = dot(β_opt, Sj * β_opt)
            trAS = tr(Ainv * Sj)

            # EFS formula: scale_est=1 for multi-parameter (no separate scale)
            a = max(0.0, rank_j / λ - trAS)

            if a > 0 && bSb > eps()
                r = a / bSb  # scale_est = 1 for multi-parameter
                log_sp_new[j] = clamp(log_sp[j] + log(max(r, 1e-15)), -15.0, 15.0)
            end

            max_change = max(max_change, abs(log_sp_new[j] - log_sp[j]))
        end

        if control.trace
            sp_str = join([@sprintf("%.4f", exp(s)) for s in log_sp_new], ", ")
            @info "Outer iteration $outer_iter: sp=[$sp_str], max_change=$(round(max_change, sigdigits=3))"
        end

        log_sp .= log_sp_new
        β_current .= β_opt

        if max_change < 1e-4
            if control.trace
                @info "Outer converged at iteration $outer_iter"
            end
            break
        end
    end

    # Compute final REML for return value
    S_final = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        S_final .+= exp(log_sp[j]) .* Sj
    end
    β_final, nll_pen_final, _, H_final, _ = mp_newton_inner(
        family, y, X_list, β_current, S_final, control;
        Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)

    F_final = _safe_cholesky(Symmetric(H_final))
    logdetH = F_final !== nothing ? 2.0 * sum(log.(diag(F_final.L))) : 0.0
    logdetS = _logdet_penalty(Sl, log_sp, p)
    reml_val = nll_pen_final + 0.5 * logdetH - 0.5 * logdetS + 0.5 * Mp * log(2π)

    return log_sp, β_final, reml_val, iterations
end

# Keep BFGS as fallback (legacy, not used by default)
function mp_bfgs_outer(family::MultiParameterFamily, y::AbstractVector,
                       X_list::Vector{Matrix{Float64}},
                       Sl::Vector{Matrix{Float64}},
                       β_init::Vector{Float64},
                       log_sp_init::Vector{Float64},
                       param_offsets::Vector{Int},
                       control::MPFitControl;
                       Mp::Int=0)
    nsp = length(log_sp_init)
    if nsp == 0
        p = length(β_init)
        S = zeros(p, p)
        β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_init, S, control)
        return Float64[], β_opt, nll_pen
    end

    log_sp = copy(log_sp_init)
    β_current = copy(β_init)

    reml_val, β_current, _ = mp_reml(log_sp, family, y, X_list, Sl, β_current,
                                      param_offsets, control; Mp=Mp)
    B = Matrix{Float64}(I, nsp, nsp)

    for outer_iter in 1:control.outer_maxit
        grad = _fd_reml_gradient(log_sp, family, y, X_list, Sl, β_current,
                                 param_offsets, control, reml_val; Mp=Mp)

        if maximum(abs, grad) < control.outer_tol
            break
        end

        direction = -(B * grad)

        step = 1.0
        log_sp_new = clamp.(log_sp .+ step .* direction, -30.0, 30.0)
        reml_new, β_new, _ = mp_reml(log_sp_new, family, y, X_list, Sl, β_current,
                                      param_offsets, control; Mp=Mp)

        for _ in 1:20
            if isfinite(reml_new) && reml_new < reml_val - 1e-4 * step * dot(grad, direction)
                break
            end
            step *= 0.5
            step < 1e-10 && break
            log_sp_new = clamp.(log_sp .+ step .* direction, -30.0, 30.0)
            reml_new, β_new, _ = mp_reml(log_sp_new, family, y, X_list, Sl, β_current,
                                          param_offsets, control; Mp=Mp)
        end

        if !isfinite(reml_new) || reml_new >= reml_val
            step = 0.01
            log_sp_new = clamp.(log_sp .- step .* grad, -30.0, 30.0)
            reml_new, β_new, _ = mp_reml(log_sp_new, family, y, X_list, Sl, β_current,
                                          param_offsets, control; Mp=Mp)
            (!isfinite(reml_new) || reml_new >= reml_val) && break
        end

        reml_rel_change = abs(reml_new - reml_val) / (abs(reml_val) + 1.0)
        sp_change = maximum(abs, log_sp_new .- log_sp)

        s_vec = log_sp_new .- log_sp
        grad_new = _fd_reml_gradient(log_sp_new, family, y, X_list, Sl, β_new,
                                     param_offsets, control, reml_new; Mp=Mp)
        y_vec = grad_new .- grad
        sy = dot(s_vec, y_vec)

        if sy > 1e-10
            ρ = 1.0 / sy
            I_mat = Matrix{Float64}(I, nsp, nsp)
            B = (I_mat - ρ * s_vec * y_vec') * B * (I_mat - ρ * y_vec * s_vec') + ρ * s_vec * s_vec'
        end

        log_sp .= log_sp_new
        β_current .= β_new
        reml_val = reml_new

        (reml_rel_change < 1e-8 && sp_change < 1e-4) && break
    end

    return log_sp, β_current, reml_val
end

function _fd_reml_gradient(log_sp, family, y, X_list, Sl, β, param_offsets, control, f0; Mp=0)
    nsp = length(log_sp)
    grad = Vector{Float64}(undef, nsp)
    h = 1e-3
    for i in 1:nsp
        log_sp_p = copy(log_sp)
        log_sp_m = copy(log_sp)
        log_sp_p[i] += h
        log_sp_m[i] -= h
        fp, _, _ = mp_reml(log_sp_p, family, y, X_list, Sl, β, param_offsets, control; Mp=Mp)
        fm, _, _ = mp_reml(log_sp_m, family, y, X_list, Sl, β, param_offsets, control; Mp=Mp)
        grad[i] = (fp - fm) / (2h)
    end
    return grad
end

# ============================================================================
# Covariance matrices
# ============================================================================

"""Compute Vp (posterior covariance) and Vc (corrected covariance)."""
function mp_covariance(family::MultiParameterFamily, y::AbstractVector,
                       X_list::Vector{Matrix{Float64}}, β::Vector{Float64},
                       S::Matrix{Float64}, param_offsets::Vector{Int})
    K = nparams(family)
    n = length(y)
    p = length(β)
    ncols = deriv_ncols(K)

    derivs = Matrix{Float64}(undef, n, ncols)
    η_list = _compute_eta(X_list, β, param_offsets, K)
    nll_derivs!(family, derivs, y, η_list)

    H0 = Matrix{Float64}(undef, p, p)
    assemble_hessian!(H0, derivs, X_list)

    H = H0 .+ S
    F = _safe_cholesky(Symmetric(H))
    if F !== nothing
        Vp = inv(F)
    else
        Vp = pinv(H)
    end

    # For now Vc = Vp (corrected covariance requires smoothing parameter uncertainty)
    Vc = copy(Vp)

    return Vp, Vc, H0
end

# ============================================================================
# User API: evgam()
# ============================================================================

"""
    EvgamControl

Control parameters for evgam fitting.
"""
const EvgamControl = MPFitControl

"""
    evgam_control(; kwargs...) → EvgamControl

Create control parameters for `evgam`. See `mp_control` for keyword arguments.
"""
const evgam_control = mp_control

"""
    evgam(formulas, data, family; control=mp_control(), sp=nothing, trace=false)

Fit a multi-parameter GAM. Alias for [`gamlss`](@ref) — any `MultiParameterFamily`
works here (GEV, GPD, EGPD, GaussianLS, GammaLS, etc.).

See [`gamlss`](@ref) for full documentation.

# Example
```julia
using GAM, DataFrames
# GEV model: location and log-scale depend on x, shape is constant
m = evgam(
    [@gam_formula(y ~ s(x, bs=:cr, k=10)),   # location μ
     @gam_formula(y ~ s(x, bs=:cr, k=8)),    # log-scale ψ
     @gam_formula(y ~ 1)],                     # shape ξ
    df, GEVFamily()
)
```
"""
function evgam(formulas, data, family::MultiParameterFamily;
               control::MPFitControl=mp_control(),
               sp=nothing, trace::Bool=control.trace)
    ctrl = MPFitControl(control.inner_maxit, control.inner_tol,
                        control.outer_maxit, control.outer_tol,
                        control.step_max, trace)
    K = nparams(family)

    # Handle single formula → replicate for all parameters
    if formulas isa FormulaTerm || formulas isa GamFormula
        formulas = fill(formulas, K)
    end
    length(formulas) == K || throw(ArgumentError(
        "Expected $K formulas for $(typeof(family)), got $(length(formulas))"))

    # Extract response from first formula
    cols = Tables.columntable(data)
    y = _extract_response(formulas[1], cols)
    n = length(y)

    # Build design matrices and smooth terms for each parameter
    X_list = Vector{Matrix{Float64}}(undef, K)
    smooths_list = Vector{Vector{ConstructedSmooth}}(undef, K)
    offset = 0

    for k in 1:K
        Xk, smoothsk = _build_design_matrix(formulas[k], cols, n, offset)
        X_list[k] = Xk
        smooths_list[k] = smoothsk
        offset += size(Xk, 2)
    end

    p = offset  # total coefficients
    param_offsets = cumsum([0; [size(X, 2) for X in X_list]])

    # Build penalty matrices
    Sl = build_penalty_matrices(smooths_list, param_offsets)
    nsp = length(Sl)
    Ain, bin, Aeq, beq = _global_linear_constraints(smooths_list, p)

    # Count null space dimension for REML constant
    Mp = sum(1 + sum(sm.null_dim for sm in smooths; init=0) for smooths in smooths_list)

    # Initial values
    η_init = initial_eta(family, y)
    β_init = zeros(p)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        pk = e - s + 1
        # Set intercept to mean of initial η, rest to 0
        β_init[s] = mean(η_init[k])
    end

    # Smoothing parameters — use Hessian-based initialization (matching R evgam)
    if sp !== nothing
        log_sp = Float64.(sp)
    else
        log_sp = _init_log_sp_hessian(family, y, X_list, Sl, β_init, param_offsets, nsp)
    end

    # Fit
    if sp !== nothing || nsp == 0
        # Fixed smoothing parameters — just inner Newton
        S = zeros(p, p)
        for (j, Sj) in enumerate(Sl)
            S .+= exp(log_sp[j]) .* Sj
        end
        β_opt, nll_pen, g, H, conv = mp_newton_inner(family, y, X_list, β_init, S, ctrl;
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)
        reml_val = nll_pen
        iterations = 0
    else
        # Estimate smoothing parameters via EFS
        log_sp, β_opt, reml_val, iterations = mp_efs_outer(family, y, X_list, Sl, β_init,
            log_sp, param_offsets, ctrl;
            Mp=Mp, Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)
        conv = true
    end

    # Final fitted values
    η_fit = _compute_eta(X_list, β_opt, param_offsets, K)

    # Build final penalty for covariance computation
    S = zeros(p, p)
    for (j, Sj) in enumerate(Sl)
        if j <= length(log_sp)
            S .+= exp(log_sp[j]) .* Sj
        end
    end

    # Covariance
    Vp, Vc, H0 = mp_covariance(family, y, X_list, β_opt, S, param_offsets)

    # EDF
    edf = diag(Vp * H0)

    # NLL at optimum
    nll_val = nll_total(family, y, η_fit)

    # idpars
    idpars = Vector{Int}(undef, p)
    for k in 1:K
        s = param_offsets[k] + 1
        e = param_offsets[k + 1]
        idpars[s:e] .= k
    end

    # LAML for model comparison
    laml = mp_laml(family, y, X_list, β_opt, S, Sl, log_sp, param_offsets; Mp=Mp)

    return MultiParameterModel(
        family, β_opt, η_fit, X_list, smooths_list, log_sp,
        edf, Vp, Vc, nll_val, reml_val, laml, y, n, conv, iterations, idpars, param_offsets
    )
end

# ============================================================================
# Internal helpers for design matrix construction
# ============================================================================

function _extract_response(formula, cols)
    if formula isa GamFormula
        resp_sym = formula.response
    else
        lhs = formula.lhs
        resp_sym = lhs isa Term ? lhs.sym : Symbol(string(lhs))
    end
    return Float64.(Tables.getcolumn(cols, resp_sym))
end

function _build_design_matrix(formula, cols, n, global_offset)
    if formula isa GamFormula
        return _build_gam_design(formula, cols, n, global_offset)
    else
        return _build_parametric_design(formula, cols, n, global_offset)
    end
end

function _build_gam_design(gf::GamFormula, cols, n, global_offset)
    # Use existing setup_gam infrastructure to build design matrix + smooths
    # Build intercept + parametric + smooth terms
    smooth_specs = gf.smooth_specs

    # Parametric part: intercept
    X_parts = Matrix{Float64}[ones(n, 1)]
    col_offset = global_offset + 1  # intercept occupies 1 column

    # Smooth terms — smooth_construct already absorbs constraints
    smooths = ConstructedSmooth[]
    for spec in smooth_specs
        cs = smooth_construct(spec, cols)

        # Update parameter indices to global
        k_eff = size(cs.X, 2)
        cs.first_para = col_offset + 1
        cs.last_para = col_offset + k_eff
        col_offset += k_eff

        push!(X_parts, cs.X)
        push!(smooths, cs)
    end

    X = hcat(X_parts...)
    return X, smooths
end

function _build_parametric_design(formula::FormulaTerm, cols, n, global_offset)
    # Use setup_gam to detect smooth function terms (cr(), tp(), etc.) in @formula
    rhs_terms = _flatten_rhs(formula.rhs)
    has_smooth = any(t -> (t isa AppliedSmoothTerm || t isa SmoothTerm ||
        (t isa StatsModels.FunctionTerm && _is_smooth_function(t.f))),
        rhs_terms)

    if has_smooth
        # Delegate to setup_gam for proper smooth detection, then reindex
        # cols is already a column table from Tables.columntable
        y, X_full, X_para, smooths, n_parametric = setup_gam(formula, cols)
        # Reindex smooths to global offset
        col_offset = global_offset + n_parametric
        for sm in smooths
            k = size(sm.X, 2)
            sm.first_para = col_offset + 1
            sm.last_para = col_offset + k
            col_offset += k
        end
        return X_full, smooths
    else
        # Pure parametric formula (intercept only or simple terms)
        X = ones(n, 1)
        return X, ConstructedSmooth[]
    end
end
