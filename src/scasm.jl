# Linear-constraint smooth fitting (`bs=:sc`, `bs=:scad`, and `pc` constraints).
#
# This backend targets mgcv::scasm-style functionality, but it is intentionally
# implemented as a Julia-native constrained PIRLS / QP solver rather than as a
# line-for-line port of mgcv's `pcls()` warm-start algorithm.

function scasm_control(;
    epsilon::Real = 1e-7,
    maxit::Int = 200,
    outer_maxit::Int = 200,
    trace::Bool = false,
    gamma::Real = 1.0,
)
    return gam_control(
        epsilon = epsilon,
        maxit = maxit,
        outer_maxit = outer_maxit,
        trace = trace,
        gamma = gamma,
        sp_optimizer = :efs,
    )
end

function _scasm_objective_method(method::Symbol)
    if method == :UBRE
        throw(ArgumentError(
            "method=:UBRE is not yet supported for linear-constraint fits; use :GCV, :REML, or :ML.",
        ))
    end
    return method
end

function _is_feasible(beta::AbstractVector{<:Real}, Ain, bin, Aeq, beq; tol::Float64 = 1e-6)
    if Ain !== nothing && bin !== nothing && size(Ain, 1) > 0
        if minimum(Ain * beta .- bin) < -tol
            return false
        end
    end
    if Aeq !== nothing && beq !== nothing && size(Aeq, 1) > 0
        if maximum(abs, Aeq * beta .- beq) > tol
            return false
        end
    end
    return true
end

function _global_linear_constraints(smooths::AbstractVector{<:ConstructedSmooth}, p::Int; offset::Int = 0)
    Ain_global = nothing
    bin_global = nothing
    Aeq_global = nothing
    beq_global = nothing

    for sm in smooths
        idx = (sm.first_para - offset):(sm.last_para - offset)
        if sm.Ain !== nothing && sm.bin !== nothing && size(sm.Ain, 1) > 0
            Ablk = zeros(size(sm.Ain, 1), p)
            Ablk[:, idx] .= sm.Ain
            Ain_global, bin_global = _append_constraint_block(Ain_global, bin_global, Ablk, sm.bin)
        end
        if sm.Aeq !== nothing && sm.beq !== nothing && size(sm.Aeq, 1) > 0
            Ablk = zeros(size(sm.Aeq, 1), p)
            Ablk[:, idx] .= sm.Aeq
            Aeq_global, beq_global = _append_constraint_block(Aeq_global, beq_global, Ablk, sm.beq)
        end
    end

    return Ain_global, bin_global, Aeq_global, beq_global
end

function _global_linear_constraints(smooths_list::AbstractVector{<:AbstractVector{<:ConstructedSmooth}}, p::Int)
    Ain_global = nothing
    bin_global = nothing
    Aeq_global = nothing
    beq_global = nothing
    for smooths in smooths_list
        Ain_k, bin_k, Aeq_k, beq_k = _global_linear_constraints(smooths, p)
        Ain_global, bin_global = _append_constraint_block(Ain_global, bin_global, Ain_k, bin_k)
        Aeq_global, beq_global = _append_constraint_block(Aeq_global, beq_global, Aeq_k, beq_k)
    end
    return Ain_global, bin_global, Aeq_global, beq_global
end

function _per_param_linear_constraints(smooths_list::AbstractVector{<:AbstractVector{<:ConstructedSmooth}},
                                       param_offsets::AbstractVector{<:Integer})
    K = length(smooths_list)
    Ain_list = Vector{Union{Matrix{Float64}, Nothing}}(undef, K)
    bin_list = Vector{Union{Vector{Float64}, Nothing}}(undef, K)
    Aeq_list = Vector{Union{Matrix{Float64}, Nothing}}(undef, K)
    beq_list = Vector{Union{Vector{Float64}, Nothing}}(undef, K)
    for k in 1:K
        pk = param_offsets[k + 1] - param_offsets[k]
        Ain_k, bin_k, Aeq_k, beq_k = _global_linear_constraints(
            smooths_list[k], pk; offset = param_offsets[k])
        Ain_list[k] = Ain_k
        bin_list[k] = bin_k
        Aeq_list[k] = Aeq_k
        beq_list[k] = beq_k
    end
    return Ain_list, bin_list, Aeq_list, beq_list
end

function _regularize_qp_hessian(H::Matrix{Float64})
    p = size(H, 1)
    P = 0.5 .* (H .+ H')
    _safe_cholesky(Symmetric(P)) !== nothing && return P, 0.0

    diag_scale = max(opnorm(P, Inf), maximum(abs, diag(P)), 1.0)
    last_ridge = 0.0
    for ridge_mult in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
        ridge = diag_scale * ridge_mult
        P_try = copy(P)
        @inbounds for i in 1:p
            P_try[i, i] += ridge
        end
        _safe_cholesky(Symmetric(P_try)) !== nothing && return P_try, ridge
        last_ridge = ridge
    end

    P_fallback = copy(P)
    fallback_ridge = max(last_ridge, diag_scale)
    @inbounds for i in 1:p
        P_fallback[i, i] += fallback_ridge
    end
    return P_fallback, fallback_ridge
end

function _solve_constrained_qp(H::Matrix{Float64}, f::Vector{Float64}, Ain, bin, Aeq, beq;
                               warm_start::Union{Vector{Float64}, Nothing} = nothing,
                               eps_abs::Float64 = 1e-7,
                               eps_rel::Float64 = 1e-7)
    p = size(H, 1)

    if (Ain === nothing || size(Ain, 1) == 0) && (Aeq === nothing || size(Aeq, 1) == 0)
        H_reg = copy(H)
        @inbounds for i in 1:p
            H_reg[i, i] += 1e-8
        end
        return cholesky(Symmetric(H_reg)) \ f
    end

    P_base, base_ridge = _regularize_qp_hessian(H)
    obj_scale = max(opnorm(P_base, Inf), maximum(abs, f), 1.0)
    P_base ./= obj_scale
    q = -f ./ obj_scale

    A_blocks = Matrix{Float64}[]
    l = Float64[]
    u = Float64[]

    if Aeq !== nothing && beq !== nothing && size(Aeq, 1) > 0
        push!(A_blocks, Aeq)
        append!(l, beq)
        append!(u, beq)
    end
    if Ain !== nothing && bin !== nothing && size(Ain, 1) > 0
        push!(A_blocks, Ain)
        append!(l, bin)
        append!(u, fill(Inf, length(bin)))
    end

    A = sparse(vcat(A_blocks...))
    last_status = ""
    last_ridge = base_ridge
    for ridge in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
        P = copy(P_base)
        if ridge > 0
            @inbounds for i in 1:p
                P[i, i] += ridge
            end
        end
        last_ridge = base_ridge + ridge * obj_scale

        try
            model = OSQP.Model()
            result = redirect_stderr(devnull) do
                OSQP.setup!(model;
                    P = sparse(Symmetric(P)),
                    q = q,
                    A = A,
                    l = l,
                    u = u,
                    verbose = false,
                    polish = true,
                    eps_abs = eps_abs,
                    eps_rel = eps_rel,
                    max_iter = 20000,
                    adaptive_rho = true,
                    scaled_termination = true,
                )
                if warm_start !== nothing
                    OSQP.warm_start!(model; x = warm_start)
                end
                OSQP.solve!(model)
            end
            status = lowercase(String(result.info.status))
            if occursin("solved", status)
                return Vector{Float64}(result.x)
            end
            last_status = String(result.info.status)
        catch err
            last_status = sprint(showerror, err)
        end
    end
    throw(ErrorException("OSQP failed with status $last_status after diagonal regularization up to $last_ridge"))
end

function scasm_pirls(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution, link::GLM.Link;
    Ain = nothing,
    bin = nothing,
    Aeq = nothing,
    beq = nothing,
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::GamControl = scasm_control())

    n, p = size(X)

    beta = zeros(p)
    beta_new = zeros(p)
    eta = zeros(n)
    eta_new = zeros(n)
    mu = zeros(n)
    mu_new = zeros(n)
    dmu_deta = zeros(n)
    w = zeros(n)
    z = zeros(n)
    XtWz = zeros(p)
    A = zeros(p, p)
    Xw = similar(X)
    wz_buf = zeros(n)
    penalty_buf = zeros(p)

    if start !== nothing
        copyto!(beta, start)
        mul!(eta, X, beta)
        eta .+= offset
    else
        @inbounds for i in 1:n
            mu[i] = _mustart(family, y[i], weights[i])
            eta[i] = GLM.linkfun(link, mu[i])
        end
    end

    @inbounds for i in 1:n
        mu[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta[i]))
    end

    converged = false
    n_iter = 0
    mul!(penalty_buf, S_total, beta)
    pdev_old = _deviance(family, y, mu, weights) + dot(beta, penalty_buf)
    feasible_old = _is_feasible(beta, Ain, bin, Aeq, beq)

    for iter in 1:control.maxit
        n_iter = iter

        @inbounds for i in 1:n
            dm = GLM.mueta(link, eta[i])
            dmu_deta[i] = dm
            vm = _variance_scalar(family, mu[i])
            w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
            z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / dm
        end

        _build_penalized_system!(A, XtWz, X, w, z, S_total, p, n, Xw, wz_buf)
        beta_candidate = _solve_constrained_qp(A, XtWz, Ain, bin, Aeq, beq;
            warm_start = iter == 1 ? start : beta,
            eps_abs = max(control.epsilon, 1e-8),
            eps_rel = max(control.epsilon, 1e-8))
        copyto!(beta_new, beta_candidate)

        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        @inbounds for i in 1:n
            mu_new[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_new[i]))
        end

        dev_new = _deviance(family, y, mu_new, weights)
        mul!(penalty_buf, S_total, beta_new)
        penalty_new = dot(beta_new, penalty_buf)
        pdev_new = dev_new + penalty_new

        div_thresh = 10.0 * (0.1 + abs(pdev_old)) * sqrt(eps())
        accepted_step = (pdev_new - pdev_old <= div_thresh) &&
                        _is_feasible(beta_new, Ain, bin, Aeq, beq)
        if iter > 1 && feasible_old && pdev_new - pdev_old > div_thresh
            beta_trial = copy(beta_new)
            eta_trial = similar(eta_new)
            mu_trial = similar(mu_new)
            for _ in 1:50
                accepted_step && break
                beta_trial .= 0.5 .* beta .+ 0.5 .* beta_trial
                mul!(eta_trial, X, beta_trial)
                eta_trial .+= offset
                @inbounds for i in 1:n
                    mu_trial[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_trial[i]))
                end
                dev_trial = _deviance(family, y, mu_trial, weights)
                mul!(penalty_buf, S_total, beta_trial)
                pdev_trial = dev_trial + dot(beta_trial, penalty_buf)
                if pdev_trial - pdev_old <= div_thresh && _is_feasible(beta_trial, Ain, bin, Aeq, beq)
                    copyto!(beta_new, beta_trial)
                    copyto!(eta_new, eta_trial)
                    copyto!(mu_new, mu_trial)
                    dev_new = dev_trial
                    pdev_new = pdev_trial
                    accepted_step = true
                end
            end

            if !accepted_step
                copyto!(beta_new, beta)
                copyto!(eta_new, eta)
                copyto!(mu_new, mu)
                dev_new = _deviance(family, y, mu, weights)
                pdev_new = pdev_old
            end
        end

        scale_check = _needs_scale_estimate(family) ? dev_new / max(n - p, 1) : 1.0
        crit = abs(pdev_new - pdev_old) / (abs(scale_check) + abs(pdev_new))

        copyto!(beta, beta_new)
        copyto!(eta, eta_new)
        copyto!(mu, mu_new)
        pdev_old = pdev_new
        feasible_old = _is_feasible(beta, Ain, bin, Aeq, beq)

        if crit < control.epsilon
            converged = true
            break
        end
    end

    dev_final = _deviance(family, y, mu, weights)
    @inbounds for i in 1:n
        dm = GLM.mueta(link, eta[i])
        vm = _variance_scalar(family, mu[i])
        w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
    end

    pearson = 0.0
    @inbounds for i in 1:n
        vm = _variance_scalar(family, mu[i])
        pearson += weights[i] * (y[i] - mu[i])^2 / max(vm, eps())
    end

    _build_XtWX_plus_S!(A, X, w, S_total, p, n, Xw)
    A_chol_final = try
        cholesky(Symmetric(A))
    catch
        A_reg = copy(A)
        @inbounds for i in 1:p
            A_reg[i, i] += 1e-8
        end
        cholesky(Symmetric(A_reg))
    end

    XtWX = similar(A)
    @inbounds for j in 1:p, k in 1:p
        XtWX[j, k] = A[j, k] - S_total[j, k]
    end
    edf_vec, hat_diag = penalty_edf(X, w, S_total; XtWX = XtWX, A_chol = A_chol_final)
    R = Matrix(A_chol_final.U)

    return PirlsResult(
        beta, mu, eta, w, dev_final, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end

function scasm_outer_iteration(
    X::Matrix{Float64},
    y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::UnivariateDistribution,
    link::GLM.Link;
    Ain = nothing,
    bin = nothing,
    Aeq = nothing,
    beq = nothing,
    method::Symbol = :REML,
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    control::GamControl = scasm_control(),
    start::Union{Vector{Float64}, Nothing} = nothing,
)
    n, p = size(X)
    n_sp = length(penalty.sp)

    if n_sp == 0
        S_total = zeros(p, p)
        result = scasm_pirls(X, y, S_total, family, link;
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq,
            weights = weights, offset = offset, start = start, control = control)
        return Float64[], result
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing
    Xw_buf = similar(X)
    A_buf = zeros(p, p)
    A_buf_copy = zeros(p, p)
    Ainv_buf = zeros(p, p)
    S_total = zeros(p, p)
    efs_mult = 1.0

    for outer_iter in 1:control.outer_maxit
        total_penalty!(S_total, penalty, log_sp, p)
        result = scasm_pirls(X, y, S_total, family, link;
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq,
            weights = weights, offset = offset,
            start = prev_result === nothing ? start : prev_result.coefficients,
            control = control)

        beta = result.coefficients
        w = result.working_weights
        edf_total = sum(result.edf_vec)
        scale_est = _needs_scale_estimate(family) ? max(result.pearson / (n - edf_total), 1e-10) : 1.0

        _build_XtWX_plus_S!(A_buf, X, w, S_total, p, n, Xw_buf)
        copyto!(A_buf_copy, A_buf)
        A_chol = try
            cholesky!(Symmetric(A_buf_copy))
        catch
            A_reg = copy(A_buf)
            @inbounds for i in 1:p
                A_reg[i, i] += 1e-8
            end
            cholesky(Symmetric(A_reg))
        end
        fill!(Ainv_buf, 0.0)
        @inbounds for i in 1:p
            Ainv_buf[i, i] = 1.0
        end
        ldiv!(A_chol, Ainv_buf)
        Ainv = Ainv_buf

        log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty, scale_est, efs_mult)
        max_change = maximum(abs.(log_sp_new .- log_sp))

        if outer_iter > 1 && max_change > control.epsilon
            score_old = _efs_reml_score(X, y, log_sp, penalty, family, link, weights,
                result, method, scale_est, control.gamma, n, p)
            score_new = _efs_reml_score(X, y, log_sp_new, penalty, family, link, weights,
                result, method, scale_est, control.gamma, n, p)

            if score_new > score_old + control.epsilon * abs(score_old)
                for _halve in 1:4
                    efs_mult *= 0.5
                    log_sp_new = _efs_sp_update(log_sp, beta, Ainv, penalty,
                        scale_est, efs_mult)
                    score_new = _efs_reml_score(X, y, log_sp_new, penalty, family, link, weights,
                        result, method, scale_est, control.gamma, n, p)
                    score_new <= score_old + control.epsilon * abs(score_old) &&
                        break
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
        prev_result = result
        if max_change < control.epsilon * 10
            break
        end
    end

    total_penalty!(S_total, penalty, log_sp, p)
    final_result = scasm_pirls(X, y, S_total, family, link;
        Ain = Ain, bin = bin, Aeq = Aeq, beq = beq,
        weights = weights, offset = offset,
        start = prev_result === nothing ? start : prev_result.coefficients,
        control = control)
    return log_sp, final_result
end

function _fit_scasm(y, X, smooths, n_parametric, f, data, family, link, method, weights, control;
                    start::Union{AbstractVector{<:Real}, Nothing} = nothing,
                    offset = nothing)
    n, p = size(X)
    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch("weights length $(length(wts)) ≠ data length $n"))
    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch("offset length $(length(off)) ≠ data length $n"))
    control.sp_optimizer == :efs || throw(ArgumentError(
        "Linear-constraint fits currently support only control.sp_optimizer = :efs."
    ))
    method_eff = _scasm_objective_method(method)
    start_vec = start === nothing ? nothing : Float64.(start)
    if start_vec !== nothing && length(start_vec) != p
        throw(DimensionMismatch("start length $(length(start_vec)) ≠ coefficient length $p"))
    end

    penalty = setup_penalties(smooths, n_parametric)
    _initial_sp(X, penalty)
    Ain, bin, Aeq, beq = _global_linear_constraints(smooths, p)

    log_sp, result = scasm_outer_iteration(X, y, smooths, penalty, family, link;
        Ain = Ain, bin = bin, Aeq = Aeq, beq = beq,
        method = method_eff, weights = wts, offset = off, control = control,
        start = start_vec)

    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X' * Diagonal(result.working_weights) * X
    A = XtWX + S_total
    A_chol = try
        cholesky(Symmetric(A))
    catch
        A_reg = copy(A)
        @inbounds for i in 1:p
            A_reg[i, i] += 1e-8
        end
        cholesky(Symmetric(A_reg))
    end
    Vp = inv(A_chol)
    F = Vp * XtWX
    Ve = Symmetric(F * Vp * F') |> Matrix

    if _needs_scale_estimate(family)
        scale_est = result.pearson / (n - edf_total_val)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    null_dev = _null_deviance(family, y, wts)
    reml_val, _ = reml_score(X, y, penalty, log_sp, family, link,
        wts, result; method = method_eff, gamma = control.gamma)

    return GamModel(
        f,
        y, X,
        result.coefficients,
        result.fitted_values,
        result.linear_predictor,
        wts,
        family, link,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total_val,
        scale_est,
        result.deviance,
        null_dev,
        reml_val,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged,
        0,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end

function scasm(f::FormulaTerm, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = scasm_control())
    Base.depwarn(
        "scasm(...) is deprecated; use gam(...). Linear-constraint bases (:sc/:scad/pc) " *
        "and SCAM shape-constrained bases dispatch automatically through gam().",
        :scasm,
    )
    return gam(f, data; family = family, link = link, method = method,
        weights = weights, control = control)
end

function scasm(gf::GamFormula, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = scasm_control())
    Base.depwarn(
        "scasm(...) is deprecated; use gam(...). Linear-constraint bases (:sc/:scad/pc) " *
        "and SCAM shape-constrained bases dispatch automatically through gam().",
        :scasm,
    )
    return gam(gf, data; family = family, link = link, method = method,
        weights = weights, control = control)
end
