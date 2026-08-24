# Penalized IRLS for Extended Families
#
# Modified P-IRLS that works with ExtendedFamily types, including
# estimation of extra parameters (NB theta, Beta phi) during iteration.

"""
    _dd_stationary_polish!(beta, eta, mu, X, y, offset, weights, S_total,
                           family, link, control) -> Bool

Verify stationarity of the penalized deviance F(β) = D(β) + β'Sβ at the
current iterate and, if the gradient is not negligible, run damped Newton
steps on F using the *unclamped* deviance derivatives until stationarity
(max|∇F| ≤ tol·(1 + |F|)) or the iteration budget is exhausted. Returns
whether stationarity was achieved. Mutates `beta`, `eta`, `mu` in place.

This exists because the clamped IRLS working system (weight floor at eps,
±40 pseudodata clamp) discards the gradient contribution of low-curvature
observations, so a deviance plateau does not imply a minimum — most
acutely for the ELF (quantile) family, whose curvature vanishes outside a
narrow sigmoid band while its gradient stays constant.
"""
function _dd_stationary_polish!(beta::Vector{Float64}, eta::Vector{Float64},
    mu::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64},
    offset::Vector{Float64}, weights::Vector{Float64},
    S_total::Matrix{Float64}, family::ExtendedFamily, link::GLM.Link,
    control::GamControl; maxit::Int = 100, tol_scale::Real = 1.0)

    n, p = size(X)
    tol_g = max(sqrt(control.epsilon), 1e-6) * tol_scale

    Sb = S_total * beta
    f_val = _deviance(family, y, mu, weights) + dot(beta, Sb)

    wnewt = Vector{Float64}(undef, n)
    geta = Vector{Float64}(undef, n)

    for _ in 1:maxit
        dd = _family_Dd(family, y, mu, weights; level = 0)
        Dmu = dd[:Dmu]
        Dmu2 = dd[:Dmu2]

        @inbounds for i in 1:n
            dme = GLM.mueta(link, eta[i])
            gi = Dmu[i] * dme
            ci = Dmu2[i] * dme^2 + Dmu[i] * _d2mu_deta2(link, mu[i], eta[i])
            geta[i] = isfinite(gi) ? gi : 0.0
            wnewt[i] = isfinite(ci) ? max(ci, 0.0) : 0.0
        end

        gvec = X' * geta .+ 2.0 .* Sb
        gmax = maximum(abs, gvec)
        gmax <= tol_g * (1.0 + abs(f_val)) && return true

        # Newton system: PSD-floored curvature + penalty, ridge-protected
        H = X' * Diagonal(wnewt) * X .+ 2.0 .* S_total
        F = nothing
        base = max(1e-8 * maximum(abs, diag(H)), 1e-10)
        for attempt in 0:8
            Ht = copy(H)
            @inbounds for j in 1:p
                Ht[j, j] += base * 10.0^attempt
            end
            F = try
                cholesky(Symmetric(Ht))
            catch
                nothing
            end
            F !== nothing && break
        end
        F === nothing && return false

        δ = F \ gvec
        step = 1.0
        improved = false
        beta_t = similar(beta)
        eta_t = similar(eta)
        mu_t = similar(mu)
        for _halve in 1:30
            @inbounds for j in 1:p
                beta_t[j] = beta[j] - step * δ[j]
            end
            mul!(eta_t, X, beta_t)
            eta_t .+= offset
            @inbounds for i in 1:n
                mu_t[i] = _clamp_mu(family, GLM.linkinv(link, eta_t[i]))
            end
            f_t = _deviance(family, y, mu_t, weights) +
                  dot(beta_t, S_total * beta_t)
            if isfinite(f_t) && f_t < f_val - 1e-12 * (abs(f_val) + 1.0)
                copyto!(beta, beta_t)
                copyto!(eta, eta_t)
                copyto!(mu, mu_t)
                f_val = f_t
                Sb = S_total * beta
                improved = true
                break
            end
            step *= 0.5
        end
        # No descent available along the Newton direction: accept as
        # (approximately) stationary only under a looser tolerance.
        improved || return gmax <= 10.0 * tol_g * (1.0 + abs(f_val))
    end
    return false
end

"""
    pirls_extended(X, y, S_total, family::ExtendedFamily, link::GLM.Link;
        weights, offset, start, control)

Run penalized IRLS for an extended family. Similar to `pirls()` but
dispatches variance/deviance/link through ExtendedFamily methods and
periodically estimates extra parameters (theta, phi, etc.).
"""
function pirls_extended(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::ExtendedFamily,
    link::GLM.Link;
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::GamControl = gam_control(),
    Ain = nothing,
    bin = nothing,
    Aeq = nothing,
    beq = nothing,
    polish_maxit::Int = 100,
    polish_tol_scale::Real = 1.0)

    n, p = size(X)

    # Initialize
    if start !== nothing
        beta = copy(start)
        eta = X * beta .+ offset
    else
        mu_init = _initialize_mu(family, y)
        eta = GLM.linkfun.(Ref(link), mu_init)
        beta = zeros(p)
        beta[1] = mean(eta)
        eta .= X * beta .+ offset
    end

    mu = GLM.linkinv.(Ref(link), eta)
    mu .= _clamp_mu(family, mu)
    dev_old = _deviance(family, y, mu, weights)

    converged = false
    n_iter = 0
    feasible_old = _is_feasible(beta, Ain, bin, Aeq, beq)

    # Pre-allocate working buffers
    dmu_deta = Vector{Float64}(undef, n)
    var_mu = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    Xw = similar(X)           # n×p buffer for √w-scaled X
    wz_buf = similar(z)       # n-vector buffer for √w-scaled z
    XtWX = Matrix{Float64}(undef, p, p)
    A = Matrix{Float64}(undef, p, p)
    beta_new = Vector{Float64}(undef, p)
    beta_step = Vector{Float64}(undef, p)
    eta_new = Vector{Float64}(undef, n)
    mu_new = Vector{Float64}(undef, n)
    rhs = Vector{Float64}(undef, p)

    # Check if family provides Dd-based working quantities (like R's gam.fit3)
    has_Dd = _has_Dd(family)

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response
        if has_Dd
            # Use the extended-family deviance derivatives to form the
            # working response/weights from a second-order expansion in η.
            dd = _family_Dd(family, y, mu, weights; level=0)
            Dmu = dd[:Dmu]    # gradient of deviance w.r.t. mu
            Dmu2 = dd[:Dmu2]  # observed curvature of deviance w.r.t. mu

            dmu_deta .= GLM.mueta.(Ref(link), eta)

            @inbounds for i in 1:n
                Deta_i = Dmu[i] * dmu_deta[i]
                Deta2_i = Dmu2[i] * dmu_deta[i]^2 + Dmu[i] * _d2mu_deta2(link, mu[i], eta[i])

                if !(isfinite(Deta_i) && isfinite(Deta2_i))
                    w[i] = eps()
                    z[i] = eta[i] - offset[i]
                    continue
                end

                # Note the asymmetry when Deta2_i < 0: the weight is floored at
                # eps() (Fisher-like) while the Newton step in z keeps the raw
                # (clamped) ratio — a negative-curvature observation pushes the
                # working response but carries ~zero weight.
                denom = abs(Deta2_i) > eps() ? Deta2_i : copysign(eps(), Deta2_i == 0.0 ? 1.0 : Deta2_i)
                w[i] = clamp(0.5 * Deta2_i, eps(), 1e10)
                z[i] = eta[i] - offset[i] - clamp(Deta_i / denom, -40.0, 40.0)
            end
        else
            # Fallback: standard IRLS working weights
            dmu_deta .= GLM.mueta.(Ref(link), eta)
            var_mu .= _variance(family, mu)

            @inbounds for i in 1:n
                w[i] = clamp(weights[i] * dmu_deta[i]^2 / max(var_mu[i], eps()), eps(), 1e10)
                z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / dmu_deta[i]
            end
        end

        _build_penalized_system!(A, rhs, X, w, z, S_total, p, n, Xw, wz_buf)
        if (Ain === nothing || size(Ain, 1) == 0) && (Aeq === nothing || size(Aeq, 1) == 0)
            A_chol = cholesky(Symmetric(A))
            ldiv!(beta_new, A_chol, rhs)
        else
            beta_new .= _solve_constrained_qp(A, rhs, Ain, bin, Aeq, beq;
                warm_start = iter == 1 ? start : beta,
                eps_abs = max(control.epsilon, 1e-8),
                eps_rel = max(control.epsilon, 1e-8))
        end

        # Update: eta_new = X * beta_new + offset
        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        @inbounds for i in 1:n
            mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
        end
        dev_new = _deviance(family, y, mu_new, weights)

        # Step halving if deviance increased
        step_factor = 1.0
        accepted_step = isfinite(dev_new) &&
                        dev_new <= dev_old + control.epsilon * abs(dev_old) &&
                        _is_feasible(beta_new, Ain, bin, Aeq, beq)
        if !((Ain !== nothing && size(Ain, 1) > 0) || (Aeq !== nothing && size(Aeq, 1) > 0)) || feasible_old
            for _ in 1:25
                accepted_step && break
                step_factor *= 0.5
                if step_factor < 1e-8
                    break
                end
                @inbounds for j in 1:p
                    beta_step[j] = beta[j] + step_factor * (beta_new[j] - beta[j])
                end
                if !_is_feasible(beta_step, Ain, bin, Aeq, beq)
                    continue
                end
                mul!(eta_new, X, beta_step)
                eta_new .+= offset
                @inbounds for i in 1:n
                    mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
                end
                dev_new = _deviance(family, y, mu_new, weights)
                accepted_step = isfinite(dev_new) &&
                                dev_new <= dev_old + control.epsilon * abs(dev_old)
            end

            if accepted_step && step_factor < 1.0
                @inbounds for j in 1:p
                    beta_new[j] = beta[j] + step_factor * (beta_new[j] - beta[j])
                end
                mul!(eta_new, X, beta_new)
                eta_new .+= offset
                @inbounds for i in 1:n
                    mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
                end
                dev_new = _deviance(family, y, mu_new, weights)
            elseif !accepted_step
                copyto!(beta_new, beta)
                copyto!(eta_new, eta)
                copyto!(mu_new, mu)
                dev_new = dev_old
            end
        end

        beta .= beta_new
        eta .= eta_new
        mu .= mu_new
        feasible_old = _is_feasible(beta, Ain, bin, Aeq, beq)

        # Estimate extra parameter periodically (every 3 iterations after burn-in).
        # Scale uses n − p (not n − edf: the EDF is only available after the
        # final factorization), which is conservative since p ≥ edf.
        if iter >= 3 && iter % 3 == 0 && _has_extra_param(family)
            scale = _estimates_scale(family) ? max(dev_new / (n - p), 1e-10) : 1.0
            estimate_theta!(family, y, mu, weights, scale)
            dev_new = _deviance(family, y, mu, weights)
        end

        # Convergence check
        crit = abs(dev_new - dev_old) / (abs(dev_new) + 0.1)
        dev_old = dev_new

        if crit < control.epsilon && feasible_old
            # For Dd families the working system clamps low-curvature
            # observations (weight floor + ±40 pseudodata clamp), so the
            # deviance can plateau at a non-stationary point (the round-4
            # ELF defect: far-from-quantile observations carry a constant
            # gradient but ~zero curvature, and the clamped IRLS drops
            # their pull). Require stationarity of the penalized deviance
            # before declaring convergence, polishing with unclamped
            # Newton steps if needed.
            if has_Dd && !((Ain !== nothing && size(Ain, 1) > 0) ||
                           (Aeq !== nothing && size(Aeq, 1) > 0))
                converged = _dd_stationary_polish!(beta, eta, mu, X, y,
                    offset, weights, S_total, family, link, control;
                    maxit = polish_maxit, tol_scale = polish_tol_scale)
                dev_old = _deviance(family, y, mu, weights)
            else
                converged = true
            end
            break
        end
    end

    # Final extra parameter estimation
    if _has_extra_param(family)
        scale = _estimates_scale(family) ? max(dev_old / max(n - p, 1.0), 1e-10) : 1.0
        estimate_theta!(family, y, mu, weights, scale)
        dev_old = _deviance(family, y, mu, weights)
    end

    # Final quantities — reuse buffers
    if has_Dd
        dd_final = _family_Dd(family, y, mu, weights; level=0)
        dmu_deta .= GLM.mueta.(Ref(link), eta)
        curv_mu = haskey(dd_final, :EDmu2) ? dd_final[:EDmu2] : dd_final[:Dmu2]
        @inbounds for i in 1:n
            w[i] = clamp(0.5 * curv_mu[i] * dmu_deta[i]^2, eps(), 1e10)
        end
        pearson = 0.0
        var_mu .= _variance(family, mu)
        @inbounds for i in 1:n
            pearson += weights[i] * (y[i] - mu[i])^2 / max(var_mu[i], eps())
        end
    else
        dmu_deta .= GLM.mueta.(Ref(link), eta)
        var_mu .= _variance(family, mu)
        @inbounds for i in 1:n
            w[i] = clamp(weights[i] * dmu_deta[i]^2 / max(var_mu[i], eps()), eps(), 1e10)
        end
        pearson = 0.0
        @inbounds for i in 1:n
            pearson += weights[i] * (y[i] - mu[i])^2 / max(var_mu[i], eps())
        end
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
    @inbounds for j in 1:p, k in 1:p
        XtWX[j, k] = A[j, k] - S_total[j, k]
    end
    edf_vec, hat_diag = penalty_edf(X, w, S_total; XtWX = XtWX, A_chol = A_chol_final)
    R = Matrix(A_chol_final.U)

    return PirlsResult(
        beta, mu, eta, w, dev_old, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end
