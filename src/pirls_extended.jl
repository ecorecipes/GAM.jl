# Penalized IRLS for Extended Families
#
# Modified P-IRLS that works with ExtendedFamily types, including
# estimation of extra parameters (NB theta, Beta phi) during iteration.

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
    control::GamControl = gam_control())

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

    # Pre-allocate working buffers
    dmu_deta = Vector{Float64}(undef, n)
    var_mu = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    Xw = similar(X)           # n×p buffer for √w-scaled X
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
            # Use family's Dd derivatives for proper working weights
            # (matches R's gam.fit3 for extended families)
            dd = _family_Dd(family, y, mu, weights; level=0)
            Dmu = dd[:Dmu]    # gradient of deviance w.r.t. mu
            Dmu2 = dd[:Dmu2]  # 2nd deriv of deviance w.r.t. mu (working weight basis)

            dmu_deta .= GLM.mueta.(Ref(link), eta)

            @inbounds for i in 1:n
                # Convert mu-derivatives to eta-derivatives via chain rule
                Deta2_i = Dmu2[i] * dmu_deta[i]^2
                w[i] = clamp(Deta2_i, eps(), 1e10)
                # Working response: z = eta + Deta / Deta2, clamped to prevent instability
                Deta_i = Dmu[i] * dmu_deta[i]
                delta = Deta_i / max(Deta2_i, eps())
                z[i] = eta[i] - offset[i] + clamp(delta, -40.0, 40.0)
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

        # Build X'WX via scaled X: Xw = √w .* X, then XtWX = Xw'Xw
        @inbounds for j in 1:p, i in 1:n
            Xw[i, j] = sqrt(w[i]) * X[i, j]
        end
        mul!(XtWX, Xw', Xw)

        # A = X'WX + S_total
        @inbounds for j in 1:p, k in 1:p
            A[k, j] = XtWX[k, j] + S_total[k, j]
        end

        # RHS = X'Wz
        @inbounds for j in 1:p
            s = 0.0
            for i in 1:n
                s += X[i, j] * w[i] * z[i]
            end
            rhs[j] = s
        end

        # Solve: beta_new = A \ rhs
        A_chol = cholesky(Symmetric(A))
        ldiv!(beta_new, A_chol, rhs)

        # Update: eta_new = X * beta_new + offset
        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        @inbounds for i in 1:n
            mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
        end
        dev_new = _deviance(family, y, mu_new, weights)

        # Step halving if deviance increased
        step_factor = 1.0
        for _ in 1:25
            if isfinite(dev_new) && dev_new <= dev_old + control.epsilon * abs(dev_old)
                break
            end
            step_factor *= 0.5
            if step_factor < 1e-8
                break
            end
            @inbounds for j in 1:p
                beta_step[j] = beta[j] + step_factor * (beta_new[j] - beta[j])
            end
            mul!(eta_new, X, beta_step)
            eta_new .+= offset
            @inbounds for i in 1:n
                mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
            end
            dev_new = _deviance(family, y, mu_new, weights)
        end

        if step_factor < 1.0
            @inbounds for j in 1:p
                beta_new[j] = beta[j] + step_factor * (beta_new[j] - beta[j])
            end
            mul!(eta_new, X, beta_new)
            eta_new .+= offset
            @inbounds for i in 1:n
                mu_new[i] = _clamp_mu(family, GLM.linkinv(link, eta_new[i]))
            end
            dev_new = _deviance(family, y, mu_new, weights)
        end

        beta .= beta_new
        eta .= eta_new
        mu .= mu_new

        # Estimate extra parameter periodically (every 3 iterations after burn-in)
        if iter >= 3 && iter % 3 == 0 && _has_extra_param(family)
            scale = _estimates_scale(family) ? max(dev_new / (n - p), 1e-10) : 1.0
            estimate_theta!(family, y, mu, weights, scale)
            dev_new = _deviance(family, y, mu, weights)
        end

        # Convergence check
        crit = abs(dev_new - dev_old) / (abs(dev_new) + 0.1)
        dev_old = dev_new

        if crit < control.epsilon
            converged = true
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
        @inbounds for i in 1:n
            w[i] = clamp(dd_final[:Dmu2][i] * dmu_deta[i]^2, eps(), 1e10)
        end
        # Pearson using Dd-based variance
        pearson = 0.0
        @inbounds for i in 1:n
            v_i = max(dd_final[:Dmu2][i] > 0 ? 1.0 / dd_final[:Dmu2][i] : 1.0, eps())
            pearson += weights[i] * (y[i] - mu[i])^2 * v_i
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

    # EDF and hat matrix
    edf_vec, hat_diag = penalty_edf(X, w, S_total)

    # R factor — reuse A buffer (already has XtWX + S_total from last iteration)
    @inbounds for j in 1:p, k in 1:p
        A[k, j] = XtWX[k, j] + S_total[k, j]
    end
    R = Matrix(cholesky(Symmetric(A)).U)

    return PirlsResult(
        beta, mu, eta, w, dev_old, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end
