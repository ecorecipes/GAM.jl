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

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response
        dmu_deta = GLM.mueta.(Ref(link), eta)
        var_mu = _variance(family, mu)

        # Working weights: w_i * (dmu/deta)^2 / V(mu)
        w = weights .* dmu_deta .^ 2 ./ max.(var_mu, eps())
        w .= clamp.(w, eps(), 1e10)

        # Working response
        z = eta .- offset .+ (y .- mu) ./ dmu_deta

        # Solve penalized WLS: (X'WX + S) β = X'Wz
        XtW = X' * Diagonal(w)
        XtWX = XtW * X
        A = XtWX + S_total
        A_sym = Symmetric((A + A') / 2)
        A_chol = cholesky(A_sym)
        beta_new = A_chol \ (XtW * z)

        # Update
        eta_new = X * beta_new .+ offset
        mu_new = GLM.linkinv.(Ref(link), eta_new)
        mu_new .= _clamp_mu(family, mu_new)
        dev_new = _deviance(family, y, mu_new, weights)

        # Step halving if deviance increased
        step_factor = 1.0
        for _ in 1:25
            if isfinite(dev_new) && dev_new <= dev_old + control.epsilon * abs(dev_old)
                break
            end
            step_factor *= 0.5
            beta_try = beta .+ step_factor .* (beta_new .- beta)
            eta_new = X * beta_try .+ offset
            mu_new = GLM.linkinv.(Ref(link), eta_new)
            mu_new .= _clamp_mu(family, mu_new)
            dev_new = _deviance(family, y, mu_new, weights)
            if step_factor < 1e-8
                break
            end
        end

        if step_factor < 1.0
            beta_new = beta .+ step_factor .* (beta_new .- beta)
            eta_new = X * beta_new .+ offset
            mu_new = GLM.linkinv.(Ref(link), eta_new)
            mu_new .= _clamp_mu(family, mu_new)
            dev_new = _deviance(family, y, mu_new, weights)
        end

        beta .= beta_new
        eta .= eta_new
        mu .= mu_new

        # Estimate extra parameter periodically (every 3 iterations after burn-in)
        if iter >= 3 && iter % 3 == 0 && _has_extra_param(family)
            scale = _estimates_scale(family) ? max(dev_new / (n - sum(ones(p))), 1e-10) : 1.0
            estimate_theta!(family, y, mu, weights, scale)
            # Recompute deviance after theta update
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
        scale = _estimates_scale(family) ? max(dev_old / max(n - sum(ones(p)), 1.0), 1e-10) : 1.0
        estimate_theta!(family, y, mu, weights, scale)
        dev_old = _deviance(family, y, mu, weights)
    end

    # Final quantities
    dmu_deta = GLM.mueta.(Ref(link), eta)
    var_mu = _variance(family, mu)
    w = weights .* dmu_deta .^ 2 ./ max.(var_mu, eps())
    w .= clamp.(w, eps(), 1e10)

    # Pearson statistic
    pearson = sum(weights .* (y .- mu) .^ 2 ./ max.(var_mu, eps()))

    # EDF and hat matrix
    edf_vec, hat_diag = penalty_edf(X, w, S_total)

    # R factor
    A = X' * Diagonal(w) * X + S_total
    R = Matrix(cholesky(Symmetric(A)).U)

    return PirlsResult(
        beta, mu, eta, w, dev_old, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end
