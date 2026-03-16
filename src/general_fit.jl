# General smooth model fitting — Wood, Pya & Säfken (2016) JASA 111(516):1548-1575
#
# Implements gam.fit5: direct Newton on penalized likelihood (no PIRLS).
# Works for ANY likelihood expressible via Distributions.logpdf + a link function.
# For standard exponential families this gives identical results to PIRLS.
#
# Key differences from pirls.jl:
#   - Inner iteration: full Newton on ℓ(β) - ½β'S^λβ
#   - Derivatives via Distributions.jl logpdf + ForwardDiff through link
#   - Handles indefinite Hessians with perturbation (gam.fit5 stabilization)

using ForwardDiff

"""
    general_fit(X, y, S_total, family, link; weights, start, control) -> PirlsResult, loglik, neg_hessian

Fit a penalized model by direct Newton iteration on the penalized log-likelihood:
    max_β [ ℓ(β) - ½ β'S_total β ]

This is the inner loop of gam.fit5 (Wood/Pya/Säfken 2016). For standard
exponential families with canonical link, this gives identical results to PIRLS.

Uses analytical derivatives for standard families, ForwardDiff for general ones.

Returns (PirlsResult, log_likelihood, neg_hessian_of_loglik).
"""
function general_fit(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution, link::GLM.Link;
    weights::Vector{Float64} = ones(length(y)),
    start::Union{Nothing, Vector{Float64}} = nothing,
    control::GamControl = gam_control())

    n, p = size(X)

    # Initialize coefficients
    if start !== nothing
        beta = copy(start)
    else
        beta = zeros(p)
        mu_init = [_mustart(family, y[i], weights[i]) for i in 1:n]
        eta_init = GLM.linkfun.(Ref(link), mu_init)
        beta[1] = mean(eta_init)
    end

    eta = X * beta
    mu = _clamp_mu(family, GLM.linkinv.(Ref(link), eta))

    # Compute initial penalized log-likelihood
    ll = _total_loglik(family, y, mu, weights)
    pen = 0.5 * dot(beta, S_total * beta)
    pll = ll - pen

    converged = false
    neg_hess = zeros(p, p)
    n_iter = 0
    pll_old = pll

    for iter in 1:control.maxit
        n_iter = iter
        # Compute gradient and Hessian of log-likelihood w.r.t. η
        dl_deta, d2l_deta2 = _loglik_eta_derivs(family, link, y, mu, eta, weights)

        # Gradient of penalized log-likelihood: ∂(ℓ-pen)/∂β = X'(∂ℓ/∂η) - S·β
        grad = X' * dl_deta - S_total * beta

        # Negative Hessian of log-likelihood: -∂²ℓ/∂β² = X' diag(-d²ℓ/dη²) X
        neg_d2l = max.(-d2l_deta2, 1e-10)  # ensure positive for stability
        neg_hess .= X' * Diagonal(neg_d2l) * X

        # Penalized Hessian: Hp = neg_hess + S_total
        Hp = neg_hess + S_total

        # Stabilization: handle indefinite Hessian (gam.fit5 strategy)
        # Diagonal preconditioning for numerical stability
        D = diag(Hp)
        indefinite = false

        if any(!isfinite, D)
            @warn "Non-finite Hessian diagonal at iteration $iter"
            break
        end

        if minimum(D) <= 0
            # Hessian not positive definite — add ridge
            ridge = max(abs(minimum(D)), sqrt(eps()) * maximum(D))
            Hp = Hp + Diagonal(fill(ridge, p))
            D = diag(Hp)
            indefinite = true
        end

        # Diagonal preconditioning: D^{-1/2} Hp D^{-1/2}
        Dinvsqrt = D .^ (-0.5)
        Hp_scaled = Diagonal(Dinvsqrt) * Hp * Diagonal(Dinvsqrt)

        # Cholesky factorize
        local L
        try
            L = cholesky(Symmetric(Hp_scaled))
        catch
            Hp_scaled = Symmetric(Hp_scaled +
                Diagonal(fill(1e-6 * maximum(abs.(Hp_scaled)), p)))
            try
                L = cholesky(Hp_scaled)
            catch
                @warn "Cholesky failed at iteration $iter"
                break
            end
        end

        # Newton step: δ = Hp⁻¹ · grad (with preconditioning)
        step = Dinvsqrt .* (L \ (Dinvsqrt .* grad))

        # Limit step length (gam.fit5 style)
        c_norm = max(norm(beta), 1.0)
        s_norm = norm(step)
        if s_norm > c_norm
            step .*= c_norm / s_norm
        end

        # Line search with step halving
        accept = false
        for khalf in 0:30
            beta_new = beta + step
            eta_new = X * beta_new
            mu_new = _clamp_mu(family, GLM.linkinv.(Ref(link), eta_new))

            ll_new = _total_loglik(family, y, mu_new, weights)
            pen_new = 0.5 * dot(beta_new, S_total * beta_new)
            pll_new = ll_new - pen_new

            if pll_new >= pll - abs(pll) * eps() || khalf == 30
                pll_old = pll
                beta = beta_new
                eta = eta_new
                mu = mu_new
                ll = ll_new
                pll = pll_new
                accept = true
                break
            end
            step .*= 0.5
        end

        if !accept
            break
        end

        # Convergence check
        rel_change = abs(pll - pll_old) / max(1.0, abs(pll))
        if rel_change < control.epsilon && maximum(abs.(step)) < control.epsilon * 10
            converged = true
            break
        end
    end

    # Final deviance and Pearson statistic
    dev = _deviance(family, y, mu, weights)
    pearson = sum(weights[i] * ((y[i] - mu[i])^2 / _variance_scalar(family, mu[i]))
                  for i in 1:n)

    # EDF and hat matrix from the final Hessian
    _, d2l_deta2_final = _loglik_eta_derivs(family, link, y, mu, eta, weights)
    w_final = max.(-d2l_deta2_final, 1e-10)

    XtWX = X' * Diagonal(w_final) * X
    A = XtWX + S_total
    A_chol = cholesky(Symmetric(A))
    F = A_chol \ XtWX
    edf_vec = diag(F)
    # hat_diag: h_ii = x_i' A⁻¹ (X'WX) A⁻¹ x_i ≈ diag(X F A⁻¹ X')
    # Simpler: h_ii = w_i * x_i' A⁻¹ x_i  (for hat matrix of penalized WLS)
    Ainv_Xt = A_chol \ X'
    hat_diag = vec(sum(X .* Ainv_Xt', dims=2) .* w_final)

    R_mat = try
        Matrix(A_chol.U)
    catch
        zeros(p, p)
    end

    result = PirlsResult(
        beta, mu, eta, w_final, dev, pearson,
        converged, n_iter, R_mat, hat_diag, edf_vec)

    return result, ll, neg_hess
end

# ═══════════════════════════════════════════════════════════════════════
# Log-likelihood and derivatives via Distributions.jl + ForwardDiff
# ═══════════════════════════════════════════════════════════════════════

"""
    _make_distribution(family, mu) -> Distribution

Construct a Distributions.jl distribution instance parameterized by mean μ.
This is the core mapping from GLM family + mean to a proper distribution.
For families with unknown scale (Normal, Gamma), uses unit scale — the scale
parameter is profiled out in the REML/LAML objective.
"""
_make_distribution(::Normal, mu::Real) = Normal(mu, 1.0)
_make_distribution(::Poisson, mu::Real) = Poisson(max(mu, 1e-10))
_make_distribution(::Union{Binomial,Bernoulli}, mu::Real) = Bernoulli(clamp(mu, 1e-10, 1 - 1e-10))
_make_distribution(::Gamma, mu::Real) = Gamma(1.0, max(mu, 1e-10))   # shape=1, scale=μ
_make_distribution(::InverseGaussian, mu::Real) = InverseGaussian(max(mu, 1e-10), 1.0)

"""
    _total_loglik(family, y, mu, weights) -> Float64

Total log-likelihood: Σ wᵢ · logpdf(D(μᵢ), yᵢ)

Uses Distributions.jl for all families. The distribution is constructed via
`_make_distribution`, which maps a GLM family + mean to a proper distribution.
"""
function _total_loglik(family::UnivariateDistribution, y::Vector{Float64},
    mu::Vector{Float64}, weights::Vector{Float64})
    ll = 0.0
    @inbounds for i in eachindex(y)
        d = _make_distribution(family, mu[i])
        ll += weights[i] * logpdf(d, y[i])
    end
    return ll
end

"""
    _loglik_eta_derivs(family, link, y, mu, eta, weights) -> (dl_deta, d2l_deta2)

First and second derivatives of log-likelihood w.r.t. linear predictor η.

    dl_deta[i]   = wᵢ · ∂ℓᵢ/∂ηᵢ
    d2l_deta2[i] = wᵢ · ∂²ℓᵢ/∂ηᵢ²

Primary implementation: ForwardDiff through `logpdf(D(linkinv(η)), y)`.
Analytical fast-paths are provided for canonical link + standard family
combinations where the derivatives have simple closed forms.
"""
function _loglik_eta_derivs(family::UnivariateDistribution, link::GLM.Link,
    y::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    dl = similar(y)
    d2l = similar(y)

    @inbounds for i in 1:n
        yi = y[i]
        wi = weights[i]

        # ℓᵢ(η) = logpdf(D(linkinv(η)), yᵢ)
        function ll_of_eta(η::T) where T <: Real
            mu_val = GLM.linkinv(link, η)
            d = _make_distribution(family, mu_val)
            return logpdf(d, yi)
        end

        dl[i] = wi * ForwardDiff.derivative(ll_of_eta, eta[i])
        d2l[i] = wi * ForwardDiff.derivative(
            η -> ForwardDiff.derivative(ll_of_eta, η), eta[i])
    end
    return dl, d2l
end

# ─── Analytical fast-paths for canonical links ───────────────────────
# These give identical results to the ForwardDiff path but avoid AD overhead.

function _loglik_eta_derivs(family::Normal, link::IdentityLink,
    y::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    dl = similar(y)
    d2l = similar(y)
    @inbounds for i in 1:n
        dl[i] = weights[i] * (y[i] - mu[i])
        d2l[i] = -weights[i]
    end
    return dl, d2l
end

function _loglik_eta_derivs(family::Poisson, link::LogLink,
    y::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    dl = similar(y)
    d2l = similar(y)
    @inbounds for i in 1:n
        mu_i = max(mu[i], 1e-10)
        dl[i] = weights[i] * (y[i] - mu_i)
        d2l[i] = -weights[i] * mu_i
    end
    return dl, d2l
end

function _loglik_eta_derivs(family::BinomialLike, link::LogitLink,
    y::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    dl = similar(y)
    d2l = similar(y)
    @inbounds for i in 1:n
        mu_i = clamp(mu[i], 1e-10, 1.0 - 1e-10)
        dl[i] = weights[i] * (y[i] - mu_i)
        d2l[i] = -weights[i] * mu_i * (1.0 - mu_i)
    end
    return dl, d2l
end

function _loglik_eta_derivs(family::Gamma, link::InverseLink,
    y::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64}, weights::Vector{Float64})
    n = length(y)
    dl = similar(y)
    d2l = similar(y)
    @inbounds for i in 1:n
        mu_i = max(mu[i], 1e-10)
        y_i = max(y[i], 1e-10)
        # η = 1/μ → ℓ = -yη + log(η), ∂ℓ/∂η = μ-y, ∂²ℓ/∂η² = -μ²
        dl[i] = weights[i] * (mu_i - y_i)
        d2l[i] = -weights[i] * mu_i^2
    end
    return dl, d2l
end

# ═══════════════════════════════════════════════════════════════════════
# LAML objective for the general method
# ═══════════════════════════════════════════════════════════════════════

"""
    laml_score(X, y, penalty, log_sp, family, link, weights, pirls_result, loglik, neg_hessian;
               method=:REML, gamma=1.0) -> (neg_V, grad)

Compute the Laplace Approximate Marginal Likelihood and its gradient.

For general likelihoods (scale=1):
  V(ρ) = ℓ(β̂) - ½ β̂'S^λ β̂ + ½ log|S^λ|_+ - ½ log|H| + Mp/2 log(2π)

For Gaussian with unknown scale, uses profiled REML (equivalent to reml_score).

Returns (neg_V, gradient_wrt_log_sp). neg_V is to be MINIMIZED.
"""
function laml_score(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup, log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64}, result::PirlsResult,
    loglik::Float64, neg_hessian::Matrix{Float64};
    method::Symbol = :REML, gamma::Real = 1.0)

    n, p = size(X)
    beta = result.coefficients
    n_sp = length(log_sp)

    S_total = total_penalty(penalty, log_sp, p)

    # Penalized Hessian: H = -∂²ℓ/∂β² + S^λ
    H = neg_hessian + S_total
    H_chol = cholesky(Symmetric(H))
    log_det_H = logdet(H_chol)

    # Log pseudo-determinant of penalty
    log_det_S = _log_penalty_det(penalty, log_sp)

    # Null space dimension
    Mp = sum(b.stop - b.start + 1 - b.rank for b in penalty.blocks; init=0)

    # Penalty contribution
    pen = dot(beta, S_total * beta)

    if _needs_scale_estimate(family)
        # Profiled REML — equivalent to existing reml_score for Gaussian
        edf_total = sum(result.edf_vec)
        scale_est = max(result.pearson / (n - edf_total), 1e-10)
        Dp = result.deviance + pen
        ls = _log_saturated_likelihood(family, y, weights, scale_est)

        neg_V = (Dp / (2 * scale_est) - ls) / gamma +
                0.5 * log_det_H -
                0.5 * log_det_S -
                0.5 * Mp * (log(2π * scale_est) - log(gamma))

        grad = _reml_gradient(X, result.working_weights, S_total, H_chol,
            beta, result.fitted_values, y, penalty, log_sp,
            result.deviance, scale_est, n, p, method, gamma,
            family, link, weights)

        return neg_V, grad
    else
        # General LAML (Poisson, Binomial, etc.)
        V = loglik - 0.5 * pen + 0.5 * log_det_S - 0.5 * log_det_H + 0.5 * Mp * log(2π)
        neg_V = -V

        grad = _reml_gradient(X, result.working_weights, S_total, H_chol,
            beta, result.fitted_values, y, penalty, log_sp,
            result.deviance, 1.0, n, p, method, gamma,
            family, link, weights)

        return neg_V, grad
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Outer iteration using general fit
# ═══════════════════════════════════════════════════════════════════════

"""
    outer_iteration_general(X, y, smooths, penalty, family, link; ...) -> (log_sp, PirlsResult)

Outer smoothing parameter optimization using the general (Newton) inner
iteration from Wood/Pya/Säfken (2016). Drop-in replacement for outer_iteration.
"""
function outer_iteration_general(X::Matrix{Float64}, y::Vector{Float64},
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
        result, _, _ = general_fit(X, y, S_total, family, link;
            weights=weights, control=control)
        return penalty.sp, result
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing
    prev_ll = 0.0
    prev_neg_hess = zeros(p, p)
    score_hist = Float64[]

    for outer_iter in 1:control.outer_maxit
        S_total = total_penalty(penalty, log_sp, p)

        start = prev_result === nothing ? nothing : prev_result.coefficients
        result, ll, neg_hess = general_fit(X, y, S_total, family, link;
            weights=weights, start=start, control=control)

        if !result.converged && control.trace
            @warn "Newton inner did not converge at outer iteration $outer_iter"
        end

        # LAML score and gradient
        cur_score, cur_grad = laml_score(X, y, penalty, log_sp, family, link,
            weights, result, ll, neg_hess; method=method, gamma=control.gamma)

        # FD Hessian diagonal for Newton step
        h_fd = 1e-4
        hess_diag = zeros(n_sp)
        for j in 1:n_sp
            lsp_p = copy(log_sp); lsp_p[j] += h_fd
            S_p = total_penalty(penalty, lsp_p, p)
            r_p, ll_p, nh_p = general_fit(X, y, S_p, family, link;
                weights=weights, start=result.coefficients, control=control)
            _, grad_p = laml_score(X, y, penalty, lsp_p, family, link,
                weights, r_p, ll_p, nh_p; method=method, gamma=control.gamma)
            hess_diag[j] = (grad_p[j] - cur_grad[j]) / h_fd
        end

        # Newton step with clamping
        log_sp_new = copy(log_sp)
        for j in 1:n_sp
            if hess_diag[j] > eps()
                step = -cur_grad[j] / hess_diag[j]
            else
                step = -sign(cur_grad[j]) * min(abs(cur_grad[j]) * 2.0, 2.0)
            end
            step = clamp(step, -5.0, 5.0)
            log_sp_new[j] = clamp(log_sp[j] + step, -15.0, 15.0)
        end

        # Step halving
        S_trial = total_penalty(penalty, log_sp_new, p)
        r_trial, ll_trial, nh_trial = general_fit(X, y, S_trial, family, link;
            weights=weights, start=result.coefficients, control=control)
        trial_score, _ = laml_score(X, y, penalty, log_sp_new, family, link,
            weights, r_trial, ll_trial, nh_trial; method=method, gamma=control.gamma)

        for _ in 1:30
            if trial_score <= cur_score; break; end
            log_sp_new .= (log_sp .+ log_sp_new) ./ 2.0
            S_trial = total_penalty(penalty, log_sp_new, p)
            r_trial, ll_trial, nh_trial = general_fit(X, y, S_trial, family, link;
                weights=weights, start=result.coefficients, control=control)
            trial_score, _ = laml_score(X, y, penalty, log_sp_new, family, link,
                weights, r_trial, ll_trial, nh_trial; method=method, gamma=control.gamma)
        end
        max_change = maximum(abs.(log_sp_new .- log_sp))

        push!(score_hist, trial_score)

        if control.trace
            println("General outer $outer_iter: score=$(@sprintf("%.6f", trial_score)), " *
                    "sp=[$(join([@sprintf("%.4f", exp(s)) for s in log_sp_new], ", "))]")
        end

        log_sp .= log_sp_new
        prev_result = result
        prev_ll = ll
        prev_neg_hess = neg_hess

        if max_change < control.epsilon * 10
            break
        end
        if length(score_hist) > 3 && max_change < 0.05
            recent = score_hist[max(1, end-3):end]
            if maximum(abs.(diff(recent))) < control.epsilon
                break
            end
        end
    end

    # Final solve
    penalty.sp .= log_sp
    S_total = total_penalty(penalty, log_sp, p)
    final_result, _, _ = general_fit(X, y, S_total, family, link;
        weights=weights,
        start=prev_result === nothing ? nothing : prev_result.coefficients,
        control=control)

    return log_sp, final_result
end
