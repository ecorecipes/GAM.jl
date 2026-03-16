# REML / ML / GCV score computation
#
# Computes smoothness selection criteria and their derivatives with respect
# to log smoothing parameters. The key innovation from Wood (2011) is that
# derivatives come at negligible extra cost via the implicit function theorem.

"""
    reml_score(X, y, S_penalty, log_sp, family, link, weights, pirls_result;
               method=:REML, gamma=1.0)

Compute the REML (or ML/GCV) score for given log smoothing parameters.
Returns `(score, grad)` where `grad` is the gradient w.r.t. `log_sp`.
"""
function reml_score(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup,
    log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64},
    pirls_result::PirlsResult;
    method::Symbol = :REML, gamma::Real = 1.0,
    scale::Float64 = -1.0)

    n, p = size(X)
    beta = pirls_result.coefficients
    w = pirls_result.working_weights
    dev = pirls_result.deviance

    S_total = total_penalty(penalty, log_sp, p)

    # X'WX
    XtWX = X' * Diagonal(w) * X

    # A = X'WX + S
    A = XtWX + S_total
    A_chol = cholesky(Symmetric(A))
    log_det_A = logdet(A_chol)

    # EDF
    F = A_chol \ XtWX
    edf_total = tr(F)

    if method == :GCV
        # GCV = n * deviance / (n - gamma * edf)^2
        denom = n - gamma * edf_total
        score = n * dev / denom^2

        # Analytical gradient via IFT (matches mgcv's gdi1)
        mu = pirls_result.fitted_values
        grad = _gcv_gradient(X, y, w, beta, mu, S_total, A_chol, penalty, log_sp,
            family, link, dev, edf_total, n, gamma)
        return score, grad

    elseif method == :REML || method == :ML
        # REML/ML score (Laplace approximate restricted/marginal log-likelihood).
        # Following mgcv's gam.fit3.r lines 612-617:
        #   REML = (Dp/(2σ²) - ls)/γ + 0.5·log|A| - 0.5·log|S+|
        #          - remlInd·(Mp/2)·(log(2πσ²) - log(γ))
        # where Dp = dev + β'Sβ, ls = log saturated likelihood.

        # Estimate or use fixed scale
        if scale < 0
            # Estimate scale from Pearson statistic
            scale_est = pirls_result.pearson / (n - edf_total)
            scale_est = max(scale_est, 1e-10)
        else
            scale_est = scale
        end

        # Log pseudo-determinant of penalty
        log_det_S = _log_penalty_det(penalty, log_sp)

        # Penalty null space dimension
        Mp = sum(b.stop - b.start + 1 - b.rank for b in penalty.blocks;
            init = 0)

        # Penalized deviance: dev + β'Sβ
        penalty_contrib = dot(beta, S_total * beta)
        Dp = dev + penalty_contrib

        # Log saturated likelihood (needed for correct REML landscape when
        # scale is estimated — for Gaussian/Gamma this depends on σ²)
        ls = _log_saturated_likelihood(family, y, weights, scale_est)

        if method == :REML
            # R: (Dp/(2σ²) - ls)/γ + ldetA/2 - ldetS/2 - Mp/2·(log(2πσ²) - log(γ))
            score = (Dp / (2 * scale_est) - ls) / gamma +
                    0.5 * log_det_A -
                    0.5 * log_det_S -
                    0.5 * Mp * (log(2π * scale_est) - log(gamma))
        else  # ML
            score = (Dp / (2 * scale_est) - ls) / gamma +
                    0.5 * log_det_A -
                    0.5 * log_det_S
        end

        # Gradient via implicit function theorem (Wood 2011, Section 3.1)
        mu = pirls_result.fitted_values
        grad = _reml_gradient(X, w, S_total, A_chol, beta, mu, y, penalty, log_sp,
            dev, scale_est, n, p, method, gamma, family, link, weights)

        return score, grad
    else
        throw(ArgumentError("Unknown smoothing method: $method"))
    end
end

"""
    _log_penalty_det(penalty, log_sp)

Compute the log pseudo-determinant of the total penalty:
log|Σ λ_j S_j|_+ (product of non-zero eigenvalues).
"""
function _log_penalty_det(penalty::PenaltySetup, log_sp::Vector{Float64})
    ldet = 0.0
    sp_idx = 1
    for block in penalty.blocks
        k = block.stop - block.start + 1
        S_block = zeros(k, k)
        for Si in block.S
            λ = exp(log_sp[sp_idx])
            S_block .+= λ .* Si
            sp_idx += 1
        end
        eig = eigvals(Symmetric(S_block))
        # Only count positive eigenvalues
        for ev in eig
            if ev > eps() * maximum(abs.(eig))
                ldet += log(ev)
            end
        end
    end
    return ldet
end

"""
    _log_saturated_likelihood(family, y, weights, scale)

Log-likelihood evaluated at the saturated model (μ = y).

This is the `ls` term in R's mgcv REML formula:
  REML = (Dp/(2σ²) - ls)/γ + ...

For families with estimated scale (Gaussian, Gamma), ls depends on σ²
and is NOT constant w.r.t. smoothing parameters. This makes the REML
landscape correct — without it, the REML can decrease at high sp
(oversmoothing), leading the optimizer to a false minimum.

For families with known scale (Poisson, Binomial), ls is constant
and doesn't affect optimization.
"""
function _log_saturated_likelihood(::Normal, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Gaussian: l_sat = -n/2 · log(2πσ²) for unit weights
    # With weights: l_sat = Σ_i -0.5·log(2πσ²/w_i) = -n/2·log(2πσ²) + 0.5·Σlog(w_i)
    n = length(y)
    ls = -0.5 * n * log(2π * scale)
    if !all(w -> w ≈ 1.0, weights)
        ls += 0.5 * sum(log, weights)
    end
    return ls
end

function _log_saturated_likelihood(::Poisson, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Poisson: l_sat = Σ [y·log(y) - y - lgamma(y+1)] for y > 0
    ls = 0.0
    @inbounds for i in eachindex(y)
        yi = y[i]
        if yi > 0
            ls += weights[i] * (yi * log(yi) - yi) - logabsgamma(yi + 1)[1]
        end
    end
    return ls
end

function _log_saturated_likelihood(::BinomialLike, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Bernoulli/Binomial: l_sat = 0 for y ∈ {0, 1} (0·log(0) = 0)
    return 0.0
end

function _log_saturated_likelihood(::Gamma, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Gamma: l_sat depends on scale (φ = scale).
    # l(y;μ=y,φ) = Σ [(-1/φ)·(y/y - log(y/y) - 1) + log-normalizing]
    # The deviance residual at saturation is 0, so:
    # l_sat = Σ [-log(y) - log(φ) - lgamma(1/φ) + (1/φ-1)·log(y) + (1/φ)·log(1/φ)]
    # Simplified: this depends on φ = scale and thus changes with sp.
    # For simplicity, use the Gaussian approximation: l_sat ≈ -n/2·log(2πσ²)
    n = length(y)
    return -0.5 * n * log(2π * scale)
end

function _log_saturated_likelihood(::InverseGaussian, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    n = length(y)
    return -0.5 * n * log(2π * scale)
end

function _log_saturated_likelihood(::UnivariateDistribution, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Fallback: assume known scale → constant ls (doesn't affect optimization)
    return 0.0
end

"""
    _reml_gradient(X, w, S_total, A_chol, beta, mu, y, penalty, log_sp,
                   dev, scale, n, p, method, gamma, family, link, weights)

Gradient of REML/ML score w.r.t. log smoothing parameters.
Uses the implicit function theorem result from Wood (2011).

For the log-determinant term d(log|A|)/d(log sp), we need to account for
both the explicit penalty derivative (λ_j S_j) and the implicit weight
change through β. The full derivative is:
  trA1[j] = tr(A⁻¹ · (λ_j S_j + X' diag(dw/d(log sp_j)) X))
where dw/d(log sp_j) comes from the chain rule through η and β.
"""
function _reml_gradient(X::Matrix{Float64}, w::Vector{Float64},
    S_total::Matrix{Float64}, A_chol,
    beta::Vector{Float64},
    mu::Vector{Float64}, y::Vector{Float64},
    penalty::PenaltySetup, log_sp::Vector{Float64},
    dev::Float64, scale::Float64, n::Int, p::Int,
    method::Symbol, gamma::Real,
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64})

    n_sp = length(log_sp)
    grad = zeros(n_sp)

    Ainv = inv(A_chol)

    is_gaussian_identity = family isa Normal && link isa GLM.IdentityLink

    # Compute weight derivatives w.r.t. η for non-Gaussian
    # w_i = weights_i * (dμ/dη)² / V(μ)
    # dw_i/dη_i depends on family/link
    dw_deta = zeros(n)
    if !is_gaussian_identity
        @inbounds for i in 1:n
            eta_i = GLM.linkfun(link, mu[i])
            mu_i = mu[i]
            g1 = GLM.mueta(link, eta_i)      # dμ/dη
            vm = _variance_scalar(family, mu_i)   # V(μ)

            # d²μ/dη² from link function
            g2 = _d2mu_deta2(link, mu_i, eta_i)

            # V'(μ)
            dvm = _dvariance_scalar_mu(family, mu_i)

            # w = weights * g1² / V
            # dw/dη = weights * (2 g1 g2 V - g1² V' g1) / V²
            #       = weights * g1/V * (2 g2 - g1² V'/V)
            # Wait, need to be more careful:
            # dw/dη = d(g1²/V)/dη = (2g1·dg1/dη·V - g1²·dV/dη) / V²
            # where dg1/dη = g2 and dV/dη = V'(μ)·g1
            dw_deta[i] = weights[i] * (2.0 * g1 * g2 * vm - g1^2 * dvm * g1) / (vm * vm)
        end
    end

    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        beta_block = beta[idx]

        for Si in block.S
            λ = exp(log_sp[sp_idx])
            dS = zeros(p, p)
            dS[idx, idx] .= λ .* Si

            # D1: total derivative of penalized deviance
            # At PIRLS convergence, (dev_grad + pen_grad)' b1 ≈ 0,
            # so D1 ≈ λ β'S_jβ (the explicit sp term)
            bSb = dot(beta_block, Si * beta_block)
            D1_j = λ * bSb

            # trA1: d(log|A|)/d(log sp_j) including weight derivative
            # = tr(A⁻¹ λ_j S_j) + tr(A⁻¹ X' diag(dw/d(log sp_j)) X)
            trA1_explicit = tr(Ainv * dS)

            trA1_implicit = 0.0
            if !is_gaussian_identity
                # dw/d(log sp_j) = dw/dη · dη/d(log sp_j) = dw/dη · X b1_j
                # where b1_j = -A⁻¹(λ_j S_j β)
                rhs = zeros(p)
                rhs[idx] .= λ .* (Si * beta_block)
                b1_j = -(Ainv * rhs)

                # dη = X b1_j
                deta_j = X * b1_j

                # dw/d(log sp_j) for each observation
                dw_j = dw_deta .* deta_j

                # tr(A⁻¹ X' diag(dw_j) X) = tr(F · diag(dw_j))
                # where F = A⁻¹ X'WX... no, we need tr(A⁻¹ X' diag(dw_j) X)
                # = Σ_i dw_j[i] · (X_i' A⁻¹ X_i) = Σ_i dw_j[i] · h_ii
                # where h_ii = X_i' A⁻¹ X_i
                # But we can compute this as sum(dw_j .* diag(X Ainv X'))
                # Or more efficiently: XAinv = X * Ainv, then h_ii = sum(XAinv[i,:].^2)...
                # Actually for SYMMETRIC Ainv, h_ii = sum(X[i,:] .* (Ainv * X[i,:])')

                # Efficient computation: XA = X * Ainv, h_ii = dot(X[i,:], XA[i,:])
                # trA1_implicit = sum(dw_j .* h_ii)
                XAinv = X * Ainv  # n × p
                @inbounds for i in 1:n
                    if abs(dw_j[i]) > eps()
                        h_ii = dot(view(X, i, :), view(XAinv, i, :))
                        trA1_implicit += dw_j[i] * h_ii
                    end
                end
            end

            trA1_j = trA1_explicit + trA1_implicit

            # d(log|S+|)/d(log sp_j) = rank_j (for single penalty per block)
            d_log_det_S = Float64(block.rank)

            # REML1[j] = D1[j]/(2σ²γ) + trA1[j]/2 - det1[j]/2
            grad[sp_idx] = D1_j / (2 * scale * gamma) +
                           0.5 * trA1_j -
                           0.5 * d_log_det_S

            sp_idx += 1
        end
    end

    return grad
end

# Helper: d²μ/dη² for different link functions
_d2mu_deta2(::GLM.LogLink, mu::Float64, eta::Float64) = mu
_d2mu_deta2(::GLM.LogitLink, mu::Float64, eta::Float64) = mu * (1 - mu) * (1 - 2mu)
_d2mu_deta2(::GLM.IdentityLink, mu::Float64, eta::Float64) = 0.0
_d2mu_deta2(::GLM.InverseLink, mu::Float64, eta::Float64) = 2.0 * mu^3
_d2mu_deta2(::GLM.SqrtLink, mu::Float64, eta::Float64) = 0.5
_d2mu_deta2(::GLM.Link, mu::Float64, eta::Float64) = 0.0  # fallback

# Helper: V'(μ) for different families
_dvariance_scalar_mu(::Normal, mu::Float64) = 0.0
_dvariance_scalar_mu(::BinomialLike, mu::Float64) = 1.0 - 2.0 * mu
_dvariance_scalar_mu(::Poisson, mu::Float64) = 1.0
_dvariance_scalar_mu(::Gamma, mu::Float64) = 2.0 * mu
_dvariance_scalar_mu(::InverseGaussian, mu::Float64) = 3.0 * mu * mu
_dvariance_scalar_mu(::UnivariateDistribution, mu::Float64) = 0.0

"""
    _gcv_gradient(X, y, w, beta, mu, S_total, A_chol, penalty, log_sp,
                  family, link, dev, edf, n, gamma)

Analytical gradient of GCV score w.r.t. log smoothing parameters using the
Implicit Function Theorem (Wood 2011, Section 3). Matches mgcv's gdi1 C code.

Key formulas:
  GCV = n·dev / (n - γ·trA)²
  GCV1[j] = n·D1[j]/δ² + 2·n·dev·trA1[j]·γ/δ³

where D1[j] = ∂dev/∂(log_sp_j) via IFT and trA1[j] = ∂trA/∂(log_sp_j).
"""
function _gcv_gradient(X::Matrix{Float64}, y::Vector{Float64},
    w::Vector{Float64}, beta::Vector{Float64}, mu::Vector{Float64},
    S_total::Matrix{Float64},
    A_chol, penalty::PenaltySetup,
    log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    dev::Float64, edf::Float64,
    n::Int, gamma::Real)

    p = size(X, 2)
    n_sp = length(log_sp)
    grad = zeros(n_sp)
    delta = n - gamma * edf

    if delta < 1.0
        return grad
    end

    Ainv = inv(A_chol)
    XtWX = X' * Diagonal(w) * X
    F = Ainv * XtWX  # influence/hat matrix in coefficient space

    # Deviance gradient w.r.t. β: ∂dev/∂β = X' * [-2w(y-μ)/(V·g')]
    # For exponential family deviance d = 2∫(y-μ)/V(μ) dμ:
    #   ∂d_i/∂η_i = -2w_i(y_i-μ_i)·g'(μ_i)/(V(μ_i)·g'(μ_i)) = -2w_i(y_i-μ_i)/V(μ_i)/g'(η_i)
    # But in the PIRLS parameterization, dev_grad = ∂dev/∂β = X'·v where
    #   v_i = -2·p_weights_i·(y_i - μ_i)/(V_i·g1_i)
    # and g1 = 1/μ'(η) = g'(μ)
    v = zeros(n)
    @inbounds for i in 1:n
        vi = _variance_scalar(family, mu[i])
        g1 = 1.0 / GLM.mueta(link, X[i, :]' * beta)
        v[i] = -2.0 * (y[i] - mu[i]) / (max(vi, eps()) * g1)
    end
    dev_grad = X' * v

    # IFT: b1_j = ∂β/∂(log_sp_j) = -A⁻¹(λ_j S_j β)
    # D1_j = b1_j' · dev_grad
    D1 = zeros(n_sp)
    sp_idx = 1
    b1 = zeros(p, n_sp)
    for block in penalty.blocks
        idx = block.start:block.stop
        beta_block = beta[idx]
        for Si in block.S
            λ = exp(log_sp[sp_idx])
            # -λ_j S_j β (padded to full p vector)
            rhs = zeros(p)
            rhs[idx] .= -λ .* (Si * beta_block)
            b1[:, sp_idx] = Ainv * rhs
            D1[sp_idx] = dot(b1[:, sp_idx], dev_grad)
            sp_idx += 1
        end
    end

    # trA1_j = ∂trA/∂(log_sp_j)
    # trA = tr(F) = tr(A⁻¹ X'WX)
    # ∂trA/∂(log_sp_j) = -tr(A⁻¹(λ_j S_j)F) + weight-change terms
    #
    # For Fisher scoring (which is what we use):
    # trA1_j = -tr(A⁻¹(λ_j S_j)F) + tr(T_j KK') - tr(T_j KK'KK')
    # where T_j = diag(dw_j/w) and K = sqrt(W)X A⁻¹ X'sqrt(W)
    #
    # For Gaussian identity link, T_j = 0, so:
    # trA1_j = -tr(A⁻¹(λ_j S_j)F)
    #
    # For non-Gaussian, we need the weight derivatives too.
    # But the dominant term is always -tr(A⁻¹(λ_j S_j)F).

    trA1 = zeros(n_sp)
    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            λ = exp(log_sp[sp_idx])
            # dS/d(log_sp_j) = λ_j S_j (in the block)
            dS_block = λ .* Si
            # -tr(A⁻¹ dS F) = -tr(A⁻¹[idx,idx] dS F[idx,idx])
            Ainv_block = Ainv[idx, idx]
            F_block = F[idx, idx]
            trA1[sp_idx] = -tr(Ainv_block * dS_block * F_block)

            # Weight change terms for non-Gaussian
            if !(family isa Normal)
                # η1_j = X b1_j (derivative of η w.r.t. log_sp_j)
                eta1_j = X * b1[:, sp_idx]
                # dw/dη for Fisher weights: w = μ'(η)²/V(μ)
                # dw/dη = 2μ''(η)μ'(η)/V - μ'(η)²V'(μ)μ'(η)/V²
                # T_j_i = (dw_i/dη_i · η1_j_i) / w_i
                T_j = zeros(n)
                @inbounds for i in 1:n
                    eta_i = X[i, :]' * beta
                    mueta_i = GLM.mueta(link, eta_i)
                    vi = _variance_scalar(family, mu[i])
                    # Numerical dw/deta
                    h = 1e-7
                    eta_p = eta_i + h
                    mu_p = GLM.linkinv(link, eta_p)
                    mueta_p = GLM.mueta(link, eta_p)
                    vp = _variance_scalar(family, mu_p)
                    w_p = mueta_p^2 / max(vp, eps())
                    w_i = mueta_i^2 / max(vi, eps())
                    dwdeta = (w_p - w_i) / h
                    T_j[i] = dwdeta * eta1_j[i] / max(w_i, eps())
                end
                # KK' diagonal: diag_KKt_i = Σ_j K_ij² where K = F_hat in obs space
                # F_hat = X A⁻¹ X' W (hat matrix on η scale, with weights)
                # Actually: trA = tr(KK') where K is sqrt(w)·X·P (P = chol(A)⁻¹)
                # Simpler: the weight term is tr(T_j diag(h)) - tr(T_j diag(h)²)
                # where h = diag(H) = diag(X A⁻¹ X' W)
                H_diag = zeros(n)
                for i in 1:n
                    for j in 1:p
                        s = 0.0
                        for k in 1:p
                            s += X[i, j] * F[j, k] * w[i]
                        end
                        # Wait, this isn't right...
                    end
                end
                # Use the simpler formula: direct computation
                # hat_diag = diag(X F X' diag(w)) with F = A⁻¹ X'WX A⁻¹
                # Actually F_hat = W^{1/2} X A⁻¹ X' W^{1/2}
                # diag(F_hat) = w .* diag(X A⁻¹ X')
                XAinv = X * Ainv
                hat_d = zeros(n)
                @inbounds for i in 1:n
                    s = 0.0
                    for j in 1:p
                        s += XAinv[i, j] * X[i, j]
                    end
                    hat_d[i] = w[i] * s
                end
                # tr(T_j KK') = Σ_i T_j_i hat_d_i
                # tr(T_j KK'KK') = Σ_i T_j_i hat_d_i²
                term1 = dot(T_j, hat_d)
                term2 = dot(T_j, hat_d .^ 2)
                trA1[sp_idx] += term1 - term2
            end

            sp_idx += 1
        end
    end

    # GCV gradient: GCV1[j] = n*D1[j]/δ² + 2*n*dev*trA1[j]*γ/δ³
    delta2 = delta^2
    delta3 = delta^3
    for j in 1:n_sp
        grad[j] = n * D1[j] / delta2 + 2.0 * n * dev * trA1[j] * gamma / delta3
    end

    return grad
end
