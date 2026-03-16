# Shape-Constrained Additive Models (SCAM)
#
# Fits GAMs with shape constraints (monotonicity, convexity/concavity)
# using SCOP-splines (Pya & Wood, 2015). The key modification vs standard
# PIRLS is that constrained coefficients are exponentiated (via exp())
# to ensure positivity, with chain-rule corrections in the gradient/Hessian.
# Matches R scam's default (not.exp=FALSE).

# ============================================================================
# Softplus (notExp) and derivatives — alternative positivity transform
# ============================================================================

"""
    softplus(x; b=1.0, threshold=20.0) -> Float64

Softplus function: `(1/b) * log(1 + exp(b*x))`.
Reverts to identity for `b*x > threshold` for numerical stability.
Alternative to `exp()` for positivity constraints (R's `not.exp=TRUE` mode).
"""
function softplus(x::Real; b::Float64 = 1.0, threshold::Float64 = 20.0)
    bx = b * x
    return bx < threshold ? log1p(exp(bx)) / b : x
end

"""
    softplus_d1(x; b=1.0, threshold=20.0) -> Float64

First derivative of softplus: sigmoid `σ(bx) = exp(bx)/(1+exp(bx))`.
"""
function softplus_d1(x::Real; b::Float64 = 1.0, threshold::Float64 = 20.0)
    bx = b * x
    if bx < threshold
        ebx = exp(bx)
        return ebx / (1.0 + ebx)
    else
        return 1.0
    end
end

"""
    softplus_d2(x; b=1.0, threshold=20.0) -> Float64

Second derivative of softplus: `b * exp(bx) / (1+exp(bx))²`.
"""
function softplus_d2(x::Real; b::Float64 = 1.0, threshold::Float64 = 20.0)
    bx = b * x
    if bx < threshold
        ebx = exp(bx)
        d = 1.0 + ebx
        return b * ebx / (d * d)
    else
        return 0.0
    end
end

"""
    softplus_d3(x; b=1.0, threshold=20.0) -> Float64

Third derivative of softplus: `b² * exp(bx) * (1 - exp(2bx)) / (1+exp(bx))⁴`.
"""
function softplus_d3(x::Real; b::Float64 = 1.0, threshold::Float64 = 20.0)
    bx = b * x
    if bx < threshold
        ebx = exp(bx)
        d = 1.0 + ebx
        return b * b * ebx * (1.0 - ebx * ebx) / (d * d * d * d)
    else
        return 0.0
    end
end

# ============================================================================
# Shape-constrained PIRLS (Newton's method with positivity reparameterization)
# ============================================================================

"""
    ScamControl

Control parameters for SCAM fitting.
"""
struct ScamControl
    epsilon::Float64      # convergence tolerance
    maxit::Int            # max Newton iterations
    outer_maxit::Int      # max outer iterations for smoothing parameters
    trace::Bool           # print progress
    gamma::Float64        # GCV inflation factor
    not_exp::Bool         # if true, use softplus instead of exp (default false)
end

"""
    scam_control(; kwargs...) -> ScamControl
"""
function scam_control(;
    epsilon::Float64 = 1e-7,
    maxit::Int = 200,
    outer_maxit::Int = 200,
    trace::Bool = false,
    gamma::Float64 = 1.0,
    not_exp::Bool = false,
)
    return ScamControl(epsilon, maxit, outer_maxit, trace, gamma, not_exp)
end

"""
    scam_pirls(X, y, S_total, family, link, p_ident; kwargs...) -> NamedTuple

Penalized IRLS with positivity constraints via exp reparameterization.
`p_ident` is a BitVector indicating which coefficients must be positive.

Uses Fisher scoring (PIRLS) with chain-rule corrected design matrix:
- Transform: β̃[iv] = exp(β[iv]) (or softplus if not_exp=true)
- Effective design: X̃ = X * diag(C), where C[iv] = exp(β[iv])
- Working model: (X̃'WX̃ + S)δ = X̃'Wz

EDF is computed using R's full Newton formula with E matrix correction
and eigenvalue check for positive definiteness (Pya & Wood, 2015).
"""
function scam_pirls(
    X::Matrix{Float64},
    y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution,
    link::GLM.Link,
    p_ident::BitVector;
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::ScamControl = scam_control(),
)
    n, p = size(X)
    use_softplus = control.not_exp

    iv = findall(p_ident)  # indices of constrained coefficients
    has_constraints = !isempty(iv)

    # Pre-allocate
    beta = zeros(p)
    beta_t = zeros(p)  # β̃ = transformed beta
    Cdiag = ones(p)    # d(transform)/dβ for chain rule
    C1diag = zeros(p)  # second derivative of transform

    # Initialize coefficients
    if start !== nothing
        copyto!(beta, start)
    end

    # Apply transform
    _apply_transform!(beta_t, beta, iv, use_softplus)
    eta = X * beta_t .+ offset
    mu = [_clamp_mu_scalar(family, GLM.linkinv(link, e)) for e in eta]
    dev_old = _deviance(family, y, mu, weights)

    # Penalty square root for augmented system
    S_eig = eigen(Symmetric(S_total))
    rS_vals = sqrt.(max.(S_eig.values, 0.0))
    rS = Diagonal(rS_vals) * S_eig.vectors'  # p × p

    converged = false
    n_iter = 0

    for iter in 1:(control.maxit)
        n_iter = iter

        # Update chain-rule diagonal: C = d(transform)/dβ
        _update_Cdiag!(Cdiag, C1diag, beta, beta_t, iv, use_softplus)

        # Effective design matrix (chain-rule corrected)
        X_eff = X .* Cdiag'  # n × p, each column j scaled by Cdiag[j]

        # Newton weights: w = w1 * alpha, where
        # w1 = weights / (V(μ) * g'(η)²)  (Fisher weights)
        # alpha = 1 + (y-μ) * (V'(μ)/V(μ) + g''(η)/g'(η))
        eta_tilde = X_eff * beta  # linearized linear predictor
        w1 = zeros(n)
        w = zeros(n)
        alpha = ones(n)
        z = zeros(n)
        @inbounds for i in 1:n
            dm = GLM.mueta(link, eta[i])
            vm = _variance_scalar(family, mu[i])
            g_deriv = 1.0 / dm
            w1[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
            dvar = _dvariance_scalar(family, mu[i])
            d2g = _d2link_scalar(link, mu[i])
            alpha[i] = 1.0 + (y[i] - mu[i]) * (dvar / max(vm, eps()) + d2g * dm)
            w[i] = w1[i] * alpha[i]
            # Pseudodata based on linearized η̃
            z[i] = eta_tilde[i] + (y[i] - mu[i]) * g_deriv
        end

        # E matrix diagonal (Pya & Wood eq. from Appendix 2)
        E_diag = zeros(p)
        @inbounds for j in 1:p
            for i in 1:n
                E_diag[j] += X[i, j] * w1[i] * (1.0 / GLM.mueta(link, eta[i])) * (y[i] - mu[i])
            end
            E_diag[j] *= C1diag[j]
        end

        # Check Newton-step positive definiteness via eigenvalue test
        abs_w = abs.(w)
        I_minus = zeros(n)  # indicators for negative Newton weights
        @inbounds for i in 1:n
            if w[i] < 0
                I_minus[i] = 1.0
                z[i] = eta_tilde[i] - (y[i] - mu[i]) / (GLM.mueta(link, eta[i]) * alpha[i])
            else
                z[i] = eta_tilde[i] + (y[i] - mu[i]) / (GLM.mueta(link, eta[i]) * alpha[i])
            end
        end

        # QR of augmented system with Newton weights
        sqw = sqrt.(abs_w)
        wX = sqw .* X_eff  # n × p
        wX_aug = vcat(wX, rS)  # (n+p) × p
        Q_fact = qr(wX_aug, ColumnNorm())
        R_qr = Q_fact.R
        piv = Q_fact.p
        rpiv = sortperm(piv)

        # Check rank and compute R inverse
        r_diag = abs.(diag(R_qr))
        tol = maximum(r_diag) * sqrt(eps()) * p
        good_cols = r_diag .> tol

        if all(good_cols)
            R_inv = (R_qr \ I(p))[rpiv, :]
        else
            # SVD fallback for rank-deficient system
            R_unpiv = R_qr[:, rpiv]
            svd_r = svd(R_unpiv)
            d_inv = zeros(length(svd_r.S))
            good_sv = svd_r.S .> maximum(svd_r.S) * sqrt(eps())
            d_inv[good_sv] .= 1.0 ./ svd_r.S[good_sv]
            R_inv = svd_r.V * Diagonal(d_inv) * svd_r.U'
        end

        # Eigenvalue check for positive definiteness
        tR_inv = R_inv'
        QtQRER = tR_inv * Diagonal(E_diag) * R_inv
        if any(I_minus .> 0)
            Q_mat = Matrix(Q_fact.Q)[1:n, :]
            QtQRER += 2.0 * (I_minus .* Q_mat)' * (I_minus .* Q_mat)
        end

        ei = eigen(Symmetric(QtQRER))
        d_eig = ei.values
        ok1 = any(d_eig .> 1)

        if ok1
            # Not positive definite: fall back to Fisher scoring (alpha=1)
            sqw_fisher = sqrt.(w1)
            wX_fisher = sqw_fisher .* X_eff
            wX_aug_fisher = vcat(wX_fisher, rS)
            wz_fisher = sqw_fisher .* (eta_tilde .+ (y .- mu) ./ [GLM.mueta(link, e) for e in eta])
            wz_aug = vcat(wz_fisher, zeros(p))
            Q_fact = qr(wX_aug_fisher, ColumnNorm())
        else
            # Newton step is OK
            wz = sqw .* z
            wz_aug = vcat(wz, zeros(p))
        end

        # Solve for new beta
        qty = Q_fact.Q' * wz_aug
        R_qr2 = Q_fact.R
        piv2 = Q_fact.p
        r_diag2 = abs.(diag(R_qr2))
        tol2 = maximum(r_diag2) * sqrt(eps()) * p

        if all(r_diag2 .> tol2)
            beta_piv = R_qr2 \ qty[1:p]
        else
            svd_R2 = svd(R_qr2)
            d_inv2 = zeros(length(svd_R2.S))
            good2 = svd_R2.S .> maximum(svd_R2.S) * sqrt(eps())
            d_inv2[good2] .= 1.0 ./ svd_R2.S[good2]
            beta_piv = svd_R2.V * (d_inv2 .* (svd_R2.U' * qty[1:p]))
        end
        beta_new = zeros(p)
        beta_new[piv2] .= beta_piv

        # Apply transform
        beta_t_new = copy(beta_new)
        _apply_transform!(beta_t_new, beta_new, iv, use_softplus)

        # Compute new eta, mu, deviance
        eta_new = X * beta_t_new .+ offset
        mu_new = [_clamp_mu_scalar(family, GLM.linkinv(link, e)) for e in eta_new]
        dev_new = _deviance(family, y, mu_new, weights)

        # Step halving if deviance increased
        if !isfinite(dev_new) || dev_new > dev_old + control.epsilon * abs(dev_old)
            for _ in 1:25
                beta_new .= 0.5 .* beta .+ 0.5 .* beta_new
                _apply_transform!(beta_t_new, beta_new, iv, use_softplus)
                eta_new .= X * beta_t_new .+ offset
                mu_new .= [_clamp_mu_scalar(family, GLM.linkinv(link, e)) for e in eta_new]
                dev_new = _deviance(family, y, mu_new, weights)
                if isfinite(dev_new) && dev_new <= dev_old + control.epsilon * abs(dev_old)
                    break
                end
            end
        end

        # Convergence check
        coef_change = maximum(abs.(beta_new .- beta)) / (1.0 + maximum(abs.(beta)))
        dev_change = abs(dev_new - dev_old) / (abs(dev_new) + 0.1)

        # Update
        copyto!(beta, beta_new)
        copyto!(beta_t, beta_t_new)
        eta .= eta_new
        mu .= mu_new
        dev_old = dev_new

        if dev_change < control.epsilon || (coef_change < control.epsilon * 10.0)
            converged = true
            break
        end
    end

    # ========================================================================
    # Post-fitting: compute EDF using R's full Newton formula with E correction
    # ========================================================================
    _update_Cdiag!(Cdiag, C1diag, beta, beta_t, iv, use_softplus)
    X_eff = X .* Cdiag'

    # Newton weights at convergence
    w1_final = zeros(n)
    w_final = zeros(n)
    alpha_final = ones(n)
    @inbounds for i in 1:n
        dm = GLM.mueta(link, eta[i])
        vm = _variance_scalar(family, mu[i])
        g_deriv = 1.0 / dm
        w1_final[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
        dvar = _dvariance_scalar(family, mu[i])
        d2g = _d2link_scalar(link, mu[i])
        alpha_final[i] = 1.0 + (y[i] - mu[i]) * (dvar / max(vm, eps()) + d2g * dm)
        w_final[i] = w1_final[i] * alpha_final[i]
    end

    # E matrix at convergence
    E_diag_final = zeros(p)
    @inbounds for j in 1:p
        for i in 1:n
            E_diag_final[j] += X[i, j] * w1_final[i] * (1.0 / GLM.mueta(link, eta[i])) * (y[i] - mu[i])
        end
        E_diag_final[j] *= C1diag[j]
    end

    abs_w_final = abs.(w_final)
    I_minus_final = zeros(n)
    I_plus_final = ones(n)
    @inbounds for i in 1:n
        if w_final[i] < 0
            I_minus_final[i] = 1.0
            I_plus_final[i] = -1.0
        end
    end

    # QR of augmented system for EDF
    sqw_edf = sqrt.(abs_w_final)
    wX1 = sqw_edf .* X_eff
    wX_aug_edf = vcat(wX1, rS)
    qf = qr(wX_aug_edf, ColumnNorm())
    R_edf = qf.R; piv_edf = qf.p; rpiv_edf = sortperm(piv_edf)
    r_diag_edf = abs.(diag(R_edf))
    tol_edf = maximum(r_diag_edf) * sqrt(eps())

    if all(r_diag_edf .> tol_edf)
        R_inv_edf = (R_edf \ I(p))[rpiv_edf, :]
    else
        R_unpiv_edf = R_edf[:, rpiv_edf]
        svd_edf = svd(R_unpiv_edf)
        d_inv_edf = zeros(length(svd_edf.S))
        good_edf = svd_edf.S .> maximum(svd_edf.S) * sqrt(eps())
        d_inv_edf[good_edf] .= 1.0 ./ svd_edf.S[good_edf]
        R_inv_edf = svd_edf.V * Diagonal(d_inv_edf) * svd_edf.U'
    end

    tR_inv_edf = R_inv_edf'
    QtQRER_edf = tR_inv_edf * Diagonal(E_diag_final) * R_inv_edf
    if any(I_minus_final .> 0)
        Q_mat_edf = Matrix(qf.Q)[1:n, :]
        QtQRER_edf += 2.0 * (I_minus_final .* Q_mat_edf)' * (I_minus_final .* Q_mat_edf)
    end

    ei_edf = eigen(Symmetric(QtQRER_edf))
    d_eig_edf = ei_edf.values
    ok1_edf = any(d_eig_edf .> 1)

    local P_edf::Matrix{Float64}
    local K_edf::Matrix{Float64}

    if ok1_edf
        # Fisher fallback for EDF
        sqw_f = sqrt.(w1_final)
        wX_f = sqw_f .* X_eff
        wX_aug_f = vcat(wX_f, rS)
        qf_f = qr(wX_aug_f, ColumnNorm())
        R_f = qf_f.R; piv_f = qf_f.p; rpiv_f = sortperm(piv_f)
        r_diag_f = abs.(diag(R_f)); tol_f = maximum(r_diag_f) * sqrt(eps())
        if all(r_diag_f .> tol_f)
            P_edf = (R_f \ I(p))[rpiv_f, :]
            K_edf = Matrix(qf_f.Q)[1:n, :]
        else
            R_unpiv_f = R_f[:, rpiv_f]
            s_f = svd(R_unpiv_f)
            di_f = zeros(length(s_f.S))
            gf_f = s_f.S .> maximum(s_f.S) * sqrt(eps())
            di_f[gf_f] .= 1.0 ./ s_f.S[gf_f]
            P_edf = s_f.V * Diagonal(di_f) * s_f.U'
            K_edf = Matrix(qf_f.Q)[1:n, :] * s_f.U * Diagonal([g ? 1.0 : 0.0 for g in gf_f])
        end
    else
        # Newton EDF with eigenvalue correction
        Id_inv_r = zeros(p)
        for j in 1:p
            v = 1.0 - d_eig_edf[j]
            Id_inv_r[j] = v > eps() ? 1.0 / sqrt(v) : 0.0
        end
        V_edf = ei_edf.vectors
        P_edf = R_inv_edf * V_edf * Diagonal(Id_inv_r)
        K_edf = Matrix(qf.Q)[1:n, :] * V_edf * Diagonal(Id_inv_r)
    end

    # EDF: edf = rowSums(P * t(K' * diag(L * I_plus) * wX1))
    L_final = [1.0 / a for a in alpha_final]
    KtILQ1R = (L_final .* I_plus_final .* K_edf)' * wX1
    edf_vec = vec(sum(P_edf .* KtILQ1R'; dims = 2))

    # Hat matrix diagonal (approximate, for diagnostics)
    hat_diag = vec(sum(K_edf .^ 2; dims = 2))

    # Use Fisher weights for covariance matrix (Vp = (X'W1X + S)^{-1})
    A = X_eff' * Diagonal(w1_final) * X_eff + S_total
    R_factor = try
        cholesky(Symmetric(A)).U
    catch
        cholesky(Symmetric(A + 1e-6 * I(p))).U
    end

    # Pearson statistic
    pearson = sum(i -> weights[i] * (y[i] - mu[i])^2 / max(_variance_scalar(family, mu[i]), eps()),
        1:n)

    return (
        coefficients = beta,
        coefficients_t = beta_t,
        fitted_values = copy(mu),
        linear_predictor = copy(eta),
        deviance = _deviance(family, y, mu, weights),
        working_weights = w1_final,  # Fisher weights for covariance
        hat_diag = hat_diag,
        edf_vec = edf_vec,
        R = Matrix(R_factor),
        pearson = pearson,
        converged = converged,
        iterations = n_iter,
        Cdiag = copy(Cdiag),
    )
end

"""Apply constraint transform in-place: exp (default) or softplus."""
function _apply_transform!(beta_t::Vector{Float64}, beta::Vector{Float64},
    iv::Vector{Int}, use_softplus::Bool)
    copyto!(beta_t, beta)
    if use_softplus
        for j in iv
            beta_t[j] = softplus(beta[j])
        end
    else
        for j in iv
            beta_t[j] = exp(beta[j])
        end
    end
end

"""Update Cdiag (first derivative) and C1diag (second derivative) of the constraint transform."""
function _update_Cdiag!(Cdiag::Vector{Float64}, C1diag::Vector{Float64},
    beta::Vector{Float64}, beta_t::Vector{Float64},
    iv::Vector{Int}, use_softplus::Bool)
    fill!(C1diag, 0.0)
    fill!(Cdiag, 1.0)
    if use_softplus
        for j in iv
            Cdiag[j] = softplus_d1(beta[j])
            C1diag[j] = softplus_d2(beta[j])
        end
    else
        # exp: all derivatives equal exp(beta) = beta_t
        for j in iv
            Cdiag[j] = beta_t[j]  # exp(β) = β̃
            C1diag[j] = beta_t[j]  # d²/dβ² exp(β) = exp(β)
        end
    end
end

# ============================================================================
# Variance derivative and second link derivative helpers
# ============================================================================

function _dvariance_scalar(family::Normal, mu::Real)
    return 0.0  # Var(Y) = σ², constant
end

function _dvariance_scalar(family::Poisson, mu::Real)
    return 1.0  # Var(Y) = μ, so V'(μ) = 1
end

function _dvariance_scalar(family::BinomialLike, mu::Real)
    return 1.0 - 2.0 * mu  # Var = μ(1-μ), V' = 1-2μ
end

function _dvariance_scalar(family::Gamma, mu::Real)
    return 2.0 * mu  # Var = μ², V' = 2μ
end

function _dvariance_scalar(family::InverseGaussian, mu::Real)
    return 3.0 * mu^2  # Var = μ³, V' = 3μ²
end

function _dvariance_scalar(family::UnivariateDistribution, mu::Real)
    # Numerical fallback
    h = max(abs(mu) * 1e-7, 1e-10)
    return (_variance_scalar(family, mu + h) - _variance_scalar(family, mu - h)) / (2h)
end

function _d2link_scalar(link::IdentityLink, mu::Real)
    return 0.0  # g(μ) = μ, g'' = 0
end

function _d2link_scalar(link::LogLink, mu::Real)
    return -1.0 / (mu * mu)  # g(μ) = log(μ), g'' = -1/μ²
end

function _d2link_scalar(link::LogitLink, mu::Real)
    return (2.0 * mu - 1.0) / (mu * mu * (1.0 - mu)^2)
end

function _d2link_scalar(link::InverseLink, mu::Real)
    return 2.0 / (mu * mu * mu)  # g(μ) = 1/μ, g'' = 2/μ³
end

function _d2link_scalar(link::SqrtLink, mu::Real)
    return -0.25 / (mu^1.5)  # g(μ) = √μ, g'' = -1/(4μ^{3/2})
end

function _d2link_scalar(link::GLM.Link, mu::Real)
    # Numerical fallback
    h = max(abs(mu) * 1e-7, 1e-10)
    g1_plus = 1.0 / GLM.mueta(link, GLM.linkfun(link, mu + h))
    g1_minus = 1.0 / GLM.mueta(link, GLM.linkfun(link, mu - h))
    return (g1_plus - g1_minus) / (2h)
end

# ============================================================================
# SCAM outer iteration: smoothing parameter estimation
# ============================================================================

"""
    scam_outer_iteration(X, y, smooths, penalty, family, link, p_ident; kwargs...)

Outer iteration for SCAM: optimize smoothing parameters using GCV or REML,
with inner Newton loop using scam_pirls.
"""
function scam_outer_iteration(
    X::Matrix{Float64},
    y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::UnivariateDistribution,
    link::GLM.Link,
    p_ident::BitVector;
    method::Symbol = :GCV,
    weights::Vector{Float64} = ones(length(y)),
    control::ScamControl = scam_control(),
)
    n, p = size(X)
    n_sp = length(penalty.sp)

    scale_known = !_needs_scale_estimate(family)
    gamma = control.gamma
    sig2 = scale_known ? 1.0 : -1.0

    # -- Helper: fit at given log(sp) with cold start (avoids false local optima) --
    function _scam_eval(rho)
        S_e = total_penalty(penalty, rho, p)
        r_e = scam_pirls(X, y, S_e, family, link, p_ident;
            weights = weights, control = control)
        edf_e = sum(r_e.edf_vec)
        dev = r_e.deviance
        if scale_known
            sc = dev / n - sig2 + 2.0 * gamma * edf_e * sig2 / n
        else
            sc = n * dev / max(n - gamma * edf_e, 1.0)^2
        end
        return sc, r_e
    end

    if n_sp == 0
        sc0, r0 = _scam_eval(zeros(0))
        return Float64[], r0
    end

    # -- Optimize each sp via golden section search --
    # Cold-start each PIRLS to avoid false optima from warm-starting across
    # distant sp values (SCAM PIRLS is non-convex due to exp transform).
    log_sp = zeros(n_sp)
    best_score = Inf
    best_result = nothing

    for outer in 1:control.outer_maxit
        old_log_sp = copy(log_sp)

        for j in 1:n_sp
            golden = (sqrt(5.0) - 1.0) / 2.0
            a = -8.0; b = 15.0
            c = b - golden * (b - a)
            d = a + golden * (b - a)

            rho_c = copy(log_sp); rho_c[j] = c
            sc_c, r_c = _scam_eval(rho_c)
            rho_d = copy(log_sp); rho_d[j] = d
            sc_d, r_d = _scam_eval(rho_d)

            for _ in 1:100
                if sc_c < sc_d
                    b = d
                    d = c; sc_d = sc_c; r_d = r_c
                    c = b - golden * (b - a)
                    rho_c = copy(log_sp); rho_c[j] = c
                    sc_c, r_c = _scam_eval(rho_c)
                else
                    a = c
                    c = d; sc_c = sc_d; r_c = r_d
                    d = a + golden * (b - a)
                    rho_d = copy(log_sp); rho_d[j] = d
                    sc_d, r_d = _scam_eval(rho_d)
                end
                (b - a) < 1e-5 && break
            end

            if sc_c < sc_d
                log_sp[j] = c
                if sc_c < best_score
                    best_score = sc_c
                    best_result = r_c
                end
            else
                log_sp[j] = d
                if sc_d < best_score
                    best_score = sc_d
                    best_result = r_d
                end
            end
        end

        sp_change = maximum(abs.(log_sp .- old_log_sp))
        sp_change < 1e-5 && break
    end

    # Final fit at optimal sp
    if best_result === nothing
        _, best_result = _scam_eval(log_sp)
    end

    return log_sp, best_result
end

# ============================================================================
# Main scam() function
# ============================================================================

"""
    scam(formula, data; family=Normal(), link=nothing, method=:GCV,
         weights=nothing, control=scam_control())

Fit a shape-constrained additive model (SCAM). Uses SCOP-splines for
smooth terms with shape constraints (monotonicity, convexity/concavity).

# Shape-constrained smooth types
- `s(x, bs=:mpi)` — monotone increasing
- `s(x, bs=:mpd)` — monotone decreasing
- `s(x, bs=:cv)` — concave
- `s(x, bs=:cx)` — convex
- `s(x, bs=:micx)` — monotone increasing + convex
- `s(x, bs=:micv)` — monotone increasing + concave
- `s(x, bs=:mdcx)` — monotone decreasing + convex
- `s(x, bs=:mdcv)` — monotone decreasing + concave

Unconstrained smooth types (`:tp`, `:cr`, `:ps`, etc.) can also be used
alongside constrained ones.

# Example
```julia
using GAM, DataFrames

n = 200
x = sort(rand(n))
y = 2 .* x .+ 0.5 .* x.^2 .+ 0.1 .* randn(n)
df = DataFrame(x=x, y=y)
m = scam(@gam_formula(y ~ s(x, bs=:mpi, k=10)), df)
```
"""
function scam(f::FormulaTerm, data; kwargs...)
    gf = GamFormula(f)
    return scam(gf, data; kwargs...)
end

function scam(gf::GamFormula, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :GCV,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::ScamControl = scam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    _validate_gam_family(family)
    link_eff = link === nothing ? GLM.canonicallink(family) : link
    _validate_link(link_eff, family)

    # Bayesian dispatch
    if priors !== nothing
        f = term(gf.response) ~ term(1)
        return _fit_scam_bayes(f, gf, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains,
            weights = weights)
    end

    method in (:GCV, :UBRE, :REML) ||
        throw(ArgumentError("method must be :GCV, :UBRE, or :REML, got :$method"))

    y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = family)
    f = term(gf.response) ~ term(1)
    n, p = size(X)

    # Build global p_ident
    p_ident = build_p_ident(smooths, n_parametric, p)

    if !any(p_ident)
        # No shape constraints — fall back to standard GAM
        return _fit_gam(y, X, smooths, n_parametric, f, data, family, link_eff,
            method == :GCV ? :GCV : method == :UBRE ? :GCV : :REML,
            :pirls,
            weights === nothing ? nothing : Float64.(weights),
            gam_control(
                epsilon = control.epsilon,
                maxit = control.maxit,
                outer_maxit = control.outer_maxit,
                trace = control.trace,
                gamma = control.gamma,
            ))
    end

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric)

    # Outer iteration
    log_sp, result = scam_outer_iteration(X, y, smooths, penalty, family, link_eff, p_ident;
        method = method, weights = wts, control = control)

    # Post-processing
    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    # Covariance matrices using the effective (chain-rule corrected) design
    Cdiag = result.Cdiag
    X_eff = X * Diagonal(Cdiag)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X_eff' * Diagonal(result.working_weights) * X_eff
    A = XtWX + S_total
    A_chol = try
        cholesky(Symmetric(A))
    catch
        cholesky(Symmetric(A + 1e-6 * I))
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

    # REML/GCV score
    gcv_score = n * result.deviance / (n - control.gamma * edf_total_val)^2

    return GamModel(
        f,
        y, X,
        result.coefficients_t,  # store transformed (actual) coefficients
        result.fitted_values,
        result.linear_predictor,
        wts,
        family, link_eff,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total_val,
        scale_est,
        result.deviance,
        null_dev,
        gcv_score,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged,
        0,
        length(smooths),
        n_parametric,
        gam_control(
            epsilon = control.epsilon,
            maxit = control.maxit,
            outer_maxit = control.outer_maxit,
            trace = control.trace,
            gamma = control.gamma,
        ),
        Tables.columntable(data),
    )
end
