# Shape-Constrained Additive Models (SCAM)
#
# Fits GAMs with shape constraints (monotonicity, convexity/concavity)
# using SCOP-splines (Pya & Wood, 2015). The key modification vs standard
# PIRLS is that constrained coefficients are exponentiated (via exp())
# to ensure positivity, with chain-rule corrections in the gradient/Hessian.
# Matches R scam's default (not.exp=FALSE).

"""
Smoothing-parameter search range for shape-constrained (SCAM) fits, in log
units.

Deliberately narrower than the global `LOG_SP_BOUND` of 30. That wider
bound exists so the SHRINKAGE bases (`:ts`, `:cs`) can penalise their null
space hard enough to drop a term entirely, which needs log-lambda around 22.6.
Shape-constrained bases have no null-space penalty and no such requirement, and
SCAM selects by a coarse grid scan followed by golden-section refinement on a
GCV surface that is often multimodal for constrained fits: widening the search
made the bracketing land on a WORSE local minimum (GCV 0.10114 against mgcv's
0.10080, where GAM.jl had previously matched or beaten it).

So the range is a property of the search, not of the representable
smoothing parameters, and the two are kept separate.
"""
const SCAM_LOG_SP_BOUND = 15.0

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
    isfinite(dev_old) || (dev_old = Inf)
    prev_valid = all(_valid_mu_scalar(family, GLM.linkinv(link, e)) for e in eta)

    # Shared step-acceptance policy. SCAM's divergence tolerance is relative
    # (its Newton step in β̃-space is calibrated against the raw deviance),
    # unlike gam.fit3's absolute penalized-deviance threshold.
    step_spec = PirlsStepControl(;
        threshold = _pirls_relative_threshold(control.epsilon),
        max_halvings = 25)

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

        # Compute new eta, mu, deviance, tracking family-domain validity.
        # Step halving uses the shared acceptance policy: SCAM keeps its raw
        # deviance criterion (its transformed Newton step is calibrated
        # against it), but now inherits valid-μ enforcement — a proposal may
        # not leave the family's mean domain while the previous iterate was
        # inside it (mgcv's validmu). Without this, Gamma/InverseGaussian
        # fits under the canonical inverse link "converged" onto silently
        # clamped means with a nonsense scale.
        eta_new = X * beta_t_new .+ offset
        mu_new = similar(eta_new)

        recompute_scam! = function (b)
            _apply_transform!(beta_t_new, b, iv, use_softplus)
            eta_new .= X * beta_t_new .+ offset
            ok = true
            @inbounds for i in 1:n
                li = GLM.linkinv(link, eta_new[i])
                ok &= _valid_mu_scalar(family, li)
                mu_new[i] = _clamp_mu_scalar(family, li)
            end
            return (_deviance(family, y, mu_new, weights), ok)
        end

        dev_new, valid_new, step_ok, _ =
            pirls_halve!(beta_new, beta, recompute_scam!, step_spec,
                dev_old, prev_valid)

        if !step_ok
            # Keep the last in-domain iterate rather than accept divergence
            copyto!(beta_new, beta)
            dev_new, valid_new = recompute_scam!(beta_new)
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
        prev_valid = valid_new

        # `pirls` guards this criterion with `n_halvings <= 1` (see
        # src/pirls.jl). That guard is deliberately NOT used here, and the
        # reason is measured rather than assumed: halving is pervasive and
        # legitimate in SCAM, because the exponentiated-coefficient transform
        # makes the raw Newton step routinely overshoot. Instrumenting the SCAM
        # suite found 417 of 598 convergence declarations came after 2 or more
        # halvings (61 at 16, 25 at 18), alongside 2968 halved iterations that
        # did not converge. Requiring a near-full step there does not sharpen
        # convergence, it prevents it: on a 38-fit control sweep it flipped 6
        # fits to `converged = false` and drove one to `edf = NaN`.
        #
        # What IS wrong is treating an outright step FAILURE as convergence.
        # When `pirls_halve!` exhausts all 25 halvings it returns
        # `step_ok = false`, and the branch above then resets
        # `beta_new = beta` and falls through WITHOUT breaking -- so
        # `dev_change` is exactly 0 and the old code reported success for a
        # step that had failed completely. Requiring `step_ok` closes that hole
        # while leaving legitimately damped steps alone.
        if (dev_change < control.epsilon || (coef_change < control.epsilon * 10.0)) &&
           step_ok
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
    # Escalating-ridge recovery, shared with the other fitters: a single
    # fixed ridge aborts the fit where `gam` would recover.
    R_factor = _protected_cholesky!(copy(A)).U

    # Pearson statistic
    pearson = sum(i -> weights[i] * (y[i] - mu[i])^2 / max(_variance_scalar(family, mu[i]), eps()),
        1:n)

    # Honest reporting of family-domain violations (mirrors `pirls`): if the
    # final iterate still maps outside the family's mean domain, the fitted
    # means are boundary-clamped and the scale estimate is meaningless, so
    # this is not a converged fit.
    n_invalid = count(i -> !_valid_mu_scalar(family, GLM.linkinv(link, eta[i])), 1:n)
    if n_invalid > 0
        @warn "SCAM P-IRLS finished with $n_invalid observation(s) whose " *
              "linear predictor maps outside the $(nameof(typeof(family))) " *
              "mean domain under $(nameof(typeof(link))); fitted values are " *
              "clamped to the boundary there and the scale estimate is not " *
              "usable. Consider a different link (e.g. LogLink)." maxlog = 1
        converged = false
    end

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

Outer iteration for SCAM smoothing-parameter selection.

- `method = :GCV` (default) or `:UBRE`: direct minimization of the GCV/UBRE
  criterion by a cyclic coarse global scan over each log smoothing parameter
  followed by golden-section refinement inside the bracketing interval
  (matching R scam, which optimizes GCV/UBRE rather than REML).
- `method = :REML` or `:EFS`: Extended Fellner-Schall updates (REML-flavored),
  with inner Newton loop using `scam_pirls`.

Any other `method` throws an `ArgumentError`.
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
    offset::Vector{Float64} = zeros(length(y)),
    control::ScamControl = scam_control(),
)
    n, p = size(X)
    n_sp = length(penalty.sp)

    method in (:GCV, :UBRE, :REML, :EFS) || throw(ArgumentError(
        "SCAM method must be :GCV, :UBRE, :REML, or :EFS, got :$method"))
    if method == :UBRE && _needs_scale_estimate(family)
        throw(ArgumentError(
            ":UBRE assumes a known scale parameter; use :GCV or :REML " *
            "for families with an estimated scale."))
    end

    if n_sp == 0
        S_total = zeros(p, p)
        result = scam_pirls(X, y, S_total, family, link, p_ident;
            weights = weights, offset = offset, control = control)
        return Float64[], result, 0, NaN
    end

    if method == :GCV || method == :UBRE
        return _scam_criterion_outer(X, y, penalty, family, link, p_ident;
            method = method, weights = weights, offset = offset,
            control = control)
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing
    Xw_buf = similar(X)
    A_buf = zeros(p, p)

    n_outer = 0
    score_prev = Inf
    for outer in 1:control.outer_maxit
        n_outer = outer
        S_total = total_penalty(penalty, log_sp, p)

        result = scam_pirls(X, y, S_total, family, link, p_ident;
            weights = weights, offset = offset,
            start = prev_result === nothing ? nothing : prev_result.coefficients,
            control = control)

        beta = result.coefficients
        w = result.working_weights

        # Scale estimation
        edf_total = sum(result.edf_vec)
        if _needs_scale_estimate(family)
            scale_est = max(result.pearson / (n - edf_total), _scale_floor(y))
        else
            scale_est = 1.0
        end

        # EFS update (same formula as standard GAM outer iteration), but with
        # the chain-rule-corrected effective design X_eff = X * diag(Cdiag)
        # (matches the penalized information used for Vp post-fit).
        X_eff_efs = X .* result.Cdiag'
        _build_XtWX_plus_S!(A_buf, X_eff_efs, w, S_total, p, n, Xw_buf)
        A_chol = _protected_cholesky!(copy(A_buf))
        Ainv = inv(A_chol)

        log_sp_new = copy(log_sp)
        sp_idx = 1
        max_change = 0.0

        for block in penalty.blocks
            # Per-penalty λⱼ·∂log|S_λ|₊/∂λⱼ: equals block.rank only for
            # single-penalty blocks (all current SCAM smooths); the general
            # form future-proofs multi-penalty constrained blocks the same
            # way the core and bam EFS loops were fixed.
            ldet_derivs = _block_logdet_derivs(block,
                view(log_sp, sp_idx:(sp_idx + length(block.S) - 1)))

            for (jS, Si) in enumerate(block.S)
                if penalty.fixed[sp_idx]
                    sp_idx += 1
                    continue
                end
                λ = exp(log_sp[sp_idx])

                # Sub-penalty extent, not block width: a narrow penalty at an
                # offset would otherwise raise `DimensionMismatch` here. For a
                # full-width penalty (every SCAM block today) this range IS
                # `block.start:block.stop`, so the arithmetic is unchanged.
                # Views rather than copies, since this now runs per penalty.
                idx = _sub_penalty_idx(block, jS)
                beta_block = view(beta, idx)
                Ainv_block = view(Ainv, idx, idx)

                # Expressions left exactly as they were: `dot(b, S, b)` and
                # `dot(A, S)` are mathematically equal but sum in a different
                # order, which would perturb the last bits.
                bSb = dot(beta_block, Si * beta_block)
                # tr(A⁻¹S) = Σᵢⱼ A⁻¹ᵢⱼSᵢⱼ for symmetric S — O(p²), not O(p³)
                trVS = sum(Ainv_block .* Si)

                a = max(0.0, ldet_derivs[jS] / λ - trVS)

                if a > 0 && bSb > eps() * max(sum(abs2, beta_block), eps())
                    r = scale_est * a / bSb
                    log_sp_new[sp_idx] = clamp(
                        log_sp[sp_idx] + log(max(r, 1e-15)),
                        -SCAM_LOG_SP_BOUND, SCAM_LOG_SP_BOUND)
                end

                max_change = max(max_change,
                    abs(log_sp_new[sp_idx] - log_sp[sp_idx]))
                sp_idx += 1
            end
        end

        log_sp .= log_sp_new
        prev_result = result

        # Score-based convergence, as in the standard and bam outer loops:
        # the criterion can be numerically flat over a wide range of log-sp,
        # in which case the sp-change test alone keeps walking the ridge
        # (and can reach the ±15 clamp) long after the fit stopped changing.
        score_cur = result.deviance / max(n - sum(result.edf_vec), 1.0)^2
        if max_change < 1e-5 ||
           (outer > 1 && abs(score_cur - score_prev) <
                         1e-8 * (abs(score_prev) + 0.1))
            break
        end
        score_prev = score_cur
    end

    # Final fit at converged sp
    S_total = total_penalty(penalty, log_sp, p)
    final_result = scam_pirls(X, y, S_total, family, link, p_ident;
        weights = weights, offset = offset,
        start = prev_result === nothing ? nothing : prev_result.coefficients,
        control = control)

    return log_sp, final_result, n_outer, NaN
end

"""
    _scam_criterion_outer(X, y, penalty, family, link, p_ident; kwargs...)

Direct GCV/UBRE smoothing-parameter selection for SCAM, cycled over the log
smoothing parameters. Each coordinate is optimized by a coarse global scan
over the full log-sp range (bracketing the global minimum — the criterion
along a coordinate need not be unimodal, and pure golden section from cold
brackets has been observed to converge to a criterion-worse point than
R scam's optimum) followed by golden-section refinement inside the bracket.
Every criterion evaluation is a full constrained PIRLS fit, warm-started
from the incumbent coefficients. Returns
`(log_sp, final_result, n_cycles, criterion)`.
"""
function _scam_criterion_outer(
    X::Matrix{Float64},
    y::Vector{Float64},
    penalty::PenaltySetup,
    family::UnivariateDistribution,
    link::GLM.Link,
    p_ident::BitVector;
    method::Symbol = :GCV,
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    control::ScamControl = scam_control(),
)
    n, p = size(X)
    n_sp = length(penalty.sp)
    log_sp = copy(penalty.sp)
    warm = Ref{Union{Nothing, Vector{Float64}}}(nothing)

    # `cold = true` ignores the warm start. Warm starts are only safe for
    # LOCAL moves: warm-starting a distant sp evaluation can terminate the
    # constrained PIRLS prematurely and report an inconsistent (deviance, edf)
    # pair, i.e. a spuriously low criterion (observed: the warm evaluation at
    # a huge sp reported the wiggly fit's deviance with the linear fit's edf).
    # Unconverged warm evaluations are refit cold, and only converged results
    # update the warm start.
    fit_at = function (ls::Vector{Float64}; cold::Bool = false)
        S_total = total_penalty(penalty, ls, p)
        result = scam_pirls(X, y, S_total, family, link, p_ident;
            weights = weights, offset = offset,
            start = cold ? nothing : warm[],
            control = control)
        if !cold && !result.converged
            result = scam_pirls(X, y, S_total, family, link, p_ident;
                weights = weights, offset = offset, control = control)
        end
        if result.converged
            warm[] = result.coefficients
        end
        return result
    end

    score_of = function (result)
        edf = sum(result.edf_vec)
        if method == :GCV
            return n * result.deviance / max(n - control.gamma * edf, 1e-8)^2
        else  # :UBRE with known scale φ = 1 (mgcv: D/n + 2γφ·edf/n − φ)
            return result.deviance / n + 2.0 * control.gamma * edf / n - 1.0
        end
    end

    invphi = (sqrt(5.0) - 1.0) / 2.0
    n_cycles = 0
    # Fixed STEP, not fixed count: this is a coarse global scan that brackets
    # the optimum for the golden-section refinement below, so its usefulness
    # depends on resolution. Tying `length` to `LOG_SP_BOUND` silently halved
    # that resolution (2.5 -> 5.0 log units) when the bound was widened from 15
    # to 30 for the shrinkage bases, coarsening every SCAM fit for a reason
    # that has nothing to do with SCAM.
    const_step = 2.5
    grid = collect(-SCAM_LOG_SP_BOUND:const_step:SCAM_LOG_SP_BOUND)
    for cycle in 1:min(control.outer_maxit, 6)
        n_cycles = cycle
        max_change = 0.0
        for j in 1:n_sp
            penalty.fixed[j] && continue
            score_j = function (ls_j::Float64; cold::Bool = false)
                ls = copy(log_sp)
                ls[j] = ls_j
                return score_of(fit_at(ls; cold = cold))
            end
            # Coarse global scan with COLD starts (independent fits give the
            # honest criterion landscape; incumbent included so a cycle can
            # never move to a worse point), then warm-started golden-section
            # refinement in the bracket around the scan minimum.
            cand = sort(unique(vcat(grid, log_sp[j])))
            scores = [score_j(v; cold = true) for v in cand]
            ibest = argmin(scores)
            xbest, fbest = cand[ibest], scores[ibest]
            a = cand[max(ibest - 1, 1)]
            b = cand[min(ibest + 1, length(cand))]
            c = b - invphi * (b - a)
            d = a + invphi * (b - a)
            fc = score_j(c)
            fd = score_j(d)
            for _ in 1:25
                if fc < fd
                    b, d, fd = d, c, fc
                    c = b - invphi * (b - a)
                    fc = score_j(c)
                else
                    a, c, fc = c, d, fd
                    d = a + invphi * (b - a)
                    fd = score_j(d)
                end
                (b - a) < 0.05 && break
            end
            xg, fg = fc < fd ? (c, fc) : (d, fd)
            if fg < fbest
                xbest, fbest = xg, fg
            end
            max_change = max(max_change, abs(xbest - log_sp[j]))
            log_sp[j] = xbest
        end
        if control.trace
            @info "SCAM $method cycle $cycle: log_sp=$(round.(log_sp, digits=3)), max_change=$(round(max_change, sigdigits=3))"
        end
        max_change < 0.05 && break
    end

    final_result = fit_at(log_sp)
    if !final_result.converged
        final_result = fit_at(log_sp; cold = true)
    end
    return log_sp, final_result, n_cycles, score_of(final_result)
end

# ============================================================================
# Internal SCAM fitting core (called by gam() auto-detection and scam())
# ============================================================================

"""
    _fit_scam(y, X, smooths, n_parametric, f, data, family, link, method, weights, control)

Internal SCAM fitting function. Builds the global `p_ident` vector,
runs the SCAM outer iteration (EFS + constrained PIRLS), and assembles
the `GamModel` result. Called by `gam()` when shape constraints are
detected and by `scam()` directly.
"""
function _fit_scam(y, X, smooths, n_parametric, f, data, family, link, method, weights, control;
    offset = nothing)
    n, p = size(X)

    # Build global p_ident
    p_ident = build_p_ident(smooths, n_parametric, p)

    if !any(p_ident)
        # No shape constraints — fall back to standard GAM (which accepts
        # :REML, :ML, :GCV, and :UBRE directly)
        return _fit_gam(y, X, smooths, n_parametric, f, data, family, link,
            method in (:REML, :ML, :GCV, :UBRE) ? method : :REML,
            :pirls, weights,
            gam_control(
                epsilon = control.epsilon,
                maxit = control.maxit,
                outer_maxit = control.outer_maxit,
                trace = control.trace,
                gamma = control.gamma,
            ); offset = offset)
    end

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))
    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch(
        "offset length $(length(off)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric)

    # Outer iteration
    log_sp, result, outer_iters, criterion_val = scam_outer_iteration(X, y, smooths, penalty, family, link, p_ident;
        method = method, weights = wts, offset = off, control = control)

    # Post-processing
    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    # Covariance matrices using the effective (chain-rule corrected) design
    Cdiag = result.Cdiag
    X_eff = X * Diagonal(Cdiag)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X_eff' * Diagonal(result.working_weights) * X_eff
    A = XtWX + S_total
    A_chol = _protected_cholesky!(copy(A))
    Vp = inv(A_chol)
    F = Vp * XtWX
    # Frequentist covariance Ve = F·Vp (mgcv's Ve <- F %*% Vb), not F·Vp·F'
    Ve = Symmetric(F * Vp) |> Matrix

    if _needs_scale_estimate(family)
        scale_est = result.pearson / (n - edf_total_val)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    # Delta method through the shape-constraint transform: Vp/Ve above are the
    # covariance of the UNCONSTRAINED working parameters β, but the model
    # stores the transformed coefficients β̃ (exp(β) on constrained entries).
    # Following R scam's `Vp.t = C %*% Vb %*% C` with C = diag(∂β̃/∂β)
    # (Pya & Wood 2015), transform so that stderror/confint/predict-SEs refer
    # to the stored coefficients.
    C_delta = Diagonal(Cdiag)
    Vp = Matrix(Symmetric(C_delta * Vp * C_delta))
    Ve = Matrix(Symmetric(C_delta * Ve * C_delta))

    null_dev = _null_deviance(family, y, wts)

    # No REML/LAML score is computed for SCAM fits, so the reml slot is NaN.
    # For :GCV/:UBRE fits the optimized criterion value is stored in the
    # model's `criterion` field (NaN for the EFS/:REML path).

    return GamModel(
        f,
        y, X,
        result.coefficients_t,  # store transformed (actual) coefficients
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
        NaN,
        criterion_val,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged,
        outer_iters,
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

# ============================================================================
# Main scam() function — convenience wrapper around gam()
# ============================================================================

"""
    scam(formula, data; family=Normal(), link=nothing, method=:GCV,
         weights=nothing, control=scam_control())

Fit a shape-constrained additive model (SCAM). Uses SCOP-splines for
smooth terms with shape constraints (monotonicity, convexity/concavity).

This is a convenience wrapper around `gam()` with SCAM-specific defaults.
Calling `gam()` with shape-constrained smooth terms will automatically
use the SCAM fitting algorithm.

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

# Notes
- Standard errors are conditional on the estimated smoothing parameters and
  constraints, as in R's scam. Interval coverage for the fitted *function* is
  close to nominal: a 400-replicate study of a monotone truth gave 0.948
  (± 0.006) for nominal-95% pointwise intervals. (A parametric bootstrap of
  the *coefficients* gives reported-SE/empirical-sd ratios near 0.6, but that
  measures coefficient spread under smoothing-parameter re-selection through
  the `exp` reparameterization, inflated by the constraint boundary — it is
  not the quantity that governs interval coverage.)
- SCAM fits store `NaN` in the model's `reml` field — no REML/LAML score is
  computed. For `method = :GCV`/`:UBRE` the optimized criterion value is
  stored in the model's `criterion` field; it can also be recomputed as
  `nobs(m) * deviance(m) / (nobs(m) − γ·edf)²`.

# Example
```julia
using GAM, DataFrames

n = 200
x = sort(rand(n))
y = 2 .* x .+ 0.5 .* x.^2 .+ 0.1 .* randn(n)
df = DataFrame(x=x, y=y)

# These are equivalent:
m = scam(@formulak(y ~ s(x, bs=:mpi, k=10)), df)
m = gam(@formulak(y ~ s(x, bs=:mpi, k=10)), df)
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
    nchains::Int = 4,
    seed::Union{Integer, Nothing} = nothing)

    # Input validation
    _validate_data_lengths(data)
    _validate_response_in_data(gf.response, data)
    _validate_has_smooths(gf.smooth_specs)
    _validate_scam_has_constraints(gf.smooth_specs)
    _validate_formula_smooths(gf.smooth_specs, data)
    _validate_gam_family(family)
    link_eff = link === nothing ? GLM.canonicallink(family) : link
    _validate_link(link_eff, family)

    # Bayesian dispatch
    if priors !== nothing
        return _fit_scam_bayes(gf, gf, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains,
            weights = weights, seed = seed)
    end

    method in (:GCV, :UBRE, :REML) ||
        throw(ArgumentError("method must be :GCV, :UBRE, or :REML, got :$method"))

    y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = family)
    _validate_response(y, family)

    return _fit_scam(y, X, smooths, n_parametric, gf, data, family, link_eff,
        method, weights, control)
end
