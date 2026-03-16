# Penalized Iteratively Reweighted Least Squares (P-IRLS)
#
# Core inner loop for GAM fitting. Extends GLM's IRLS with penalty terms.
# At each iteration, solves:
#   (X'WX + Σ λ_j S_j) β = X'W z
# where z = η + (y - μ)/g'(μ) is the working response and
# W = diag(w_i * [g'(μ_i)]² / V(μ_i)) are the working weights.
#
# Reference: Wood (2011) JRSSB 73(1), Algorithm in Section 3.

"""
    PirlsResult

Result of P-IRLS inner iteration for fixed smoothing parameters.
"""
struct PirlsResult
    coefficients::Vector{Float64}
    fitted_values::Vector{Float64}
    linear_predictor::Vector{Float64}
    working_weights::Vector{Float64}
    deviance::Float64
    pearson::Float64
    converged::Bool
    iterations::Int
    R::Matrix{Float64}           # R factor of augmented system
    hat_diag::Vector{Float64}    # diagonal of hat matrix
    edf_vec::Vector{Float64}     # per-parameter EDF
end

# Bernoulli is NOT a subtype of Binomial in Distributions.jl, so we need
# a union type to dispatch correctly for binary response models.
const BinomialLike = Union{Binomial, Bernoulli}

# Scalar variance functions to avoid allocations
_variance_scalar(::Normal, mu::Float64) = 1.0
_variance_scalar(::BinomialLike, mu::Float64) = mu * (1.0 - mu)
_variance_scalar(::Poisson, mu::Float64) = mu
_variance_scalar(::Gamma, mu::Float64) = mu * mu
_variance_scalar(::InverseGaussian, mu::Float64) = mu * mu * mu
_variance_scalar(::UnivariateDistribution, mu::Float64) = 1.0

# Scalar clamp functions
_clamp_mu_scalar(::Normal, mu::Float64) = mu
_clamp_mu_scalar(::BinomialLike, mu::Float64) = clamp(mu, eps(), 1.0 - eps())
_clamp_mu_scalar(::Poisson, mu::Float64) = max(mu, eps())
_clamp_mu_scalar(::Gamma, mu::Float64) = max(mu, eps())
_clamp_mu_scalar(::InverseGaussian, mu::Float64) = max(mu, eps())
_clamp_mu_scalar(::UnivariateDistribution, mu::Float64) = mu

# Family-specific initialization (matches R's family$initialize)
_mustart(::Normal, y::Float64, w::Float64) = y
_mustart(::Poisson, y::Float64, w::Float64) = y + 0.1
_mustart(::BinomialLike, y::Float64, w::Float64) = (w * y + 0.5) / (w + 1.0)
_mustart(::Gamma, y::Float64, w::Float64) = max(y, eps())
_mustart(::InverseGaussian, y::Float64, w::Float64) = max(y, eps())
_mustart(::UnivariateDistribution, y::Float64, w::Float64) = y

"""
    pirls(X, y, S_total, family, link; weights, offset, start, control)

Run penalized IRLS to convergence for fixed penalty matrix `S_total`.
"""
function pirls(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution, link::GLM.Link;
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::GamControl = gam_control())

    n, p = size(X)

    # Pre-allocate working buffers
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
    Xw = similar(X)  # buffer for sqrt(W)*X
    wz_buf = zeros(n) # buffer for w.*z

    # Initialize
    if start !== nothing
        copyto!(beta, start)
        mul!(eta, X, beta)
        eta .+= offset
    else
        # Family-specific mustart (matches R's family$initialize)
        @inbounds for i in 1:n
            mu[i] = _mustart(family, y[i], weights[i])
            eta[i] = GLM.linkfun(link, mu[i])
        end
    end

    @inbounds for i in 1:n
        mu[i] = GLM.linkinv(link, eta[i])
    end

    # Initial penalized deviance for step control — use null model (β=0)
    # to match R's gam.fit3 (lines 283-285): old.pdev computed from null.coef
    null_coef = zeros(p)
    null_eta = X * null_coef .+ offset
    null_mu = [_clamp_mu_scalar(family, GLM.linkinv(link, e)) for e in null_eta]
    pdev_old = _deviance(family, y, null_mu, weights) + dot(null_coef, S_total * null_coef)

    converged = false
    n_iter = 0
    # Store old beta/eta for step halving (R's coefold/etaold)
    # R initializes these to null.coef/null.eta, not mustart
    beta_old = copy(null_coef)
    eta_old = copy(null_eta)

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response (in-place, scalar ops)
        @inbounds for i in 1:n
            dm = GLM.mueta(link, eta[i])
            dmu_deta[i] = dm
            vm = _variance_scalar(family, mu[i])
            w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
            z[i] = eta[i] - offset[i] + (y[i] - mu[i]) / dm
        end

        # Build A = X'WX + S_total using BLAS (in-place)
        _build_penalized_system!(A, XtWz, X, w, z, S_total, p, n, Xw, wz_buf)

        # Solve via Cholesky
        A_chol = cholesky!(Symmetric(A))
        ldiv!(beta_new, A_chol, XtWz)

        # Update eta, mu
        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        @inbounds for i in 1:n
            mu_new[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_new[i]))
        end
        dev_new = _deviance(family, y, mu_new, weights)
        penalty_new = dot(beta_new, S_total * beta_new)
        pdev_new = dev_new + penalty_new

        # Step halving if penalized deviance increased (matches R's gam.fit3)
        div_thresh = 10.0 * (0.1 + abs(pdev_old)) * sqrt(eps())
        if pdev_new - pdev_old > div_thresh
            for ii in 1:100
                @inbounds for j in 1:p
                    beta_new[j] = (beta_new[j] + beta_old[j]) / 2.0
                end
                @inbounds for i in 1:n
                    eta_new[i] = (eta_new[i] + eta_old[i]) / 2.0
                    mu_new[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_new[i]))
                end
                dev_new = _deviance(family, y, mu_new, weights)
                penalty_new = dot(beta_new, S_total * beta_new)
                pdev_new = dev_new + penalty_new
                if pdev_new - pdev_old <= div_thresh
                    break
                end
            end
        end

        # Convergence check on penalized deviance (R's gam.fit3 line 447)
        scale_check = _needs_scale_estimate(family) ? dev_new / max(n - p, 1) : 1.0
        crit = abs(pdev_new - pdev_old) / (abs(scale_check) + abs(pdev_new))

        copyto!(beta_old, beta_new)
        copyto!(eta_old, eta_new)
        copyto!(beta, beta_new)
        copyto!(eta, eta_new)
        copyto!(mu, mu_new)
        pdev_old = pdev_new

        if crit < control.epsilon
            converged = true
            break
        end
    end

    # Final unpenalized deviance
    dev_final = _deviance(family, y, mu, weights)

    # Final quantities
    @inbounds for i in 1:n
        dm = GLM.mueta(link, eta[i])
        vm = _variance_scalar(family, mu[i])
        w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
    end

    # Pearson statistic
    pearson = 0.0
    @inbounds for i in 1:n
        vm = _variance_scalar(family, mu[i])
        pearson += weights[i] * (y[i] - mu[i])^2 / max(vm, eps())
    end

    # EDF and hat matrix — reuse A which has X'WX+S from the inner loop
    # Rebuild with final weights
    _build_XtWX_plus_S!(A, X, w, S_total, p, n, Xw)

    # Cholesky of A for R factor and EDF
    A_chol_final = cholesky(Symmetric(A))

    # Extract XtWX = A - S for EDF computation (avoid n×p allocation)
    XtWX = similar(A)
    @inbounds for j in 1:p, k in 1:p
        XtWX[j, k] = A[j, k] - S_total[j, k]
    end

    edf_vec, hat_diag = penalty_edf(X, w, S_total;
        XtWX = XtWX, A_chol = A_chol_final)

    R = Matrix(A_chol_final.U)

    return PirlsResult(
        beta, mu, eta, w, dev_final, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end

"""
    pirls_gaussian(X, y, S_total, XtX, Xty; weights) -> PirlsResult

Direct solve for Gaussian family with identity link (no IRLS iteration needed).
β = (X'WX + S)⁻¹ X'Wy where W=diag(weights).
Accepts pre-computed X'X and X'y to avoid O(np²) recomputation.
"""
function pirls_gaussian(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    XtX::Matrix{Float64}, Xty::Vector{Float64};
    weights::Vector{Float64} = ones(length(y)))
    n, p = size(X)

    # For weighted case: X'WX = Σ w_i x_i x_i', X'Wy = Σ w_i x_i y_i
    # With uniform weights=1, XtWX = XtX, XtWy = Xty
    uniform = all(w -> w ≈ 1.0, weights)
    if uniform
        XtWX = XtX
        XtWy = Xty
    else
        # Recompute with weights (rare for standard Gaussian GAM)
        Xw = similar(X)
        @inbounds for i in 1:n
            sw = sqrt(weights[i])
            for j in 1:p
                Xw[i, j] = X[i, j] * sw
            end
        end
        XtWX = zeros(p, p)
        BLAS.syrk!('U', 'T', 1.0, Xw, 0.0, XtWX)
        @inbounds for j in 1:p, k in (j + 1):p
            XtWX[k, j] = XtWX[j, k]
        end
        XtWy = X' * (weights .* y)
    end

    # A = X'WX + S, solve A β = X'Wy
    A = XtWX + S_total
    A_chol = cholesky(Symmetric(A))
    beta = A_chol \ XtWy

    # Fitted values and linear predictor
    eta = X * beta
    mu = copy(eta)

    # Working weights (= weights for Gaussian/identity)
    w = copy(weights)

    # Deviance = Σ w_i (y_i - μ_i)²
    dev = 0.0
    @inbounds for i in 1:n
        dev += weights[i] * (y[i] - mu[i])^2
    end

    # Pearson = deviance for Gaussian
    pearson = dev

    # EDF and hat diag — reuse pre-computed quantities
    edf_vec, hat_diag = penalty_edf(X, w, S_total;
        XtWX = XtWX, A_chol = A_chol)

    R = Matrix(A_chol.U)

    return PirlsResult(
        beta, mu, eta, w, dev, pearson,
        true, 1, R, hat_diag, edf_vec,
    )
end

"""Build A = X'WX + S and rhs = X'Wz in-place using BLAS."""
function _build_penalized_system!(A::Matrix{Float64}, rhs::Vector{Float64},
    X::Matrix{Float64}, w::Vector{Float64}, z::Vector{Float64},
    S::Matrix{Float64}, p::Int, n::Int,
    Xw::Matrix{Float64} = similar(X), wz::Vector{Float64} = similar(z))

    @inbounds for i in 1:n
        sw = sqrt(w[i])
        wz[i] = w[i] * z[i]
        for j in 1:p
            Xw[i, j] = X[i, j] * sw
        end
    end

    # A = Xw' * Xw via BLAS syrk, then add S
    BLAS.syrk!('U', 'T', 1.0, Xw, 0.0, A)
    @inbounds for j in 1:p
        for k in (j + 1):p
            A[k, j] = A[j, k]
        end
        for k in 1:p
            A[j, k] += S[j, k]
        end
    end

    mul!(rhs, X', wz)
end

"""Build A = X'WX + S in-place (no rhs) using BLAS."""
function _build_XtWX_plus_S!(A::Matrix{Float64},
    X::Matrix{Float64}, w::Vector{Float64},
    S::Matrix{Float64}, p::Int, n::Int,
    Xw::Matrix{Float64} = similar(X))

    @inbounds for i in 1:n
        sw = sqrt(w[i])
        for j in 1:p
            Xw[i, j] = X[i, j] * sw
        end
    end

    BLAS.syrk!('U', 'T', 1.0, Xw, 0.0, A)
    @inbounds for j in 1:p
        for k in (j + 1):p
            A[k, j] = A[j, k]
        end
        for k in 1:p
            A[j, k] += S[j, k]
        end
    end
end

# Distribution-specific helpers

function _deviance(d::Normal, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        r = y[i] - mu[i]
        dev += wt[i] * r * r
    end
    return dev
end

function _deviance(d::BinomialLike, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        mui = clamp(mu[i], eps(), 1 - eps())
        yi = y[i]
        di = 0.0
        if yi > 0
            di += yi * log(yi / mui)
        end
        if yi < 1
            di += (1 - yi) * log((1 - yi) / (1 - mui))
        end
        dev += wt[i] * di
    end
    return 2 * dev
end

function _deviance(d::Poisson, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        yi = y[i]
        if yi > 0
            dev += wt[i] * 2 * (yi * log(yi / mui) - (yi - mui))
        else
            dev += wt[i] * 2 * mui
        end
    end
    return dev
end

function _deviance(d::Gamma, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        dev += wt[i] * 2 * (-log(y[i] / mui) + (y[i] - mui) / mui)
    end
    return dev
end

function _deviance(d::InverseGaussian, y, mu, wt)
    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        r = y[i] - mu[i]
        dev += wt[i] * r * r / (mu[i] * mu[i] * max(y[i], eps()))
    end
    return dev
end

# Fallback for other distributions
function _deviance(d::UnivariateDistribution, y, mu, wt)
    ll = 0.0
    for i in eachindex(y, mu, wt)
        ll += wt[i] * logpdf(d, y[i])
    end
    return -2 * ll
end

function _variance(d::Normal, mu)
    return ones(length(mu))
end

function _variance(d::BinomialLike, mu)
    return mu .* (1 .- mu)
end

function _variance(d::Poisson, mu)
    return copy(mu)
end

function _variance(d::Gamma, mu)
    return mu .^ 2
end

function _variance(d::InverseGaussian, mu)
    return mu .^ 3
end

function _variance(d::UnivariateDistribution, mu)
    return ones(length(mu))
end

function _clamp_mu(::Normal, mu)
    return mu
end

function _clamp_mu(::BinomialLike, mu)
    return clamp.(mu, eps(), 1 - eps())
end

function _clamp_mu(::Poisson, mu)
    return max.(mu, eps())
end

function _clamp_mu(::Gamma, mu)
    return max.(mu, eps())
end

function _clamp_mu(::InverseGaussian, mu)
    return max.(mu, eps())
end

function _clamp_mu(::UnivariateDistribution, mu)
    return mu
end
