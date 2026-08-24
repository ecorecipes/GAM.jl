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

# Scalar clamp functions. These map an unclamped mean into the family's
# domain; they must return a *finite* value, since an inverse link at η = 0
# yields μ = ±Inf and a clamp that passes Inf through is not a clamp — the
# non-finite value then propagates into the working system as NaN.
const _MU_UPPER = 1.0 / eps()

_clamp_mu_positive(mu::Float64) =
    isnan(mu) ? eps() : clamp(mu, eps(), _MU_UPPER)

_clamp_mu_scalar(::Normal, mu::Float64) = mu
_clamp_mu_scalar(::BinomialLike, mu::Float64) =
    isnan(mu) ? 0.5 : clamp(mu, eps(), 1.0 - eps())
_clamp_mu_scalar(::Poisson, mu::Float64) = _clamp_mu_positive(mu)
_clamp_mu_scalar(::Gamma, mu::Float64) = _clamp_mu_positive(mu)
_clamp_mu_scalar(::InverseGaussian, mu::Float64) = _clamp_mu_positive(mu)
_clamp_mu_scalar(::UnivariateDistribution, mu::Float64) = mu

# Family-domain validity of an (unclamped) mean, mirroring mgcv's validmu():
# used to force step-halving away from iterates whose linear predictor maps
# outside the family's mean domain (e.g. Gamma with the canonical inverse
# link when η crosses zero).
_valid_mu_scalar(::Normal, mu::Float64) = isfinite(mu)
_valid_mu_scalar(::BinomialLike, mu::Float64) = isfinite(mu) && 0.0 < mu < 1.0
_valid_mu_scalar(::Poisson, mu::Float64) = isfinite(mu) && mu > 0.0
_valid_mu_scalar(::Gamma, mu::Float64) = isfinite(mu) && mu > 0.0
_valid_mu_scalar(::InverseGaussian, mu::Float64) = isfinite(mu) && mu > 0.0
_valid_mu_scalar(::UnivariateDistribution, mu::Float64) = isfinite(mu)

"""
    PirlsStepControl(; validity, threshold, accept, max_halvings)

Step-acceptance policy shared by the P-IRLS variants (`pirls`, `pirls_bam`,
`scasm_pirls`, `scam_pirls`).

Each variant differs in how it *proposes* a coefficient vector (dense solve,
chunked accumulation, constrained QP, transformed Newton step) but they all
accept or reject that proposal the same way: reject if the objective rose by
more than a tolerance, or if the proposal left the family's mean domain while
the previous iterate was inside it, then halve toward the previous iterate.

Keeping that rule in one place is what stops a fix landing in one loop and
not the others — the failure mode that let shape-constrained fits accept
out-of-domain means long after `pirls` had been taught not to.

# Fields
- `validity`: `(family, μ) -> Bool`, mgcv's `validmu` analogue
- `threshold`: `(obj_old) -> Float64`, divergence tolerance
- `accept`: `(obj_new, obj_old, thresh, valid_new, prev_valid) -> Bool`
- `max_halvings`: halvings before declaring step failure
"""
struct PirlsStepControl{T, A}
    threshold::T
    accept::A
    max_halvings::Int
end

"""Default acceptance: objective must not rise past `thresh`, and a valid
previous iterate may not be traded for an invalid proposal (mgcv gam.fit3)."""
_pirls_default_accept(obj_new, obj_old, thresh, valid_new, prev_valid) =
    isfinite(obj_new) && (obj_new - obj_old <= thresh) &&
    !(prev_valid && !valid_new)

"""gam.fit3's divergence tolerance, `10(0.1 + |pdev|)√eps`."""
_pirls_gamfit3_threshold(obj_old) = 10.0 * (0.1 + abs(obj_old)) * sqrt(eps())

"""Relative tolerance used by the constrained/transformed variants."""
_pirls_relative_threshold(eps_rel) = obj_old -> eps_rel * abs(obj_old)

function PirlsStepControl(; threshold = _pirls_gamfit3_threshold,
    accept = _pirls_default_accept, max_halvings::Int = 100)
    return PirlsStepControl(threshold, accept, max_halvings)
end

"""
    pirls_halve!(beta_new, beta_old, recompute!, spec, obj_old, prev_valid)
        -> (obj, valid, accepted, halvings)

Accept `beta_new`, or repeatedly interpolate it toward `beta_old`
(`β ← (β + β_old)/2`, i.e. step factors 1, ½, ¼, …) until `spec.accept`
is satisfied or `spec.max_halvings` is exhausted.

`recompute!(beta) -> (objective, valid)` is supplied by the caller and is
responsible for refreshing whatever state the objective depends on (η, μ,
transformed coefficients, …) — that is the only part which genuinely differs
between the P-IRLS variants.

Guarantees the returned iterate is never out-of-domain when `prev_valid` is
`true` and any halving succeeded.
"""
function pirls_halve!(beta_new::Vector{Float64}, beta_old::Vector{Float64},
    recompute!, spec::PirlsStepControl, obj_old::Float64, prev_valid::Bool)

    thresh = spec.threshold(obj_old)
    obj, valid = recompute!(beta_new)
    spec.accept(obj, obj_old, thresh, valid, prev_valid) &&
        return (obj, valid, true, 0)

    p = length(beta_new)
    for h in 1:(spec.max_halvings)
        @inbounds for j in 1:p
            beta_new[j] = 0.5 * (beta_new[j] + beta_old[j])
        end
        obj, valid = recompute!(beta_new)
        spec.accept(obj, obj_old, thresh, valid, prev_valid) &&
            return (obj, valid, true, h)
    end
    return (obj, valid, false, spec.max_halvings)
end

"""
    _pirls_working!(w, z, y, mu, eta, offset, weights, family, link;
                    dmu_deta = nothing)

Working weights and working response for one P-IRLS iteration, shared by the
dense (`pirls`), chunked (`pirls_bam`) and constrained (`scasm_pirls`)
variants.

`dμ/dη` is floored at `eps()` in magnitude, mirroring R's `family\$mu.eta`:
without it a saturated observation (|η| large, or η → 0 under an inverse
link) drives `z = η + (y − μ)/(dμ/dη)` to `±Inf`/`NaN` and poisons the whole
normal-equation system — which is how an out-of-domain iterate turns into an
opaque LAPACK or QP failure rather than a step rejection.
"""
function _pirls_working!(w::Vector{Float64}, z::Vector{Float64},
    y::Vector{Float64}, mu::Vector{Float64}, eta::Vector{Float64},
    offset::Vector{Float64}, weights::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link;
    dmu_deta::Union{Vector{Float64}, Nothing} = nothing)

    @inbounds for i in eachindex(y)
        dm = GLM.mueta(link, eta[i])
        if !isfinite(dm) || abs(dm) < eps()
            dm = isfinite(dm) ? (dm < 0 ? -eps() : eps()) :
                 (dm < 0 ? -1 / eps() : 1 / eps())
        end
        dmu_deta === nothing || (dmu_deta[i] = dm)
        vm = _variance_scalar(family, mu[i])
        w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
        zi = eta[i] - offset[i] + (y[i] - mu[i]) / dm
        z[i] = isfinite(zi) ? zi : eta[i] - offset[i]
    end
    return nothing
end

"""
    _protected_cholesky!(A) -> Cholesky

Cholesky factorization of `Symmetric(A)` (mutating `A`), with an escalating
ridge fallback when `A` is numerically indefinite (rank deficiency,
concurvity, or λ → 0 boundaries). Mirrors mgcv's use of regularized solves
where plain Cholesky would abort the whole fit.
"""
function _protected_cholesky!(A::Matrix{Float64})
    A_save = copy(A)
    try
        return cholesky!(Symmetric(A))
    catch e
        e isa LinearAlgebra.PosDefException || rethrow()
    end
    p = size(A_save, 1)
    ridge = max(tr(A_save) / p, 1.0) * 1e-10
    for _ in 1:8
        copyto!(A, A_save)
        @inbounds for i in 1:p
            A[i, i] += ridge
        end
        try
            return cholesky!(Symmetric(A))
        catch e
            e isa LinearAlgebra.PosDefException || rethrow()
            ridge *= 100.0
        end
    end
    throw(LinearAlgebra.PosDefException(1))
end

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
    penalty_buf = zeros(p) # buffer for S_total * beta

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
    # For links whose null predictor is outside the mean domain (Gamma or
    # InverseGaussian with the canonical inverse link put η = 0 → μ = ∞) the
    # null baseline is not finite. A non-finite baseline makes the divergence
    # test vacuous, so fall back to the in-domain starting iterate.
    null_coef = zeros(p)
    null_eta = X * null_coef .+ offset
    null_mu = [_clamp_mu_scalar(family, GLM.linkinv(link, e)) for e in null_eta]
    pdev_old = _deviance(family, y, null_mu, weights)
    if !isfinite(pdev_old)
        # No usable baseline: admit the first proposal unconditionally (the
        # divergence test only becomes meaningful once a finite penalized
        # deviance exists) while still rejecting non-finite proposals.
        pdev_old = Inf
    end

    converged = false
    n_iter = 0
    # Store old beta/eta for step halving (R's coefold/etaold)
    # R initializes these to null.coef/null.eta, not mustart
    beta_old = copy(null_coef)
    eta_old = copy(null_eta)
    # Validity (mgcv validmu) of the previous iterate's unclamped means:
    # enforcement below only halves toward iterates that are themselves valid.
    prev_valid = all(_valid_mu_scalar(family, GLM.linkinv(link, e)) for e in null_eta)

    # gam.fit3 step-acceptance policy (shared with bam/scasm/scam variants)
    step_spec = PirlsStepControl(; max_halvings = 100)

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response (in-place, scalar ops).
        # dμ/dη is floored at eps() in magnitude (as in R's family$mu.eta),
        # so saturated observations (|η| huge, dμ/dη underflowing to 0)
        # cannot give z = ±Inf and poison X'Wz.
        _pirls_working!(w, z, y, mu, eta, offset, weights, family, link;
            dmu_deta = dmu_deta)

        # Build A = X'WX + S_total using BLAS (in-place)
        _build_penalized_system!(A, XtWz, X, w, z, S_total, p, n, Xw, wz_buf)

        # Solve via Cholesky, with escalating ridge fallback for
        # near-singular A (rank deficiency / λ → 0 boundaries)
        A_chol = _protected_cholesky!(A)
        ldiv!(beta_new, A_chol, XtWz)

        # Update eta, mu (tracking family-domain validity, mgcv's validmu)
        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        valid_new = true
        @inbounds for i in 1:n
            li = GLM.linkinv(link, eta_new[i])
            valid_new &= _valid_mu_scalar(family, li)
            mu_new[i] = _clamp_mu_scalar(family, li)
        end
        dev_new = _deviance(family, y, mu_new, weights)
        mul!(penalty_buf, S_total, beta_new)
        penalty_new = dot(beta_new, penalty_buf)
        pdev_new = dev_new + penalty_new

        # Step halving if the penalized deviance increased (R's gam.fit3), or
        # if the proposal left the family's mean domain while the previous
        # iterate was valid (mgcv halves on !validmu(mu) the same way).
        # The acceptance rule lives in `pirls_halve!`, shared with the bam,
        # scasm and scam variants.
        recompute! = function (b)
            mul!(eta_new, X, b)
            eta_new .+= offset
            ok = true
            @inbounds for i in 1:n
                li = GLM.linkinv(link, eta_new[i])
                ok &= _valid_mu_scalar(family, li)
                mu_new[i] = _clamp_mu_scalar(family, li)
            end
            dev_new = _deviance(family, y, mu_new, weights)
            mul!(penalty_buf, S_total, b)
            return (dev_new + dot(b, penalty_buf), ok)
        end

        pdev_new, valid_new, step_ok, _ =
            pirls_halve!(beta_new, beta_old, recompute!, step_spec,
                pdev_old, prev_valid)
        dev_new = _deviance(family, y, mu_new, weights)

        if !step_ok
            # mgcv raises "step failure" here; keep the previous
            # (best) iterate and stop rather than accepting divergence
            copyto!(beta, beta_old)
            copyto!(eta, eta_old)
            @inbounds for i in 1:n
                mu[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta[i]))
            end
            @warn "P-IRLS step failure: penalized deviance could not be " *
                  "reduced after $(step_spec.max_halvings) step halvings; " *
                  "returning last stable iterate" maxlog = 1
            converged = false
            break
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
        prev_valid = valid_new

        if crit < control.epsilon
            converged = true
            break
        end
    end

    # Warn if the final iterate still contains boundary-clamped means
    # (unclamped linkinv outside the family domain) — mirrors mgcv, which
    # warns when validmu enforcement cannot be fully satisfied.
    n_invalid = count(i -> !_valid_mu_scalar(family, GLM.linkinv(link, eta[i])), 1:n)
    if n_invalid > 0
        @warn "P-IRLS converged with $n_invalid observation(s) whose linear " *
              "predictor maps outside the $(nameof(typeof(family))) mean domain " *
              "under $(nameof(typeof(link))); fitted values are clamped to the " *
              "boundary there. Consider a different link (e.g. LogLink)." maxlog = 1
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
    A_chol_final = _protected_cholesky!(copy(A))

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

    # A = X'WX + S, solve A β = X'Wy (protected against numerically
    # indefinite A at λ → 0 boundaries)
    A = XtWX + S_total
    A_chol = _protected_cholesky!(A)
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

# Unsupported distributions must fail loudly: a generic logpdf-based
# "deviance" here would silently ignore the fitted means. Family support is
# gated upstream by _validate_gam_family, so this is a safety net only.
function _deviance(d::UnivariateDistribution, y, mu, wt)
    throw(ArgumentError(
        "no deviance implementation for family $(nameof(typeof(d))); " *
        "supported families are Normal, Poisson, Bernoulli/Binomial, " *
        "Gamma, and InverseGaussian (plus the ExtendedFamily types)"))
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

# Vector clamps used by `predict(:response)`; these must agree with the
# scalar `_clamp_mu_scalar` used during fitting, or predictions and fitted
# values disagree exactly where the link leaves the mean domain.
function _clamp_mu(::Poisson, mu)
    return _clamp_mu_positive.(mu)
end

function _clamp_mu(::Gamma, mu)
    return _clamp_mu_positive.(mu)
end

function _clamp_mu(::InverseGaussian, mu)
    return _clamp_mu_positive.(mu)
end

function _clamp_mu(::UnivariateDistribution, mu)
    return mu
end

# Extended families (non-UnivariateDistribution): no generic clamp
_clamp_mu(::Any, mu) = mu

