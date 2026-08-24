# REML / ML / GCV score computation
#
# Computes smoothness selection criteria and their derivatives with respect
# to log smoothing parameters. The key innovation from Wood (2011) is that
# derivatives come at negligible extra cost via the implicit function theorem.

"""
    reml_score(X, y, S_penalty, log_sp, family, link, weights, pirls_result;
               method=:REML, gamma=1.0, scale=-1.0, compute_gradient=true)

Compute the REML (or ML/GCV) score for given log smoothing parameters.
Returns `(score, grad)` where `grad` is the gradient w.r.t. `log_sp`.
Pass `compute_gradient=false` to skip the (relatively expensive) analytical
gradient when only the score is needed; `grad` is then a zero vector.

# Gradient accuracy

The analytic gradient is the total derivative of the *profiled* score — the
score at the penalized MLE β̂(ρ) — obtained by the implicit function theorem
(Wood 2011, §3.1), including the implicit weight-change terms `dW/dρ` for
non-Gaussian families. Validated against central differences in
`test/test_reml_gradient.jl`:

| Regime | Max relative error |
|:-------|:-------------------|
| Known-scale families (Poisson, Binomial), incl. weights/offsets/tensor | `~1e-9` |
| Estimated-scale families with `scale` supplied explicitly | `~1e-8` |
| **Estimated-scale families with `scale` estimated (default)** | **up to `1e-1`** |

The last row is a genuine limitation, not noise. When `scale < 0` the scale is
re-estimated from `pirls_result` as `pearson/(n − edf)` at every `ρ`, so the
profiled score acquires a `dσ̂²/dρ` dependence that the gradient does not
include. Treating σ² as constant is valid only when σ̂²(ρ) satisfies the
score's own stationarity condition `∂score/∂σ² = 0` (envelope theorem), which
the Pearson/Fletcher estimator does not. Supplying the REML-profiling scale
`σ̂² = Dp/(n − Mp)` restores agreement for Gaussian (relative error `4e-10`);
for Gamma a residual `~5e-3` remains because its saturated likelihood carries
digamma terms, so `Dp/(n − Mp)` is not the profiling value there either.

**Consequence for callers.** Use the gradient freely for known-scale families
or when passing `scale` explicitly. A future exact Newton-REML optimizer must
either profile the scale analytically (as mgcv does) or add the `dσ̂²/dρ` chain
term; no current fitting path consumes this gradient.
"""
function reml_score(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup,
    log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64},
    pirls_result::PirlsResult;
    method::Symbol = :REML, gamma::Real = 1.0,
    scale::Float64 = -1.0,
    compute_gradient::Bool = true)

    n, p = size(X)
    beta = pirls_result.coefficients
    w = pirls_result.working_weights
    dev = pirls_result.deviance

    S_total = total_penalty(penalty, log_sp, p)

    # X'WX
    XtWX = X' * Diagonal(w) * X

    # A = X'WX + S — protected Cholesky so a boundary-sp fit that converged
    # does not throw PosDefException while computing its final score
    A = XtWX + S_total
    A_chol = _protected_cholesky!(A)
    log_det_A = logdet(A_chol)

    # EDF
    F = A_chol \ XtWX
    edf_total = tr(F)

    if method == :GCV
        # GCV = n * deviance / (n - gamma * edf)^2
        denom = n - gamma * edf_total
        score = n * dev / denom^2

        if !compute_gradient
            return score, zeros(length(log_sp))
        end
        # Analytical gradient via IFT (matches mgcv's gdi1)
        mu = pirls_result.fitted_values
        grad = _gcv_gradient(X, y, w, beta, mu,
            pirls_result.linear_predictor, S_total, A_chol, penalty, log_sp,
            family, link, dev, edf_total, n, gamma, weights)
        return score, grad

    elseif method == :UBRE
        # UBRE/AIC-type criterion for known scale σ² (mgcv's ubre):
        #   dev/n + 2γ·edf·σ²/n − σ²
        sigma2 = scale > 0 ? scale : 1.0
        score = dev / n + 2 * gamma * edf_total * sigma2 / n - sigma2
        return score, zeros(length(log_sp))

    elseif method == :REML || method == :ML
        # REML/ML score (Laplace approximate restricted/marginal log-likelihood).
        # Following mgcv's gam.fit3.r lines 612-617:
        #   REML = (Dp/(2σ²) - ls)/γ + 0.5·log|A| - 0.5·log|S+|
        #          - remlInd·(Mp/2)·(log(2πσ²) - log(γ))
        # where Dp = dev + β'Sβ, ls = log saturated likelihood.

        # Null space dimension of the total penalty over ALL p coefficients
        # (unpenalized parametric terms included, as in mgcv)
        Mp = p - sum(b.rank for b in penalty.blocks; init = 0)

        # Penalized deviance: dev + β'Sβ (needed before the scale, because the
        # ML profiling scale below is a function of it)
        penalty_contrib = dot(beta, S_total * beta)
        Dp = dev + penalty_contrib

        # Estimate or use fixed scale. Families with known dispersion
        # (Poisson, Binomial) always use φ = 1, matching the criterion the
        # outer iteration actually optimized.
        if scale < 0
            if _needs_scale_estimate(family)
                if method == :ML
                    # mgcv PROFILES the scale, and the profiling equation differs
                    # by criterion. Setting mgcv's d(score)/d(log φ) to zero
                    # (R/gam.fit3.r:629):
                    #     dlr.dlphi <- (-Dp/(2*scale) - ls[2]*scale)/gamma
                    #                  - Mp/2*remlInd
                    # For Gaussian, ls = -(n/2)·log(2πφ) so ls[2] = -n/2, giving
                    #     REML (remlInd=1): φ̂ = Dp/(n - Mp)
                    #     ML   (remlInd=0): φ̂ = Dp/n
                    # Measured on the reference Gaussian fit at mgcv's own sp:
                    # mgcv's ML score 74.62946433 is reproduced to 3.81e-16 with
                    # Dp/n, but only to 6.77e-5 with pearson/(n - edf). Note
                    # `b$scale` REPORTS the REML-style scale even for an ML fit,
                    # so it cannot be used to infer the scoring scale.
                    #
                    # Confined to :ML — the REML path keeps pearson/(n - edf)
                    # exactly as before, so no REML fit's selected sp moves.
                    # Gamma/InverseGaussian have digamma terms in `ls` and need
                    # the root solved numerically; see `_ml_profiled_scale`.
                    scale_est = _ml_profiled_scale(family, y, weights, Dp, n,
                        pirls_result.pearson / (n - edf_total))
                else
                    scale_est = pirls_result.pearson / (n - edf_total)
                end
                # Response-relative floor (see _scale_floor): an absolute
                # 1e-10 clip distorts the REML surface itself for responses
                # on tiny scales, moving the optimum.
                scale_est = max(scale_est, _scale_floor(y))
            else
                scale_est = 1.0
            end
        else
            scale_est = scale
        end

        # Log pseudo-determinant of penalty
        log_det_S = _log_penalty_det(penalty, log_sp)

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
            # ML differs from REML by MORE than dropping the `Mp` term.
            #
            # mgcv's score expression is shared (R/gam.fit3.r:615-616):
            #     REML <- (Dp/(2*scale) - ls[1])/gamma + oo$rank.tol/2 - rp$det/2
            #             - remlInd*(Mp/2*(log(2*pi*scale)-log(gamma)))
            # with `remlInd` 1 for REML and 0 for ML (:545-546). But `oo$rank.tol`
            # is NOT the same quantity in the two cases: the `REML` flag is passed
            # down to `gdi1` (:563), which branches on its sign (src/gdi.c:2729-2741).
            # For REML it returns log|X'WX+S| over all p coefficients; for ML it
            # calls `MLpenalty1`, documented at src/gdi.c:1536-1551 as obtaining
            # "the version of log|X'WX+S| that applies to ML" by projecting the
            # Hessian "into the range space of the penalty".
            #
            # `MLpenalty1` drops the null-space COLUMNS from the R factor of
            # A = X'WX + S (src/gdi.c:1560-1570) and accumulates
            #     ldetXWXS = 2*Σ log|R̃_ii|   (src/gdi.c:1658-1660)
            # over the qM = p − Mp remaining columns. Since (R'R)_ij is the inner
            # product of columns i and j of R, dropping the null columns yields
            # exactly the principal submatrix of A on the penalty's range space.
            #
            # So: REML uses log|A|, ML uses log|Yᵀ A Y| with Y the range-space
            # basis. Dropping only the `Mp` term (the previous behaviour here)
            # applied a correction far too small — measured REML−ML gap 0.63
            # against mgcv's 4.53 on the reference Gaussian fit.
            Y = _penalty_range_basis(penalty, p)
            A_ml = Symmetric(Y' * (XtWX + S_total) * Y)
            log_det_A_ml = logdet(_protected_cholesky!(Matrix(A_ml)))
            score = (Dp / (2 * scale_est) - ls) / gamma +
                    0.5 * log_det_A_ml -
                    0.5 * log_det_S
        end

        if !compute_gradient
            return score, zeros(length(log_sp))
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
    _penalty_range_basis(penalty, p) -> Matrix{Float64}

Orthonormal basis `Y` (p × (p−Mp)) for the **range space of the total penalty**,
reproducing mgcv's `totalPenaltySpace` (R/gam.fit3.r:2661-2685).

mgcv forms `St` as the sum of the penalty blocks each rescaled by its own
Frobenius norm — `St[k0:k1,k0:k1] += S[[i]]/sqrt(sum(S[[i]]*S[[i]]))` (:2677) —
so that no block dominates, eigen-decomposes it, and splits on
`es\$values > max(es\$values)*.Machine\$double.eps^.66` (:2680). `Y` is the range
space and `Z` the null space, with `Mp = ncol(Z)`.

Two consequences worth stating, because both matter downstream:

  * The rescaling uses **no smoothing parameters**, so this basis is fixed for
    the whole optimisation — mgcv computes it once in `estimate.gam`
    (R/mgcv.r:1921-1924) and reuses it at every trial `ρ`.
  * Only the *span* of `Y` is used here (via `logdet(Y'AY)`), which is invariant
    to rotation and sign within the range space. R's `eigen` returns descending
    eigenvalues where LAPACK returns ascending, and the two pick different bases
    of any degenerate eigenspace, but neither difference can affect the result.
"""
function _penalty_range_basis(penalty::PenaltySetup, p::Int)
    St = zeros(Float64, p, p)
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            nrm = sqrt(sum(abs2, Si))
            nrm > 0 || continue
            @inbounds for j in eachindex(idx), k in eachindex(idx)
                St[idx[j], idx[k]] += Si[j, k] / nrm
            end
        end
    end
    es = eigen(Symmetric(St))
    mx = maximum(es.values)
    mx > 0 || return zeros(Float64, p, 0)
    keep = es.values .> mx * eps()^0.66
    return es.vectors[:, keep]
end

"""
    _log_penalty_det(penalty, log_sp)

Compute the log pseudo-determinant of the total penalty:
log|Σ λ_j S_j|_+ (product of non-zero eigenvalues).
"""
function _log_penalty_det(penalty::PenaltySetup, log_sp::AbstractVector)
    T = promote_type(Float64, eltype(log_sp))
    ldet = zero(T)
    sp_idx = 1
    for block in penalty.blocks
        k = block.stop - block.start + 1
        nS = length(block.S)
        # Factor out the largest λ in the block:
        #   log|Σλⱼ Sⱼ|₊ = r·log(λmax) + log|Σ(λⱼ/λmax)Sⱼ|₊
        # so the eigen threshold operates on ratios λⱼ/λmax ≤ 1 instead of
        # raw λ values spanning up to e³⁰ — genuine small eigenvalues of a
        # weakly-weighted margin are no longer dropped. (mgcv goes further
        # with the full similarity-transform reparameterization of
        # Wood 2011 / gam.reparam; that remains future work.)
        lsp_max = log_sp[sp_idx]
        for j in 1:nS
            lsp_max = max(lsp_max, log_sp[sp_idx + j - 1])
        end
        S_block = zeros(T, k, k)
        for Si in block.S
            λr = exp(log_sp[sp_idx] - lsp_max)
            @inbounds for j in 1:k, m in 1:k
                S_block[j, m] += λr * Si[j, m]
            end
            sp_idx += 1
        end
        eig = eigvals(Symmetric(S_block))
        thresh = eps(real(T)) * maximum(abs.(eig))
        for ev in eig
            if ev > thresh
                ldet += log(ev) + lsp_max
            end
        end
    end
    return ldet
end

"""
    _ml_profiled_scale(family, y, weights, Dp, n, phi0) -> Float64

Scale that maximises the **ML** criterion, i.e. mgcv's profiled `scale` for
`method="ML"`.

mgcv profiles the scale out rather than plugging in a moment estimator, and the
profiling equation depends on the criterion. Setting its derivative w.r.t.
`log φ` to zero (R/gam.fit3.r:629, with `remlInd = 0` for ML) gives

    -Dp/(2φ) - φ·ls'(φ) = 0,     ls'(φ) = d(ls)/dφ

For Gaussian `ls = -(n/2)·log(2πφ)`, so `ls'(φ) = -n/(2φ)` and the equation
collapses to the closed form `φ̂ = Dp/n` (the REML analogue, carrying the extra
`-Mp/2`, is `Dp/(n - Mp)`). Gamma and InverseGaussian have digamma terms in
`ls`, so no closed form exists and the root is found numerically below.

`phi0` seeds the bracket and is returned unchanged if the solve fails, so a
pathological fit degrades to the previous behaviour rather than throwing.
"""
function _ml_profiled_scale(family::UnivariateDistribution, y::Vector{Float64},
    weights::Vector{Float64}, Dp::Float64, n::Int, phi0::Float64)

    # Gaussian: exact closed form, and avoids the numerical solve entirely.
    family isa Normal && return Dp / n

    (isfinite(phi0) && phi0 > 0) || return phi0

    function dls(phi::Float64)
        h = max(1e-6 * phi, 1e-14)
        (_log_saturated_likelihood(family, y, weights, phi + h) -
         _log_saturated_likelihood(family, y, weights, phi - h)) / (2h)
    end
    g(phi::Float64) = -Dp / (2 * phi) - phi * dls(phi)

    lo = hi = phi0
    glo = g(lo)
    isfinite(glo) || return phi0
    bracketed = false
    for _ in 1:60
        lo /= 1.5
        hi *= 1.5
        gl, gh = g(lo), g(hi)
        (isfinite(gl) && isfinite(gh)) || return phi0
        if gl * gh <= 0
            bracketed = true
            break
        end
    end
    bracketed || return phi0

    # Bisect in log φ — the root spans orders of magnitude across datasets.
    for _ in 1:100
        mid = sqrt(lo * hi)
        gm = g(mid)
        isfinite(gm) || return phi0
        if g(lo) * gm <= 0
            hi = mid
        else
            lo = mid
        end
        hi / lo < 1 + 1e-13 && break
    end
    phi = sqrt(lo * hi)
    return (isfinite(phi) && phi > 0) ? phi : phi0
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
            ls += weights[i] * (yi * log(yi) - yi - logabsgamma(yi + 1)[1])
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
    # Exact Gamma saturated log-likelihood. With shape αᵢ = wᵢ/φ and μ = y:
    #   lᵢ = αᵢ·log(αᵢ) − αᵢ − log(yᵢ) − lgamma(αᵢ)
    # Depends on φ = scale, which keeps the REML landscape correct when
    # the scale is estimated.
    phi = max(scale, _scale_floor(y))
    ls = 0.0
    @inbounds for i in eachindex(y)
        a = weights[i] / phi
        ls += a * log(a) - a - log(max(y[i], eps())) - logabsgamma(a)[1]
    end
    return ls
end

function _log_saturated_likelihood(::InverseGaussian, y::Vector{Float64},
    weights::Vector{Float64}, scale::Float64)
    # Exact: at μ = y the IG exponent vanishes, leaving
    #   lᵢ = 0.5·[log(wᵢ/φ) − log(2π·yᵢ³)]
    phi = max(scale, _scale_floor(y))
    ls = 0.0
    @inbounds for i in eachindex(y)
        ls += 0.5 * (log(weights[i] / phi) - log(2π * max(y[i], eps())^3))
    end
    return ls
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

`scale` is treated as **constant** here. That is exact when the scale is
known (Poisson, Binomial) or supplied by the caller, and both the explicit
and implicit terms above then reproduce central differences of the profiled
score to ~1e-9. It is *not* exact when the caller lets `reml_score` estimate
the scale, because σ̂²(ρ) then varies with ρ and its chain term is omitted —
see the `reml_score` docstring for the measured error and the two ways to
fix it. `test/test_reml_gradient.jl` pins both regimes.
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

    # The determinant in the score differs by criterion: REML uses log|A| over
    # all p coefficients, ML uses log|YᵀAY| over the penalty's range space (see
    # the ML branch of `reml_score` for the mgcv source trail). Differentiating
    #     d/dρⱼ log|YᵀAY| = tr[(YᵀAY)⁻¹ Yᵀ (dA/dρⱼ) Y] = tr[Y(YᵀAY)⁻¹Yᵀ · dA/dρⱼ]
    # so the ML gradient is the REML one with A⁻¹ replaced by the projected
    # operator below, in the determinant terms ONLY. `b1_j = dβ/dρⱼ` keeps the
    # true A⁻¹: the coefficient derivative is a property of the fit, not of the
    # criterion being optimised.
    det_op = Ainv
    if method === :ML
        Y_rng = _penalty_range_basis(penalty, p)
        A_full = X' * Diagonal(w) * X + S_total
        det_op = Y_rng * (Symmetric(Y_rng' * A_full * Y_rng) \ Y_rng')
    end

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

            # Working weight w = weights·g1²/V with g1 = dμ/dη. By the
            # quotient rule, with dg1/dη = g2 and dV/dη = V′(μ)·g1:
            #   dw/dη = weights·(2·g1·g2·V − g1³·V′(μ)) / V²
            dw_deta[i] = weights[i] * (2.0 * g1 * g2 * vm - g1^2 * dvm * g1) / (vm * vm)
        end
    end

    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        beta_block = beta[idx]

        # λⱼ·tr(S_λ⁺Sⱼ) per penalty: equals block.rank for single-penalty
        # blocks; differs per margin for multi-penalty (tensor) blocks.
        ldet_derivs = _block_logdet_derivs(block,
            view(log_sp, sp_idx:(sp_idx + length(block.S) - 1)))

        for (j_pen, Si) in enumerate(block.S)
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
            trA1_explicit = tr(det_op * dS)

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
                XAinv = X * det_op  # n × p (det_op === Ainv unless method === :ML)
                @inbounds for i in 1:n
                    if abs(dw_j[i]) > eps()
                        h_ii = dot(view(X, i, :), view(XAinv, i, :))
                        trA1_implicit += dw_j[i] * h_ii
                    end
                end
            end

            trA1_j = trA1_explicit + trA1_implicit

            # d(log|S+|)/d(log sp_j) = λⱼ·tr(S_λ⁺Sⱼ)
            d_log_det_S = ldet_derivs[j_pen]

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
# η = √μ ⇒ μ = η², d²μ/dη² = 2 (constant)
_d2mu_deta2(::GLM.SqrtLink, mu::Float64, eta::Float64) = 2.0
# Fallback: 0 second derivative degrades Newton weights to Fisher-type
# weights for links without an analytic entry (e.g. probit, cloglog)
_d2mu_deta2(::GLM.Link, mu::Float64, eta::Float64) = 0.0

# Helper: V'(μ) for different families
_dvariance_scalar_mu(::Normal, mu::Float64) = 0.0
_dvariance_scalar_mu(::BinomialLike, mu::Float64) = 1.0 - 2.0 * mu
_dvariance_scalar_mu(::Poisson, mu::Float64) = 1.0
_dvariance_scalar_mu(::Gamma, mu::Float64) = 2.0 * mu
_dvariance_scalar_mu(::InverseGaussian, mu::Float64) = 3.0 * mu * mu
_dvariance_scalar_mu(::UnivariateDistribution, mu::Float64) = 0.0

"""
    _gcv_gradient(X, y, w, beta, mu, eta, S_total, A_chol, penalty, log_sp,
                  family, link, dev, edf, n, gamma)

Analytical gradient of GCV score w.r.t. log smoothing parameters using the
Implicit Function Theorem (Wood 2011, Section 3). Matches mgcv's gdi1 C code.
`eta` is the fitted linear predictor (offset included).

Key formulas:
  GCV = n·dev / (n - γ·trA)²
  GCV1[j] = n·D1[j]/δ² + 2·n·dev·trA1[j]·γ/δ³

where D1[j] = ∂dev/∂(log_sp_j) via IFT and trA1[j] = ∂trA/∂(log_sp_j).
"""
function _gcv_gradient(X::Matrix{Float64}, y::Vector{Float64},
    w::Vector{Float64}, beta::Vector{Float64}, mu::Vector{Float64},
    eta::Vector{Float64},
    S_total::Matrix{Float64},
    A_chol, penalty::PenaltySetup,
    log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    dev::Float64, edf::Float64,
    n::Int, gamma::Real,
    prior_weights::Vector{Float64} = ones(length(y)))

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
        g1 = 1.0 / GLM.mueta(link, eta[i])
        v[i] = -2.0 * prior_weights[i] * (y[i] - mu[i]) / (max(vi, eps()) * g1)
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
                    eta_i = eta[i]
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
                # Weight term: tr(T_j diag(h)) - tr(T_j diag(h)²)
                # where h = diag(F_hat), F_hat = W^{1/2} X A⁻¹ X' W^{1/2},
                # i.e. diag(F_hat) = w .* diag(X A⁻¹ X')
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

# ============================================================================
# Smoothing-parameter uncertainty correction: Vc, edf2
# ============================================================================
#
# Wood, Pya & Säfken (2016, JASA 111:1548-1563, §6.2). The Bayesian covariance
# Vp = A⁻¹φ conditions on ρ̂ = log λ̂ as if it were known; Vc widens it to
# account for having estimated ρ. mgcv builds it in `gam.fit3.post.proc`
# (mgcv 1.9.4) as
#
#     Vc = Vb + db.drho %*% Vρ %*% t(db.drho) + scale * Vb.corr(...)
#        = Vb +          Vc1                 +        Vc2
#
# term by term, with A = X'WX + S_λ, S_λ = Σⱼ λⱼ Sⱼ, φ the scale:
#
# Vb   = A⁻¹φ                              — the usual Bayesian covariance.
#
# Vc1  — mgcv: `rV <- (d * t(ev$vectors))[, 1:M]; Vc <- crossprod(rV %*% t(db.drho))`
#        where `ev <- eigen(hess)` and `d <- 1/sqrt(d)` with non-positive
#        eigenvalues zeroed. Multiplying out, this is
#            Vc1 = (dβ̂/dρ) · Vρ · (dβ̂/dρ)ᵀ,   Vρ = [hess⁺]₁:M,₁:M
#        the delta-method propagation of Var(ρ̂) into β̂.
#
#        dβ̂/dρⱼ follows from implicit differentiation of the penalized score
#        ∂/∂β[l(β) − ½β'S_λβ] = 0 at the optimum:
#            (−X'WX − S_λ)·dβ̂/dρⱼ − λⱼ Sⱼ β̂ = 0
#        so  dβ̂/dρⱼ = −A⁻¹ λⱼ Sⱼ β̂.
#        This already accounts for W's dependence on β — it is the Hessian of
#        the penalized likelihood, not a frozen-weight approximation.
#        Verified against mgcv's own `object$db.drho` to 8.3e-13 relative.
#
# Vc2  — mgcv: `Vb.corr(R, L, lsp0, S, off, dw.drho, w = NULL, lsp, Vr, ...)`.
#        Note `w = NULL` is passed *literally* at the call site, which sends
#        Vb.corr down its `dH[[i]] <- H * 0` branch: the `dw.drho` argument is
#        therefore ignored entirely and dH_i = λᵢSᵢ. Vb.corr then forms
#            dR_i = −U⁻¹ · dchol(λᵢSᵢ, U) · U⁻¹ = d(U⁻¹)/dρᵢ,   A = UᵀU
#        (the derivative of the "square root" of A⁻¹), and `vcorr` returns
#            Vc2 = φ · Σᵢⱼ Vr[i,j] · dR_i · dR_jᵀ
#        i.e. the extra variance from Vb itself moving with ρ̂.
#
# hess — the Hessian of the *negative* REML/LAML score. mgcv optimizes over
#        (ρ, log φ) jointly when the scale is estimated, so `hess` is
#        (M+1)×(M+1) with log φ last; `Vb.corr` drops that row/column
#        (`drop.scale`) before using Vr. This code reconstructs the same
#        matrix from quantities GAM.jl already has:
#          H_ρρ block : H_prof + h·hᵀ/H_φφ, where H_prof is the Hessian of the
#                       scale-profiled REML score. This inversion of the Schur
#                       complement is exact — [H⁻¹]₁:M,₁:M = H_prof⁻¹ was
#                       checked against mgcv to 2.2e-16 — so Vc1 needs only
#                       H_prof, and the full matrix is needed only for Vr.
#          hⱼ = ∂²(−l_r)/∂ρⱼ∂logφ = −λⱼ β̂'Sⱼβ̂ / (2φγ)   (envelope theorem on
#                       Dp(ρ) = dev + β̂'S_λβ̂, since ∂Dp/∂ρⱼ = λⱼβ̂'Sⱼβ̂).
#                       Matches mgcv's hess[1:M, M+1] to 3.3e-7.
#          H_φφ       = second derivative of the score's φ-part in log φ; taken
#                       by central differences so that families whose saturated
#                       log-likelihood carries digamma terms (Gamma, inverse
#                       Gaussian) are handled without a bespoke derivation.
#                       For Gaussian this reduces to Dp/(2φ) = (n−Mp)/2 at the
#                       optimum, reproducing mgcv's hess[M+1,M+1] exactly.
#
# edf2 — mgcv: `edf2 <- rowSums(Vc * crossprod(R))/scale`, with `crossprod(R)`
#        the X'WX of the fit, then `if (sum(edf2) > sum(edf1)) edf2 <- edf1`.
#        The cap is applied to the whole vector at once (edf2 is *replaced by*
#        edf1, not clamped elementwise) — that is mgcv's actual branch.
#
# The whole construction was validated by reproducing mgcv's edf, edf1, edf2,
# Vp and Vc from scratch in R to ~1e-12 relative before any of it was written
# in Julia.

"""
    _dchol(dA, U) -> Matrix{Float64}

Derivative of the upper-triangular Cholesky factor. Given `A = UᵀU` and
`dA = ∂A/∂θ`, returns `∂U/∂θ`.

With `L = Uᵀ` and `Φ = L⁻¹ (dA) L⁻ᵀ`, the standard result is
`dL = L · tril(Φ, halved diagonal)`; this is mgcv's C routine `dchol`, which
it reproduces to 8.9e-16.
"""
function _dchol(dA::AbstractMatrix{Float64}, U::AbstractMatrix{Float64})
    p = size(U, 1)
    Lt = LowerTriangular(transpose(U))
    # Φ = L⁻¹ dA L⁻ᵀ
    Φ = Lt \ Matrix(dA)
    Φ = transpose(Lt \ transpose(Φ))
    Φ = Matrix(Φ)
    @inbounds for i in 1:p
        Φ[i, i] *= 0.5
        for j in (i + 1):p
            Φ[i, j] = 0.0        # keep the lower triangle only
        end
    end
    return Matrix(transpose(Lt * Φ))
end

"""
    _dbeta_drho(A_chol, beta, penalty, log_sp, p, free) -> Matrix{Float64}

`dβ̂/dρ`, a `p × length(free)` matrix whose columns are
`−A⁻¹ λⱼ Sⱼ β̂` for each free smoothing parameter `j`.
"""
function _dbeta_drho(A_chol::Cholesky, beta::Vector{Float64},
    penalty::PenaltySetup, log_sp::AbstractVector, p::Int,
    free::Vector{Int})

    db = zeros(p, length(free))
    v = zeros(p)
    for (col, j) in enumerate(free)
        fill!(v, 0.0)
        _accumulate_penalty_j!(v, penalty, log_sp, j, beta)
        db[:, col] = -(A_chol \ v)
    end
    return db
end

"""
    _accumulate_penalty_j!(out, penalty, log_sp, j, beta)

Add `λⱼ Sⱼ β` (the `j`-th individual penalty, not the summed `S_λ`) into `out`.
"""
function _accumulate_penalty_j!(out::Vector{Float64}, penalty::PenaltySetup,
    log_sp::AbstractVector, j::Int, beta::Vector{Float64})

    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            if sp_idx == j
                λ = exp(log_sp[j])
                @inbounds for a in eachindex(idx), b in eachindex(idx)
                    out[idx[a]] += λ * Si[a, b] * beta[idx[b]]
                end
                return out
            end
            sp_idx += 1
        end
    end
    return out
end

"""
    _penalty_block_j(penalty, log_sp, j, p) -> Matrix{Float64}

The `p × p` matrix `λⱼ Sⱼ` for a single penalty `j`, zero outside its block.
"""
function _penalty_block_j(penalty::PenaltySetup, log_sp::AbstractVector,
    j::Int, p::Int)

    out = zeros(p, p)
    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            if sp_idx == j
                λ = exp(log_sp[j])
                @inbounds for a in eachindex(idx), b in eachindex(idx)
                    out[idx[a], idx[b]] = λ * Si[a, b]
                end
                return out
            end
            sp_idx += 1
        end
    end
    return out
end

"""
    _bSb_j(penalty, log_sp, j, beta) -> Float64

The quadratic form `λⱼ β'Sⱼβ` for a single penalty, which by the envelope
theorem equals `∂Dp/∂ρⱼ`.
"""
function _bSb_j(penalty::PenaltySetup, log_sp::AbstractVector, j::Int,
    beta::Vector{Float64})

    sp_idx = 1
    for block in penalty.blocks
        idx = block.start:block.stop
        for Si in block.S
            if sp_idx == j
                λ = exp(log_sp[j])
                s = 0.0
                @inbounds for a in eachindex(idx), b in eachindex(idx)
                    s += beta[idx[a]] * Si[a, b] * beta[idx[b]]
                end
                return λ * s
            end
            sp_idx += 1
        end
    end
    return 0.0
end

"""
    _reml_score_scale_profiled(X, y, penalty, lsp, family, link, weights, res,
                               method, gamma) -> Float64

REML/ML score at `lsp` with the scale **profiled out by maximisation**, which
is mgcv's convention and the one whose curvature `Vρ` is meant to describe.

`reml_score` instead plugs in `pearson/(n − edf)`. The two agree in *value* at
the optimum — for the single-smooth Gaussian fit in `test_rcall.jl` the scales
match to 7e-7 — but not in *curvature*: their profiled Hessians differ by 3.3%
there, which propagates straight into `Vc` and `edf2`. Since only the Hessian
needs this, the fitted criterion is left alone.

Given the P-IRLS fit at `lsp`, none of `Dp`, `log|A|` or `log|S|₊` depend on
the scale, so profiling is a one-dimensional problem solved here by Newton on
`dg/du` with a bisection fallback (`u = log φ`). Solving rather than
substituting a closed form keeps families whose saturated log-likelihood
carries digamma terms — Gamma, inverse Gaussian — exact.
"""
function _reml_score_scale_profiled(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup, lsp::AbstractVector,
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64}, res::PirlsResult, method::Symbol, gamma::Real)

    p = size(X, 2)
    S_total = total_penalty(penalty, lsp, p)
    A = X' * Diagonal(res.working_weights) * X + S_total
    log_det_A = logdet(_protected_cholesky!(A))
    log_det_S = _log_penalty_det(penalty, lsp)
    Mp = p - sum(b.rank for b in penalty.blocks; init = 0)
    Dp = res.deviance + dot(res.coefficients, S_total * res.coefficients)
    fixed = 0.5 * log_det_A - 0.5 * log_det_S

    if !_needs_scale_estimate(family)
        ls = _log_saturated_likelihood(family, y, weights, 1.0)
        s = (Dp / 2 - ls) / gamma + fixed
        method === :REML && (s -= 0.5 * Mp * (log(2π) - log(gamma)))
        return s
    end

    g = function (u)
        φ = exp(u)
        s = (Dp / (2 * φ) - _log_saturated_likelihood(family, y, weights, φ)) / gamma + fixed
        method === :REML && (s -= 0.5 * Mp * (log(2π * φ) - log(gamma)))
        return s
    end

    # Newton on g'(u) from the Gaussian closed form Dp/(n − Mp); the objective
    # is smooth and unimodal in u, and the bracket guards a bad Newton step.
    n = length(y)
    u = log(max(Dp / max(n - Mp, 1), _scale_floor(y)))
    δ = 1e-5
    for _ in 1:50
        g1 = (g(u + δ) - g(u - δ)) / (2δ)
        g2 = (g(u + δ) - 2 * g(u) + g(u - δ)) / δ^2
        (isfinite(g1) && isfinite(g2) && g2 > 0) || break
        step = clamp(g1 / g2, -1.0, 1.0)
        u -= step
        abs(step) < 1e-12 && break
    end
    return isfinite(g(u)) ? g(u) : Inf
end

"""
    _profiled_reml_hessian(X, y, penalty, log_sp, family, link, weights, offset,
                           free, beta_start, control, method, gamma; h=1e-3)

Hessian of the scale-profiled REML/ML score with respect to `ρ = log λ`,
restricted to the free smoothing parameters `free`.

Two routes, chosen by whether the family's scale is estimated:

  * **Known scale** (Poisson, Binomial, or a supplied `scale`) — central
    differences of the *analytic* gradient `_reml_gradient`, `2M` warm-started
    P-IRLS refits. The gradient is exact here (pinned to ~1e-9 by
    `test/test_reml_gradient.jl`), and differencing it agrees with a pure
    second difference of the score to ~1e-7.

  * **Estimated scale** (Normal, Gamma, inverse Gaussian) — central second
    differences of the *score*, `2M² + 1` refits with point caching.
    The analytic gradient is not exact in this case: `reml_score` profiles the
    scale as `pearson/(n − edf)`, whose own ρ-dependence the envelope theorem
    does not cover, so differencing the gradient disagrees with the truth by
    ~1.4e-3 (Gaussian) to ~4.8e-2 (Gamma) relative. Differencing the score
    needs no such assumption, and is what mgcv's `hess` corresponds to — a
    finite-differenced profiled score reproduces mgcv's own Schur complement
    `H_ρρ − h hᵀ/H_φφ` to four decimals.

The result is symmetrized: the two mixed partials are computed independently
and agree only up to the differencing error.
"""
function _profiled_reml_hessian(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup, log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64}, offset::Vector{Float64},
    free::Vector{Int}, beta_start::Vector{Float64},
    control::GamControl, method::Symbol, gamma::Real; h::Float64 = 1e-3)

    p = size(X, 2)
    M = length(free)
    H = zeros(M, M)

    refit = function (lsp)
        S_total = total_penalty(penalty, lsp, p)
        return pirls(X, y, S_total, family, link;
            weights = weights, offset = offset, start = copy(beta_start),
            control = control)
    end

    if !_needs_scale_estimate(family)
        grad_at = function (lsp)
            _, g = reml_score(X, y, penalty, lsp, family, link, weights,
                refit(lsp); method = method, gamma = gamma,
                compute_gradient = true)
            return g
        end
        lsp = copy(log_sp)
        for (a, j) in enumerate(free)
            lsp[j] = log_sp[j] + h
            gp = grad_at(lsp)
            lsp[j] = log_sp[j] - h
            gm = grad_at(lsp)
            lsp[j] = log_sp[j]
            @inbounds for (b, k) in enumerate(free)
                H[b, a] = (gp[k] - gm[k]) / (2h)
            end
        end
    else
        cache = Dict{Vector{Int}, Float64}()
        # `steps` gives the perturbation in integer multiples of h on each
        # free coordinate, so the 4-point stencils share their corner points.
        score_at = function (steps::Vector{Int})
            return get!(cache, copy(steps)) do
                lsp = copy(log_sp)
                for (a, j) in enumerate(free)
                    lsp[j] = log_sp[j] + steps[a] * h
                end
                return _reml_score_scale_profiled(X, y, penalty, lsp, family,
                    link, weights, refit(lsp), method, gamma)
            end
        end
        st = zeros(Int, M)
        for a in 1:M, b in a:M
            if a == b
                st .= 0; st[a] = 2;  spp = score_at(st)
                st .= 0;             s00 = score_at(st)
                st .= 0; st[a] = -2; smm = score_at(st)
                H[a, a] = (spp - 2 * s00 + smm) / (4h^2)
            else
                st .= 0; st[a] = 1;  st[b] = 1;  spp = score_at(st)
                st .= 0; st[a] = 1;  st[b] = -1; spm = score_at(st)
                st .= 0; st[a] = -1; st[b] = 1;  smp = score_at(st)
                st .= 0; st[a] = -1; st[b] = -1; smm = score_at(st)
                v = (spp - spm - smp + smm) / (4h^2)
                H[a, b] = v
                H[b, a] = v
            end
        end
    end

    @inbounds for a in 1:M, b in (a + 1):M
        v = 0.5 * (H[a, b] + H[b, a])
        H[a, b] = v
        H[b, a] = v
    end
    return H
end

"""
    _ridged_inverse(H, ridge) -> Matrix{Float64}

`(H + ridge·I)⁻¹` computed through the symmetric eigendecomposition with
non-positive eigenvalues floored at zero first — mgcv's

    d <- ev\$values; d[d <= 0] <- 0; d <- 1/sqrt(d + 1/10)
    Vr <- crossprod(d * t(ev\$vectors))

Passing `ridge = 0` gives the Moore-Penrose pseudo-inverse on the positive
eigenspace instead (mgcv's `Vρ`), which is what keeps an indefinite or
rank-deficient Hessian — a smoothing parameter sitting on its boundary, say —
from producing a non-finite or negative-variance `Vc`.
"""
function _ridged_inverse(H::AbstractMatrix{Float64}, ridge::Float64)
    E = eigen(Symmetric(Matrix(H)))
    d = similar(E.values)
    @inbounds for i in eachindex(E.values)
        λ = E.values[i] <= 0 ? 0.0 : E.values[i]
        if ridge > 0
            d[i] = 1.0 / (λ + ridge)
        else
            d[i] = λ > 0 ? 1.0 / λ : 0.0
        end
    end
    return E.vectors * Diagonal(d) * E.vectors'
end

"""
    corrected_covariance(...) -> (Vc, edf2, Vrho)

Smoothing-parameter-uncertainty-corrected covariance `Vc` and the associated
`edf2`, following mgcv's `gam.fit3.post.proc` (see the derivation above).

Returns `(Vc, edf2, Vρ)`; `Vρ` is the covariance of `ρ̂` on the free smoothing
parameters, mgcv's `V.sp`. Returns `nothing` when there is no smoothing
parameter uncertainty to propagate — every `sp` fixed, or a criterion for which
mgcv itself leaves `Vc` unset (GCV/UBRE, where `reml.scale` is `NA`).
"""
function corrected_covariance(X::Matrix{Float64}, y::Vector{Float64},
    penalty::PenaltySetup, log_sp::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    weights::Vector{Float64}, offset::Vector{Float64},
    beta::Vector{Float64}, XtWX::Matrix{Float64}, A_chol::Cholesky,
    Vb::Matrix{Float64}, edf1::Vector{Float64}, scale::Float64,
    deviance::Float64, control::GamControl, method::Symbol, gamma::Real)

    (method === :REML || method === :ML) || return nothing

    p = size(X, 2)
    free = [j for j in eachindex(log_sp) if !penalty.fixed[j]]
    isempty(free) && return nothing
    M = length(free)

    # --- dβ̂/dρ on the free smoothing parameters
    db = _dbeta_drho(A_chol, beta, penalty, log_sp, p, free)

    # --- Hessian of the profiled REML score
    H_prof = _profiled_reml_hessian(X, y, penalty, log_sp, family, link,
        weights, offset, free, beta, control, method, gamma)
    all(isfinite, H_prof) || return nothing

    # --- Vc1 = db · Vρ · db',  Vρ = [hess⁻¹]₁:M,₁:M = H_prof⁻¹ (Schur)
    Vρ = _ridged_inverse(H_prof, 0.0)
    Vc1 = db * Vρ * db'

    # --- assemble the full (M+1)×(M+1) Hessian when the scale is estimated,
    #     so that Vr carries mgcv's ridge in the same space mgcv applies it in
    scale_estimated = _needs_scale_estimate(family)
    if scale_estimated
        Mp = p - sum(b.rank for b in penalty.blocks; init = 0)
        Dp = deviance + dot(beta, total_penalty(penalty, log_sp, p) * beta)
        # φ-part of the score as a function of u = log φ (mirrors reml_score)
        φ_part = function (u)
            φ = exp(u)
            s = (Dp / (2 * φ) - _log_saturated_likelihood(family, y, weights, φ)) / gamma
            if method === :REML
                s -= 0.5 * Mp * (log(2π * φ) - log(gamma))
            end
            return s
        end
        u0 = log(scale)
        δ = 1e-4
        H_φφ = (φ_part(u0 + δ) - 2 * φ_part(u0) + φ_part(u0 - δ)) / δ^2
        hv = [-_bSb_j(penalty, log_sp, j, beta) / (2 * scale * gamma) for j in free]

        H_full = zeros(M + 1, M + 1)
        if isfinite(H_φφ) && H_φφ > 0
            H_full[1:M, 1:M] .= H_prof .+ (hv * hv') ./ H_φφ
            H_full[1:M, M + 1] .= hv
            H_full[M + 1, 1:M] .= hv
            H_full[M + 1, M + 1] = H_φφ
        else
            H_full[1:M, 1:M] .= H_prof
            H_full[M + 1, M + 1] = 1.0
        end
        Vr = _ridged_inverse(H_full, 0.1)[1:M, 1:M]
    else
        Vr = _ridged_inverse(H_prof, 0.1)
    end

    # --- Vc2 = φ · Σᵢⱼ Vr[i,j] dR_i dR_j'
    U = UpperTriangular(Matrix(A_chol.U))
    dR = Vector{Matrix{Float64}}(undef, M)
    for (a, j) in enumerate(free)
        Z = _dchol(_penalty_block_j(penalty, log_sp, j, p), U)
        dR[a] = -Matrix((U \ Z) / U)
    end
    Vc2 = zeros(p, p)
    for a in 1:M, b in 1:M
        v = Vr[a, b]
        v == 0.0 && continue
        BLAS.gemm!('N', 'T', v * scale, dR[a], dR[b], 1.0, Vc2)
    end

    Vc = Vb .+ Vc1 .+ Vc2
    @inbounds for i in 1:p, j in 1:(i - 1)      # enforce exact symmetry
        v = 0.5 * (Vc[i, j] + Vc[j, i])
        Vc[i, j] = v
        Vc[j, i] = v
    end
    all(isfinite, Vc) || return nothing

    # --- edf2 = rowSums(Vc * X'WX)/scale, replaced wholesale by edf1 if it
    #     overshoots (mgcv's `if (sum(edf2) > sum(edf1)) edf2 <- edf1`)
    edf2 = vec(sum(Vc .* XtWX; dims = 2)) ./ scale
    if !all(isfinite, edf2) || sum(edf2) > sum(edf1)
        edf2 = copy(edf1)
    end

    return (Vc, edf2, Vρ)
end
