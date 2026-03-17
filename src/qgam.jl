# Quantile GAM (qgam) implementation
#
# Port of R's qgam package (Fasiolo et al., 2020).
# Uses the Extended Log-F (ELF) density as a smooth surrogate for the pinball
# loss, fit via standard GAM machinery with automatic learning rate calibration.
#
# References:
#   Fasiolo, M., Wood, S.N., Zaffran, M., Nedellec, R. and Goude, Y. (2020).
#   Fast calibrated additive quantile regression. JASA.

using SpecialFunctions: digamma, trigamma, logabsbeta
using Random: MersenneTwister

# ============================================================================
# Numerically stable log(1 + exp(x))
# ============================================================================

"""
    log1pexp(x)

Compute `log(1 + exp(x))` in a numerically stable way.
Uses the recipe from Mächler (2012).
"""
function log1pexp(x::Real)
    if x < -37.0
        return exp(x)
    elseif x <= 18.0
        return log1p(exp(x))
    elseif x <= 33.3
        return x + exp(-x)
    else
        return x
    end
end

function log1pexp!(out::AbstractVector, x::AbstractVector)
    @inbounds for i in eachindex(x, out)
        out[i] = log1pexp(x[i])
    end
    return out
end

# ============================================================================
# Sigmoid function and derivatives
# ============================================================================

"""
    sigmoid_derivs(x)

Compute the logistic sigmoid and its first three derivatives.
Returns `(D0, D1, D2, D3)` where `D0 = 1/(1+exp(-x))`.
"""
function sigmoid_derivs(x::Real)
    D0 = 1.0 / (1.0 + exp(-x))
    D1 = D0 * (1.0 - D0)
    D2 = D1 - 2.0 * D1 * D0
    D3 = D2 - 2.0 * D2 * D0 - 2.0 * D1 * D1
    return (D0, D1, D2, D3)
end

# ============================================================================
# ELF Family (Extended Log-F with fixed scale)
# ============================================================================

"""
    ELFFamily(; qu=0.5, co=0.1, theta=0.0, estimate_theta=false)

Extended Log-F family for quantile regression (Fasiolo et al., 2020).

The ELF density provides a smooth approximation to the pinball loss function,
enabling quantile regression within the standard GAM fitting framework.

# Parameters
- `qu`: target quantile ∈ (0, 1)
- `co`: smoothness constant (typically determined from data)
- `theta`: log learning rate log(σ). If `estimate_theta=false`, this is fixed.
- `link`: link function for the quantile location (default: identity)
- `estimate_theta`: whether to estimate θ during fitting

The ELF negative log-likelihood per observation is:
```
nll(y,μ) = [(1-τ)·λ·log(1-τ) + λ·τ·log(τ) - (1-τ)·(y-μ) + λ·log(1+exp((y-μ)/λ))] / σ
```
where τ = qu, σ = exp(θ), λ = co (with normalization).
"""
mutable struct ELFFamily <: ExtendedFamily
    qu::Float64
    co::Vector{Float64}     # can be vector (per-observation smoothness)
    theta::Float64           # log(sigma)
    estimate_theta::Bool
end

function ELFFamily(; qu::Real=0.5, co=0.1, theta::Real=0.0,
                   estimate_theta::Bool=false)
    0.0 < qu < 1.0 || throw(ArgumentError("qu must be in (0, 1), got $qu"))
    co_vec = co isa AbstractVector ? Float64.(co) : Float64[co]
    return ELFFamily(Float64(qu), co_vec, Float64(theta), estimate_theta)
end

_default_link(::ELFFamily) = IdentityLink()
_family_name(::ELFFamily) = "ELF"
_has_extra_param(f::ELFFamily) = f.estimate_theta
_estimates_scale(::ELFFamily) = false

function _null_deviance(f::ELFFamily, y, wt)
    mu = fill(quantile(y, f.qu), length(y))
    return _deviance(f, y, mu, wt)
end

function _get_co(f::ELFFamily, n::Int)
    if length(f.co) == 1
        return fill(f.co[1], n)
    else
        length(f.co) == n || throw(DimensionMismatch("co has length $(length(f.co)) but data has $n observations"))
        return f.co
    end
end

function _get_sig(f::ELFFamily, n::Int)
    sig = exp(f.theta)
    co = _get_co(f, n)
    lam = co
    mean_lam = mean(lam)
    return sig .* lam ./ mean_lam
end

# ============================================================================
# Core ELF methods
# ============================================================================

function _initialize_mu(f::ELFFamily, y)
    # Initialize to the empirical quantile
    mu_init = fill(quantile(y, f.qu), length(y))
    return mu_init
end

function _clamp_mu(::ELFFamily, mu)
    return mu  # No clamping needed for ELF (continuous response)
end

function _variance(f::ELFFamily, mu)
    # Working variance for PIRLS: use observed information
    # V(μ) ∝ σ / f_logistic((y-μ)/λ)
    # Since we don't have y here, return constant (PIRLS uses Dd-style weights)
    n = length(mu)
    sig = _get_sig(f, n)
    return sig
end

"""
    _deviance(f::ELFFamily, y, mu, wt)

Compute the ELF deviance: -2 × (log-likelihood - saturated log-likelihood).
"""
function _deviance(f::ELFFamily, y, mu, wt)
    tau = f.qu
    n = length(y)
    co = _get_co(f, n)
    sig = _get_sig(f, n)
    lam = co

    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        z = y[i] - mu[i]
        term = (1.0 - tau) * lam[i] * log1p(-tau) +
               lam[i] * tau * log(tau) -
               (1.0 - tau) * z +
               lam[i] * log1pexp(z / lam[i])
        dev += 2.0 * wt[i] * term / sig[i]
    end
    return dev
end

function _deviance_residuals(f::ELFFamily, y, mu, wt)
    tau = f.qu
    n = length(y)
    co = _get_co(f, n)
    sig = _get_sig(f, n)
    lam = co

    r = similar(y)
    @inbounds for i in eachindex(y, mu, wt)
        z = y[i] - mu[i]
        # Per-obs deviance contribution
        term = (1.0 - tau) * lam[i] * log1p(-tau) +
               lam[i] * tau * log(tau) -
               (1.0 - tau) * z +
               lam[i] * log1pexp(z / lam[i])
        di = 2.0 * wt[i] * term / sig[i]

        # Saturated: z = 0 is NOT the saturated value for ELF
        # Saturated log-lik uses y = mu case
        sat = (1.0 - tau) * lam[i] * log1p(-tau) +
              lam[i] * tau * log(tau)
        di_sat = 2.0 * wt[i] * sat / sig[i]

        residual_dev = max(di - di_sat, 0.0)
        r[i] = sign(y[i] - mu[i]) * sqrt(residual_dev)
    end
    return r
end

"""
    elf_aic(f::ELFFamily, y, mu, wt)

Compute AIC contribution (twice negative log-likelihood) for ELF.
"""
function elf_aic(f::ELFFamily, y, mu, wt)
    tau = f.qu
    n = length(y)
    co = _get_co(f, n)
    sig = _get_sig(f, n)
    lam = co

    aic = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        z = y[i] - mu[i]
        # Full log-likelihood (not just deviance)
        ll_i = -(1.0 - tau) * z / sig[i] +
               lam[i] * log1pexp(z / lam[i]) / sig[i] +
               log(lam[i] * exp(logabsbeta(lam[i] * (1.0 - tau) / sig[i], tau * lam[i] / sig[i])[1]))
        aic += 2.0 * wt[i] * ll_i
    end
    return aic
end

# ============================================================================
# Dd structure — derivatives for Newton/PIRLS iteration
# ============================================================================

"""
    elf_Dd(f::ELFFamily, y, mu, wt; level=0)

Compute derivatives of the ELF deviance w.r.t. μ and θ.

Returns a NamedTuple with fields depending on `level`:
- Level 0: `Dmu`, `Dmu2`, `EDmu2` (needed for IRLS)
- Level 1: + `Dth`, `Dmuth`, `Dmu3`, `Dmu2th` (needed for 1st derivative of REML)
- Level 2: + `Dmu4`, `Dth2`, `Dmuth2`, `Dmu2th2`, `Dmu3th` (needed for 2nd derivative)
"""
function elf_Dd(f::ELFFamily, y, mu, wt; level::Int=0)
    tau = f.qu
    n = length(y)
    co = _get_co(f, n)
    sig = _get_sig(f, n)
    lam = co

    Dmu = similar(y)
    Dmu2 = similar(y)

    @inbounds for i in eachindex(y, mu, wt)
        z = y[i] - mu[i]
        pl = 1.0 / (1.0 + exp(-z / lam[i]))  # plogis(z, 0, lam)

        Dmu[i] = -2.0 * wt[i] * (pl - 1.0 + tau) / sig[i]

        dl = pl * (1.0 - pl) / lam[i]  # dlogis(z, 0, lam) — logistic PDF
        Dmu2[i] = 2.0 * wt[i] * dl / sig[i]
    end

    EDmu2 = copy(Dmu2)  # use observed information (following R qgam)

    result = Dict{Symbol, Any}(:Dmu => Dmu, :Dmu2 => Dmu2, :EDmu2 => EDmu2)

    if level > 0
        Dth = similar(y)
        Dmuth = similar(y)
        Dmu3 = similar(y)
        Dmu2th = similar(y)

        @inbounds for i in eachindex(y, mu, wt)
            z = y[i] - mu[i]
            zl = z / lam[i]
            pl = 1.0 / (1.0 + exp(-zl))
            dl = pl * (1.0 - pl) / lam[i]

            _, D1, D2, _ = sigmoid_derivs(zl)

            term = (1.0 - tau) * lam[i] * log1p(-tau) +
                   lam[i] * tau * log(tau) -
                   (1.0 - tau) * z +
                   lam[i] * log1pexp(zl)

            Dth[i] = -2.0 * wt[i] * term / sig[i]
            Dmuth[i] = -Dmu[i]
            Dmu3[i] = -(2.0 * wt[i] * D2) / (sig[i] * lam[i]^2)
            Dmu2th[i] = -Dmu2[i]
        end

        result[:Dth] = Dth
        result[:Dmuth] = Dmuth
        result[:Dmu3] = Dmu3
        result[:Dmu2th] = Dmu2th
    end

    if level > 1
        Dmu4 = similar(y)
        Dth2 = similar(y)
        Dmuth2 = similar(y)
        Dmu2th2 = similar(y)
        Dmu3th = similar(y)

        @inbounds for i in eachindex(y, mu, wt)
            z = y[i] - mu[i]
            zl = z / lam[i]
            _, D1, D2, D3 = sigmoid_derivs(zl)

            Dmu4[i] = (2.0 * wt[i] * D3) / (sig[i] * lam[i]^3)
            Dth2[i] = -result[:Dth][i]
            Dmuth2[i] = Dmu[i]
            Dmu2th2[i] = Dmu2[i]
            Dmu3th[i] = -result[:Dmu3][i]
        end

        result[:Dmu4] = Dmu4
        result[:Dth2] = Dth2
        result[:Dmuth2] = Dmuth2
        result[:Dmu2th2] = Dmu2th2
        result[:Dmu3th] = Dmu3th
    end

    return result
end

# ============================================================================
# Log saturated likelihood and derivatives w.r.t. theta
# ============================================================================

"""
    elf_ls(f::ELFFamily, y, wt)

Compute log saturated likelihood and its derivatives w.r.t. θ = log(σ).
"""
function elf_ls(f::ELFFamily, y, wt)
    tau = f.qu
    n = length(y)
    co = _get_co(f, n)
    sig = _get_sig(f, n)
    lam = co

    ls = 0.0
    lsth = 0.0
    lsth2 = 0.0

    @inbounds for i in eachindex(y, wt)
        α = lam[i] * (1.0 - tau) / sig[i]
        β_param = lam[i] * tau / sig[i]
        γ = lam[i] / sig[i]

        ls += wt[i] * (
            (1.0 - tau) * lam[i] * log1p(-tau) / sig[i] +
            lam[i] * tau * log(tau) / sig[i] -
            log(lam[i]) - logabsbeta(α, β_param)[1]
        )

        dα = digamma(α)
        dβ = digamma(β_param)
        dγ = digamma(γ)

        lsth += wt[i] * (
            -(1.0 - tau) * log1p(-tau) -
            tau * log(tau) +
            (1.0 - tau) * dα +
            tau * dβ -
            dγ
        ) * lam[i] / sig[i]

        tα = trigamma(α)
        tβ = trigamma(β_param)
        tγ = trigamma(γ)

        lsth2 += -wt[i] * (
            -(1.0 - tau) * log1p(-tau) -
            tau * log(tau) +
            (1.0 - tau) * dα +
            tau * dβ -
            dγ
        ) * lam[i] / sig[i] -
        wt[i] * (
            (1.0 - tau)^2 * tα +
            tau^2 * tβ -
            tγ
        ) * lam[i]^2 / sig[i]^2
    end

    return (ls=ls, lsth1=lsth, lsth2=lsth2)
end

# ============================================================================
# Theta estimation (for when estimate_theta=true)
# ============================================================================

function estimate_theta!(f::ELFFamily, y, mu, wt, scale)
    if !f.estimate_theta
        return
    end
    # Newton step on the REML-like objective for theta
    n = length(y)
    ls_result = elf_ls(f, y, wt)
    # Also need the deviance derivative w.r.t. theta
    dd = elf_Dd(f, y, mu, wt; level=1)

    dev_th = sum(dd[:Dth])
    total_grad = ls_result.lsth1 - 0.5 * dev_th
    total_hess = ls_result.lsth2

    if abs(total_hess) > 1e-10
        step = -total_grad / total_hess
        step = clamp(step, -1.0, 1.0)  # conservative step
        f.theta += step
    end
end

# ============================================================================
# Pinball loss
# ============================================================================

"""
    pinball_loss(y, mu, qu; reduce=true)

Compute the pinball (check) loss for quantile regression.

`L_τ(y, μ) = (τ - 1{y < μ}) · (y - μ)`

If `reduce=true` (default), returns the sum. Otherwise returns per-observation losses.
"""
function pinball_loss(y::AbstractVector, mu::AbstractVector, qu::Real; reduce::Bool=true)
    0.0 < qu < 1.0 || throw(ArgumentError("qu must be in (0, 1)"))
    n = length(y)
    length(mu) == n || throw(DimensionMismatch("y and mu must have same length"))

    if reduce
        loss = 0.0
        @inbounds for i in 1:n
            d = y[i] - mu[i]
            loss += d < 0.0 ? -(1.0 - qu) * d : qu * d
        end
        return loss
    else
        losses = similar(y)
        @inbounds for i in 1:n
            d = y[i] - mu[i]
            losses[i] = d < 0.0 ? -(1.0 - qu) * d : qu * d
        end
        return losses
    end
end

# ============================================================================
# Main qgam API
# ============================================================================

"""
    qgam(formula, data, qu; lsig=nothing, err=nothing, control=gam_control(), kwargs...)

Fit a smooth additive quantile regression model for a single quantile.

Uses the Extended Log-F (ELF) density as a smooth surrogate for the pinball loss,
following Fasiolo et al. (2020). The model is fit using the standard GAM machinery
(penalized IRLS with REML smoothing parameter selection).

# Arguments
- `formula`: GAM formula (e.g., `@gam_formula(y ~ s(x, k=20))`)
- `data`: data table
- `qu`: target quantile ∈ (0, 1)
- `lsig`: log learning rate. If `nothing`, it is calibrated automatically.
- `err`: error bound for quantile curve. If `nothing`, estimated from data.
- `control`: `GamControl` for fitting

# Returns
A `GamModel` with the ELF family fit.

# Examples
```julia
fit = qgam(@gam_formula(y ~ s(x, k=20)), data, 0.5)
```
"""
function qgam(formula, data, qu::Real;
              lsig::Union{Nothing, Real}=nothing,
              err::Union{Nothing, Real}=nothing,
              control::GamControl=gam_control(),
              kwargs...)
    0.0 < qu < 1.0 || throw(ArgumentError("qu must be in (0, 1)"))

    # Step 1: Initial Gaussian fit to estimate variance
    gauss_fit = gam(formula, data; control=control, kwargs...)
    y = gauss_fit.y
    n = length(y)
    mu_gauss = gauss_fit.fitted_values
    var_hat = mean((y .- mu_gauss) .^ 2)

    # Step 2: Compute error parameter if not provided
    if err === nothing
        err = _get_err_param(qu, var_hat, n)
    end

    # Step 3: Compute smoothness constant co
    co = err * sqrt(2π * var_hat) / (2 * log(2))

    # Step 4: Determine learning rate
    if lsig === nothing
        lsig = _tune_learn_fast(formula, data, qu, co, var_hat, gauss_fit;
                                control=control, kwargs...)
    end

    # Step 5: Fit the quantile model with ELF family
    elf = ELFFamily(qu=qu, co=fill(co, n), theta=Float64(lsig))
    fit = gam(formula, data; family=elf, control=control, kwargs...)

    return fit
end

"""
    mqgam(formula, data, qu; kwargs...)

Fit smooth additive quantile regression models for multiple quantiles.

# Arguments
- `formula`: GAM formula
- `data`: data table
- `qu`: vector of quantiles ∈ (0, 1)
- `lsig`: log learning rate (scalar or vector of same length as qu)
- `co`: smoothness constant (scalar)

# Returns
A `NamedTuple` with `fits` (Dict of qu => GamModel) and shared structure.
"""
function mqgam(formula, data, qu::AbstractVector{<:Real};
               lsig::Union{Nothing, Real}=nothing,
               co::Union{Nothing, Real}=nothing,
               control::GamControl=gam_control(),
               kwargs...)
    # Get response to compute n
    gf = formula isa GamFormula ? formula : GamFormula(formula)
    resp_name = gf.response
    y = Tables.getcolumn(Tables.columns(data), resp_name)
    n = length(y)

    fits = Dict{Float64, Any}()
    for q in qu
        if co !== nothing && lsig !== nothing
            elf = ELFFamily(qu=Float64(q), co=fill(Float64(co), n), theta=Float64(lsig))
            fits[Float64(q)] = gam(formula, data; family=elf, control=control, kwargs...)
        else
            fits[Float64(q)] = qgam(formula, data, q; lsig=lsig, control=control, kwargs...)
        end
    end
    return (fits=fits, quantiles=Float64.(qu))
end

# ============================================================================
# Internal: Error parameter selection
# ============================================================================

"""
Automatic selection of the error parameter `err` following Fasiolo et al. (2020).
"""
function _get_err_param(qu::Real, var_hat::Real, n::Int)
    # Heuristic from qgam: err decreases with n, scaled by quantile extremity
    qu_adj = min(qu, 1.0 - qu)
    err = min(0.5, 5.0 / sqrt(n) / qu_adj)
    return clamp(err, 0.01, 0.5)
end

# ============================================================================
# Internal: Learning rate calibration (bootstrap + Brent search, matching R's tuneLearnFast)
# ============================================================================

"""
Learning rate calibration using bootstrap out-of-sample pinball loss,
optimized via Brent's method. Matches R qgam's tuneLearnFast approach.
"""
function _tune_learn_fast(formula, data, qu::Real, co::Real, var_hat::Real, gauss_fit;
                          control::GamControl=gam_control(),
                          K::Int=20, kwargs...)
    y = gauss_fit.y
    n = length(y)

    # Initial search range centered on sqrt(var_hat)
    sig_init = sqrt(var_hat)
    lsig_center = log(sig_init)
    lsig_lo = lsig_center - 3.0
    lsig_hi = lsig_center + 3.0

    # Generate K bootstrap weight vectors with reproducible RNG
    # Use a deterministic seed based on data characteristics for reproducibility
    boot_rng = MersenneTwister(hash((n, qu, round(var_hat, digits=6))))
    boot_weights = Vector{Vector{Float64}}(undef, K)
    for b in 1:K
        # Multinomial bootstrap: sample indices with replacement, convert to counts
        idx = rand(boot_rng, 1:n, n)
        w = zeros(n)
        for i in idx
            w[i] += 1.0
        end
        boot_weights[b] = w
    end

    # Objective: mean out-of-sample pinball loss over bootstrap samples
    function _boot_pinball_loss(lsig_val)
        elf = ELFFamily(qu=qu, co=fill(co, n), theta=lsig_val)
        total_loss = 0.0
        n_valid = 0

        for b in 1:K
            w = boot_weights[b]
            try
                # Fit on bootstrap (weighted) sample
                fit_b = gam(formula, data; family=elf, weights=w, control=control, kwargs...)
                mu_b = fit_b.fitted_values

                # Evaluate pinball loss only on out-of-bag observations (w[i] == 0)
                oob_loss = 0.0
                n_oob = 0
                for i in 1:n
                    if w[i] == 0.0
                        d = y[i] - mu_b[i]
                        oob_loss += d < 0.0 ? -(1.0 - qu) * d : qu * d
                        n_oob += 1
                    end
                end
                if n_oob > 0
                    total_loss += oob_loss / n_oob
                    n_valid += 1
                end
            catch
                continue
            end
        end

        return n_valid > 0 ? total_loss / n_valid : Inf
    end

    # Brent's method: golden section search with parabolic interpolation
    best_lsig, _ = _brent_minimize(_boot_pinball_loss, lsig_lo, lsig_hi; tol=0.1, maxiter=20)

    return best_lsig
end

"""
Brent's method for 1D minimization on [a, b].
Returns (x_min, f_min).
"""
function _brent_minimize(f, a::Real, b::Real; tol::Real=1e-4, maxiter::Int=50)
    golden = 0.3819660112501051  # (3 - √5) / 2

    x = w = v = a + golden * (b - a)
    fx = fw = fv = f(x)

    d = e = 0.0

    for iter in 1:maxiter
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        if abs(x - midpoint) <= tol2 - 0.5 * (b - a)
            return x, fx
        end

        # Try parabolic interpolation
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0
                p = -p
            else
                q = -q
            end

            if abs(p) < abs(0.5 * q * e) && p > q * (a - x) && p < q * (b - x)
                e = d
                d = p / q
            else
                e = (x < midpoint ? b : a) - x
                d = golden * e
            end
        else
            e = (x < midpoint ? b : a) - x
            d = golden * e
        end

        u = abs(d) >= tol1 ? x + d : x + copysign(tol1, d)
        fu = f(u)

        if fu <= fx
            if u < x
                b = x
            else
                a = x
            end
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else
            if u < x
                a = u
            else
                b = u
            end
            if fu <= fw || w == x
                v, fv = w, fw
                w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end

    return x, fx
end
