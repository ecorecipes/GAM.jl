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
_has_Dd(::ELFFamily) = false
_family_Dd(f::ELFFamily, y, mu, wt; level=0) = elf_Dd(f, y, mu, wt; level=level)

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
    mean_lam = mean(co)
    return sig .* co ./ mean_lam
end

# Precomputed constants for tight inner loops
function _elf_constants(f::ELFFamily, n::Int)
    tau = f.qu
    co = _get_co(f, n)
    sig_base = exp(f.theta)
    mean_lam = mean(co)
    log_tau = log(tau)
    log1m_tau = log1p(-tau)
    return (tau=tau, co=co, sig_base=sig_base, mean_lam=mean_lam,
            log_tau=log_tau, log1m_tau=log1m_tau)
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
_clamp_mu(::ELFFamily, mu::Float64) = mu  # scalar method

function _variance(f::ELFFamily, mu)
    # Working variance for PIRLS: use observed information
    # V(μ) ∝ σ / f_logistic((y-μ)/λ)
    # Since we don't have y here, return constant (PIRLS uses Dd-style weights)
    n = length(mu)
    sig = _get_sig(f, n)
    return sig
end

# Scalar variance: for use in tight loops
function _elf_var_i(f::ELFFamily, i::Int, mean_lam::Float64)
    sig_base = exp(f.theta)
    co_val = length(f.co) == 1 ? f.co[1] : f.co[i]
    return sig_base * co_val / mean_lam
end

"""
    _deviance(f::ELFFamily, y, mu, wt)

Compute the ELF deviance: -2 × (log-likelihood - saturated log-likelihood).
"""
function _deviance(f::ELFFamily, y, mu, wt)
    tau = f.qu
    n = length(y)
    co = f.co
    sig_base = exp(f.theta)
    is_scalar_co = length(co) == 1
    co_val = is_scalar_co ? co[1] : 0.0
    mean_lam = is_scalar_co ? co_val : mean(co)
    log_tau = log(tau)
    log1m_tau = log1p(-tau)
    one_m_tau = 1.0 - tau

    dev = 0.0
    @inbounds for i in eachindex(y, mu, wt)
        lam_i = is_scalar_co ? co_val : co[i]
        sig_i = sig_base * lam_i / mean_lam
        z = y[i] - mu[i]
        term = one_m_tau * lam_i * log1m_tau +
               lam_i * tau * log_tau -
               one_m_tau * z +
               lam_i * log1pexp(z / lam_i)
        dev += 2.0 * wt[i] * term / sig_i
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
    qgam(formulas, data, qu; co=nothing, err=nothing, links=[IdentityLink(), LogLink()],
         method=:efs, control=mp_control(), gamlss_ctrl=gamlss_control(), kwargs...)

Fit a smooth additive quantile regression model for a single quantile.

With a single formula, `qgam` fits the standard fixed-scale `ELFFamily` model
and returns a `GamModel`.

With a vector of two formulas, `qgam` fits a location-scale quantile model using
`ELFLSSFamily` and returns a `MultiParameterModel`. The first formula models the
target quantile location `μ`, and the second models the covariate-dependent scale
`σ`.

Uses the Extended Log-F (ELF) density as a smooth surrogate for the pinball loss,
following Fasiolo et al. (2020).

# Arguments
- `formula` / `formulas`: GAM formula, or two formulas for `μ` and `σ`
- `data`: data table
- `qu`: target quantile ∈ (0, 1)
- `lsig`: log learning rate for single-formula ELF fits. If `nothing`, it is calibrated automatically.
- `co`: ELF smoothness constant for two-formula ELFLSS fits. If `nothing`, it is initialized from the response variance.
- `err`: error bound for quantile curve. If `nothing`, estimated from data.

# Examples
```julia
fit = qgam(@formulak(y ~ s(x, k=20)), data, 0.5)

fit_lss = qgam([
    @formulak(y ~ s(x, k=20)),
    @formulak(y ~ 0 + s(x, k=10))
], data, 0.9)
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

    # Step 5: Fit the quantile model with ELF family, starting from Gaussian fit
    elf = ELFFamily(qu=qu, co=fill(co, n), theta=Float64(lsig))
    fit = gam(formula, data; family=elf, start=gauss_fit.coefficients,
              control=control, kwargs...)

    return fit
end

function _qgam_default_co(y::AbstractVector, qu::Real; err::Union{Nothing, Real}=nothing)
    y_float = Float64.(y)
    n = length(y_float)
    var_hat = max(var(y_float; corrected=false), eps(Float64))
    err_val = err === nothing ? _get_err_param(qu, var_hat, n) : Float64(err)
    return err_val * sqrt(2π * var_hat) / (2 * log(2))
end

function qgam(formulas::AbstractVector, data, qu::Real;
              co::Union{Nothing, Real}=nothing,
              err::Union{Nothing, Real}=nothing,
              links::Vector{<:GLM.Link}=[IdentityLink(), LogLink()],
              method::Symbol=:efs,
              control::MPFitControl=mp_control(),
              gamlss_ctrl::GamlssControl=gamlss_control(),
              kwargs...)
    0.0 < qu < 1.0 || throw(ArgumentError("qu must be in (0, 1)"))
    length(formulas) == 2 || throw(ArgumentError(
        "ELFLSS qgam expects exactly 2 formulas (mu and sigma), got $(length(formulas))."))
    length(links) == 2 || throw(ArgumentError(
        "ELFLSS qgam expects exactly 2 links (mu and sigma), got $(length(links))."))

    y = _extract_response(formulas[1], Tables.columntable(data))
    co_val = co === nothing ? _qgam_default_co(y, qu; err=err) : Float64(co)

    fam = ELFLSSFamily(qu=qu, co=co_val, links=links)
    return gam(formulas, data, fam; method=method, control=control,
               gamlss_ctrl=gamlss_ctrl, kwargs...)
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
    resp_name = if formula isa GamFormula
        formula.response
    elseif formula isa FormulaTerm
        Symbol(formula.lhs)
    else
        error("Unsupported formula type: $(typeof(formula))")
    end
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

Performance optimizations vs naive approach:
1. Pre-builds X, smooths, penalty once — avoids re-parsing formula per fit
2. Warm-starts each bootstrap fit from the previous one (not just Gaussian)
3. Parallelizes bootstrap samples across threads
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
    boot_rng = MersenneTwister(hash((n, qu, round(var_hat, digits=6))))
    boot_weights = Vector{Vector{Float64}}(undef, K)
    for b in 1:K
        idx = rand(boot_rng, 1:n, n)
        w = zeros(n)
        for i in idx
            w[i] += 1.0
        end
        boot_weights[b] = w
    end

    # Pre-build model matrices and penalties ONCE (main optimization #1)
    X, smooths, n_parametric, penalty_template = _precompute_gam_matrices(formula, data)
    link = IdentityLink()  # ELF uses identity link
    p = size(X, 2)

    # Use Gaussian SP as fixed SP for bootstrap fits (avoids full outer loop)
    gauss_sp = gauss_fit.sp
    gauss_coef = gauss_fit.coefficients

    # Quiet control for bootstrap fits — minimal iterations
    boot_control = gam_control(
        epsilon=control.epsilon, maxit=control.maxit,
        outer_maxit=min(control.outer_maxit, 10),
        trace=false, sp_optimizer=control.sp_optimizer,
        gamma=control.gamma)

    # Objective: mean out-of-sample pinball loss over bootstrap samples
    function _boot_pinball_loss(lsig_val)
        elf = ELFFamily(qu=qu, co=fill(co, n), theta=lsig_val)

        # Pre-compute S_total once (SP is fixed from Gaussian fit)
        S_total = total_penalty(penalty_template, gauss_sp, p)

        # Parallel bootstrap evaluation (optimization #3)
        losses = zeros(K)
        valid = zeros(Bool, K)
        nthreads = Threads.nthreads()

        if nthreads > 1
            Threads.@threads for b in 1:K
                loss, ok = _eval_one_bootstrap_fast(
                    X, y, S_total, elf, link,
                    boot_weights[b], gauss_coef, boot_control, qu)
                losses[b] = loss
                valid[b] = ok
            end
        else
            for b in 1:K
                loss, ok = _eval_one_bootstrap_fast(
                    X, y, S_total, elf, link,
                    boot_weights[b], gauss_coef, boot_control, qu)
                losses[b] = loss
                valid[b] = ok
            end
        end

        n_valid = sum(valid)
        return n_valid > 0 ? sum(losses[valid]) / n_valid : Inf
    end

    if control.trace
        println("Estimating learning rate. Each dot corresponds to a loss evaluation. ")
        print("qu = $qu")
    end

    # Wrap to show progress dots
    function _traced_loss(lsig_val)
        result = _boot_pinball_loss(lsig_val)
        control.trace && print(".")
        return result
    end

    best_lsig, _ = _brent_minimize(_traced_loss, lsig_lo, lsig_hi; tol=0.1, maxiter=20)

    if control.trace
        println("done ")
    end

    return best_lsig
end

"""
Fast bootstrap evaluation: fit ELF GAM with FIXED smoothing parameters (from Gaussian fit).
Skips the entire outer iteration — only runs one PIRLS solve.
This is valid for calibration since we only need approximate fitted values
to evaluate the OOB pinball loss.
"""
function _eval_one_bootstrap_fast(X, y, S_total, elf, link,
                                  boot_w, start_coefs, control, qu)
    n = length(y)
    try
        # Single PIRLS solve with fixed SP — no outer loop
        result = pirls_extended(X, y, S_total, elf, link;
            weights=boot_w, start=start_coefs, control=control)

        mu_b = result.fitted_values

        # Evaluate pinball loss on out-of-bag observations
        oob_loss = 0.0
        n_oob = 0
        @inbounds for i in 1:n
            if boot_w[i] == 0.0
                d = y[i] - mu_b[i]
                oob_loss += d < 0.0 ? -(1.0 - qu) * d : qu * d
                n_oob += 1
            end
        end
        if n_oob > 0
            return (oob_loss / n_oob, true)
        else
            return (0.0, false)
        end
    catch
        return (0.0, false)
    end
end

"""
Pre-compute model matrices and smooths from a formula + data, for reuse across
multiple fits (e.g., bootstrap calibration).
Returns (X, smooths, n_parametric, penalty).
"""
function _precompute_gam_matrices(formula, data)
    # Use setup_gam with Normal family (just builds matrices, family doesn't matter)
    y, X, X_para, smooths, n_parametric = setup_gam(formula, data; family=Normal())
    penalty = setup_penalties(smooths, n_parametric)
    _initial_sp(X, penalty)
    return X, smooths, n_parametric, penalty
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

# ============================================================================
# qdo — apply function to a single quantile from mqgam result
# ============================================================================

"""
    qdo(mqfit, qu, f=identity, args...; kwargs...)

Apply function `f` to a single quantile model extracted from an `mqgam` result.

# Arguments
- `mqfit`: result from `mqgam()` (a NamedTuple with `fits` and `quantiles`)
- `qu`: target quantile to extract
- `f`: function to apply (default: identity, returns the model itself)
- Additional positional and keyword arguments are forwarded to `f`.

# Examples
```julia
mq = mqgam(@formulak(y ~ s(x, k=20)), data, [0.1, 0.5, 0.9])
qdo(mq, 0.5)                          # extract the median fit
qdo(mq, 0.1, predict)                 # predictions at 10th percentile
qdo(mq, 0.9, predict, newdata; type=:response)
```
"""
function qdo(mqfit::NamedTuple, qu::Real, f=identity, args...; kwargs...)
    q = Float64(qu)
    haskey(mqfit.fits, q) || throw(ArgumentError(
        "Quantile $q not found in mqgam result. Available: $(sort(collect(keys(mqfit.fits))))"))
    model = mqfit.fits[q]
    return f(model, args...; kwargs...)
end

# ============================================================================
# cqcheck — quantile calibration check
# ============================================================================

"""
    CQCheckResult

Result of a quantile calibration check (1D binned analysis).

# Fields
- `bin_mid`: midpoints of bins along the conditioning variable
- `bin_lo`: lower edges of bins
- `bin_hi`: upper edges of bins
- `proportions`: observed proportion of y < μ̂ in each bin
- `ci_lower`: lower bound of binomial CI at level `lev`
- `ci_upper`: upper bound of binomial CI at level `lev`
- `bin_sizes`: number of observations in each bin
- `target_qu`: the target quantile
- `lev`: significance level used for CI
- `flagged`: boolean vector — true where proportion is outside CI
"""
struct CQCheckResult
    bin_mid::Vector{Float64}
    bin_lo::Vector{Float64}
    bin_hi::Vector{Float64}
    proportions::Vector{Float64}
    ci_lower::Vector{Float64}
    ci_upper::Vector{Float64}
    bin_sizes::Vector{Int}
    target_qu::Float64
    lev::Float64
    flagged::Vector{Bool}
end

"""
    cqcheck(model::GamModel, v::AbstractVector; nbin=10, lev=0.05, y=nothing)

Check quantile calibration by binning observations along variable `v`.

For a well-calibrated quantile model at level τ, the proportion of observations
with y < μ̂ should be approximately τ in every region of covariate space.
This function bins observations along `v` and checks whether the observed
proportion in each bin is consistent with the target quantile (using a
binomial confidence interval).

# Arguments
- `model`: a fitted qgam `GamModel` (must have `ELFFamily`)
- `v`: conditioning variable (numeric vector of length n)
- `nbin`: number of bins (default: 10)
- `lev`: significance level for binomial CI (default: 0.05)
- `y`: response vector (default: extracted from model)

# Returns
A `CQCheckResult` with per-bin calibration diagnostics.

# Examples
```julia
fit = qgam(@formulak(y ~ s(x, k=20)), data, 0.5)
res = cqcheck(fit, data.x; nbin=10)
res.flagged  # which bins have miscalibration
```
"""
function _cqcheck(mu::AbstractVector, y_obs::AbstractVector, v::AbstractVector, qu::Real;
                  nbin::Int=10, lev::Real=0.05)
    n = length(y_obs)
    length(mu) == n || throw(DimensionMismatch("fitted values and y must have same length"))
    length(v) == n || throw(DimensionMismatch("conditioning variable v must have length $n"))

    # Binary indicator: y < μ̂
    below = [y_obs[i] < mu[i] ? 1.0 : 0.0 for i in 1:n]

    # Create equal-width bins
    v_min, v_max = extrema(v)
    bounds = range(v_min, v_max; length=nbin + 1)

    # Assign observations to bins
    bin_idx = zeros(Int, n)
    for i in 1:n
        for b in 1:nbin
            if v[i] >= bounds[b] && (b == nbin ? v[i] <= bounds[b + 1] : v[i] < bounds[b + 1])
                bin_idx[i] = b
                break
            end
        end
        if bin_idx[i] == 0
            bin_idx[i] = nbin  # edge case: exactly at max
        end
    end

    # Compute per-bin statistics
    bin_mid = Float64[]
    bin_lo = Float64[]
    bin_hi = Float64[]
    proportions = Float64[]
    ci_lower = Float64[]
    ci_upper = Float64[]
    bin_sizes = Int[]
    flagged = Bool[]

    for b in 1:nbin
        mask = bin_idx .== b
        bsize = sum(mask)
        bsize > 0 || continue

        n_below = sum(below[mask])
        prop = n_below / bsize

        # Binomial CI: use normal approximation for large samples, exact for small
        lb = quantile(Binomial(bsize, qu), lev / 2) / bsize
        ub = quantile(Binomial(bsize, qu), 1.0 - lev / 2) / bsize

        push!(bin_mid, (bounds[b] + bounds[b + 1]) / 2)
        push!(bin_lo, bounds[b])
        push!(bin_hi, bounds[b + 1])
        push!(proportions, prop)
        push!(ci_lower, lb)
        push!(ci_upper, ub)
        push!(bin_sizes, bsize)
        push!(flagged, prop < lb || prop > ub)
    end

    return CQCheckResult(bin_mid, bin_lo, bin_hi, proportions, ci_lower, ci_upper,
                         bin_sizes, qu, lev, flagged)
end

function cqcheck(model::GamModel, v::AbstractVector;
                 nbin::Int=10, lev::Real=0.05,
                 y::Union{Nothing, AbstractVector}=nothing)
    mu = model.fitted_values
    y_obs = y === nothing ? model.y : y
    fam = model.family
    qu = if fam isa ELFFamily
        fam.qu
    else
        @warn "Model does not use ELFFamily; assuming qu=0.5"
        0.5
    end
    return _cqcheck(mu, y_obs, v, qu; nbin=nbin, lev=lev)
end

function _elflss_location(model::MultiParameterModel)
    model.family isa ELFLSSFamily || throw(ArgumentError(
        "_elflss_location requires a MultiParameterModel with ELFLSSFamily, got $(typeof(model.family))."))
    fam = model.family
    return _apply_link_inv.(Ref(fam.links[1]), model.fitted_eta[1])
end

function cqcheck(model::MultiParameterModel, v::AbstractVector;
                 nbin::Int=10, lev::Real=0.05,
                 y::Union{Nothing, AbstractVector}=nothing)
    model.family isa ELFLSSFamily || throw(ArgumentError(
        "cqcheck is only defined for MultiParameterModel fits using ELFLSSFamily, got $(typeof(model.family))."))
    mu = _elflss_location(model)
    y_obs = y === nothing ? model.y : y
    return _cqcheck(mu, y_obs, v, model.family.qu; nbin=nbin, lev=lev)
end

function Base.show(io::IO, r::CQCheckResult)
    n_flagged = sum(r.flagged)
    println(io, "Quantile Calibration Check (target τ = $(r.target_qu), level = $(r.lev))")
    println(io, "  $(length(r.bin_mid)) bins, $(n_flagged) flagged")
    println(io, "")
    println(io, "  Bin midpoint  | Proportion | CI lower | CI upper | Size | Flag")
    println(io, "  " * "─"^65)
    for i in eachindex(r.bin_mid)
        flag_str = r.flagged[i] ? " *" : "  "
        @printf(io, "  %12.4f  | %10.4f | %8.4f | %8.4f | %4d |%s\n",
                r.bin_mid[i], r.proportions[i], r.ci_lower[i], r.ci_upper[i],
                r.bin_sizes[i], flag_str)
    end
end

# ============================================================================
# check_qgam — comprehensive qgam diagnostics (like R's check.qgam)
# ============================================================================

"""
    QGamCheck

Result of comprehensive qgam diagnostic checks.

# Fields
- `target_qu`: target quantile
- `actual_proportion`: actual proportion of negative residuals
- `integrated_abs_bias`: mean |F(μ̂) - F(μ₀)| — quantile bias from smoothed loss
- `bias_values`: per-observation bias values
- `calibration`: `CQCheckResult` from cqcheck on fitted values
"""
struct QGamCheck
    target_qu::Float64
    actual_proportion::Float64
    integrated_abs_bias::Float64
    bias_values::Vector{Float64}
    calibration::CQCheckResult
end

"""
    check_qgam(model::GamModel; nbin=10, lev=0.05)

Comprehensive diagnostic check for a fitted qgam model.

Computes:
1. Proportion of negative residuals vs target quantile
2. Bias due to smoothed loss: |F(μ̂) - F(μ₀)| using the logistic approximation
3. Calibration check (cqcheck) on fitted values

# Returns
A `QGamCheck` with diagnostic results.

# Examples
```julia
fit = qgam(@formulak(y ~ s(x, k=20)), data, 0.5)
chk = check_qgam(fit)
chk.actual_proportion   # should be ≈ 0.5
chk.integrated_abs_bias # should be small (< 0.05)
```
"""
function _check_qgam(mu::AbstractVector, y::AbstractVector, qu::Real, lam::Real;
                     nbin::Int=10, lev::Real=0.05)
    n = length(y)
    res = y .- mu

    # Actual proportion of negative residuals
    actual_prop = mean(res .< 0)

    # Bias from smoothed loss: bias_i = logistic((μ̂_i - y_i) / λ) - I(μ̂_i > y_i)
    # The logistic CDF F(x) = 1/(1+exp(-x/λ)) applied to residual
    bias = similar(mu)
    for i in 1:n
        r = mu[i] - y[i]
        bias[i] = 1.0 / (1.0 + exp(-r / lam)) - (r > 0 ? 1.0 : 0.0)
    end
    iab = mean(abs.(bias))

    # Calibration check on fitted values
    cal = _cqcheck(mu, y, mu, qu; nbin=nbin, lev=lev)

    return QGamCheck(qu, actual_prop, iab, bias, cal)
end

function check_qgam(model::GamModel; nbin::Int=10, lev::Real=0.05)
    fam = model.family
    qu = fam isa ELFFamily ? fam.qu : 0.5
    lam = fam isa ELFFamily ? (length(fam.co) == 1 ? fam.co[1] : mean(fam.co)) : 0.1
    return _check_qgam(model.fitted_values, model.y, qu, lam; nbin=nbin, lev=lev)
end

function check_qgam(model::MultiParameterModel; nbin::Int=10, lev::Real=0.05)
    model.family isa ELFLSSFamily || throw(ArgumentError(
        "check_qgam is only defined for MultiParameterModel fits using ELFLSSFamily, got $(typeof(model.family))."))
    mu = _elflss_location(model)
    return _check_qgam(mu, model.y, model.family.qu, model.family.co; nbin=nbin, lev=lev)
end

function Base.show(io::IO, chk::QGamCheck)
    println(io, "qgam Diagnostic Check")
    println(io, "  Target quantile:     $(chk.target_qu)")
    @printf(io, "  Actual proportion:   %.4f\n", chk.actual_proportion)
    @printf(io, "  Integrated |bias|:   %.6f\n", chk.integrated_abs_bias)
    println(io, "")
    show(io, chk.calibration)
end

# ============================================================================
# Quantile residuals for qgam models
# ============================================================================

"""
    quantile_residuals(model::GamModel)

Compute randomized quantile residuals for a qgam model.

For a quantile regression model at level τ, the quantile residual is:
- `r_i = Φ⁻¹(F_ELF(y_i | μ̂_i))`

where F_ELF is the ELF CDF and Φ⁻¹ is the normal quantile function.
Well-specified models produce residuals that are approximately standard normal.

For non-ELF models, returns standard response residuals.
"""
function quantile_residuals(model::GamModel)
    fam = model.family
    if !(fam isa ELFFamily)
        return model.y .- model.fitted_values
    end

    y = model.y
    mu = model.fitted_values
    n = length(y)
    qu = fam.qu
    sig = exp(fam.theta)
    co = length(fam.co) == 1 ? fill(fam.co[1], n) : fam.co

    qres = similar(y)
    for i in 1:n
        lam = co[i]
        # ELF CDF: integral of ELF density from -∞ to y
        # F_ELF(y|μ) is the logistic CDF with location μ and scale λ
        # shifted by the asymmetry of τ
        z = (y[i] - mu[i]) / lam
        # CDF of the ELF: p = τ·sigmoid(z/τ) for z < 0, τ + (1-τ)·(1-sigmoid(-z/(1-τ))) for z ≥ 0
        # Simplified: F(y|μ) = logistic(z) is the symmetric approximation
        p = 1.0 / (1.0 + exp(-z))
        # Clamp to avoid Inf in quantile transform
        p = clamp(p, 1e-10, 1 - 1e-10)
        qres[i] = quantile(Normal(), p)
    end

    return qres
end

function quantile_residuals(model::MultiParameterModel)
    model.family isa ELFLSSFamily || throw(ArgumentError(
        "quantile_residuals is only defined for MultiParameterModel fits using ELFLSSFamily, got $(typeof(model.family))."))
    fam = model.family
    y = model.y
    mu = _elflss_location(model)
    qres = similar(y)
    for i in eachindex(y, mu)
        p = 1.0 / (1.0 + exp(-(y[i] - mu[i]) / fam.co))
        p = clamp(p, 1e-10, 1 - 1e-10)
        qres[i] = quantile(Normal(), p)
    end
    return qres
end

# ============================================================================
# ELFLSS Family — Extended Log-F Location-Scale (2-parameter quantile model)
# ============================================================================

"""
    ELFLSSFamily(; qu=0.5, co=0.1)

Extended Log-F Location-Scale family for 2-parameter quantile regression.

This is a GAMLSS-style model with two linear predictors:
- η₁ → μ (quantile location, identity link)
- η₂ → log(σ) (log learning rate, log link)

The ELF density with varying σ per observation allows the learning rate to
adapt across covariate space, potentially improving calibration.

NLL per observation:
```
nll(y, μ, σ) = (1-τ)·z - (co/σ)·log(1+exp(z·σ/co)) + log(co·B(co(1-τ)/σ, co·τ/σ))
```
where z = (y - μ)/σ and B is the beta function.

# Arguments
- `qu`: target quantile ∈ (0, 1)
- `co`: smoothness constant (typically data-dependent)

Use directly with `gam()` / `gamlss()`, or via `qgam([mu_formula, sigma_formula], ...)`:
```julia
fam = ELFLSSFamily(qu=0.5, co=0.2)
m = gam([@formulak(y ~ s(x, k=20)), @formulak(y ~ 0 + s(x, k=10))],
        data, fam)
```
"""
struct ELFLSSFamily <: MultiParameterFamily
    qu::Float64
    co::Float64
    links::Vector{GLM.Link}
end

function ELFLSSFamily(; qu::Real=0.5, co::Real=0.1,
                       links=[IdentityLink(), LogLink()])
    0.0 < qu < 1.0 || throw(ArgumentError("qu must be in (0, 1), got $qu"))
    co > 0 || throw(ArgumentError("co must be positive, got $co"))
    return ELFLSSFamily(Float64(qu), Float64(co), links)
end

nparams(::ELFLSSFamily) = 2
param_names(::ELFLSSFamily) = ["mu", "sigma"]
param_links(f::ELFLSSFamily) = [_link_symbol(l) for l in f.links]

function nll_obs(f::ELFLSSFamily, y_i, η_vec)
    tau = f.qu
    co = f.co

    mu = _apply_link_inv(f.links[1], η_vec[1])
    sig = _apply_link_inv(f.links[2], η_vec[2])
    sig = max(sig, 1e-10)

    lam = co / sig
    z = (y_i - mu) / sig

    # ELF NLL: -log p(y|μ,σ) (sign-flipped log-density)
    # log-density = (1-τ)·z - lam·log(1+exp(z/lam)) - log(sig·lam·B(lam(1-τ), lam·τ))
    lpxp = log1pexp(z / lam)
    log_norm = _elf_log_normalizer(lam, tau)

    nll = -((1.0 - tau) * z - lam * lpxp - log_norm - log(sig))
    return nll
end

"""
Compute log normalizing constant for ELF density:
log(λ · B(λ(1-τ), λτ))
"""
function _elf_log_normalizer(lam::Real, tau::Real)
    a = lam * (1.0 - tau)
    b = lam * tau
    # logabsbeta returns (log|B|, sign)
    lab, _ = logabsbeta(a, b)
    return log(lam) + lab
end

function initial_eta(f::ELFLSSFamily, y::AbstractVector)
    n = length(y)
    mu_init = GLM.linkfun(f.links[1], median(y))
    sig_init = GLM.linkfun(f.links[2], max(std(y), 0.01))
    return [fill(mu_init, n), fill(sig_init, n)]
end
