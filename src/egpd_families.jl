# Extended Generalized Pareto Distribution (EGPD) families for multi-parameter GAMs
#
# The EGPD extends the standard GPD by composing its CDF with a transformation G:
#   F_EGPD(y) = G(F_GPD(y; σ, ξ))
#
# where F_GPD(y; σ, ξ) = 1 - (1 + ξy/σ)^(-1/ξ) is the standard GPD CDF.
#
# Different choices of G give different EGPD models:
#   Model 1 (Power):      G(u) = u^κ                          — 3 params (ψ, ξ, log κ)
#   Model 2 (Mixture):    G(u) = p·u^κ₁ + (1-p)·u^κ₂         — 5 params (ψ, ξ, log κ₁, log dκ, logit p)
#   Model 3 (Beta-type):  G(u) = 1 - B_inv((1-u)^δ; 1/δ, 2)  — 3 params (ψ, ξ, log δ)
#   Model 4 (Beta-kappa): G combines beta and power transforms  — 4 params (ψ, ξ, log δ, log κ)
#
# All models use log-scale ψ = log(σ) with identity link for ξ (shape).
# Additional parameters use log or logit links to ensure positivity/boundedness.
#
# Reference: Naveau et al. (2016), Tencaliec et al. (2020)
# Implementation follows the egpd R package by Ahmad et al.

using SpecialFunctions: logbeta, lbeta
using Distributions: Beta, cdf as dist_cdf, logpdf as dist_logpdf

# Threshold below which |ξ| is treated as 0 (exponential limit of the GPD).
# Used consistently by nll_obs AND the exact-derivative routines so the
# objective and its derivatives agree on which branch is active.
const _EGPD_XI_EPS = 1e-6

"""
    _egpd_zero_row!(out, j)

Zero out row `j` of the per-observation derivative matrix. Used for
out-of-support points (NLL is the constant 1e20 there, so all derivatives
are finite zeros — consistent with how GPD handles out-of-support data).
"""
@inline function _egpd_zero_row!(out::Matrix{Float64}, j::Int)
    @inbounds for c in 1:size(out, 2)
        out[j, c] = 0.0
    end
    return nothing
end

"""
    _egpd_ad_derivs_row!(out, j, fam, yj, ηv)

Fill gradient + upper-triangle Hessian columns of `out` row `j` by
ForwardDiff through `nll_obs` (which contains the small-|ξ| exponential-limit
branch). Used as the small-|ξ| fallback for the hand-coded exact derivative
routines: correctness over speed near ξ = 0.
"""
function _egpd_ad_derivs_row!(out::Matrix{Float64}, j::Int,
                              fam::MultiParameterFamily, yj::Real,
                              ηv::Vector{Float64})
    K = length(ηv)
    f = η -> nll_obs(fam, yj, η)
    g = ForwardDiff.gradient(f, ηv)
    H = ForwardDiff.hessian(f, ηv)
    @inbounds for k in 1:K
        out[j, k] = g[k]
    end
    col = K
    @inbounds for c in 1:K, r in 1:c
        col += 1
        out[j, col] = H[r, c]
    end
    return nothing
end

# ============================================================================
# EGPD Model 1 — Power transformation G(u) = u^κ
# ============================================================================

"""
    EGPD1Family()

Extended GPD Model 1 (power transformation).

The density is:
    f(y) = g(F_GPD(y)) · f_GPD(y)

where G(u) = u^κ, so g(u) = κ u^(κ-1), and:
    -log f(y) = ψ + (1/ξ+1)log(1+ξy/σ) + (1-κ)log(1 - (1+ξy/σ)^(-1/ξ)) - log(κ)

Three parameters:
- η₁ = ψ = log(σ) (log-scale)
- η₂ = ξ (shape, identity link)
- η₃ = log(κ) (log-kappa)
"""
struct EGPD1Family <: MultiParameterFamily end

nparams(::EGPD1Family) = 3
param_names(::EGPD1Family) = ["logscale", "shape", "logkappa"]
param_links(::EGPD1Family) = [:log, :identity, :log]

function nll_obs(::EGPD1Family, yi::Real, η::AbstractVector)
    ψ = η[1]
    ξ = η[2]
    lκ = η[3]
    σ = exp(ψ)
    κ = exp(lκ)

    t = ξ * yi / σ
    t <= -1 && return oftype(η[1], 1e20)

    if abs(ξ) < _EGPD_XI_EPS
        z = yi / σ
        F_gpd = 1 - exp(-z)
        F_gpd <= 0 && return oftype(η[1], 1e20)
        log_F = log(F_gpd)
        nll_gpd = ψ + z
    else
        inv_xi = 1 / ξ
        log1pt = log1p(t)
        surv = (1 + t)^(-inv_xi)
        F_gpd = 1 - surv
        F_gpd <= 0 && return oftype(η[1], 1e20)
        log_F = log(F_gpd)
        nll_gpd = ψ + (inv_xi + 1) * log1pt
    end

    # NLL = NLL_GPD + (1 - κ)·log(F_GPD) - log(κ)
    # Matches C++: (1-exp(lkappa))·log(1 - s^(-1/ξ)) + (1+1/ξ)·log(1+t) + ψ - lκ
    return nll_gpd + (1 - κ) * log_F - lκ
end

function initial_eta(::EGPD1Family, y::AbstractVector)
    n = length(y)
    σ0 = max(mean(y), 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.1
    lκ0 = 0.0  # κ = 1 (standard GPD)
    return [fill(ψ0, n), fill(ξ0, n), fill(lκ0, n)]
end


# ============================================================================
# EGPD Model 2 — Mixture power transformation G(u) = p·u^κ₁ + (1-p)·u^κ₂
# ============================================================================

"""
    EGPD2Family()

Extended GPD Model 2 (mixture of two power transformations).

G(u) = p·u^κ₁ + (1-p)·u^κ₂ with κ₂ ≥ κ₁ (enforced via reparameterization).

Five parameters:
- η₁ = ψ = log(σ)
- η₂ = ξ (shape)
- η₃ = log(κ₁) (log first power)
- η₄ = log(κ₂ - κ₁) (log difference, ensures κ₂ ≥ κ₁)
- η₅ = logit(p) (mixture weight)
"""
struct EGPD2Family <: MultiParameterFamily end

nparams(::EGPD2Family) = 5
param_names(::EGPD2Family) = ["logscale", "shape", "logkappa1", "logdkappa", "logitp"]
param_links(::EGPD2Family) = [:log, :identity, :log, :log, :logit]

function nll_obs(::EGPD2Family, yi::Real, η::AbstractVector)
    ψ = η[1]
    ξ = η[2]
    lκ1 = η[3]
    ldκ = η[4]
    logit_p = η[5]

    σ = exp(ψ)
    κ1 = exp(lκ1)
    # κ2 = κ1 + exp(ldκ), computed via log-sum-exp for stability
    mx = max(lκ1, ldκ)
    lκ2 = mx + log1p(exp(-abs(lκ1 - ldκ)))
    κ2 = exp(lκ2)
    p = 1 / (1 + exp(-logit_p))

    t = ξ * yi / σ
    t <= -1 && return oftype(η[1], 1e20)

    if abs(ξ) < _EGPD_XI_EPS
        z = yi / σ
        F_gpd = 1 - exp(-z)
        nll_gpd = ψ + z
    else
        inv_xi = 1 / ξ
        log1pt = log1p(t)
        F_gpd = 1 - (1 + t)^(-inv_xi)
        nll_gpd = ψ + (inv_xi + 1) * log1pt
    end

    F_gpd <= 0 && return oftype(η[1], 1e20)

    # g(u) = p·κ₁·u^(κ₁-1) + (1-p)·κ₂·u^(κ₂-1)
    g_val = p * κ1 * F_gpd^(κ1 - 1) + (1 - p) * κ2 * F_gpd^(κ2 - 1)
    g_val <= 0 && return oftype(η[1], 1e20)

    return nll_gpd - log(g_val)
end

function initial_eta(::EGPD2Family, y::AbstractVector)
    n = length(y)
    σ0 = max(mean(y), 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.05
    lκ1_0 = 0.0   # κ₁ = 1
    ldκ_0 = 0.0    # dκ = 1, so κ₂ = 2
    logit_p_0 = 0.0  # p = 0.5
    return [fill(ψ0, n), fill(ξ0, n), fill(lκ1_0, n), fill(ldκ_0, n), fill(logit_p_0, n)]
end


# ============================================================================
# EGPD Model 3 — Beta-type transformation
# ============================================================================

"""
    EGPD3Family()

Extended GPD Model 3 (beta-type transformation).

G(u) = 1 - I_beta((1-u)^δ; 1/δ, 2) where I_beta is the regularized
incomplete beta function.

Equivalently, the NLL per observation:
    -log f(y) = (1/ξ+1)log(1+ξy/σ) + log(δ) + ψ - log(1 - (1+ξy/σ)^(-δ/ξ)) - log(1+δ)

Three parameters:
- η₁ = ψ = log(σ)
- η₂ = ξ (shape)
- η₃ = log(δ) (log-delta)
"""
struct EGPD3Family <: MultiParameterFamily end

nparams(::EGPD3Family) = 3
param_names(::EGPD3Family) = ["logscale", "shape", "logdelta"]
param_links(::EGPD3Family) = [:log, :identity, :log]

function nll_obs(::EGPD3Family, yi::Real, η::AbstractVector)
    ψ = η[1]
    ξ = η[2]
    lδ = η[3]
    σ = exp(ψ)
    δ = exp(lδ)

    t = ξ * yi / σ
    t <= -1 && return oftype(η[1], 1e20)

    if abs(ξ) < _EGPD_XI_EPS
        # Exponential limit: (1+ξz)^(-1/ξ) → exp(-z) as ξ → 0
        z = yi / σ
        nll_base = ψ + z
        surv_pow = exp(-δ * z)
    else
        inv_xi = 1 / ξ
        log1pt = log1p(t)
        nll_base = ψ + (inv_xi + 1) * log1pt
        surv_pow = (1 + t)^(-δ / ξ)  # (1+t)^(-δ/ξ)
    end

    # NLL = (1/ξ+1)log(1+t) + log(δ) + ψ - log(1 - (1+t)^(-δ/ξ)) - log(1+δ)
    one_minus_sp = 1 - surv_pow
    one_minus_sp <= 0 && return oftype(η[1], 1e20)

    return nll_base + lδ - log(one_minus_sp) - log1p(δ)
end

function initial_eta(::EGPD3Family, y::AbstractVector)
    n = length(y)
    σ0 = max(mean(y), 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.1
    lδ0 = 0.0  # δ = 1
    return [fill(ψ0, n), fill(ξ0, n), fill(lδ0, n)]
end


# ============================================================================
# EGPD Model 4 — Beta-kappa transformation
# ============================================================================

"""
    EGPD4Family()

Extended GPD Model 4 (beta-kappa transformation).

Combines the Beta-type transformation (model 3) with a power transformation.
G(u) = [1 - I_beta((1-u)^δ; 1/δ, 2)]^(κ/2)

NLL per observation:
    (1-κ/2)·log(1 - (1-1/((1+δ)·s^(δ/ξ)))·(1+δ)/(s^(1/ξ)·δ))
    + (1+1/ξ)·log(1+t) + log(2) + log(δ) + ψ - log(κ) - log(1-1/s^(δ/ξ)) - log(1+δ)

where s = (1+ξy/σ) and t = ξy/σ.

Four parameters:
- η₁ = ψ = log(σ)
- η₂ = ξ (shape)
- η₃ = log(δ)
- η₄ = log(κ)
"""
struct EGPD4Family <: MultiParameterFamily end

nparams(::EGPD4Family) = 4
param_names(::EGPD4Family) = ["logscale", "shape", "logdelta", "logkappa"]
param_links(::EGPD4Family) = [:log, :identity, :log, :log]

function nll_obs(::EGPD4Family, yi::Real, η::AbstractVector)
    ψ = η[1]
    ξ = η[2]
    lδ = η[3]
    lκ = η[4]
    σ = exp(ψ)
    δ = exp(lδ)
    κ = exp(lκ)

    t = ξ * yi / σ
    t <= -1 && return oftype(η[1], 1e20)

    if abs(ξ) < _EGPD_XI_EPS
        z = yi / σ
        nll_base = ψ + z
        s_delta_xi = exp(-δ * z)  # (1+t)^(-δ/ξ) ≈ exp(-δ·z) for small ξ
        s_inv_xi = exp(-z)        # (1+t)^(-1/ξ) ≈ exp(-z)
    else
        inv_xi = 1 / ξ
        log1pt = log1p(t)
        nll_base = ψ + (inv_xi + 1) * log1pt
        s = 1 + t
        s_delta_xi = s^(-δ * inv_xi)
        s_inv_xi = s^(-inv_xi)
    end

    δ_plus_1 = 1 + δ
    one_minus_sdx = 1 - s_delta_xi
    one_minus_sdx <= 0 && return oftype(η[1], 1e20)

    # The beta-type CDF contribution: I_beta((1-F)^δ; 1/δ, 2)
    # Simplified: inner = 1 - (1 - 1/(δ_plus_1 · s^(δ/ξ))) · δ_plus_1 / (s^(1/ξ) · δ)
    inner_beta = (1 - 1 / (δ_plus_1 * (1 / s_delta_xi))) * δ_plus_1 / ((1 / s_inv_xi) * δ)
    G3_val = 1 - inner_beta
    G3_val <= 0 && return oftype(η[1], 1e20)

    # NLL = (1 - κ/2)·log(G3) + nll_base + log(2) + lδ - lκ - log(1 - s^(-δ/ξ)) - log(1+δ)
    return (1 - κ / 2) * log(G3_val) + nll_base + log(2) + lδ - lκ - log(one_minus_sdx) - log(δ_plus_1)
end

function initial_eta(::EGPD4Family, y::AbstractVector)
    n = length(y)
    σ0 = max(mean(y), 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.1
    lδ0 = 0.0  # δ = 1
    lκ0 = 0.0  # κ = 1 (reduces to model 3 when κ=2... start with κ=1)
    return [fill(ψ0, n), fill(ξ0, n), fill(lδ0, n), fill(lκ0, n)]
end


# ============================================================================
# Hand-coded exact derivatives (translated from egpd/src/egpd.cpp)
# These override the AD default for performance.
# ============================================================================

"""
    egpd1_nll_derivs_exact!(out, y, ψvec, ξvec, lκvec)

Hand-coded EGPD1 per-obs derivatives. Translation of egpd1d12 from egpd.cpp.
out is n×9: [d_ψ, d_ξ, d_lκ, d_ψψ, d_ψξ, d_ψlκ, d_ξξ, d_ξlκ, d_lκlκ]
"""
function egpd1_nll_derivs_exact!(out::Matrix{Float64}, y::AbstractVector,
                                  ψvec::AbstractVector, ξvec::AbstractVector,
                                  lκvec::AbstractVector)
    n = length(y)
    @inbounds for j in 1:n
        yj = y[j]
        lpsi = ψvec[j]
        xi = ξvec[j]
        lkappa = lκvec[j]

        # Out-of-support: NLL is the constant 1e20, derivatives are zero
        if yj <= 0 || 1 + xi * yj / exp(lpsi) <= 0
            _egpd_zero_row!(out, j)
            continue
        end
        # Small-|ξ| exponential-limit branch: AD through nll_obs (which uses
        # the same _EGPD_XI_EPS threshold), avoiding 1/ξ blow-up
        if abs(xi) < _EGPD_XI_EPS
            _egpd_ad_derivs_row!(out, j, EGPD1Family(), yj, [lpsi, xi, lkappa])
            continue
        end

        ee1 = exp(lpsi)
        ee2 = xi * yj
        ee3 = ee2 / ee1
        ee4 = 1 + ee3
        ee5 = 1 / xi
        ee6 = 1 + ee5
        ee7 = ee4^ee5
        ee8 = 1 - 1 / ee7
        ee9 = ee4^ee6
        ee10 = log1p(ee3)
        ee11 = exp(lkappa)
        ee12 = ee4 * ee1
        ee14 = 1 - ee11
        ee16 = ee10 / (xi * ee7) - yj / (ee9 * ee1)
        ee17 = ee10 / xi
        ee18 = yj * ee6
        ee19 = ee4^(ee5 + 2)
        ee20 = xi * ee6
        ee21 = ee8 * ee9
        ee23 = 1 / ee9
        ee25 = ee11 * log(ee8)
        ee26 = ee10 / (xi * ee9)
        ee27 = ee2 / ee12
        ee28 = ee18 / ee12
        ee29 = ee18 / (ee19 * ee1)
        ee31 = yj / ee12 - 2 * ee17

        out[j, 1] = 1 - yj * (ee14 / (ee8 * ee7) + ee20) / ee12
        out[j, 2] = ee28 - (ee14 * ee16 / ee8 + ee17) / xi
        out[j, 3] = -(1 + ee25)
        # Hessian — column-major upper triangle: H[ψψ], H[ψξ], H[ξξ], H[ψlκ], H[ξlκ], H[lκlκ]
        out[j, 4] = yj * (ee14 * (ee23 - yj * (1 / (ee8 * ee4^(2 * ee6)) +
            ee20 / ee19) / ee1) / ee8 - ee20 * (ee27 - 1) / ee4) / ee1
        out[j, 5] = -(yj * (((ee16 / ee21 + ee26) / xi - ee29) * ee14 / ee8 +
            ((1 - ee27) * ee6 - ee5) / ee4) / ee1)
        # Swap cols 6,7 from C++ row-major to Julia column-major
        out[j, 6] = -(((((ee16 / ee8 + ee17) * ee16 + ee31 / ee7) / xi +
            yj * ((ee23 - ee26) / xi + ee29) / ee1) * ee14 / ee8 + ee31 / xi) / xi +
            yj * (1 / (xi^2) + ee28) / ee12)
        out[j, 7] = yj * ee11 / (ee21 * ee1)
        out[j, 8] = ee11 * ee16 / (xi * ee8)
        out[j, 9] = -ee25
    end
    return out
end

"""
    egpd3_nll_derivs_exact!(out, y, ψvec, ξvec, lδvec)

Hand-coded EGPD3 per-obs derivatives. Translation of egpd3d12 from egpd.cpp.
out is n×9: [d_ψ, d_ξ, d_lδ, d_ψψ, d_ψξ, d_ψlδ, d_ξξ, d_ξlδ, d_lδlδ]
"""
function egpd3_nll_derivs_exact!(out::Matrix{Float64}, y::AbstractVector,
                                  ψvec::AbstractVector, ξvec::AbstractVector,
                                  lδvec::AbstractVector)
    n = length(y)
    @inbounds for j in 1:n
        yj = y[j]
        lpsi = ψvec[j]
        xi = ξvec[j]
        ldelta = lδvec[j]

        # Out-of-support: NLL is the constant 1e20, derivatives are zero
        if yj <= 0 || 1 + xi * yj / exp(lpsi) <= 0
            _egpd_zero_row!(out, j)
            continue
        end
        # Small-|ξ| exponential-limit branch via AD through nll_obs
        if abs(xi) < _EGPD_XI_EPS
            _egpd_ad_derivs_row!(out, j, EGPD3Family(), yj, [lpsi, xi, ldelta])
            continue
        end

        ee1 = exp(lpsi)
        ee2 = xi * yj
        ee3 = ee2 / ee1
        ee4 = 1 + ee3
        ee5 = exp(ldelta)
        ee6 = 1 / xi
        ee7 = ee5 / xi
        ee8 = ee4^ee7
        ee9 = 1 + ee6
        ee10 = 1 / ee8
        ee11 = log1p(ee3)
        ee12 = 1 - ee10
        ee13 = ee5 - 1
        ee14 = ee4^ee9
        ee15 = ee13 / xi
        ee16 = ee4 * ee1
        ee17 = ee4^ee6
        ee18 = ee4^ee15
        ee21 = ee11 / (xi * ee17) - yj / (ee14 * ee1)
        ee22 = xi * ee12
        ee23 = yj * ee9
        ee25 = ee4^(ee6 + 2) * ee1
        ee26 = 1 + ee5
        ee27 = 1 + ee7
        ee28 = xi^2
        ee30 = (2 * ee5 - 1) / xi
        ee31 = 1 / ee18
        ee32 = 1 / ee14
        ee33 = 1 / ee4^ee27
        ee34 = 2 * ee7
        ee36 = xi * ee9
        ee37 = ee2 / ee16
        ee38 = ee23 / ee25
        ee40 = yj / ee16 - 2 * (ee11 / xi)

        out[j, 1] = 1 + yj * (ee5 / (ee12 * ee8) - ee36) / ee16
        out[j, 2] = (ee5 * ee21 / (ee22 * ee4^(ee15 - 1)) + ee23 / ee1) / ee4 -
            ee11 / ee28
        out[j, 3] = 1 - (1 / ee26 + ee11 / (ee22 * ee8)) * ee5
        # Hessian — column-major upper triangle: H[ψψ], H[ψξ], H[ξξ], H[ψlδ], H[ξlδ], H[lδlδ]
        out[j, 4] = -(yj * (((ee32 - ee2 * ee9 / ee25) / ee18 - yj * (ee13 / ee4^(2 +
            ee7) + ee5 / (ee12 * ee4^(2 * ee27))) / ee1) * ee5 / ee12 +
            ee36 * (ee37 - 1) / ee4) / ee1)
        out[j, 5] = -(yj * (((1 - ee37) * ee9 - ee6) / ee4 - ((ee13 / ee4^(ee15 +
            1) + ee5 / (ee12 * ee4^(ee30 + 1))) * ee21 / xi +
            (ee11 / (ee28 * ee14) - ee38) / ee18) * ee5 / ee12) / ee1)
        # Swap cols 6,7 from C++ row-major to Julia column-major
        out[j, 6] = ((((ee40 / ee17 + ee11 * ee21 / xi) / xi + yj * ((ee32 -
            ee11 / (xi * ee14)) / xi + ee38) / ee1) / ee18 + (ee13 / ee4^((ee5 -
            2) / xi) + ee5 / (ee12 * ee4^(2 * ee15))) * ee21^2 / xi) * ee5 / ee12 -
            ee40 / xi) / xi - yj * (1 / ee28 + ee23 / ee16) / ee16
        out[j, 7] = -(yj * ((1 / (ee12 * ee4^(1 + ee34)) + ee33) * ee5 * ee11 / xi -
            ee33) * ee5 / (ee12 * ee1))
        out[j, 8] = (ee31 - (1 / (ee12 * ee4^ee30) + ee31) * ee5 * ee11 / xi) * ee5 * ee21 / ee22
        out[j, 9] = (((1 / (ee12 * ee4^ee34) + ee10) * ee5 * ee11 / xi -
            ee10) * ee11 / ee22 - (1 - ee5 / ee26) / ee26) * ee5
    end
    return out
end

"""
    egpd4_nll_derivs_exact!(out, y, ψvec, ξvec, lδvec, lκvec)

Hand-coded EGPD4 per-obs derivatives. Translation of egpd4d12 from egpd.cpp.
out is n×14: [d_ψ, d_ξ, d_lδ, d_lκ, d_ψψ, d_ψξ, d_ψlδ, d_ψlκ, d_ξξ, d_ξlδ, d_ξlκ, d_lδlδ, d_lδlκ, d_lκlκ]
"""
function egpd4_nll_derivs_exact!(out::Matrix{Float64}, y::AbstractVector,
                                  ψvec::AbstractVector, ξvec::AbstractVector,
                                  lδvec::AbstractVector, lκvec::AbstractVector)
    n = length(y)
    @inbounds for j in 1:n
        yj = y[j]
        lpsi = ψvec[j]
        xi = ξvec[j]
        ldelta = lδvec[j]
        lkappa = lκvec[j]

        # Out-of-support: NLL is the constant 1e20, derivatives are zero
        if yj <= 0 || 1 + xi * yj / exp(lpsi) <= 0
            _egpd_zero_row!(out, j)
            continue
        end
        # Small-|ξ| exponential-limit branch via AD through nll_obs
        if abs(xi) < _EGPD_XI_EPS
            _egpd_ad_derivs_row!(out, j, EGPD4Family(), yj,
                                 [lpsi, xi, ldelta, lkappa])
            continue
        end

        ee1 = exp(ldelta)
        ee2 = exp(lpsi)
        ee3 = xi * yj
        ee4 = ee3 / ee2
        ee5 = 1 + ee4
        ee6 = ee1 / xi
        ee7 = 1 + ee1
        ee8 = ee5^ee6
        ee9 = 1 / xi
        ee10 = log1p(ee4)
        ee11 = ee7 * ee8
        ee12 = ee5^ee9
        ee13 = 1 / ee11
        ee14 = 1 - ee13
        ee15 = 1 + ee6
        ee16 = ee5^ee15
        ee17 = ee14 * ee7
        ee18 = 1 / ee1
        ee19 = ee18 - 1
        ee20 = ee19 * ee1
        ee21 = xi * ee8
        ee22 = 1 + ee9
        ee23 = 1 / ee8
        ee24 = ee10 / ee21
        ee25 = ee20 / xi
        ee27 = ee5^ee25
        ee28 = 1 - ee17 / (ee12 * ee1)
        ee29 = ee5^ee22
        ee30 = 1 - ee23
        ee31 = exp(lkappa)
        ee33 = 1 / ee16
        ee34 = ee1 * ee10
        ee35 = ee24 - yj / (ee16 * ee2)
        ee36 = ee1 - 1
        ee38 = 1 - ee7 / ee1
        ee39 = ee5 * ee2
        ee41 = 1 - 0.5 * ee31
        ee42 = ee13 + ee24
        ee43 = ee36 / xi
        ee44 = ee7 / xi
        ee45 = ee5^(2 + ee6)
        ee46 = xi * ee16
        ee48 = ee5^(ee44 + 1)
        ee49 = ee5^ee43
        ee50 = ee45 * ee2
        ee53 = ee38 * ee14 / ee12 + ee42 / ee12
        ee55 = ee17 / (ee27 * ee1) - 1 / ee12
        ee57 = ee17 / (ee29 * ee1) - 1 / ee48
        ee59 = ee34 / ee46
        ee61 = ee10 / (xi * ee12) - yj / (ee29 * ee2)
        ee62 = ee10 / xi
        ee64 = ee5^(ee9 + 2) * ee2
        ee66 = xi^2
        ee67 = yj * ee22
        ee69 = yj * ee15 / ee50
        ee71 = yj / ee39 - 2 * ee62
        ee72 = (ee20 / ee27 - 1 / ee27) * ee10
        ee73 = ee33 - ee59
        ee74 = 2 * ee1
        ee75 = ee34 / ee21
        ee77 = (ee71 / ee8 + ee34 * ee35 / xi) / xi + yj * (ee73 / xi + ee69) / ee2
        ee78 = ee38 * ee1
        ee79 = ee7 * ee12
        ee80 = (ee74 - 1) / xi
        ee81 = ee35^2
        ee82 = 0.5 * (ee31 * log(ee28))
        ee83 = 1 / ee49
        ee84 = 1 / ee29
        ee85 = ee33 - ee3 * ee15 / ee50
        ee86 = ee33 + ee59
        ee87 = ee23 - (ee23 + ee75)
        ee88 = 2 * ee6
        ee90 = ee34 / (ee66 * ee16) - ee69
        ee91 = xi * ee30
        ee92 = xi * ee22
        ee93 = ee3 / ee39
        ee94 = ee67 / ee39
        ee95 = ee67 / ee64

        out[j, 1] = 1 - yj * (ee57 * ee41 / ee28 + (ee92 - ee1 / (ee30 * ee8)) / ee5) / ee2
        out[j, 2] = ee94 - (ee55 * ee41 * ee35 / ee28 + ee62 - ee1 * ee61 / (ee30 * ee49)) / xi
        out[j, 3] = 1 - (ee53 * ee41 / ee28 + (1 / ee7 + ee10 / (ee91 * ee8)) * ee1)
        out[j, 4] = -(ee82 + 1)

        # Hessian — compute in C++ row-major order, then assign to Julia column-major cols
        # C++ order: H11(5), H12(6), H13(7), H14(8), H22(9), H23(10), H24(11), H33(12), H34(13), H44(14)
        # Julia order: H11(5), H12(6), H22(7), H13(8), H23(9), H33(10), H14(11), H24(12), H34(13), H44(14)
        h11 = -(yj * (((ee84 - ee3 * ee22 / ee64) / ee49 - yj * (ee36 / ee45 +
            ee1 / (ee30 * ee5^(2 * ee15))) / ee2) * ee1 / ee30 +
            (ee85 / ee12 + yj * (ee57^2 / ee28 - 2 / ee5^(ee44 +
            2)) / ee2 - (ee85 / ee27 - yj * ee19 * ee1 / ee64) * ee14 * ee7 / ee1) * ee41 / ee28 +
            ee92 * (ee93 - 1) / ee5) / ee2)
        h12 = -(yj * (((ee55 * ee57 / ee28 - 2 / ee29) * ee35 / xi +
            (ee20 * ee35 / (xi * ee5^(ee25 + 1)) + ee90 / ee27) * ee14 * ee7 / ee1 -
            ee90 / ee12) * ee41 / ee28 + ((1 - ee93) * ee22 -
            ee9) / ee5 - ((ee36 / ee5^(ee43 + 1) + ee1 / (ee30 * ee5^(ee80 +
            1))) * ee61 / xi + (ee10 / (ee66 * ee29) - ee95) / ee49) * ee1 / ee30) / ee2)
        h13 = -(yj * (((((ee33 - ee86) / ee27 - ee72 / ee46) * ee7 / ee1 +
            ee38 / ee29) * ee14 + ee53 * ee57 / ee28 + ee42 / ee29 - (ee78 / (ee7 * ee48) +
            (ee33 - (1 / (ee7 * ee16) + ee10 / ee46) * ee1) / ee12)) * ee41 / ee28 +
            ((1 / (ee30 * ee5^(1 + ee88)) +
            ee33) * ee1 * ee10 / xi - ee33) * ee1 / ee30) / ee2)
        h14 = 0.5 * (yj * ee57 * ee31 / (ee28 * ee2))
        h22 = -((((ee77 / ee27 + ee20 * ee81 / (xi * ee5^((ee18 -
            2) * ee1 / xi))) * ee14 * ee7 / ee1 + (ee55^2 / ee28 -
            2 / ee27) * ee81 / xi - ee77 / ee12) * ee41 / ee28 + ee71 / xi -
            (((ee71 / ee12 + ee10 * ee61 / xi) / xi + yj * ((ee84 - ee10 / (xi * ee29)) / xi +
            ee95) / ee2) / ee49 + (ee36 / ee5^((ee1 - 2) / xi) +
            ee1 / (ee30 * ee5^(2 * ee43))) * ee61^2 / xi) * ee1 / ee30) / xi +
            yj * (1 / ee66 + ee94) / ee39)
        h23 = -((((((ee87 * ee10 / xi + yj * (ee86 - ee33) / ee2) / ee27 -
            ee72 * ee35 / xi) * ee7 / ee1 + ee38 * ee35 / ee27) * ee14 +
            (ee53 * ee55 / ee28 + ee42 / ee27 - ee78 / ee79) * ee35 - ((ee23 -
            ee75) * ee10 / xi - (ee1 * ee35 / ee7 + yj * ee73 / ee2)) / ee12) * ee41 / ee28 -
            (ee83 - (1 / (ee30 * ee5^ee80) + ee83) * ee1 * ee10 / xi) * ee1 * ee61 / ee30) / xi)
        h24 = 0.5 * (ee55 * ee31 * ee35 / (xi * ee28))
        h33 = (((1 / (ee30 * ee5^ee88) + ee23) * ee1 * ee10 / xi -
            ee23) * ee10 / ee91 - (1 - ee1 / ee7) / ee7) * ee1 - (ee53^2 / ee28 +
            ((ee23 - ee42 * ee1) * ee10 / xi + (ee23 -
            (2 / ee11 + ee24) * ee1) / ee7) / ee12 + 2 * (ee38 * ee42 * ee1 / ee79) -
            ((ee87 / ee27 - ee72 / ee21) * ee7 * ee10 / xi + (1 + ee74 -
            2 * ee7) / ee12) * ee14 / ee1) * ee41 / ee28
        h34 = 0.5 * (ee53 * ee31 / ee28)
        h44 = -ee82

        # Assign in Julia column-major order: H11, H12, H22, H13, H23, H33, H14, H24, H34, H44
        out[j, 5]  = h11
        out[j, 6]  = h12
        out[j, 7]  = h22
        out[j, 8]  = h13
        out[j, 9]  = h23
        out[j, 10] = h33
        out[j, 11] = h14
        out[j, 12] = h24
        out[j, 13] = h34
        out[j, 14] = h44
    end
    return out
end

# ============================================================================
# nll_derivs! overrides using hand-coded exact derivatives
# ============================================================================

function nll_derivs!(::EGPD1Family, out::Matrix{Float64}, y::AbstractVector,
                     η_list::Vector{<:AbstractVector})
    egpd1_nll_derivs_exact!(out, y, η_list[1], η_list[2], η_list[3])
    return out
end

function nll_derivs!(::EGPD3Family, out::Matrix{Float64}, y::AbstractVector,
                     η_list::Vector{<:AbstractVector})
    egpd3_nll_derivs_exact!(out, y, η_list[1], η_list[2], η_list[3])
    return out
end

function nll_derivs!(::EGPD4Family, out::Matrix{Float64}, y::AbstractVector,
                     η_list::Vector{<:AbstractVector})
    egpd4_nll_derivs_exact!(out, y, η_list[1], η_list[2], η_list[3], η_list[4])
    return out
end
