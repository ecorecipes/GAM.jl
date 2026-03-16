# Extreme value distribution families for multi-parameter GAMs
#
# GEV(μ, ψ, ξ) — Generalized Extreme Value with location μ, log-scale ψ, shape ξ
# GPD(ψ, ξ) — Generalized Pareto with log-scale ψ, shape ξ
#
# Each family provides nll_obs (AD-differentiable) and initial_eta.
# nll_derivs! is computed automatically via ForwardDiff.
# Hand-coded exact derivatives are kept as _exact! functions for reference/testing.

# ============================================================================
# GEV family
# ============================================================================

"""
    GEVFamily()

Generalized Extreme Value distribution family for multi-parameter GAMs.

Three parameters:
- η₁ = μ (location, identity link)
- η₂ = ψ = log(σ) (log-scale, so σ = exp(ψ) > 0)
- η₃ = ξ (shape, identity link)

The GEV density for ξ ≠ 0:
    -log f(y) = ψ + (1/ξ + 1) log(1 + ξ(y-μ)/σ) + (1 + ξ(y-μ)/σ)^(-1/ξ)

For ξ = 0 (Gumbel limit):
    -log f(y) = ψ + (y-μ)/σ + exp(-(y-μ)/σ)
"""
struct GEVFamily <: MultiParameterFamily end

nparams(::GEVFamily) = 3
param_names(::GEVFamily) = ["location", "logscale", "shape"]
param_links(::GEVFamily) = [:identity, :log, :identity]

function nll_obs(::GEVFamily, yi::Real, η::AbstractVector)
    μ = η[1]
    σ = exp(η[2])
    ξ = η[3]
    z = (yi - μ) / σ
    if abs(ξ) < 1e-8
        # Gumbel limit
        return η[2] + z + exp(-z)
    end
    t = 1 + ξ * z
    t <= 0 && return oftype(η[1], 1e20)
    return η[2] + (1/ξ + 1) * log(t) + t^(-1/ξ)
end

function initial_eta(::GEVFamily, y::AbstractVector)
    n = length(y)
    μ0 = mean(y)
    σ0 = max(std(y) * sqrt(6) / π, 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.1
    return [fill(μ0, n), fill(ψ0, n), fill(ξ0, n)]
end

# ============================================================================
# GPD family
# ============================================================================

"""
    GPDFamily(; threshold=0.0)

Generalized Pareto distribution family for multi-parameter GAMs.

Two parameters:
- η₁ = ψ = log(σ) (log-scale)
- η₂ = ξ (shape, identity link)

Exceedances y > 0 over threshold. The GPD density for ξ ≠ 0:
    -log f(y) = ψ + (1/ξ + 1) log(1 + ξy/σ)

For ξ = 0 (Exponential limit):
    -log f(y) = ψ + y/σ
"""
struct GPDFamily <: MultiParameterFamily
    threshold::Float64
end

GPDFamily(; threshold::Real=0.0) = GPDFamily(Float64(threshold))

nparams(::GPDFamily) = 2
param_names(::GPDFamily) = ["logscale", "shape"]
param_links(::GPDFamily) = [:log, :identity]

function nll_obs(f::GPDFamily, yi::Real, η::AbstractVector)
    yj = yi - f.threshold
    σ = exp(η[1])
    ξ = η[2]
    if abs(ξ) < 1e-8
        # Exponential limit
        return η[1] + yj / σ
    end
    t = 1 + ξ * yj / σ
    t <= 0 && return oftype(η[1], 1e20)
    return η[1] + (1/ξ + 1) * log(t)
end

function initial_eta(f::GPDFamily, y::AbstractVector)
    n = length(y)
    exc = y .- f.threshold
    σ0 = max(mean(exc), 1e-4)
    ψ0 = log(σ0)
    ξ0 = 0.1
    return [fill(ψ0, n), fill(ξ0, n)]
end

# ============================================================================
# Hand-coded exact derivatives (kept for reference/testing, no longer used
# in the main fitting path — AD via ForwardDiff is used instead)
# ============================================================================

"""
    gev_nll_derivs_exact!(out, y, μvec, ψvec, ξvec)

Compute exact GEV per-observation NLL derivatives.
Direct translation of evgam's gevd12 C++ function.
out is n×9: columns [d_μ, d_ψ, d_ξ, d_μμ, d_μψ, d_ψψ, d_μξ, d_ψξ, d_ξξ]
"""
function gev_nll_derivs_exact!(out::Matrix{Float64}, y::AbstractVector,
                               μvec::AbstractVector, ψvec::AbstractVector,
                               ξvec::AbstractVector)
    n = length(y)
    @inbounds for j in 1:n
        yj = y[j]
        μ = μvec[j]
        lpsi = ψvec[j]
        xi = ξvec[j]

        if abs(xi) > 1e-7
            ee1 = exp(lpsi)
            ee2 = yj - μ
            ee4 = xi * ee2 / ee1
            ee5 = 1.0 + ee4
            ee6 = 1.0 / xi
            ee7 = 1.0 + ee6
            ee8 = ee5^ee6          # s^(1/ξ)
            ee9 = ee5 * ee1
            ee10 = 1.0 / ee8       # s^(-1/ξ)
            ee11 = log1p(ee4)
            ee12 = ee5^ee7          # s^(1/ξ+1)
            ee13 = ee7 * ee2
            ee16 = (ee10 - xi) * ee2 / ee9 + 1.0
            ee17 = ee11 / (xi * ee8)
            ee18 = xi * ee7
            ee19 = (ee16 * ee7 - (1.0 + ee17) / xi) / ee5
            ee20 = ee13 / ee9
            ee22 = ee2 / (ee12 * ee1)
            ee23 = xi - ee10

            # Gradient
            out[j, 1] = -((ee18 - ee10) / ee9)
            out[j, 2] = (ee10 - ee18) * ee2 / ee9 + 1.0
            out[j, 3] = ((ee10 - 1.0) * ee11 / xi - ee22) / xi + ee20

            # Hessian — column-major upper triangle: H[μμ], H[μψ], H[ψψ], H[μξ], H[ψξ], H[ξξ]
            out[j, 4] = -(ee18 * ee23 / (ee5 * ee5 * ee1 * ee1))           # H[μ,μ]
            out[j, 5] = (xi * ee16 * ee7 - ee10) / ee5 / ee1                # H[μ,ψ]
            # Swap cols 6,7 from C++ row-major to Julia column-major
            out[j, 7] = -(ee19 / ee1)                                        # H[μ,ξ]
            out[j, 6] = -((ee10 + xi * (ee23 * ee2 / ee9 - 1.0) * ee7) / ee5 * ee2 / ee1) # H[ψ,ψ]
            out[j, 8] = -(ee19 * ee2 / ee1)                                  # H[ψ,ξ]
            out[j, 9] = ((((ee2 / ee9 - 2.0 * (ee11 / xi)) / ee5^(ee6 - 1.0) -
                ee2 / ee1) / ee5 + (2.0 + ee17 - ee22) * ee11 / xi) / xi +
                (ee13 / (ee5^(ee6 + 2.0) * ee1) + (1.0 / ee12 - ee11 / (xi *
                ee12)) / xi) * ee2 / ee1) / xi - (ee20 + 1.0 / (xi * xi)) * ee2 / ee9
        else
            # Gumbel limit
            ee1 = exp(lpsi)
            ee2 = yj - μ
            ee3 = ee2 / ee1
            ee5 = exp(-ee3)
            ee7 = (ee3 - 1.0) * ee5 + 1.0
            ee8 = ee5 - 1.0

            out[j, 1] = ee8 / ee1
            out[j, 2] = ee8 * ee2 / ee1 + 1.0
            out[j, 3] = 0.0
            out[j, 4] = ee5 / (ee1 * ee1)     # H[μ,μ]
            out[j, 5] = ee7 / ee1               # H[μ,ψ]
            out[j, 6] = ee7 * ee2 / ee1          # H[ψ,ψ]
            out[j, 7] = 0.0                       # H[μ,ξ]
            out[j, 8] = 0.0                       # H[ψ,ξ]
            out[j, 9] = 0.0                       # H[ξ,ξ]
        end
    end
    return out
end

"""
    gpd_nll_derivs_exact!(out, y, ψvec, ξvec)

Compute exact GPD per-observation NLL derivatives.
Direct translation of evgam's gpdd12 C++ function.
out is n×5: columns [d_ψ, d_ξ, d_ψψ, d_ψξ, d_ξξ]
"""
function gpd_nll_derivs_exact!(out::Matrix{Float64}, y::AbstractVector,
                               ψvec::AbstractVector, ξvec::AbstractVector)
    n = length(y)
    @inbounds for j in 1:n
        yj = y[j]
        lpsi = ψvec[j]
        xi = ξvec[j]

        if abs(xi) > 1e-7
            ee1 = exp(lpsi)
            ee2 = xi * yj
            ee3 = ee2 / ee1
            ee4 = (1.0 + ee3) * ee1
            ee5 = 1.0 / xi
            ee6 = 1.0 + ee5
            ee7 = xi * xi
            ee8 = log1p(ee3)
            ee9 = ee2 * ee6
            ee10 = ee2 / ee4
            ee12 = yj * ee6 / ee4

            # Gradient
            out[j, 1] = 1.0 - ee9 / ee4
            out[j, 2] = ee12 - ee8 / ee7

            # Hessian
            out[j, 3] = -(ee9 * (ee10 - 1.0) / ee4)
            out[j, 4] = -(yj * ((1.0 - ee10) * ee6 - ee5) / ee4)
            out[j, 5] = -((yj / ee4 - 2.0 * (ee8 / xi)) / ee7 + yj * (1.0 / ee7 + ee12) / ee4)
        else
            # Exponential limit
            ee1 = exp(lpsi)
            z = yj / ee1

            out[j, 1] = 1.0 - z
            out[j, 2] = 0.0
            out[j, 3] = z * (z - 1.0)
            out[j, 4] = 0.0
            out[j, 5] = 0.0
        end
    end
    return out
end

# ============================================================================
# Performance overrides: use exact C++-equivalent derivatives for GEV/GPD
# These override the AD fallback from multiparameter.jl for speed.
# New families can omit these and rely on the AD default.
# ============================================================================

function nll_total(::GEVFamily, y::AbstractVector, η_list::Vector{<:AbstractVector})
    n = length(y)
    nllh = 0.0
    @inbounds for j in 1:n
        μ = η_list[1][j]
        ψ = η_list[2][j]
        ξ = η_list[3][j]
        if abs(ξ) > 1e-7
            t = ξ * (y[j] - μ) / exp(ψ)
            t <= -1.0 && return 1e20
            nllh += ψ + (1/ξ + 1) * log1p(t) + (1 + t)^(-1/ξ)
        else
            z = (y[j] - μ) / exp(ψ)
            nllh += ψ + z + exp(-z)
        end
    end
    return nllh
end

function nll_total(f::GPDFamily, y::AbstractVector, η_list::Vector{<:AbstractVector})
    n = length(y)
    nllh = 0.0
    @inbounds for j in 1:n
        yj = y[j] - f.threshold
        ψ = η_list[1][j]
        ξ = η_list[2][j]
        if abs(ξ) > 1e-7
            t = ξ * yj / exp(ψ)
            t <= -1.0 && return 1e20
            nllh += ψ + (1/ξ + 1) * log1p(t)
        else
            nllh += ψ + yj / exp(ψ)
        end
    end
    return nllh
end

function nll_derivs!(::GEVFamily, out::Matrix{Float64}, y::AbstractVector,
                     η_list::Vector{<:AbstractVector})
    # Use hand-coded exact derivatives (translated from evgam's gevC.cpp).
    # These match ForwardDiff AD to machine precision (~5e-12 max error)
    # and are ~42x faster per call.
    gev_nll_derivs_exact!(out, y, η_list[1], η_list[2], η_list[3])
    return out
end

function nll_derivs!(f::GPDFamily, out::Matrix{Float64}, y::AbstractVector,
                     η_list::Vector{<:AbstractVector})
    y_exc = y .- f.threshold
    gpd_nll_derivs_exact!(out, y_exc, η_list[1], η_list[2])
    return out
end
