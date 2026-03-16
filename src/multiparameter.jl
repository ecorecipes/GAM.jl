# Multi-parameter distribution models for GAM.jl
#
# Framework for distributions where each parameter (location, scale, shape, etc.)
# has its own GAM formula and design matrix. Covers evgam (GEV, GPD),
# egpd (extended GPD), and gamlss-style models.
#
# Architecture:
#   - Each distribution defines K parameters with link functions
#   - Per-observation NLL via nll_obs(family, y_i, η_vec)
#   - Derivatives computed automatically via ForwardDiff (AD)
#   - Families can optionally override nll_derivs! for hand-tuned speed
#   - Block gradient/Hessian assembly from per-obs derivatives and design matrices
#   - Inner Newton for penalized NLL of coefficients β given smoothing params λ
#   - Outer BFGS/EFS for REML optimization of log(λ)

using LinearAlgebra
using DifferentiationInterface
import ForwardDiff

const _ad_backend = AutoForwardDiff()

# ============================================================================
# Abstract types
# ============================================================================

"""
    MultiParameterFamily

Abstract supertype for multi-parameter distributions where each distribution
parameter has its own linear predictor. To implement a new family, define:

**Required:**
    nparams(::MyFamily) → Int
    param_names(::MyFamily) → Vector{String}
    param_links(::MyFamily) → Vector{Symbol}  # :identity, :log, :logit
    nll_obs(::MyFamily, y_i, η_vec) → Float64  (per-observation NLL, AD-differentiable)
    initial_eta(::MyFamily, y) → Vector{Vector{Float64}}

**Optional (for performance — AD fallback is used otherwise):**
    nll_derivs!(::MyFamily, out, y, η_list) → out  (hand-coded derivatives)
"""
abstract type MultiParameterFamily end

"""Number of distribution parameters for this family."""
function nparams end

"""Human-readable names for each parameter."""
function param_names end

"""Link functions for each parameter (e.g., [:identity, :log, :identity])."""
function param_links end

"""
    nll_obs(family, y_i, η_vec) → Float64

Per-observation negative log-likelihood. `η_vec` is a K-vector of linear
predictor values for this observation. Must be AD-differentiable (no
in-place mutation, no type restrictions on η_vec elements).

This is the **only** function new families must define for derivatives —
the framework computes gradients and Hessians automatically via ForwardDiff.
"""
function nll_obs end

"""
    nll_total(family, y, η_list) → Float64

Total negative log-likelihood over all observations.
η_list[k] is the n-vector of linear predictor values for parameter k.
Default: sums nll_obs over observations.
"""
function nll_total(family::MultiParameterFamily, y::AbstractVector,
                   η_list::Vector{<:AbstractVector})
    K = nparams(family)
    n = length(y)
    total = 0.0
    η_vec = Vector{Float64}(undef, K)
    @inbounds for i in 1:n
        for k in 1:K
            η_vec[k] = η_list[k][i]
        end
        total += nll_obs(family, y[i], η_vec)
    end
    return total
end

"""
    nll_derivs!(family, out, y, η_list) → out

Fill n × (K + K(K+1)/2) matrix `out` with per-observation derivatives.
Columns 1:K are gradients ∂ℓ/∂η_k.
Remaining columns are upper-triangle Hessian: ∂²ℓ/∂η_i∂η_j for i ≤ j,
stored in order (1,1), (1,2), (2,2), (1,3), (2,3), (3,3), ...

Default implementation uses ForwardDiff via DifferentiationInterface.
Families can override this for hand-tuned performance.
"""
function nll_derivs!(family::MultiParameterFamily, out::Matrix{Float64},
                     y::AbstractVector, η_list::Vector{<:AbstractVector})
    K = nparams(family)
    n = length(y)
    nc = deriv_ncols(K)

    @inbounds for i in 1:n
        η_vec = [η_list[k][i] for k in 1:K]
        f_i = η -> nll_obs(family, y[i], η)
        val, grad, hess = value_gradient_and_hessian(f_i, _ad_backend, η_vec)

        # Fill gradient columns
        for k in 1:K
            out[i, grad_col(k)] = grad[k]
        end
        # Fill Hessian columns (upper triangle)
        for j in 1:K
            for ii in 1:j
                out[i, hess_col(K, ii, j)] = hess[ii, j]
            end
        end
    end
    return out
end

"""
    initial_eta(family, y) → Vector{Vector{Float64}}

Compute reasonable starting values for each linear predictor.
"""
function initial_eta end

# ============================================================================
# Derivative column indexing
# ============================================================================

"""
Number of columns in the per-observation derivative matrix for K parameters.
K gradient columns + K(K+1)/2 Hessian columns.
"""
deriv_ncols(K::Int) = K + div(K * (K + 1), 2)

"""
Index into the Hessian columns of the derivative matrix.
For K parameters, gradient occupies columns 1:K.
Hessian column for (i,j) where i ≤ j is at position K + upper_tri_index(i,j,K).
"""
function hess_col(K::Int, i::Int, j::Int)
    i, j = minmax(i, j)
    # Upper triangle stored column-major: (1,1),(1,2),(2,2),(1,3),(2,3),(3,3),...
    return K + div((j - 1) * j, 2) + i
end

"""Gradient column index (simply the parameter index)."""
grad_col(k::Int) = k

# ============================================================================
# Block gradient/Hessian assembly
# ============================================================================

"""
    assemble_gradient(derivs, X_list, idpars) → g

Assemble the full gradient vector from per-observation derivatives and
block design matrices.

    g[block_k] = X_k' * derivs[:, k]

# Arguments
- `derivs`: n × ncols matrix of per-observation derivatives
- `X_list`: vector of design matrices [X_1, ..., X_K]

# Returns
- `g`: p-vector where p = sum of ncol(X_k)
"""
function assemble_gradient(derivs::Matrix{Float64}, X_list::Vector{Matrix{Float64}})
    K = length(X_list)
    p = sum(size(X, 2) for X in X_list)
    g = Vector{Float64}(undef, p)
    offset = 0
    for k in 1:K
        pk = size(X_list[k], 2)
        gk = @view g[(offset+1):(offset+pk)]
        dk = @view derivs[:, grad_col(k)]
        mul!(gk, X_list[k]', dk)
        offset += pk
    end
    return g
end

"""
    assemble_hessian!(H, derivs, X_list) → H

Assemble the full Hessian matrix from per-observation derivatives and
block design matrices.

    H[block_i, block_j] = X_i' * diag(derivs[:, hess_col(i,j)]) * X_j
"""
function assemble_hessian!(H::Matrix{Float64}, derivs::Matrix{Float64},
                           X_list::Vector{Matrix{Float64}})
    K = length(X_list)
    fill!(H, 0.0)
    offsets = cumsum([0; [size(X, 2) for X in X_list]])

    # Temporary for diag(d) * X_j
    n = size(derivs, 1)
    for j in 1:K
        pj = size(X_list[j], 2)
        sj = offsets[j]
        for i in 1:j
            pi_ = size(X_list[i], 2)
            si = offsets[i]
            col = hess_col(K, i, j)
            d = @view derivs[:, col]

            # H[si+1:si+pi_, sj+1:sj+pj] = X_i' * diag(d) * X_j
            # Compute as (X_i .* d)' * X_j
            Hij = @view H[(si+1):(si+pi_), (sj+1):(sj+pj)]
            _weighted_crossprod!(Hij, X_list[i], d, X_list[j])

            if i != j
                Hji = @view H[(sj+1):(sj+pj), (si+1):(si+pi_)]
                Hji .= Hij'
            end
        end
    end
    return H
end

"""Compute X_i' * diag(d) * X_j efficiently."""
function _weighted_crossprod!(out::AbstractMatrix, Xi::Matrix{Float64},
                              d::AbstractVector, Xj::Matrix{Float64})
    n = size(Xi, 1)
    pi_ = size(Xi, 2)
    pj = size(Xj, 2)
    # Form tmp = Xi .* d, then out = tmp' * Xj
    tmp = similar(Xi)
    @inbounds for col in 1:pi_
        for row in 1:n
            tmp[row, col] = Xi[row, col] * d[row]
        end
    end
    mul!(out, tmp', Xj)
    return out
end

# ============================================================================
# Multi-parameter model result
# ============================================================================

"""
    MultiParameterModel

Result of fitting a multi-parameter GAM (e.g., evgam for GEV/GPD).

Contains fitted coefficients, smoothing parameters, variance-covariance
matrices, and sub-model information for each distribution parameter.
"""
struct MultiParameterModel{F<:MultiParameterFamily}
    family::F
    coefficients::Vector{Float64}    # concatenated β = [β_1; β_2; ...; β_K]
    fitted_eta::Vector{Vector{Float64}}  # linear predictors per parameter
    X_list::Vector{Matrix{Float64}}  # design matrices per parameter
    smooths::Vector{Vector{ConstructedSmooth}}  # smooths per parameter
    sp::Vector{Float64}              # log smoothing parameters
    edf::Vector{Float64}             # effective degrees of freedom per coefficient
    Vp::Matrix{Float64}              # posterior covariance (Bayesian)
    Vc::Matrix{Float64}              # corrected covariance (frequentist)
    nll::Float64                     # negative log-likelihood at optimum
    reml::Float64                    # REML score at optimum
    y::Vector{Float64}               # response
    nobs::Int
    converged::Bool
    idpars::Vector{Int}              # maps each coefficient to its parameter index
    param_offsets::Vector{Int}       # cumulative column counts [0, p1, p1+p2, ...]
end

"""Number of distribution parameters."""
nparams(m::MultiParameterModel) = nparams(m.family)

"""Get fitted coefficients for parameter k."""
function param_coef(m::MultiParameterModel, k::Int)
    s = m.param_offsets[k] + 1
    e = m.param_offsets[k + 1]
    return m.coefficients[s:e]
end

"""Get fitted linear predictor for parameter k."""
param_eta(m::MultiParameterModel, k::Int) = m.fitted_eta[k]

"""Get the total number of basis coefficients."""
total_p(m::MultiParameterModel) = length(m.coefficients)

# ============================================================================
# Penalty construction for multi-parameter models
# ============================================================================

"""
    build_block_penalty(smooths_list, sp, param_offsets) → S

Build the block-diagonal penalty matrix S = Σ_j λ_j S_j across all
parameters and their smooth terms.
"""
function build_block_penalty(smooths_list::Vector{Vector{ConstructedSmooth}},
                             sp::Vector{Float64},
                             param_offsets::Vector{Int})
    p = param_offsets[end]
    S = zeros(p, p)
    sp_idx = 0
    for (k, smooths) in enumerate(smooths_list)
        for sm in smooths
            for Sj in sm.S
                sp_idx += 1
                λ = exp(sp[sp_idx])
                s = sm.first_para
                e = sm.last_para
                S[s:e, s:e] .+= λ .* Sj
            end
        end
    end
    return S
end

"""Count total smoothing parameters across all parameter sub-models."""
function count_sp(smooths_list::Vector{Vector{ConstructedSmooth}})
    return sum(sum(length(sm.S) for sm in smooths; init=0) for smooths in smooths_list; init=0)
end

"""
    build_penalty_matrices(smooths_list, param_offsets) → Vector{Matrix}

Return the individual penalty matrices Sl_j (full size, zero-padded),
one per smoothing parameter.
"""
function build_penalty_matrices(smooths_list::Vector{Vector{ConstructedSmooth}},
                                param_offsets::Vector{Int})
    p = param_offsets[end]
    Sl = Matrix{Float64}[]
    for (k, smooths) in enumerate(smooths_list)
        for sm in smooths
            for Sj in sm.S
                Smat = zeros(p, p)
                s = sm.first_para
                e = sm.last_para
                Smat[s:e, s:e] .= Sj
                push!(Sl, Smat)
            end
        end
    end
    return Sl
end
