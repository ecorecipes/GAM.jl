# GAMTuringExt — Bayesian GAM fitting via Turing.jl
#
# This package extension is loaded when the user does `using GAM, Turing`.
# It provides the implementations of _fit_gam_bayes, _fit_gamlss_bayes,
# _fit_scam_bayes, and posterior summary methods for BayesGamModel.

module GAMTuringExt

using GAM
using Turing
using Turing: DynamicPPL, MCMCChains
using Distributions
using LinearAlgebra
using Statistics
using StatsBase: CoefTable
using StatsAPI
using GLM: Link, IdentityLink, LogLink, LogitLink, InverseLink, ProbitLink,
    CloglogLink, SqrtLink, linkinv, linkfun

import GAM: _fit_gam_bayes, _fit_gamlss_bayes, _fit_scam_bayes,
    _bayes_coef_means, _bayes_vcov, _bayes_coeftable, _bayes_credint

# ============================================================================
# Link function helpers for Turing models
# ============================================================================

_linkinv(::IdentityLink, η) = η
_linkinv(::LogLink, η) = exp(η)
_linkinv(::LogitLink, η) = 1 / (1 + exp(-η))
_linkinv(::InverseLink, η) = 1 / η
_linkinv(::SqrtLink, η) = η^2
_linkinv(::ProbitLink, η) = 0.5 * (1 + erf(η / sqrt(2)))
_linkinv(::CloglogLink, η) = 1 - exp(-exp(η))

# ============================================================================
# Turing model builder for standard GAM
# ============================================================================

function _build_gam_turing_model(
    X_para, smooths, y, family, link, priors;
    weights = nothing
)
    n = length(y)
    n_para = size(X_para, 2)
    n_smooth = length(smooths)

    # Pre-compute dimensions
    Xf_list = [sm.Xf for sm in smooths]
    Zs_list = [sm.Zs for sm in smooths]

    # Build full fixed-effect matrix: parametric + smooth null spaces
    X_fixed = X_para
    for Xf in Xf_list
        if size(Xf, 2) > 0
            X_fixed = hcat(X_fixed, Xf)
        end
    end
    n_fixed = size(X_fixed, 2)

    # Smooth info for the model
    n_random_blocks = sum(length(zs) for zs in Zs_list; init = 0)
    Zs_flat = Matrix{Float64}[]
    smooth_block_labels = String[]
    for (i, zs) in enumerate(Zs_list)
        for (j, Z) in enumerate(zs)
            push!(Zs_flat, Z)
            push!(smooth_block_labels, smooths[i].label)
        end
    end

    wts = weights === nothing ? ones(n) : Float64.(weights)

    DynamicPPL.@model function gam_model(
        y_obs, X_f, Zs, wts, n_f, family, link, priors
    )
        n_blocks = length(Zs)

        # Fixed effects prior
        β ~ MvNormal(zeros(n_f), 10.0 * I)

        # Smooth SD priors (one per random effect block)
        σ_s = Vector{Real}(undef, n_blocks)
        for i in 1:n_blocks
            σ_s[i] ~ GAM.get_prior(priors, :sds, smooth_block_labels[i])
        end

        # Smooth random effects (non-centered parameterization)
        z_list = Vector{Vector{Real}}(undef, n_blocks)
        for i in 1:n_blocks
            k_i = size(Zs[i], 2)
            z_list[i] ~ MvNormal(zeros(k_i), I)
        end

        # Linear predictor
        η = X_f * β
        for i in 1:n_blocks
            η = η .+ σ_s[i] .* (Zs[i] * z_list[i])
        end

        # Apply link
        μ = [_linkinv(link, η_i) for η_i in η]

        # Likelihood (dispatch on family type)
        _gam_likelihood!(y_obs, μ, wts, family, priors)

        return nothing
    end

    model = gam_model(y, X_fixed, Zs_flat, wts, n_fixed, family, link, priors)
    return model, X_fixed, Zs_flat, smooth_block_labels
end

# ============================================================================
# Likelihood dispatch
# ============================================================================

function _gam_likelihood!(y, μ, wts, family::Normal, priors)
    σ ~ GAM.get_prior(priors, :sigma)
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ / sqrt(wts[i]))
    end
end

function _gam_likelihood!(y, μ, wts, family::Poisson, priors)
    for i in eachindex(y)
        y[i] ~ Poisson(max(μ[i], 1e-10))
    end
end

function _gam_likelihood!(y, μ, wts, family::Bernoulli, priors)
    for i in eachindex(y)
        y[i] ~ Bernoulli(clamp(μ[i], 1e-10, 1 - 1e-10))
    end
end

function _gam_likelihood!(y, μ, wts, family::Binomial, priors)
    for i in eachindex(y)
        y[i] ~ Binomial(1, clamp(μ[i], 1e-10, 1 - 1e-10))
    end
end

function _gam_likelihood!(y, μ, wts, family::Gamma, priors)
    ϕ ~ GAM.get_prior(priors, :phi)
    for i in eachindex(y)
        # Gamma parameterized as shape α, scale θ where E[Y] = αθ = μ
        α = max(ϕ, 1e-10)
        θ = max(μ[i] / α, 1e-10)
        y[i] ~ Distributions.Gamma(α, θ)
    end
end

function _gam_likelihood!(y, μ, wts, family::InverseGaussian, priors)
    ϕ ~ GAM.get_prior(priors, :phi)
    for i in eachindex(y)
        y[i] ~ Distributions.InverseGaussian(max(μ[i], 1e-10), max(ϕ, 1e-10))
    end
end

# ============================================================================
# Main Bayesian fitting entry point
# ============================================================================

function GAM._fit_gam_bayes(formula, data, family, link, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4,
    weights = nothing, gam_formula = nothing)

    # Build design matrices using GAM.jl infrastructure
    if gam_formula !== nothing
        gf = gam_formula
    elseif formula isa GAM.GamFormula
        gf = formula
    else
        gf = GAM.GamFormula(formula)
    end

    # Build smooth bases
    cols = Tables.columntable(data)
    n = Tables.rowcount(data)

    X_para, smooths, labels = GAM.gam_matrices(gf, data)

    # Extract response
    y = Float64.(Tables.getcolumn(data, gf.response))

    # Build and sample Turing model
    model, X_fixed, Zs_flat, block_labels = _build_gam_turing_model(
        X_para, smooths, y, family, link, priors; weights = weights
    )

    # Select sampler
    turing_sampler = sampler === nothing ? NUTS() : sampler

    # Run MCMC
    chains = if nchains > 1
        sample(model, turing_sampler, MCMCThreads(), nsamples, nchains)
    else
        sample(model, turing_sampler, nsamples)
    end

    # Build coefficient names
    coef_names = String[]
    # Fixed effects
    para_names = gf.has_intercept ? ["(Intercept)"] : String[]
    for v in gf.parametric
        push!(para_names, string(v))
    end
    for sm in smooths
        if size(sm.Xf, 2) > 0
            for j in 1:size(sm.Xf, 2)
                push!(para_names, "$(sm.label)_f$j")
            end
        end
    end
    append!(coef_names, para_names)

    # Smooth SD names
    smooth_sd_names = ["sds($(l))" for l in block_labels]

    # Construct result
    sampler_desc = "$(typeof(turing_sampler)) ($nsamples samples × $nchains chains)"

    return GAM.BayesGamModel(
        formula, family, link,
        smooths, chains,
        coef_names, labels,
        size(X_para, 2), length(smooths),
        n, priors, sampler_desc, data
    )
end

# ============================================================================
# Posterior summary methods
# ============================================================================

function GAM._bayes_coef_means(m::GAM.BayesGamModel)
    chains = m.chains
    # Extract β parameters
    n_fixed = length(m.coef_names)
    means = Float64[]
    for i in 1:n_fixed
        sym = Symbol("β[$i]")
        if sym in names(chains)
            push!(means, mean(chains[sym]))
        end
    end
    return means
end

function GAM._bayes_vcov(m::GAM.BayesGamModel)
    chains = m.chains
    n_fixed = length(m.coef_names)
    syms = [Symbol("β[$i]") for i in 1:n_fixed]
    valid = [s for s in syms if s in names(chains)]
    n = length(valid)
    V = zeros(n, n)
    vals = [vec(chains[s].data) for s in valid]
    for i in 1:n, j in 1:n
        V[i, j] = cov(vals[i], vals[j])
    end
    return V
end

function GAM._bayes_coeftable(m::GAM.BayesGamModel)
    chains = m.chains
    n_fixed = length(m.coef_names)

    names_out = String[]
    estimates = Float64[]
    errors = Float64[]
    lower = Float64[]
    upper = Float64[]

    for i in 1:n_fixed
        sym = Symbol("β[$i]")
        if sym in names(chains)
            vals = vec(chains[sym].data)
            push!(names_out, m.coef_names[i])
            push!(estimates, mean(vals))
            push!(errors, std(vals))
            q = quantile(vals, [0.025, 0.975])
            push!(lower, q[1])
            push!(upper, q[2])
        end
    end

    return CoefTable(
        hcat(estimates, errors, lower, upper),
        ["Estimate", "Est.Error", "l-95% CI", "u-95% CI"],
        names_out
    )
end

function GAM._bayes_credint(m::GAM.BayesGamModel; level::Real = 0.95)
    chains = m.chains
    n_fixed = length(m.coef_names)
    α = (1 - level) / 2

    result = Matrix{Float64}(undef, 0, 2)
    for i in 1:n_fixed
        sym = Symbol("β[$i]")
        if sym in names(chains)
            vals = vec(chains[sym].data)
            q = quantile(vals, [α, 1 - α])
            result = vcat(result, q')
        end
    end
    return result
end

# ============================================================================
# Stub implementations for gamlss and scam Bayesian fitting
# (to be fully implemented in Phase 10.5 and 10.6)
# ============================================================================

function GAM._fit_gamlss_bayes(formulas, data, family, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4)
    error("Bayesian GAMLSS fitting is not yet implemented. Coming soon!")
end

function GAM._fit_scam_bayes(f, gf, data, family, link, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4,
    weights = nothing)
    error("Bayesian SCAM fitting is not yet implemented. Coming soon!")
end

end # module GAMTuringExt
