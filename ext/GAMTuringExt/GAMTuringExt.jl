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
using Tables
using GLM: Link, IdentityLink, LogLink, LogitLink, InverseLink, ProbitLink,
    CloglogLink, SqrtLink, linkinv, linkfun

import GAM: _fit_gam_bayes, _fit_gamlss_bayes, _fit_scam_bayes,
    _bayes_coef_means, _bayes_vcov, _bayes_coeftable, _bayes_credint

# ============================================================================
# Link function helpers (vectorized for efficiency)
# ============================================================================

_linkinv(::IdentityLink, η) = η
_linkinv(::LogLink, η) = exp(η)
_linkinv(::LogitLink, η) = 1 / (1 + exp(-η))
_linkinv(::InverseLink, η) = 1 / η
_linkinv(::SqrtLink, η) = η^2
_linkinv(::ProbitLink, η) = 0.5 * (1 + erf(η / sqrt(2)))
_linkinv(::CloglogLink, η) = 1 - exp(-exp(η))

_linkinv_vec(::IdentityLink, η) = η
_linkinv_vec(::LogLink, η) = exp.(η)
_linkinv_vec(::LogitLink, η) = 1 ./ (1 .+ exp.(.-η))
_linkinv_vec(::InverseLink, η) = 1 ./ η
_linkinv_vec(::SqrtLink, η) = η .^ 2
_linkinv_vec(::CloglogLink, η) = 1 .- exp.(.-exp.(η))

# ============================================================================
# Turing model builder for standard GAM
#
# Performance-critical design:
#   1. All random-effect blocks concatenated into one Z matrix + one z vector
#   2. Block SDs broadcast-multiplied (AD-friendly, no mutation)
#   3. Vectorized likelihoods: MvNormal for Gaussian, arraydist for others
#   4. Link inverse vectorized over η
# ============================================================================

function _build_gam_turing_model(
    X_para, smooths, y, family, link, priors;
    weights = nothing
)
    n = length(y)

    # Build full fixed-effect matrix: parametric + smooth null spaces
    Xf_blocks = Matrix{Float64}[X_para]
    for sm in smooths
        if size(sm.Xf, 2) > 0
            push!(Xf_blocks, sm.Xf)
        end
    end
    X_fixed = hcat(Xf_blocks...)
    n_fixed = size(X_fixed, 2)

    # Flatten all random-effect blocks into one combined matrix
    Zs_flat = Matrix{Float64}[]
    smooth_block_labels = String[]
    block_dims = Int[]
    for sm in smooths
        for Z in sm.Zs
            push!(Zs_flat, Z)
            push!(smooth_block_labels, sm.label)
            push!(block_dims, size(Z, 2))
        end
    end
    n_blocks = length(Zs_flat)
    total_random = sum(block_dims; init = 0)

    # Combined Z: n × total_random
    Z_combined = total_random > 0 ? hcat(Zs_flat...) : zeros(n, 0)

    # Block offset map (for scaling z by block-specific σ_s)
    block_ends = cumsum(block_dims)
    block_starts = [1; block_ends[1:end-1] .+ 1]

    wts = weights === nothing ? ones(n) : Float64.(weights)
    all_wts_one = all(w -> w ≈ 1.0, wts)

    # Family / link tags
    family_tag = _family_tag(family)
    needs_scale = family_tag in (:gaussian, :gamma, :invgaussian)
    is_identity = link isa IdentityLink

    # Resolve priors outside @model
    sds_priors = [GAM.get_prior(priors, :sds, l) for l in smooth_block_labels]
    scale_prior = needs_scale ?
        GAM.get_prior(priors, family_tag == :gaussian ? :sigma : :phi) : nothing

    # Check if all sds priors are the same (can use filldist)
    all_sds_same = n_blocks > 0 && all(p -> p == sds_priors[1], sds_priors)

    DynamicPPL.@model function gam_model(
        y_obs, X_f, Z_comb, wts,
        n_f, n_blocks, total_random, block_starts, block_ends, block_dims,
        family_tag, is_identity, link,
        sds_priors, scale_prior, all_sds_same, all_wts_one
    )
        n_obs = length(y_obs)

        # --- Fixed effects ---
        β ~ MvNormal(zeros(n_f), 10.0 * I)

        # --- Scale parameter (Gaussian σ, Gamma/IG ϕ) ---
        local σ_obs
        if scale_prior !== nothing
            σ_obs ~ scale_prior
        end

        # --- Smooth SDs ---
        local σ_s
        if n_blocks > 0
            if all_sds_same
                σ_s ~ filldist(sds_priors[1], n_blocks)
            else
                σ_s = Vector{Real}(undef, n_blocks)
                for i in 1:n_blocks
                    σ_s[i] ~ sds_priors[i]
                end
            end
        end

        # --- Random effects (non-centered, single MvNormal) ---
        local z
        if total_random > 0
            z ~ MvNormal(zeros(total_random), I)
        end

        # --- Linear predictor ---
        η = X_f * β
        if total_random > 0
            # Build scale vector: repeat each σ_s[i] for its block dimension
            scale_vec = vcat([fill(σ_s[i], block_dims[i]) for i in 1:n_blocks]...)
            η = η .+ Z_comb * (scale_vec .* z)
        end

        # --- Likelihood ---
        if family_tag == :gaussian
            if is_identity
                # Fast path: MvNormal (single logpdf, efficient AD)
                if all_wts_one
                    y_obs ~ MvNormal(η, σ_obs^2 * I)
                else
                    y_obs ~ MvNormal(η, Diagonal((σ_obs ./ sqrt.(wts)) .^ 2))
                end
            else
                μ = _linkinv_vec(link, η)
                if all_wts_one
                    y_obs ~ MvNormal(μ, σ_obs^2 * I)
                else
                    y_obs ~ MvNormal(μ, Diagonal((σ_obs ./ sqrt.(wts)) .^ 2))
                end
            end
        elseif family_tag == :poisson
            μ = _linkinv_vec(link, η)
            y_obs ~ arraydist(Poisson.(max.(μ, 1e-10)))
        elseif family_tag == :bernoulli
            μ = _linkinv_vec(link, η)
            y_obs ~ arraydist(Bernoulli.(clamp.(μ, 1e-10, 1 - 1e-10)))
        elseif family_tag == :binomial
            μ = _linkinv_vec(link, η)
            y_obs ~ arraydist(Binomial.(1, clamp.(μ, 1e-10, 1 - 1e-10)))
        elseif family_tag == :gamma
            μ = _linkinv_vec(link, η)
            α_shape = max(σ_obs, 1e-10)
            y_obs ~ arraydist(Distributions.Gamma.(α_shape, max.(μ ./ α_shape, 1e-10)))
        elseif family_tag == :invgaussian
            μ = _linkinv_vec(link, η)
            y_obs ~ arraydist(Distributions.InverseGaussian.(max.(μ, 1e-10), max(σ_obs, 1e-10)))
        end

        return nothing
    end

    model = gam_model(
        y, X_fixed, Z_combined, wts,
        n_fixed, n_blocks, total_random, block_starts, block_ends, block_dims,
        family_tag, is_identity, link,
        sds_priors, scale_prior, all_sds_same, all_wts_one
    )
    return model, X_fixed, Zs_flat, smooth_block_labels
end

_family_tag(::Normal) = :gaussian
_family_tag(::Poisson) = :poisson
_family_tag(::Bernoulli) = :bernoulli
_family_tag(::Binomial) = :binomial
_family_tag(::Gamma) = :gamma
_family_tag(::InverseGaussian) = :invgaussian
_family_tag(f) = error("Bayesian fitting not yet supported for $(typeof(f))")

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

    X_para, smooths, labels = GAM.gam_matrices(gf, data)

    # Extract response
    y = Float64.(Tables.getcolumn(Tables.columntable(data), gf.response))

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
        length(y), priors, sampler_desc, data
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
