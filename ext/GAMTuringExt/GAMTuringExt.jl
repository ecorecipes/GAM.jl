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
using Random
using Statistics
using StatsBase: CoefTable
using StatsAPI
using Tables
using GLM: Link, IdentityLink, LogLink, LogitLink, InverseLink, ProbitLink,
    CloglogLink, SqrtLink, linkinv, linkfun

import GAM: _fit_gam_bayes, _fit_gamlss_bayes, _fit_scam_bayes,
    _fit_gamm_bayes, _fit_gamm_bayes_from_parts,
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
    # Support both GamFormula and FormulaTerm (from @formula with FunctionTerm smooths)
    local gf, X_para, smooths, labels, resp_sym

    if gam_formula !== nothing
        gf = gam_formula
        X_para, smooths, labels = GAM.gam_matrices(gf, data)
        resp_sym = gf.response
    elseif formula isa GAM.GamFormula
        gf = formula
        X_para, smooths, labels = GAM.gam_matrices(gf, data)
        resp_sym = gf.response
    elseif formula isa StatsModels.FormulaTerm
        # @formula path — use the FormulaTerm overload of gam_matrices
        X_para, smooths, labels = GAM.gam_matrices(formula, data)
        resp_sym = formula.lhs isa StatsModels.Term ? formula.lhs.sym :
            Symbol(string(formula.lhs))
        gf = nothing
    else
        error("Unsupported formula type: $(typeof(formula))")
    end

    # Extract response
    y = Float64.(Tables.getcolumn(Tables.columntable(data), resp_sym))

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
    if gf !== nothing
        para_names = gf.has_intercept ? ["(Intercept)"] : String[]
        for v in gf.parametric
            push!(para_names, string(v))
        end
    else
        para_names = ["(Intercept)"]  # FormulaTerm path always has intercept
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

"""Find chain symbols for fixed-effect coefficients across all parameter blocks."""
function _find_beta_symbols(chains, n_names)
    chain_names = names(chains)

    # Try standard GAM pattern: β[1], β[2], ...
    syms_standard = [Symbol("β[$i]") for i in 1:n_names]
    if all(s -> s in chain_names, syms_standard)
        return syms_standard
    end

    # Try GAMLSS multi-parameter pattern: β_1[1], β_1[2], ..., β_2[1], ...
    syms = Symbol[]
    for s in chain_names
        str = string(s)
        if startswith(str, "β") && contains(str, "[") && endswith(str, "]")
            push!(syms, s)
        end
    end
    # Sort β symbols naturally: β_1[1], β_1[2], ..., β_2[1], β_2[2], ...
    sort!(syms; by = s -> begin
        str = string(s)
        # Extract parameter index and coefficient index
        m = match(r"β_?(\d+)?\[(\d+)\]", str)
        if m !== nothing
            k = m.captures[1] === nothing ? 0 : parse(Int, m.captures[1])
            i = parse(Int, m.captures[2])
            return (k, i)
        end
        return (999, 999)
    end)

    return syms[1:min(length(syms), n_names)]
end

function GAM._bayes_coef_means(m::GAM.BayesGamModel)
    chains = m.chains
    n_fixed = length(m.coef_names)
    syms = _find_beta_symbols(chains, n_fixed)
    means = Float64[]
    for sym in syms
        if sym in names(chains)
            push!(means, mean(chains[sym]))
        end
    end
    return means
end

function GAM._bayes_vcov(m::GAM.BayesGamModel)
    chains = m.chains
    n_fixed = length(m.coef_names)
    syms = _find_beta_symbols(chains, n_fixed)
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
    syms = _find_beta_symbols(chains, n_fixed)

    names_out = String[]
    estimates = Float64[]
    errors = Float64[]
    lower = Float64[]
    upper = Float64[]

    for (idx, sym) in enumerate(syms)
        if sym in names(chains)
            vals = vec(chains[sym].data)
            name = idx <= length(m.coef_names) ? m.coef_names[idx] : string(sym)
            push!(names_out, name)
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
    syms = _find_beta_symbols(chains, n_fixed)

    result = Matrix{Float64}(undef, 0, 2)
    for sym in syms
        if sym in names(chains)
            vals = vec(chains[sym].data)
            q = quantile(vals, [α, 1 - α])
            result = vcat(result, q')
        end
    end
    return result
end

# ============================================================================
# Bayesian GAMLSS — multi-parameter distributional models via Turing.jl
#
# Each distribution parameter gets its own linear predictor with fixed and
# random effects.  The likelihood is vectorized using arraydist.
# ============================================================================

# ── Family-specific vectorized distribution constructors ─────────────────

"""
    _gamlss_arraydist(family, params_vec) → arraydist(...)

Build a vectorized Distributions.jl `arraydist` from per-observation parameter
vectors.  `params_vec` is a Vector of K vectors, one per distribution parameter.
"""
function _gamlss_arraydist(::GAM.DistFamily{<:Normal}, pv)
    return arraydist(Normal.(pv[1], max.(pv[2], 1e-10)))
end

function _gamlss_arraydist(::GAM.GammaLocationScale, pv)
    μ = max.(pv[1], 1e-6)
    σ = max.(pv[2], 1e-6)
    α = 1 ./ (σ .^ 2)
    θ = μ .* σ .^ 2
    return arraydist(Distributions.Gamma.(max.(α, 1e-6), max.(θ, 1e-10)))
end

function _gamlss_arraydist(::GAM.BetaRegression, pv)
    μ = clamp.(pv[1], 1e-6, 1 - 1e-6)
    φ = max.(pv[2], 1e-6)
    return arraydist(Beta.(μ .* φ, (1 .- μ) .* φ))
end

function _gamlss_arraydist(::GAM.NegativeBinomialLocationScale, pv)
    μ = max.(pv[1], 1e-10)
    σ = max.(pv[2], 1e-10)
    r = 1 ./ (σ .^ 2)
    p = r ./ (r .+ μ)
    return arraydist(NegativeBinomial.(r, p))
end

function _gamlss_arraydist(::GAM.InverseGaussianLocationScale, pv)
    μ = max.(pv[1], 1e-10)
    σ = max.(pv[2], 1e-10)
    λ = μ ./ (σ .^ 2)
    return arraydist(Distributions.InverseGaussian.(μ, λ))
end

# ── Build per-parameter mixed-model matrices ─────────────────────────────

function _gamlss_matrices(formulas, data, K)
    cols = Tables.columntable(data)

    param_X = Matrix{Float64}[]
    param_smooths = Vector{GAM.SmoothMixedModel}[]
    param_labels = Vector{String}[]
    param_block_dims = Any[]  # each element is Vector{Int}

    for k in 1:K
        gf = formulas[k]
        n = Tables.rowcount(data)

        # Parametric design
        X_para = ones(n, 1)
        if gf isa GAM.GamFormula
            for v in gf.parametric
                col = Float64.(Tables.getcolumn(cols, v))
                X_para = hcat(X_para, col)
            end
            specs = gf.smooth_specs
        else
            specs = GAM.SmoothSpec[]
        end

        # Build smooth mixed-model decompositions
        smooths = GAM.SmoothMixedModel[]
        labels = String[]
        for spec in specs
            sm = GAM.smooth_construct(spec, cols)
            smm = GAM.smooth2random(sm)
            push!(smooths, smm)
            push!(labels, spec.label)
        end

        # Build combined Xf (fixed) and Z (random) for this parameter
        Xf_blocks = Matrix{Float64}[X_para]
        for smm in smooths
            if size(smm.Xf, 2) > 0
                push!(Xf_blocks, smm.Xf)
            end
        end
        X_fixed_k = hcat(Xf_blocks...)

        push!(param_X, X_fixed_k)
        push!(param_smooths, smooths)
        push!(param_labels, labels)

        # Block dimensions for this parameter
        bdims = Int[]
        for smm in smooths
            for Z in smm.Zs
                push!(bdims, size(Z, 2))
            end
        end
        push!(param_block_dims, bdims)
    end

    return param_X, param_smooths, param_labels, param_block_dims
end

# ── Turing model builder for GAMLSS ─────────────────────────────────────

function _build_gamlss_turing_model(
    param_X, param_smooths, param_labels, param_block_dims,
    y, family, priors
)
    n = length(y)
    K = length(param_X)
    links = if family isa GAM.DistFamily
        family.links
    elseif hasproperty(family, :links)
        family.links
    else
        error("Cannot extract link functions from $(typeof(family))")
    end

    # Pre-compute combined Z per parameter
    param_Z = Matrix{Float64}[]
    param_n_blocks = Int[]
    param_total_random = Int[]
    all_sds_priors = Vector{Distribution}[]

    for k in 1:K
        Zs_flat = Matrix{Float64}[]
        sds_p = Distribution[]
        for smm in param_smooths[k]
            for (j, Z) in enumerate(smm.Zs)
                push!(Zs_flat, Z)
                push!(sds_p, GAM.get_prior(priors, :sds, smm.label))
            end
        end
        nb = length(Zs_flat)
        tr = sum(param_block_dims[k]; init = 0)
        Z_comb = tr > 0 ? hcat(Zs_flat...) : zeros(n, 0)

        push!(param_Z, Z_comb)
        push!(param_n_blocks, nb)
        push!(param_total_random, tr)
        push!(all_sds_priors, sds_p)
    end

    DynamicPPL.@model function gamlss_model(
        y_obs, param_X, param_Z, param_block_dims,
        param_n_blocks, param_total_random,
        all_sds_priors, family, links, K
    )
        n_obs = length(y_obs)

        # We support up to 4 parameters (location, scale, shape1, shape2)
        # Each gets its own named β, σ_s, z to avoid dynamic symbol issues

        # Parameter 1 (always present)
        β_1 ~ MvNormal(zeros(size(param_X[1], 2)), 10.0 * I)
        η_1 = param_X[1] * β_1
        if param_total_random[1] > 0
            nb1 = param_n_blocks[1]
            σ_s_1 = Vector{Real}(undef, nb1)
            for j in 1:nb1
                σ_s_1[j] ~ all_sds_priors[1][j]
            end
            z_1 ~ MvNormal(zeros(param_total_random[1]), I)
            sv1 = vcat([fill(σ_s_1[j], param_block_dims[1][j]) for j in 1:nb1]...)
            η_1 = η_1 .+ param_Z[1] * (sv1 .* z_1)
        end
        p1 = _linkinv_vec(links[1], η_1)

        local p2, p3, p4

        # Parameter 2 (if K ≥ 2)
        if K >= 2
            β_2 ~ MvNormal(zeros(size(param_X[2], 2)), 10.0 * I)
            η_2 = param_X[2] * β_2
            if param_total_random[2] > 0
                nb2 = param_n_blocks[2]
                σ_s_2 = Vector{Real}(undef, nb2)
                for j in 1:nb2
                    σ_s_2[j] ~ all_sds_priors[2][j]
                end
                z_2 ~ MvNormal(zeros(param_total_random[2]), I)
                sv2 = vcat([fill(σ_s_2[j], param_block_dims[2][j]) for j in 1:nb2]...)
                η_2 = η_2 .+ param_Z[2] * (sv2 .* z_2)
            end
            p2 = _linkinv_vec(links[2], η_2)
        end

        # Parameter 3 (if K ≥ 3)
        if K >= 3
            β_3 ~ MvNormal(zeros(size(param_X[3], 2)), 10.0 * I)
            η_3 = param_X[3] * β_3
            if param_total_random[3] > 0
                nb3 = param_n_blocks[3]
                σ_s_3 = Vector{Real}(undef, nb3)
                for j in 1:nb3
                    σ_s_3[j] ~ all_sds_priors[3][j]
                end
                z_3 ~ MvNormal(zeros(param_total_random[3]), I)
                sv3 = vcat([fill(σ_s_3[j], param_block_dims[3][j]) for j in 1:nb3]...)
                η_3 = η_3 .+ param_Z[3] * (sv3 .* z_3)
            end
            p3 = _linkinv_vec(links[3], η_3)
        end

        # Parameter 4 (if K ≥ 4, rare)
        if K >= 4
            β_4 ~ MvNormal(zeros(size(param_X[4], 2)), 10.0 * I)
            η_4 = param_X[4] * β_4
            if param_total_random[4] > 0
                nb4 = param_n_blocks[4]
                σ_s_4 = Vector{Real}(undef, nb4)
                for j in 1:nb4
                    σ_s_4[j] ~ all_sds_priors[4][j]
                end
                z_4 ~ MvNormal(zeros(param_total_random[4]), I)
                sv4 = vcat([fill(σ_s_4[j], param_block_dims[4][j]) for j in 1:nb4]...)
                η_4 = η_4 .+ param_Z[4] * (sv4 .* z_4)
            end
            p4 = _linkinv_vec(links[4], η_4)
        end

        # Assemble params vector
        pv = if K == 2
            [p1, p2]
        elseif K == 3
            [p1, p2, p3]
        elseif K == 4
            [p1, p2, p3, p4]
        else
            [p1]
        end

        # Likelihood
        y_obs ~ _gamlss_arraydist(family, pv)

        return nothing
    end

    model = gamlss_model(
        y, param_X, param_Z, param_block_dims,
        param_n_blocks, param_total_random,
        all_sds_priors, family, links, K
    )
    return model
end

# ── GAMLSS fitting entry point ──────────────────────────────────────────

function GAM._fit_gamlss_bayes(formulas, data, family, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4)

    # Determine number of parameters
    K = if family isa GAM.DistFamily
        GAM.nparams(family)
    elseif family isa GAM.MultiParameterFamily
        GAM.nparams(family)
    else
        GAM._gamlss_nparams(family)
    end

    # Normalize formulas: single formula → replicate K times
    if formulas isa GAM.GamFormula || formulas isa StatsModels.FormulaTerm
        formulas_vec = fill(formulas, K)
    else
        formulas_vec = collect(formulas)
    end
    length(formulas_vec) == K || throw(ArgumentError(
        "Expected $K formulas for $(typeof(family)), got $(length(formulas_vec))"))

    # Wrap family if needed
    fam = if family isa GAM.MultiParameterFamily
        family
    elseif family isa Normal
        GAM.DistFamily(family, [IdentityLink(), LogLink()], ["mu", "sigma"])
    else
        error("Cannot determine GAMLSS family type for $(typeof(family))")
    end

    # Build per-parameter matrices
    param_X, param_smooths, param_labels, param_block_dims =
        _gamlss_matrices(formulas_vec, data, K)

    # Extract response
    cols = Tables.columntable(data)
    resp_sym = if formulas_vec[1] isa GAM.GamFormula
        formulas_vec[1].response
    else
        formulas_vec[1].lhs.sym
    end
    y = Float64.(Tables.getcolumn(cols, resp_sym))

    # Build Turing model
    model = _build_gamlss_turing_model(
        param_X, param_smooths, param_labels, param_block_dims,
        y, fam, priors
    )

    # Run MCMC
    turing_sampler = sampler === nothing ? NUTS() : sampler
    chains = if nchains > 1
        sample(model, turing_sampler, MCMCThreads(), nsamples, nchains)
    else
        sample(model, turing_sampler, nsamples)
    end

    # Build coefficient names
    coef_names = String[]
    all_labels = String[]
    total_smooths = 0
    for k in 1:K
        pname = GAM.param_names(fam)[k]
        n_f = size(param_X[k], 2)
        push!(coef_names, "$(pname)_(Intercept)")
        for j in 2:n_f
            push!(coef_names, "$(pname)_f$j")
        end
        for l in param_labels[k]
            push!(all_labels, "$(pname):$l")
        end
        total_smooths += length(param_smooths[k])
    end

    # Flatten all smooths
    all_smooths = GAM.SmoothMixedModel[]
    for k in 1:K
        append!(all_smooths, param_smooths[k])
    end

    sampler_desc = "$(typeof(turing_sampler)) ($nsamples samples × $nchains chains)"

    return GAM.BayesGamModel(
        formulas_vec[1], fam, nothing,
        all_smooths, chains,
        coef_names, all_labels,
        sum(size(X, 2) for X in param_X),
        total_smooths,
        length(y), priors, sampler_desc, data
    )
end

# ============================================================================
# Bayesian SCAM — shape-constrained additive models via Turing.jl
#
# Shape constraints use exp() reparameterization: constrained coefficients
# β_constrained = exp(α) where α ~ N(0, σ²_s).  The positivity of exp(α)
# combined with the constraint matrix Σ (cumsum etc.) enforces monotonicity,
# convexity, etc.
#
# Design:
#   1. Build SCAM basis as usual (X = B * Σ, penalty S, p_ident BitVector)
#   2. Use smooth2random to decompose into fixed + random
#   3. In Turing model, apply exp() to coefficients flagged by p_ident
#   4. Standard vectorized likelihood
# ============================================================================

function GAM._fit_scam_bayes(f, gf, data, family, link, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4,
    weights = nothing)

    # Build design matrices via standard setup_gam
    y, X, X_para, smooths, n_parametric = GAM.setup_gam(gf, data; family = family)
    n, p = size(X)

    # Build p_ident: which coefficients need exp transform
    p_ident = GAM.build_p_ident(smooths, n_parametric, p)

    # Determine which smooth coefficients are constrained
    # We'll work with the full basis (not smooth2random) for constrained smooths,
    # and smooth2random for unconstrained smooths
    constrained_smooths = filter(sm -> sm.p_ident !== nothing, smooths)
    unconstrained_smooths = filter(sm -> sm.p_ident === nothing, smooths)

    # Build mixed-model decomposition for unconstrained smooths only
    uc_mmods = GAM.SmoothMixedModel[]
    for sm in unconstrained_smooths
        push!(uc_mmods, GAM.smooth2random(sm))
    end

    # For constrained smooths: use the raw basis X and penalty
    # Apply the standard non-centered parameterization but with exp transform

    # Build combined matrices
    # Fixed effects: intercept + parametric terms
    X_fixed = X_para

    # Unconstrained smooth: mixed-model form (Xf + Z)
    uc_Zs = Matrix{Float64}[]
    uc_block_dims = Int[]
    uc_labels = String[]
    for smm in uc_mmods
        if size(smm.Xf, 2) > 0
            X_fixed = hcat(X_fixed, smm.Xf)
        end
        for Z in smm.Zs
            push!(uc_Zs, Z)
            push!(uc_block_dims, size(Z, 2))
            push!(uc_labels, smm.label)
        end
    end
    n_fixed = size(X_fixed, 2)
    uc_total_random = sum(uc_block_dims; init = 0)
    Z_uc = uc_total_random > 0 ? hcat(uc_Zs...) : zeros(n, 0)
    n_uc_blocks = length(uc_Zs)

    # Constrained smooth: raw basis columns (subject to exp transform)
    con_X = Matrix{Float64}[]
    con_dims = Int[]
    con_labels = String[]
    con_penalties = Matrix{Float64}[]
    for sm in constrained_smooths
        push!(con_X, sm.X)
        push!(con_dims, size(sm.X, 2))
        push!(con_labels, sm.spec.label)
        push!(con_penalties, sm.S[1])  # primary penalty
    end
    n_con_blocks = length(con_X)
    total_con = sum(con_dims; init = 0)
    X_con = total_con > 0 ? hcat(con_X...) : zeros(n, 0)

    # Resolve priors
    uc_sds_priors = [GAM.get_prior(priors, :sds, l) for l in uc_labels]
    con_sds_priors = [GAM.get_prior(priors, :sds, l) for l in con_labels]

    family_tag = _family_tag(family)
    needs_scale = family_tag in (:gaussian, :gamma, :invgaussian)
    scale_prior = needs_scale ?
        GAM.get_prior(priors, family_tag == :gaussian ? :sigma : :phi) : nothing
    is_identity = link isa IdentityLink
    wts = weights === nothing ? ones(n) : Float64.(weights)
    all_wts_one = all(w -> w ≈ 1.0, wts)

    DynamicPPL.@model function scam_model(
        y_obs, X_f, Z_uc, X_con,
        n_f, n_uc_blocks, uc_total_random, uc_block_dims,
        n_con_blocks, con_dims, total_con,
        family_tag, is_identity, link, wts, all_wts_one,
        uc_sds_priors, con_sds_priors, scale_prior
    )
        n_obs = length(y_obs)

        # Fixed effects
        β ~ MvNormal(zeros(n_f), 10.0 * I)
        η = X_f * β

        # Unconstrained smooth random effects (standard non-centered)
        if uc_total_random > 0
            local σ_uc
            σ_uc = Vector{Real}(undef, n_uc_blocks)
            for i in 1:n_uc_blocks
                σ_uc[i] ~ uc_sds_priors[i]
            end
            z_uc ~ MvNormal(zeros(uc_total_random), I)
            uc_scale = vcat([fill(σ_uc[i], uc_block_dims[i]) for i in 1:n_uc_blocks]...)
            η = η .+ Z_uc * (uc_scale .* z_uc)
        end

        # Constrained smooth coefficients: α ~ N(0, σ²I), β_con = exp(α)
        if total_con > 0
            local σ_con
            σ_con = Vector{Real}(undef, n_con_blocks)
            for i in 1:n_con_blocks
                σ_con[i] ~ con_sds_priors[i]
            end
            α_con ~ MvNormal(zeros(total_con), I)
            con_scale = vcat([fill(σ_con[i], con_dims[i]) for i in 1:n_con_blocks]...)
            β_con = exp.(con_scale .* α_con)
            η = η .+ X_con * β_con
        end

        # Scale parameter
        local σ_obs
        if scale_prior !== nothing
            σ_obs ~ scale_prior
        end

        # Likelihood (same as standard GAM)
        if family_tag == :gaussian
            if is_identity
                if all_wts_one
                    y_obs ~ MvNormal(η, σ_obs^2 * I)
                else
                    y_obs ~ MvNormal(η, Diagonal((σ_obs ./ sqrt.(wts)) .^ 2))
                end
            else
                μ = _linkinv_vec(link, η)
                y_obs ~ MvNormal(μ, σ_obs^2 * I)
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

    model = scam_model(
        y, X_fixed, Z_uc, X_con,
        n_fixed, n_uc_blocks, uc_total_random, uc_block_dims,
        n_con_blocks, con_dims, total_con,
        family_tag, is_identity, link, wts, all_wts_one,
        uc_sds_priors, con_sds_priors, scale_prior
    )

    # Run MCMC
    turing_sampler = sampler === nothing ? NUTS() : sampler
    chains = if nchains > 1
        sample(model, turing_sampler, MCMCThreads(), nsamples, nchains)
    else
        sample(model, turing_sampler, nsamples)
    end

    # Build coefficient names
    coef_names = String["(Intercept)"]
    for smm in uc_mmods
        if size(smm.Xf, 2) > 0
            for j in 1:size(smm.Xf, 2)
                push!(coef_names, "$(smm.label)_f$j")
            end
        end
    end

    all_labels = String[]
    all_smooths = GAM.SmoothMixedModel[]
    for smm in uc_mmods
        push!(all_labels, smm.label)
        push!(all_smooths, smm)
    end
    for l in con_labels
        push!(all_labels, "$l(constrained)")
    end

    sampler_desc = "$(typeof(turing_sampler)) ($nsamples samples × $nchains chains)"

    return GAM.BayesGamModel(
        f, family, link,
        all_smooths, chains,
        coef_names, all_labels,
        n_fixed, length(smooths),
        n, priors, sampler_desc, data
    )
end

# ============================================================================
# Posterior predictive utilities for BayesGamModel
# ============================================================================

"""
    GAM.posterior_samples(m::BayesGamModel; n=nothing)

Extract posterior coefficient samples from the MCMC chains.
Returns an `n_draws × n_coef` matrix.
"""
function GAM.posterior_samples(m::GAM.BayesGamModel;
    n::Union{Int, Nothing} = nothing, kwargs...)

    chains = m.chains
    n_coef = length(m.coef_names)
    beta_syms = _find_beta_symbols(chains, n_coef)
    isempty(beta_syms) && return Matrix{Float64}(undef, 0, 0)

    n_total = length(vec(chains[beta_syms[1]].data))
    beta_draws = Matrix{Float64}(undef, n_total, length(beta_syms))
    for (j, sym) in enumerate(beta_syms)
        beta_draws[:, j] = vec(chains[sym].data)
    end

    if n !== nothing && n < n_total
        idx = sort(Random.randperm(n_total)[1:n])
        beta_draws = beta_draws[idx, :]
    end

    return beta_draws
end

# ============================================================================
# Bayesian GAMM — Turing model with grouped random effects
# ============================================================================

"""
Build a Turing model for GAMM: smooth terms (via smooth2random) + explicit
grouped random effects (random intercepts/slopes).
"""
function _build_gamm_turing_model(
    X_para, smooths, random_effects, y, family, link, priors;
    weights = nothing
)
    n = length(y)

    # --- Fixed effects: parametric + smooth null spaces ---
    Xf_blocks = Matrix{Float64}[X_para]
    for sm in smooths
        if size(sm.Xf, 2) > 0
            push!(Xf_blocks, sm.Xf)
        end
    end
    X_fixed = hcat(Xf_blocks...)
    n_fixed = size(X_fixed, 2)

    # --- Smooth random-effect blocks ---
    Zs_smooth = Matrix{Float64}[]
    smooth_block_labels = String[]
    smooth_block_dims = Int[]
    for sm in smooths
        for Z in sm.Zs
            push!(Zs_smooth, Z)
            push!(smooth_block_labels, sm.label)
            push!(smooth_block_dims, size(Z, 2))
        end
    end
    n_smooth_blocks = length(Zs_smooth)
    total_smooth_random = sum(smooth_block_dims; init = 0)
    Z_smooth = total_smooth_random > 0 ? hcat(Zs_smooth...) : zeros(n, 0)

    # --- Grouped random-effect blocks ---
    Z_re_list = Matrix{Float64}[]
    re_labels = String[]
    re_block_dims = Int[]
    for cre in random_effects
        push!(Z_re_list, cre.Z)
        push!(re_labels, cre.spec.label)
        push!(re_block_dims, size(cre.Z, 2))
    end
    n_re_blocks = length(Z_re_list)
    total_re_random = sum(re_block_dims; init = 0)
    Z_re = total_re_random > 0 ? hcat(Z_re_list...) : zeros(n, 0)

    # Block offset maps
    sm_block_ends = cumsum(smooth_block_dims)
    sm_block_starts = isempty(sm_block_ends) ? Int[] :
        [1; sm_block_ends[1:end-1] .+ 1]

    re_block_ends = cumsum(re_block_dims)
    re_block_starts = isempty(re_block_ends) ? Int[] :
        [1; re_block_ends[1:end-1] .+ 1]

    wts = weights === nothing ? ones(n) : Float64.(weights)
    all_wts_one = all(w -> w ≈ 1.0, wts)

    family_tag = _family_tag(family)
    needs_scale = family_tag in (:gaussian, :gamma, :invgaussian)
    is_identity = link isa IdentityLink

    # Resolve priors
    sds_priors = [GAM.get_prior(priors, :sds, l) for l in smooth_block_labels]
    re_sd_priors = [GAM.get_prior(priors, :sds, l) for l in re_labels]
    scale_prior = needs_scale ?
        GAM.get_prior(priors, family_tag == :gaussian ? :sigma : :phi) : nothing

    DynamicPPL.@model function gamm_model(
        y_obs, X_f, Z_sm, Z_re,
        n_f, n_sm_blocks, total_sm, sm_block_starts, sm_block_ends, smooth_block_dims,
        n_re_blocks, total_re, re_block_starts, re_block_ends, re_block_dims,
        family_tag, is_identity, link,
        sds_priors, re_sd_priors, scale_prior, all_wts_one
    )
        n_obs = length(y_obs)

        # --- Fixed effects ---
        β ~ MvNormal(zeros(n_f), 10.0 * I)

        # --- Scale parameter ---
        local σ_obs
        if scale_prior !== nothing
            σ_obs ~ scale_prior
        end

        # --- Smooth SDs ---
        local σ_s
        if n_sm_blocks > 0
            σ_s = Vector{Real}(undef, n_sm_blocks)
            for i in 1:n_sm_blocks
                σ_s[i] ~ sds_priors[i]
            end
        end

        # --- Random effect SDs ---
        local σ_re
        if n_re_blocks > 0
            σ_re = Vector{Real}(undef, n_re_blocks)
            for i in 1:n_re_blocks
                σ_re[i] ~ re_sd_priors[i]
            end
        end

        # --- Smooth random effects (non-centered) ---
        local z_sm
        if total_sm > 0
            z_sm ~ MvNormal(zeros(total_sm), I)
        end

        # --- Grouped random effects (non-centered) ---
        local z_re
        if total_re > 0
            z_re ~ MvNormal(zeros(total_re), I)
        end

        # --- Linear predictor ---
        η = X_f * β

        # Add smooth random contributions
        if total_sm > 0
            scale_sm = vcat([fill(σ_s[i], smooth_block_dims[i]) for i in 1:n_sm_blocks]...)
            η = η .+ Z_sm * (scale_sm .* z_sm)
        end

        # Add grouped random effect contributions
        if total_re > 0
            scale_re = vcat([fill(σ_re[i], re_block_dims[i]) for i in 1:n_re_blocks]...)
            η = η .+ Z_re * (scale_re .* z_re)
        end

        # --- Likelihood ---
        if family_tag == :gaussian
            if is_identity
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

    model = gamm_model(
        y, X_fixed, Z_smooth, Z_re,
        n_fixed, n_smooth_blocks, total_smooth_random,
        sm_block_starts, sm_block_ends, smooth_block_dims,
        n_re_blocks, total_re_random,
        re_block_starts, re_block_ends, re_block_dims,
        family_tag, is_identity, link,
        sds_priors, re_sd_priors, scale_prior, all_wts_one
    )
    return model, X_fixed, Zs_smooth, smooth_block_labels, re_labels
end

"""
Bayesian GAMM from GammFormula: parse formula, build matrices, fit with Turing.
"""
function GAM._fit_gamm_bayes(gf::GAM.GammFormula, data, family, link, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4, weights = nothing)

    # Build GAM part
    X_para, smooths, labels = GAM.gam_matrices(gf.gam_formula, data)
    y = Float64.(Tables.getcolumn(Tables.columntable(data), gf.gam_formula.response))

    # Build random effects
    random_effects = [GAM.construct_random_effect(re, data) for re in gf.random_effects]

    return _fit_gamm_bayes_impl(y, X_para, smooths, random_effects, labels,
        gf, data, family, link, priors;
        sampler = sampler, nsamples = nsamples, nchains = nchains, weights = weights)
end

"""
Bayesian GAMM from pre-built matrices: used by the FormulaTerm dispatch path.
"""
function GAM._fit_gamm_bayes_from_parts(y, X, smooths, n_parametric, random_effects,
    formula, data, family, link, priors::GAM.PriorSpec;
    sampler = nothing, nsamples::Int = 2000, nchains::Int = 4, weights = nothing)

    # Build smooth2random representations
    sm_mixed = [GAM.smooth2random(sm) for sm in smooths]

    # Parametric part of X (first n_parametric columns)
    X_para = X[:, 1:n_parametric]
    labels = [sm.label for sm in sm_mixed]

    return _fit_gamm_bayes_impl(y, X_para, sm_mixed, random_effects, labels,
        formula, data, family, link, priors;
        sampler = sampler, nsamples = nsamples, nchains = nchains, weights = weights)
end

"""
Common implementation for Bayesian GAMM fitting.
"""
function _fit_gamm_bayes_impl(y, X_para, smooths, random_effects, labels,
    formula, data, family, link, priors;
    sampler = nothing, nsamples = 2000, nchains = 4, weights = nothing)

    # Build and sample Turing model
    model, X_fixed, Zs_smooth, smooth_labels, re_labels = _build_gamm_turing_model(
        X_para, smooths, random_effects, y, family, link, priors; weights = weights
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
    # Parametric fixed effects
    para_names = ["(Intercept)"]
    for sm in smooths
        if size(sm.Xf, 2) > 0
            for j in 1:size(sm.Xf, 2)
                push!(para_names, "$(sm.label)_f$j")
            end
        end
    end
    append!(coef_names, para_names)

    # Smooth SD names
    smooth_sd_names = ["sds($(l))" for l in smooth_labels]
    # RE SD names
    re_sd_names = ["σ_re($(l))" for l in re_labels]

    all_labels = vcat(labels, re_labels)

    sampler_desc = "$(typeof(turing_sampler)) ($nsamples samples × $nchains chains)"

    return GAM.BayesGamModel(
        formula, family, link,
        smooths, chains,
        coef_names, all_labels,
        size(X_para, 2), length(smooths),
        length(y), priors, sampler_desc, data
    )
end

# ============================================================================
# smooth_prior — Composable Turing @model for smooth terms
# ============================================================================
#
# Usage with to_submodel + prefix (DynamicPPL 0.40+):
#
#   sm = gam_smooth(:x, data; k=10)
#   @model function my_gam(y, sm)
#       β0 ~ Normal(0, 10)
#       σ ~ Exponential(1.0)
#       f ~ to_submodel(prefix(smooth_prior(sm), :s_x))
#       y ~ MvNormal(β0 .+ f, σ^2 * I)
#   end
#
# Multiple smooths:
#
#   sm1 = gam_smooth(:x1, data; k=10)
#   sm2 = gam_smooth(:x2, data; k=8, bs=:cr)
#   @model function my_gam(y, sm1, sm2)
#       β0 ~ Normal(0, 10)
#       σ ~ Exponential(1.0)
#       f1 ~ to_submodel(prefix(smooth_prior(sm1), :s_x1))
#       f2 ~ to_submodel(prefix(smooth_prior(sm2), :s_x2))
#       y ~ MvNormal(β0 .+ f1 .+ f2, σ^2 * I)
#   end

"""
    smooth_prior(sm::SmoothMixedModel; sds_prior, fixed_prior)

A Turing `@model` that samples the parameters of a smooth term and returns
the evaluated smooth function values (a vector of length n).

Use with `to_submodel` and `prefix` inside a Turing `@model` to compose smooth
terms into custom Bayesian models. Each smooth gets its own fixed-effect
coefficients, smooth SD, and random effects — all sampled automatically.

# Arguments
- `sm::SmoothMixedModel`: smooth term from `gam_smooth()` or `smooth2random(smooth_construct(...))`
- `sds_prior`: prior on the smooth SD σ_s (default: `Exponential(1.0)`)
- `fixed_prior`: prior on each fixed-effect coefficient (default: `Normal(0, 10)`)

# Returns
- `f::Vector{Float64}`: evaluated smooth at each observation, `Xf * β_f + σ_s * Zs * z`

# Example
```julia
import GAM
using Turing

sm = GAM.gam_smooth(:x, data; k=10, bs=:cr)

@model function my_model(y, sm)
    β0 ~ Normal(0, 10)
    σ ~ Exponential(1.0)
    f ~ to_submodel(prefix(GAM.smooth_prior(sm), :s_x))
    y ~ MvNormal(β0 .+ f, σ^2 * I)
end

chain = sample(my_model(y, sm), NUTS(), 1000)
```
"""
Turing.@model function GAM.smooth_prior(
    sm::GAM.SmoothMixedModel;
    sds_prior = Exponential(1.0),
    fixed_prior = Normal(0.0, 10.0),
)
    n_fixed = size(sm.Xf, 2)
    n_random = sum(size(Z, 2) for Z in sm.Zs; init = 0)

    # Sample fixed-effect coefficients (null space)
    if n_fixed > 0
        β_f ~ filldist(fixed_prior, n_fixed)
    end

    # Sample smooth SD and random effects (non-centered parameterization)
    if n_random > 0
        σ_s ~ sds_prior
        z ~ MvNormal(zeros(n_random), I)
    end

    # Evaluate: f = Xf * β_f + σ_s * Z * z
    n_obs = size(sm.Xf, 1)
    f = zeros(eltype(n_fixed > 0 ? β_f : [0.0]), n_obs)

    if n_fixed > 0
        f = f .+ sm.Xf * β_f
    end

    if n_random > 0
        offset = 0
        for Z in sm.Zs
            nz = size(Z, 2)
            z_block = z[(offset + 1):(offset + nz)]
            f = f .+ σ_s .* (Z * z_block)
            offset += nz
        end
    end

    return f
end

"""
    smooth_predictive(sm::SmoothMixedModel, Xf_new, Zs_new; sds_prior, fixed_prior)

Like `smooth_prior` but evaluates the smooth at new covariate values (for prediction).
Pass the new-data basis matrices `Xf_new` and `Zs_new` (obtained by constructing
the smooth on new data and applying smooth2random).
"""
Turing.@model function GAM.smooth_predictive(
    sm::GAM.SmoothMixedModel,
    Xf_new::AbstractMatrix,
    Zs_new::Vector{<:AbstractMatrix};
    sds_prior = Exponential(1.0),
    fixed_prior = Normal(0.0, 10.0),
)
    n_fixed = size(sm.Xf, 2)
    n_random = sum(size(Z, 2) for Z in sm.Zs; init = 0)

    if n_fixed > 0
        β_f ~ filldist(fixed_prior, n_fixed)
    end

    if n_random > 0
        σ_s ~ sds_prior
        z ~ MvNormal(zeros(n_random), I)
    end

    n_new = size(Xf_new, 1)
    f = zeros(eltype(n_fixed > 0 ? β_f : [0.0]), n_new)

    if n_fixed > 0
        f = f .+ Xf_new * β_f
    end

    if n_random > 0
        offset = 0
        for Z in Zs_new
            nz = size(Z, 2)
            z_block = z[(offset + 1):(offset + nz)]
            f = f .+ σ_s .* (Z * z_block)
            offset += nz
        end
    end

    return f
end

end # module GAMTuringExt
