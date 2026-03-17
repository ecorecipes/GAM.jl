# BayesGamModel — result type for Bayesian GAM fits
#
# Returned by gam(...; priors=PriorSpec(...)) when Turing.jl is loaded.
# Implements the StatsBase interface for posterior summaries.

# ============================================================================
# Convenience: build mixed-model matrices from formula + data
# (Placed here because it depends on GamFormula from formula.jl)
# ============================================================================

"""
    gam_matrices(formula, data; kwargs...) -> (X_parametric, smooths, smooth_labels)

Build the parametric design matrix and mixed-model smooth components from
a GAM formula and data. This is the main building block for embedding GAM
smooth terms in custom Turing.jl `@model` definitions.

Returns:
- `X_parametric`: parametric terms design matrix (including intercept)
- `smooths`: vector of `SmoothMixedModel`, one per smooth term
- `smooth_labels`: vector of string labels

# Example
```julia
X, sms, labels = gam_matrices(@gam_formula(y ~ x1 + s(x2, k=20)), data)
# X is (n × 2) — intercept + x1
# sms[1].Xf is (n × 1) — s(x2) null space
# sms[1].Zs[1] is (n × 18) — s(x2) penalized wiggle
```
"""
function gam_matrices(formula::GamFormula, data;
    knots::Union{Nothing, Dict} = nothing)

    # Build parametric part
    n = Tables.rowcount(data)
    X_para = ones(n, 1)  # intercept
    if formula.has_intercept
        para_names = ["(Intercept)"]
    else
        X_para = Matrix{Float64}(undef, n, 0)
        para_names = String[]
    end

    for v in formula.parametric
        col = Float64.(Tables.getcolumn(data, v))
        X_para = hcat(X_para, col)
        push!(para_names, string(v))
    end

    # Build smooth terms
    smooths = SmoothMixedModel[]
    labels = String[]

    for spec in formula.smooth_specs
        sm = smooth_construct(spec, data, knots)
        smm = smooth2random(sm)
        push!(smooths, smm)
        push!(labels, spec.label)
    end

    return X_para, smooths, labels
end

"""
    gam_matrices(formula::FormulaTerm, data; knots=nothing)

Build mixed-model matrices from a StatsModels `@formula`. Detects smooth
terms (FunctionTerm{typeof(s)} etc.) and converts them to SmoothMixedModel.
"""
function gam_matrices(formula::FormulaTerm, data;
    knots::Union{Nothing, Dict} = nothing)

    n = Tables.rowcount(data)
    cols = Tables.columntable(data)

    X_para = ones(n, 1)
    para_names = ["(Intercept)"]

    # Flatten RHS and separate smooths from parametric
    rhs_terms = _flatten_rhs(formula.rhs)
    smooths = SmoothMixedModel[]
    labels = String[]

    for term in rhs_terms
        if term isa AppliedSmoothTerm || term isa SmoothTerm
            spec = term isa SmoothTerm ? term.spec : term.spec
            sm = smooth_construct(spec, cols, knots)
            smm = smooth2random(sm)
            push!(smooths, smm)
            push!(labels, spec.label)
        elseif term isa StatsModels.FunctionTerm && _is_smooth_function(term.f)
            spec = _functionterm_to_smoothspec(term)
            sm = smooth_construct(spec, cols, knots)
            smm = smooth2random(sm)
            push!(smooths, smm)
            push!(labels, spec.label)
        elseif term isa Term
            col = Float64.(Tables.getcolumn(cols, term.sym))
            X_para = hcat(X_para, col)
            push!(para_names, string(term.sym))
        elseif term isa ContinuousTerm
            col = Float64.(StatsModels.modelcols(term, cols))
            X_para = hcat(X_para, col)
            push!(para_names, string(term.sym))
        end
    end

    return X_para, smooths, labels
end

"""
    gam_smooth(var::Symbol, data; bs=:tp, k=10, m=nothing, by=nothing) -> SmoothMixedModel

Convenience: build a single smooth term in mixed-model form.

# Example
```julia
smm = gam_smooth(:x, data; bs=:cr, k=20)
# smm.Xf = fixed effects (null space)
# smm.Zs[1] = random effects (penalized wiggle)
```
"""
function gam_smooth(var::Symbol, data; bs::Symbol = :tp, k::Int = 10,
    m::Union{Int, Nothing} = nothing, by::Union{Symbol, Nothing} = nothing)
    spec = s(var; bs = bs, k = k, m = m, by = by)
    sm = smooth_construct(spec, data)
    return smooth2random(sm)
end

function gam_smooth(vars::NTuple{N, Symbol}, data; bs::Symbol = :tp, k::Int = 10,
    m::Union{Int, Nothing} = nothing) where {N}
    spec = s(vars...; bs = bs, k = k, m = m)
    sm = smooth_construct(spec, data)
    return smooth2random(sm)
end

"""
    BayesGamModel

A Bayesian GAM fitted via MCMC (Turing.jl). Returned by `gam()`, `gamlss()`,
or `scam()` when a `priors` keyword argument is provided.

Implements the StatsBase interface (`coef`, `vcov`, `predict`, etc.) where
point estimates are posterior means and intervals are credible intervals.

# Fields
- `formula`: the formula used
- `family`: distribution family
- `link`: link function
- `smooths_info`: smooth term metadata (labels, mixed-model decompositions)
- `chains`: MCMC chains (MCMCChains.Chains object, stored as Any to avoid
  hard dependency)
- `coef_names`: names of all model coefficients
- `smooth_labels`: labels for smooth terms
- `n_parametric`: number of parametric coefficients
- `n_smooth`: number of smooth terms
- `n_obs`: number of observations
- `priors`: the PriorSpec used
- `sampler_info`: string describing sampler and settings
- `data`: original data (for prediction/visualization)
"""
mutable struct BayesGamModel
    formula::Any
    family::Any
    link::Any
    smooths_info::Vector{SmoothMixedModel}
    chains::Any                     # MCMCChains.Chains (Any to avoid dep)
    coef_names::Vector{String}
    smooth_labels::Vector{String}
    n_parametric::Int
    n_smooth::Int
    n_obs::Int
    priors::PriorSpec
    sampler_info::String
    data::Any
end

# ============================================================================
# StatsBase interface — posterior summaries
# ============================================================================

StatsAPI.nobs(m::BayesGamModel) = m.n_obs

const _TURING_HINT = "Turing.jl extension not loaded. Run `using Turing` first."

function StatsAPI.coef(m::BayesGamModel)
    hasmethod(_bayes_coef_means, Tuple{BayesGamModel}) || error(_TURING_HINT)
    return _bayes_coef_means(m)
end

function StatsAPI.vcov(m::BayesGamModel)
    hasmethod(_bayes_vcov, Tuple{BayesGamModel}) || error(_TURING_HINT)
    return _bayes_vcov(m)
end

function StatsAPI.coeftable(m::BayesGamModel)
    hasmethod(_bayes_coeftable, Tuple{BayesGamModel}) || error(_TURING_HINT)
    return _bayes_coeftable(m)
end

function StatsAPI.confint(m::BayesGamModel; level::Real = 0.95)
    hasmethod(_bayes_credint, Tuple{BayesGamModel}) || error(_TURING_HINT)
    return _bayes_credint(m; level = level)
end

# Declared without methods — implementations provided by GAMTuringExt.
# If the extension isn't loaded, calling these yields a MethodError.
function _bayes_coef_means end
function _bayes_vcov end
function _bayes_coeftable end
function _bayes_credint end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", m::BayesGamModel)
    println(io, "Bayesian Generalized Additive Model")
    println(io)

    if m.formula !== nothing
        println(io, "Formula: ", m.formula)
    end
    println(io, "Family:  ", _bayes_family_name(m.family))
    println(io, "Link:    ", nameof(typeof(m.link)))
    println(io, "Sampler: ", m.sampler_info)
    println(io)

    # Attempt to show coefficient table if chains are available
    if m.chains !== nothing
        try
            ct = _bayes_coeftable(m)
            println(io, "Parametric coefficients:")
            show(io, MIME("text/plain"), ct)
            println(io)
        catch
            println(io, "(Coefficient table requires Turing.jl extension)")
        end
    end

    println(io)
    if m.n_smooth > 0
        println(io, "Smooth terms: ", join(m.smooth_labels, ", "))
    end
    @printf(io, "n = %d\n", m.n_obs)
end

function Base.show(io::IO, m::BayesGamModel)
    print(io, "BayesGamModel(")
    print(io, "n_smooth=$(m.n_smooth), ")
    print(io, "n=$(m.n_obs))")
end

function _bayes_family_name(f)
    if f isa ExtendedFamily
        return _family_name(f)
    elseif f isa MultiParameterFamily
        return string(typeof(f))
    else
        return string(nameof(typeof(f)))
    end
end

# ============================================================================
# Dispatch stub — called from gam() when priors != nothing
# ============================================================================

"""
    _fit_gam_bayes(formula, data, family, link, priors; kwargs...)

Internal: fit a Bayesian GAM. This is a stub that throws an informative
error if Turing.jl is not loaded. The actual implementation is provided
by the GAMTuringExt package extension.
"""
function _fit_gam_bayes(args...; kwargs...)
    error(
        "Bayesian GAM fitting requires Turing.jl. " *
        "Please run `using Turing` before calling gam(...; priors=PriorSpec(...))."
    )
end

"""
    _fit_gamlss_bayes(args...; kwargs...)

Internal stub for Bayesian GAMLSS. Requires Turing.jl extension.
"""
function _fit_gamlss_bayes(args...; kwargs...)
    error(
        "Bayesian GAMLSS fitting requires Turing.jl. " *
        "Please run `using Turing` before calling gamlss(...; priors=PriorSpec(...))."
    )
end

"""
    _fit_scam_bayes(args...; kwargs...)

Internal stub for Bayesian SCAM. Requires Turing.jl extension.
"""
function _fit_scam_bayes(args...; kwargs...)
    error(
        "Bayesian SCAM fitting requires Turing.jl. " *
        "Please run `using Turing` before calling scam(...; priors=PriorSpec(...))."
    )
end
