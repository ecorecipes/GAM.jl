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
    X_para, _ = _build_parametric_matrix(formula, Tables.columntable(data))

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

    cols = Tables.columntable(data)
    X_para, _ = _build_parametric_matrix(formula, cols)
    smooth_terms, _ = _split_formula_terms(formula)
    smooths = SmoothMixedModel[]
    labels = String[]

    for term in smooth_terms
        if term isa AppliedSmoothTerm || term isa SmoothTerm
            spec = term isa SmoothTerm ? term.spec : term.spec
            sm = smooth_construct(spec, cols, knots)
            smm = smooth2random(sm)
            push!(smooths, smm)
            push!(labels, spec.label)
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

# Stubs for Turing extension @model functions — implemented in GAMTuringExt
"""
    smooth_prior(sm::SmoothMixedModel; sds_prior, fixed_prior)

A Turing `@model` for sampling the parameters of a single smooth term.
Returns the evaluated smooth function values (vector of length n).

Use with `to_submodel` and `prefix` to compose smooth terms into custom Bayesian models:

```julia
import GAM
using Turing

sm = GAM.gam_smooth(:x, data; k=10)
@model function my_gam(y, sm)
    β0 ~ Normal(0, 10)
    σ ~ Exponential(1.0)
    f ~ to_submodel(prefix(GAM.smooth_prior(sm), :s_x))
    y ~ MvNormal(β0 .+ f, σ^2 * I)
end
```

Requires `using Turing` to load the implementation.
"""
function smooth_prior end

"""
    smooth_predictive(sm, Xf_new, Zs_new; sds_prior, fixed_prior)

Like `smooth_prior` but evaluates the smooth at new covariate values.
Requires `using Turing`.
"""
function smooth_predictive end

"""
    BayesGamModel

A Bayesian GAM fitted via MCMC (Turing.jl). Returned by `gam()`, `gamlss()`,
`scam()`, or `gamm()` when a `priors` keyword argument is provided.

Implements posterior-summary methods such as `coef`, `vcov`, `coeftable`,
and `confint`, and stores pointwise log-likelihood draws for Bayesian
model scoring via [`waic`](@ref) and [`loo`](@ref).

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
- `loglik_obs`: posterior draw × observation matrix of pointwise log-likelihoods
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
    loglik_obs::Union{Matrix{Float64}, Nothing}
end

function BayesGamModel(
    formula,
    family,
    link,
    smooths_info,
    chains,
    coef_names,
    smooth_labels,
    n_parametric,
    n_smooth,
    n_obs,
    priors,
    sampler_info,
    data,
)
    return BayesGamModel(
        formula, family, link,
        smooths_info, chains,
        coef_names, smooth_labels,
        n_parametric, n_smooth,
        n_obs, priors, sampler_info, data,
        nothing,
    )
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

"""
    pointwise_loglikelihood(m::BayesGamModel)

Return the stored posterior draw × observation matrix of pointwise
log-likelihood contributions used for Bayesian model scoring.
"""
function pointwise_loglikelihood(m::BayesGamModel)
    m.loglik_obs === nothing && error(
        "Pointwise log-likelihood draws are not available for this model."
    )
    return m.loglik_obs
end

"""
    LOOResult

Summary of leave-one-out cross-validation (LOO) computed from the stored
pointwise posterior log-likelihood matrix using importance sampling or
Pareto-smoothed importance sampling (PSIS).

# Fields
- `elpd_loo`: estimated expected log pointwise predictive density under LOO
- `p_loo`: effective number of parameters implied by LOO
- `looic`: deviance-scale LOO criterion (`-2 * elpd_loo`)
- `se_elpd_loo`: standard error of `elpd_loo`
- `pointwise_elpd`: observation-wise ELPD contributions
- `pointwise_p`: observation-wise effective-parameter contributions
- `pareto_k`: Pareto shape diagnostics for each observation (`NaN` for plain IS-LOO)
- `n_eff`: estimated effective sample sizes for each observation (`NaN` for plain IS-LOO)
- `method`: scoring method (`:psis` or `:is`)
"""
struct LOOResult
    elpd_loo::Float64
    p_loo::Float64
    looic::Float64
    se_elpd_loo::Float64
    pointwise_elpd::Vector{Float64}
    pointwise_p::Vector{Float64}
    pareto_k::Vector{Float64}
    n_eff::Vector{Float64}
    method::Symbol
end

function LOOResult(
    elpd_loo::Real,
    p_loo::Real,
    looic::Real,
    se_elpd_loo::Real,
    pointwise_elpd::Vector{Float64},
    pointwise_p::Vector{Float64},
)
    n = length(pointwise_elpd)
    return LOOResult(
        Float64(elpd_loo),
        Float64(p_loo),
        Float64(looic),
        Float64(se_elpd_loo),
        pointwise_elpd,
        pointwise_p,
        fill(NaN, n),
        fill(NaN, n),
        :is,
    )
end

"""
    PSISKDiagnostic

Summary of Pareto-k diagnostics from PSIS-LOO.

# Fields
- `pareto_k`: Pareto shape value for each observation
- `n_eff`: effective sample-size estimate for each observation
- `warning_indices`: observations with `pareto_k > 0.7`
- `danger_indices`: observations with `pareto_k > 1.0`
"""
struct PSISKDiagnostic
    pareto_k::Vector{Float64}
    n_eff::Vector{Float64}
    warning_indices::Vector{Int}
    danger_indices::Vector{Int}
end

"""
    WAICResult

Summary of the widely applicable information criterion (WAIC) for a Bayesian
model fit.

# Fields
- `elpd_waic`: estimated expected log pointwise predictive density
- `p_waic`: effective number of parameters
- `waic`: deviance-scale WAIC (`-2 * elpd_waic`)
- `se_elpd_waic`: standard error of `elpd_waic`
- `pointwise_elpd`: observation-wise ELPD contributions
- `pointwise_p`: observation-wise effective-parameter contributions
"""
struct WAICResult
    elpd_waic::Float64
    p_waic::Float64
    waic::Float64
    se_elpd_waic::Float64
    pointwise_elpd::Vector{Float64}
    pointwise_p::Vector{Float64}
end

function _logmeanexp(x::AbstractVector{<:Real})
    isempty(x) && throw(ArgumentError("logmeanexp requires at least one value"))
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax))) - log(length(x))
end

function _score_loglik_input(m::BayesGamModel, score_name::AbstractString)
    loglik = pointwise_loglikelihood(m)
    n_draws, n_obs = size(loglik)
    n_draws > 0 || throw(ArgumentError("$score_name requires at least one posterior draw"))
    return loglik, n_draws, n_obs
end

_score_vector(x::AbstractArray) = vec(Float64.(x))
_score_vector(x::Real) = Float64[x]

function _psis_score_arrays(loglik::Matrix{Float64}; reff::Real=1.0, warn::Bool=false)
    all(isfinite, loglik) || throw(ArgumentError(
        "PSIS-LOO requires finite pointwise log-likelihood draws."
    ))
    n_draws, n_obs = size(loglik)
    log_ratios = reshape(-loglik, n_draws, 1, n_obs)
    result = PSIS.psis(log_ratios, reff; normalize = false, warn = warn)
    pareto_k = _score_vector(result.pareto_shape)
    n_eff = _score_vector(result.ess)
    return result, pareto_k, n_eff
end

function _loo_is(loglik::Matrix{Float64})
    _, n_obs = size(loglik)
    pointwise_lppd = Vector{Float64}(undef, n_obs)
    pointwise_elpd = Vector{Float64}(undef, n_obs)
    pointwise_p = Vector{Float64}(undef, n_obs)

    @inbounds for i in 1:n_obs
        ll_i = view(loglik, :, i)
        pointwise_lppd[i] = _logmeanexp(ll_i)
        pointwise_elpd[i] = -_logmeanexp(-ll_i)
        pointwise_p[i] = pointwise_lppd[i] - pointwise_elpd[i]
    end

    elpd_loo = sum(pointwise_elpd)
    p_loo = sum(pointwise_p)
    looic = -2 * elpd_loo
    se_elpd = n_obs > 1 ? sqrt(n_obs * var(pointwise_elpd; corrected = true)) : 0.0

    return LOOResult(elpd_loo, p_loo, looic, se_elpd, pointwise_elpd, pointwise_p)
end

"""
    psis_loo(m::BayesGamModel; reff=1.0, warn=false) -> LOOResult

Compute leave-one-out cross-validation using Pareto-smoothed importance
sampling (PSIS-LOO).
"""
function psis_loo(m::BayesGamModel; reff::Real=1.0, warn::Bool=false)
    loglik, _, n_obs = _score_loglik_input(m, "PSIS-LOO")
    psis_result, pareto_k, n_eff = _psis_score_arrays(loglik; reff = reff, warn = warn)

    pointwise_lppd = Vector{Float64}(undef, n_obs)
    pointwise_elpd = Vector{Float64}(undef, n_obs)
    pointwise_p = Vector{Float64}(undef, n_obs)

    @inbounds for i in 1:n_obs
        ll_i = view(loglik, :, i)
        lw_i = view(psis_result.log_weights, :, 1, i)
        pointwise_lppd[i] = _logmeanexp(ll_i)
        pointwise_elpd[i] = -_logmeanexp(lw_i)
        pointwise_p[i] = pointwise_lppd[i] - pointwise_elpd[i]
    end

    elpd_loo = sum(pointwise_elpd)
    p_loo = sum(pointwise_p)
    looic = -2 * elpd_loo
    se_elpd = n_obs > 1 ? sqrt(n_obs * var(pointwise_elpd; corrected = true)) : 0.0

    return LOOResult(
        elpd_loo,
        p_loo,
        looic,
        se_elpd,
        pointwise_elpd,
        pointwise_p,
        pareto_k,
        n_eff,
        :psis,
    )
end

"""
    loo(m::BayesGamModel; method=:psis, reff=1.0, warn=false) -> LOOResult

Compute approximate leave-one-out cross-validation from the stored pointwise
posterior log-likelihood matrix using Pareto-smoothed importance sampling by
default. Use `method=:is` for the older raw importance-sampling estimate.
"""
function loo(m::BayesGamModel; method::Symbol=:psis, reff::Real=1.0, warn::Bool=false)
    loglik, _, _ = _score_loglik_input(m, "LOO")
    if method == :psis
        return psis_loo(m; reff = reff, warn = warn)
    elseif method == :is
        return _loo_is(loglik)
    end
    throw(ArgumentError("Unsupported LOO method: $method. Use :psis or :is."))
end

"""
    pareto_k_diagnostic(x::Union{BayesGamModel, LOOResult}; threshold=0.7, danger=1.0, reff=1.0, warn=false)

Summarize Pareto-k diagnostics from a PSIS-LOO computation.
"""
function pareto_k_diagnostic(
    x::Union{BayesGamModel, LOOResult};
    threshold::Real=0.7,
    danger::Real=1.0,
    reff::Real=1.0,
    warn::Bool=false,
)
    result = x isa BayesGamModel ? psis_loo(x; reff = reff, warn = warn) : x
    result.method == :psis || throw(ArgumentError(
        "Pareto-k diagnostics require a PSIS-LOO result. Recompute with `psis_loo(...)` or `loo(...; method=:psis)`."
    ))
    warning_indices = findall(k -> isfinite(k) && k > threshold, result.pareto_k)
    danger_indices = findall(k -> isfinite(k) && k > danger, result.pareto_k)
    return PSISKDiagnostic(result.pareto_k, result.n_eff, warning_indices, danger_indices)
end

"""
    waic(m::BayesGamModel) -> WAICResult

Compute WAIC from the stored pointwise posterior log-likelihood matrix.
"""
function waic(m::BayesGamModel)
    loglik, n_draws, n_obs = _score_loglik_input(m, "WAIC")

    pointwise_lppd = Vector{Float64}(undef, n_obs)
    pointwise_p = Vector{Float64}(undef, n_obs)
    pointwise_elpd = Vector{Float64}(undef, n_obs)

    @inbounds for i in 1:n_obs
        ll_i = view(loglik, :, i)
        pointwise_lppd[i] = _logmeanexp(ll_i)
        pointwise_p[i] = n_draws > 1 ? var(ll_i; corrected = true) : 0.0
        pointwise_elpd[i] = pointwise_lppd[i] - pointwise_p[i]
    end

    elpd_waic = sum(pointwise_elpd)
    p_waic = sum(pointwise_p)
    waic_val = -2 * elpd_waic
    se_elpd = n_obs > 1 ? sqrt(n_obs * var(pointwise_elpd; corrected = true)) : 0.0

    return WAICResult(elpd_waic, p_waic, waic_val, se_elpd, pointwise_elpd, pointwise_p)
end

function Base.show(io::IO, ::MIME"text/plain", l::LOOResult)
    label = l.method == :psis ? "PSIS-LOO" : "LOO"
    println(io, label)
    @printf(io, "  elpd_loo   %12.4f\n", l.elpd_loo)
    @printf(io, "  p_loo      %12.4f\n", l.p_loo)
    @printf(io, "  looic      %12.4f\n", l.looic)
    @printf(io, "  se_elpd    %12.4f\n", l.se_elpd_loo)
    if l.method == :psis
        @printf(io, "  k>0.7      %12d\n", count(k -> isfinite(k) && k > 0.7, l.pareto_k))
        @printf(io, "  k>1.0      %12d\n", count(k -> isfinite(k) && k > 1.0, l.pareto_k))
    end
end

function Base.show(io::IO, l::LOOResult)
    print(io, "LOOResult(")
    @printf(io, "elpd_loo=%.4f, p_loo=%.4f, looic=%.4f, method=%s",
        l.elpd_loo, l.p_loo, l.looic, string(l.method))
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", d::PSISKDiagnostic)
    println(io, "Pareto-k diagnostic")
    @printf(io, "  k>0.7      %12d\n", length(d.warning_indices))
    @printf(io, "  k>1.0      %12d\n", length(d.danger_indices))
end

function Base.show(io::IO, d::PSISKDiagnostic)
    print(io, "PSISKDiagnostic(")
    @printf(io, "warnings=%d, danger=%d", length(d.warning_indices), length(d.danger_indices))
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", w::WAICResult)
    println(io, "WAIC")
    @printf(io, "  elpd_waic  %12.4f\n", w.elpd_waic)
    @printf(io, "  p_waic     %12.4f\n", w.p_waic)
    @printf(io, "  waic       %12.4f\n", w.waic)
    @printf(io, "  se_elpd    %12.4f\n", w.se_elpd_waic)
end

function Base.show(io::IO, w::WAICResult)
    print(io, "WAICResult(")
    @printf(io, "elpd_waic=%.4f, p_waic=%.4f, waic=%.4f", w.elpd_waic, w.p_waic, w.waic)
    print(io, ")")
end

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
