# Main gam() interface

# Scale estimation: Normal, Gamma, InverseGaussian estimate scale (like mgcv)
_needs_scale_estimate(::Normal) = true
_needs_scale_estimate(::Gamma) = true
_needs_scale_estimate(::InverseGaussian) = true
_needs_scale_estimate(::Any) = false
_needs_scale_estimate(f::ExtendedFamily) = _estimates_scale(f)

# ============================================================================
# Family and link validation
# ============================================================================

const _SUPPORTED_GAM_FAMILIES = (Normal, Poisson, Binomial, Bernoulli, Gamma, InverseGaussian)

const _FAMILY_HINTS = Dict{String,String}(
    "Beta" => "Use BetaFamily() for single-parameter beta regression in gam(), " *
              "or BetaRegression() for (μ, φ) GAMLSS via gamlss().",
    "NegativeBinomial" => "Use NegBinFamily() for gam(), " *
              "or NegativeBinomialLocationScale() for (μ, σ) GAMLSS via gamlss().",
    "Exponential" => "The Exponential is a special case of Gamma(shape=1). " *
              "Use Gamma() with a log link instead.",
    "LogNormal" => "GAM models the mean on a link scale. " *
              "Use Normal() with a log link: gam(...; family=Normal(), link=LogLink()).",
    "Laplace" => "Not directly supported. Consider qgam() for robust/quantile regression.",
    "Categorical" => "Not supported. For binary outcomes use Bernoulli() or Binomial().",
)

const _GAMLSS_DIST_HINTS = Dict{String,String}(
    "Gamma" => "Use GammaLocationScale() which parameterizes Gamma as (μ=mean, σ=CV).",
    "Beta" => "Use BetaRegression() which parameterizes Beta as (μ=mean, φ=precision).",
    "NegativeBinomial" => "Use NegativeBinomialLocationScale() which parameterizes NB as (μ=mean, σ=overdispersion).",
    "InverseGaussian" => "Use InverseGaussianLocationScale() which parameterizes IG as (μ=mean, σ=CV).",
    "Exponential" => "Use GammaLocationScale() (Exponential is Gamma with shape=1).",
    "Poisson" => "Poisson has only one parameter (λ=mean). Use gam() instead, " *
                 "or NegativeBinomialLocationScale() if you need overdispersion.",
    "Binomial" => "Binomial has only one parameter (p). Use gam() instead.",
    "Bernoulli" => "Bernoulli has only one parameter (p). Use gam() instead.",
)

# Valid link–family combinations (not exhaustive, but catches clear mistakes)
# Valid link–family combinations: catch clearly incompatible pairs via dispatch
_check_link_family(::GLM.Link, ::Any) = nothing  # default: no error

_check_link_family(::LogitLink, ::Normal) =
    "LogitLink constrains μ ∈ (0,1) — did you mean IdentityLink() or LogLink()?"
_check_link_family(::LogitLink, ::Poisson) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink()?"
_check_link_family(::LogitLink, ::QuasiPoissonFamily) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink()?"
_check_link_family(::LogitLink, ::Gamma) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink() or InverseLink()?"
_check_link_family(::LogitLink, ::InverseGaussian) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink()?"
_check_link_family(::InverseLink, ::Bernoulli) =
    "InverseLink is not appropriate for binary data — did you mean LogitLink()?"
_check_link_family(::InverseLink, ::Binomial) =
    "InverseLink is not appropriate for binary data — did you mean LogitLink()?"
_check_link_family(::InverseLink, ::QuasiBinomialFamily) =
    "InverseLink is not appropriate for binary/proportion data — did you mean LogitLink()?"

"""
    _validate_gam_family(family)

Check that `family` is supported by `gam()`. Throws ArgumentError with a
helpful hint for common mistakes (e.g., using Beta() instead of BetaFamily()).
"""
function _validate_gam_family(family::UnivariateDistribution)
    if family isa Union{_SUPPORTED_GAM_FAMILIES...}
        return nothing
    end
    fname = string(nameof(typeof(family)))
    hint = get(_FAMILY_HINTS, fname, "")
    msg = "Unsupported family for gam(): $(typeof(family)).\n" *
          "Supported families: Normal(), Poisson(), Binomial(), Bernoulli(), Gamma(), InverseGaussian().\n" *
          "Extended families: NegBinFamily(), QuasiPoissonFamily(), QuasiBinomialFamily(), TweedieFamily(), BetaFamily()."
    if !isempty(hint)
        msg *= "\nHint: $hint"
    end
    throw(ArgumentError(msg))
end
_validate_gam_family(::ExtendedFamily) = nothing

"""
    _validate_link(link, family)

Check that `link` is a reasonable choice for `family`. Throws ArgumentError
for clearly incompatible combinations.
"""
function _validate_link(link::GLM.Link, family)
    msg = _check_link_family(link, family)
    if msg !== nothing
        throw(ArgumentError(
            "Incompatible link $(nameof(typeof(link))) for family $(nameof(typeof(family))). $msg"))
    end
    return nothing
end

"""
    _validate_gamlss_family(family)

Check that a Distributions.jl type is supported for direct use in gamlss().
Only Normal() is supported directly; other distributions need reparameterized types.
"""
function _validate_gamlss_family(family::UnivariateDistribution)
    if hasmethod(_gamlss_nparams, Tuple{typeof(family)})
        return nothing
    end
    fname = string(nameof(typeof(family)))
    hint = get(_GAMLSS_DIST_HINTS, fname, "")
    msg = "Unsupported distribution for gamlss(): $(typeof(family)).\n" *
          "Only Normal() can be passed directly (its params match Distributions.jl).\n" *
          "For other distributions, use a reparameterized family type:\n" *
          "  GammaLocationScale(), BetaRegression(), NegativeBinomialLocationScale(),\n" *
          "  InverseGaussianLocationScale(), GEVFamily(), GPDFamily()."
    if !isempty(hint)
        msg *= "\nHint: $hint"
    end
    throw(ArgumentError(msg))
end

"""
    gam(formula, data; family=Normal(), link=nothing,
        method=:REML, weights=nothing, control=gam_control())

Fit a Generalized Additive Model. This is the primary entry point for all
GAM-family models — the fitting algorithm is selected automatically based
on the model specification.

If the model contains shape-constrained smooth terms (e.g., `mpi`, `cx`),
the SCAM fitting algorithm is used automatically. You can also call
[`scam`](@ref) explicitly.

If `family` is a [`MultiParameterFamily`](@ref) or [`DistFamily`](@ref),
the model is fit using the GAMLSS framework. You can also call
[`gamlss`](@ref) explicitly.

# Arguments
- `formula`: a StatsModels formula with smooth terms, e.g., `@formula(y ~ s(x))`
  or a `GamFormula` from `@formulak`.
- `data`: a table (DataFrame, NamedTuple of vectors, etc.)
- `family`: distribution family (default: `Normal()`).
  Accepts `UnivariateDistribution`, `ExtendedFamily`, `MultiParameterFamily`,
  or `DistFamily`.
- `link`: link function (default: canonical link for family)
- `method`: smoothing parameter estimation method (`:REML`, `:ML`, `:GCV`, `:UBRE`)
- `weights`: optional observation weights
- `control`: fitting control parameters (see [`gam_control`](@ref))

# Returns
A [`GamModel`](@ref) for standard/SCAM models, or a
[`MultiParameterModel`](@ref) for GAMLSS/multi-parameter models.

# Examples
```julia
using GAM, DataFrames

# Gaussian GAM with TPRS smooth
n = 200
x = range(0, 2π; length=n)
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)
m = gam(@formula(y ~ s(x)), df)

# Poisson GAM
counts = rand(Poisson(5), n)
df.count = counts
m2 = gam(@formula(count ~ s(x)), df; family=Poisson(), link=LogLink())

# Shape-constrained (auto-SCAM): monotone increasing
m3 = gam(@formulak(y ~ s(x, bs=:mpi)), df)

# Multi-parameter GAMLSS (auto-dispatch)
m4 = gam(@formulak(y ~ s(x)), df, GammaLocationScale())

# Multiple smooths
df.x2 = randn(n)
m5 = gam(@formula(y ~ s(x) + s(x2)), df)
```
"""
function gam(f::FormulaTerm, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    optimizer::Symbol = :pirls,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing,
    select::Bool = false,
    start::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    # Input validation
    _validate_data_lengths(data)
    _validate_gam_family(family)
    link_eff = if family isa ExtendedFamily
        link === nothing ? _default_link(family) : link
    else
        link === nothing ? GLM.canonicallink(family) : link
    end
    _validate_link(link_eff, family)

    # Bayesian dispatch: when priors are provided, use Turing backend
    if priors !== nothing
        return _fit_gam_bayes(f, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains,
            weights = weights)
    end

    method in (:REML, :ML, :GCV, :UBRE) ||
        throw(ArgumentError("method must be :REML, :ML, :GCV, or :UBRE, got :$method"))
    optimizer in (:pirls, :general) ||
        throw(ArgumentError("optimizer must be :pirls or :general, got :$optimizer"))

    if family isa ExtendedFamily
        y, X, X_para, smooths, n_parametric = setup_gam(f, data; family = Normal())
        _validate_response(y, family)
        return _fit_gam_extended(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, weights, control; start = start, offset = offset, select = select)
    else
        y, X, X_para, smooths, n_parametric = setup_gam(f, data; family = family)
        _validate_response(y, family)

        if has_linear_constraints(smooths)
            _reject_select(select, "shape/side-constrained (scasm)")
            return _fit_scasm(y, X, smooths, n_parametric, f, data, family, link_eff,
                method, weights, control;
                start = start, offset = offset)
        end

        # Auto-detect shape constraints → use SCAM fitting
        if has_shape_constraints(smooths)
            _reject_select(select, "shape-constrained (scam)")
            return _fit_scam(y, X, smooths, n_parametric, f, data, family, link_eff,
                method, weights, scam_control(
                    epsilon = control.epsilon,
                    maxit = control.maxit,
                    outer_maxit = control.outer_maxit,
                    trace = control.trace,
                    gamma = control.gamma,
                ); offset = offset)
        end

        return _fit_gam(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, optimizer, weights, control;
            offset = offset, select = select, start = start)
    end
end

# select=TRUE (double-penalty term selection) is implemented for ordinary and
# extended-family GAMs; the constrained (SCAM/SCASM) solvers do not support it.
function _reject_select(select, what)
    if select
        throw(ArgumentError("select=true is not supported for $what models"))
    end
end

function gam(gf::GamFormula, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    optimizer::Symbol = :pirls,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing,
    select::Bool = false,
    start::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    # Input validation
    _validate_data_lengths(data)
    _validate_response_in_data(gf.response, data)
    _validate_has_smooths(gf.smooth_specs)
    _validate_formula_smooths(gf.smooth_specs, data)
    _validate_gam_family(family)
    link_eff = if family isa ExtendedFamily
        link === nothing ? _default_link(family) : link
    else
        link === nothing ? GLM.canonicallink(family) : link
    end
    _validate_link(link_eff, family)

    # Bayesian dispatch: when priors are provided, use Turing backend
    if priors !== nothing
        f = term(gf.response) ~ term(1)
        return _fit_gam_bayes(f, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains,
            weights = weights, gam_formula = gf)
    end

    method in (:REML, :ML, :GCV, :UBRE) ||
        throw(ArgumentError("method must be :REML, :ML, :GCV, or :UBRE, got :$method"))
    optimizer in (:pirls, :general) ||
        throw(ArgumentError("optimizer must be :pirls or :general, got :$optimizer"))

    if family isa ExtendedFamily
        y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = Normal())
        _validate_response(y, family)
        return _fit_gam_extended(y, X, smooths, n_parametric, gf, data, family, link_eff,
            method, weights, control; start = start, offset = offset, select = select)
    else
        y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = family)
        _validate_response(y, family)

        if has_linear_constraints(smooths)
            _reject_select(select, "shape/side-constrained (scasm)")
            return _fit_scasm(y, X, smooths, n_parametric, gf, data, family, link_eff,
                method, weights, control;
                start = start, offset = offset)
        end

        # Auto-detect shape constraints → use SCAM fitting
        if has_shape_constraints(smooths)
            _reject_select(select, "shape-constrained (scam)")
            return _fit_scam(y, X, smooths, n_parametric, gf, data, family, link_eff,
                method, weights, scam_control(
                    epsilon = control.epsilon,
                    maxit = control.maxit,
                    outer_maxit = control.outer_maxit,
                    trace = control.trace,
                    gamma = control.gamma,
                ); offset = offset)
        end

        return _fit_gam(y, X, smooths, n_parametric, gf, data, family, link_eff,
            method, optimizer, weights, control;
            offset = offset, select = select, start = start)
    end
end

# ============================================================================
# gam() dispatch for multi-parameter families (GAMLSS)
# ============================================================================

"""
    gam(formula, data, family::MultiParameterFamily; kwargs...)

Fit a GAMLSS model via `gam()`. When a single formula is provided, it is
replicated for all distribution parameters. Delegates to [`gamlss`](@ref).

# Examples
```julia
m = gam(@formulak(y ~ s(x)), df, GammaLocationScale())
m = gam([@formulak(y ~ s(x)), @formulak(y ~ 1)], df, BetaRegression())
```
"""
function gam(f::Union{FormulaTerm, GamFormula}, data, family::MultiParameterFamily; kwargs...)
    K = nparams(family)
    formulas = fill(f, K)
    return gamlss(formulas, data, family; kwargs...)
end

function gam(formulas::AbstractVector, data, family::MultiParameterFamily; kwargs...)
    return gamlss(formulas, data, family; kwargs...)
end

# Positional family/link convenience (as shown in the README):
#   gam(f, df, Poisson()), gam(f, df, Poisson(), LogLink())
function gam(f::Union{FormulaTerm, GamFormula}, data,
    family::Union{UnivariateDistribution, ExtendedFamily},
    link::Union{GLM.Link, Nothing} = nothing; kwargs...)
    return gam(f, data; family = family, link = link, kwargs...)
end

function _fit_gam(y, X, smooths, n_parametric, f, data,
    family, link, method, optimizer, weights, control;
    offset = nothing, select::Bool = false,
    start::Union{AbstractVector{<:Real}, Nothing} = nothing)
    n, p = size(X)

    # Weights
    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))
    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch(
        "offset length $(length(off)) ≠ data length $n"))
    start_f = start === nothing ? nothing : Float64.(start)
    start_f === nothing || length(start_f) == p || throw(DimensionMismatch(
        "start length $(length(start_f)) ≠ number of coefficients $p"))

    # Setup penalties
    penalty = setup_penalties(smooths, n_parametric; select = select)

    # Initialize smoothing parameters (mgcv's initial.sp heuristic)
    _initial_sp(X, penalty)

    # Outer iteration: optimize smoothing parameters
    if optimizer == :general
        start_f === nothing || throw(ArgumentError(
            "start= is not supported with optimizer = :general"))
        log_sp, result, outer_converged, outer_iters =
            outer_iteration_general(X, y, smooths, penalty, family, link;
                method = method, weights = wts, offset = off, control = control)
    else
        log_sp, result, outer_converged, outer_iters =
            outer_iteration(X, y, smooths, penalty, family, link;
                method = method, weights = wts, offset = off,
                start = start_f, control = control)
    end

    if !result.converged
        @warn "GAM fit did not fully converge: P-IRLS reached its iteration " *
              "limit at the final smoothing parameters. Estimates may be " *
              "unreliable; consider increasing maxit via gam_control()."
    elseif !outer_converged && control.outer_maxit > 0
        @warn "GAM fit did not fully converge: the smoothing parameter " *
              "iteration reached outer_maxit with the parameters still " *
              "changing. Consider increasing outer_maxit via gam_control()."
    end

    # Compute per-smooth EDF
    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    # Bayesian covariance (Vp) and frequentist covariance (Ve)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X' * Diagonal(result.working_weights) * X
    A = XtWX + S_total
    A_chol = _protected_cholesky!(A)
    Vp = inv(A_chol)
    # Frequentist covariance Ve = F*Vp (mgcv: Ve <- F %*% Vb), F = (X'WX+S)^-1 X'WX
    F = Vp * XtWX
    Ve = Symmetric(F * Vp) |> Matrix

    # Scale parameter — estimated for Normal, Gamma, InverseGaussian (like
    # mgcv), using the estimator selected by control.scale_est
    if _needs_scale_estimate(family)
        scale_est = _estimate_scale(family, y, result.fitted_values, wts,
            result.pearson, result.deviance, n, edf_total_val,
            control.scale_est)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    # Null deviance (intercept + offset model when an offset is present)
    null_dev = _null_deviance(family, link, y, wts, off, control)

    # REML score (only the score is needed here — skip the gradient)
    reml_val, _ = reml_score(X, y, penalty, log_sp, family, link,
        wts, result; method = method, gamma = control.gamma,
        compute_gradient = false)

    return GamModel(
        f,
        y, X,
        result.coefficients,
        result.fitted_values,
        result.linear_predictor,
        wts,
        family, link,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total_val,
        scale_est,
        result.deviance,
        null_dev,
        reml_val,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged && outer_converged,
        outer_iters,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end

"""
    _estimate_scale(family, y, mu, wts, pearson, dev, n, edf, method) -> Float64

Scale (dispersion) estimator for the reported scale and covariance scaling.
- `:pearson` — Pearson X² / (n − edf)
- `:deviance` — deviance / (n − edf)
- `:fletcher` — Fletcher (2012) corrected Pearson (mgcv's default):
  φ̂_P/(1 + s̄), s̄ = mean(V′(μ)(y − μ)/V(μ))
"""
function _estimate_scale(family, y, mu, wts, pearson::Float64, dev::Float64,
    n::Int, edf::Float64, method::Symbol)
    denom = max(n - edf, 1.0)
    method === :deviance && return max(dev / denom, 1e-10)
    phi = pearson / denom
    if method === :fletcher
        s_bar = 0.0
        @inbounds for i in eachindex(y)
            vm = _variance_scalar(family, mu[i])
            s_bar += _dvariance_scalar_mu(family, mu[i]) * (y[i] - mu[i]) /
                     max(vm, eps())
        end
        s_bar /= length(y)
        if isfinite(s_bar) && 1.0 + s_bar > 0
            phi /= 1.0 + s_bar
        end
    end
    return max(phi, 1e-10)
end

# Convenience: fit(GamModel, formula, data; ...)
function StatsAPI.fit(::Type{GamModel}, f::FormulaTerm, data; kwargs...)
    return gam(f, data; kwargs...)
end

function StatsAPI.fit(::Type{GamModel}, gf::GamFormula, data; kwargs...)
    return gam(gf, data; kwargs...)
end

# Null model mean is the weighted mean (the MLE under prior weights,
# including binomial trial counts).
_weighted_mean(y, wt) = sum(wt .* y) / sum(wt)

"""
    _null_deviance(family, link, y, wt, off, control)

Null deviance. With a nonzero offset, the null model is the intercept-plus-
offset model (as in mgcv/R's glm), fit by an intercept-only P-IRLS; without
one, the closed-form weighted-mean null model applies.
"""
function _null_deviance(family, link::GLM.Link, y, wt, off, control::GamControl)
    if all(==(0.0), off)
        return _null_deviance(family, y, wt)
    end
    X1 = ones(length(y), 1)
    S0 = zeros(1, 1)
    r = family isa ExtendedFamily ?
        pirls_extended(X1, y, S0, family, link;
            weights = wt, offset = off, control = control) :
        pirls(X1, y, S0, family, link;
            weights = wt, offset = off, control = control)
    return r.deviance
end

function _null_deviance(family::Normal, y, wt)
    mu = _weighted_mean(y, wt)
    return sum(wt .* (y .- mu) .^ 2)
end

function _null_deviance(family::BinomialLike, y, wt)
    mu = _weighted_mean(y, wt)
    mu = clamp(mu, eps(), 1 - eps())
    return _deviance(family, y, fill(mu, length(y)), wt)
end

function _null_deviance(family::Poisson, y, wt)
    mu = _weighted_mean(y, wt)
    mu = max(mu, eps())
    return _deviance(family, y, fill(mu, length(y)), wt)
end

function _null_deviance(family::UnivariateDistribution, y, wt)
    mu = _weighted_mean(y, wt)
    return _deviance(family, y, fill(mu, length(y)), wt)
end

# ============================================================================
# Extended family fitting
# ============================================================================

function _fit_gam_extended(y, X, smooths, n_parametric, f, data,
    family::ExtendedFamily, link::GLM.Link, method, weights, control;
    start::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset = nothing, select::Bool = false)
    n, p = size(X)

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))
    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch(
        "offset length $(length(off)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric; select = select)
    _initial_sp(X, penalty)
    Ain, bin, Aeq, beq = _global_linear_constraints(smooths, p)

    log_sp, result, outer_converged, outer_iters =
        outer_iteration(X, y, smooths, penalty, family, link;
            method = method, weights = wts, offset = off, control = control,
            start = start === nothing ? nothing : Float64.(start),
            Ain = Ain, bin = bin, Aeq = Aeq, beq = beq)

    if !result.converged
        @warn "GAM fit did not fully converge: P-IRLS reached its iteration " *
              "limit at the final smoothing parameters. Estimates may be " *
              "unreliable; consider increasing maxit via gam_control()."
    elseif !outer_converged && control.outer_maxit > 0
        @warn "GAM fit did not fully converge: the smoothing parameter " *
              "iteration reached outer_maxit with the parameters still " *
              "changing. Consider increasing outer_maxit via gam_control()."
    end

    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X' * Diagonal(result.working_weights) * X
    A = XtWX + S_total
    A_chol = _protected_cholesky!(copy(A))
    Vp = inv(A_chol)
    F = Vp * XtWX
    Ve = Symmetric(F * Vp) |> Matrix

    # Scale parameter
    if _estimates_scale(family)
        scale_est = max(result.pearson / (n - edf_total_val), 1e-10)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    null_dev = _null_deviance(family, link, y, wts, off, control)

    # REML/LAML score at the converged fit (the criterion the EFS iteration
    # targets, with ls = 0: extended-family deviance is already −2(ll−ll_sat),
    # so the score omits the family's saturated-likelihood constant).
    reml_val = begin
        Dp = result.deviance + dot(result.coefficients, S_total * result.coefficients)
        Mp = p - sum(b.rank for b in penalty.blocks; init = 0)
        Dp / (2 * scale_est) + 0.5 * logdet(A_chol) -
            0.5 * _log_penalty_det(penalty, log_sp) -
            0.5 * Mp * log(2π * scale_est)
    end

    return GamModel(
        f,
        y, X,
        result.coefficients,
        result.fitted_values,
        result.linear_predictor,
        wts,
        family, link,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total_val,
        scale_est,
        result.deviance,
        null_dev,
        reml_val,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged && outer_converged,
        outer_iters,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end
