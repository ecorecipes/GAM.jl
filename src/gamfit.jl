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
_check_link_family(::LogitLink, ::Gamma) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink() or InverseLink()?"
_check_link_family(::LogitLink, ::InverseGaussian) =
    "LogitLink constrains μ ∈ (0,1) — did you mean LogLink()?"
_check_link_family(::InverseLink, ::Bernoulli) =
    "InverseLink is not appropriate for binary data — did you mean LogitLink()?"
_check_link_family(::InverseLink, ::Binomial) =
    "InverseLink is not appropriate for binary data — did you mean LogitLink()?"

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
          "Extended families: NegBinFamily(), TweedieFamily(), BetaFamily()."
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
    gam(formula::FormulaTerm, data; family=Normal(), link=nothing,
        method=:REML, weights=nothing, control=gam_control())

Fit a Generalized Additive Model.

# Arguments
- `formula`: a StatsModels formula with smooth terms, e.g., `@formula(y ~ s(x))`
- `data`: a table (DataFrame, NamedTuple of vectors, etc.)
- `family`: distribution family (default: `Normal()`)
- `link`: link function (default: canonical link for family)
- `method`: smoothing parameter estimation method (`:REML`, `:ML`, `:GCV`, `:UBRE`)
- `weights`: optional observation weights
- `control`: fitting control parameters (see [`gam_control`](@ref))

# Returns
A [`GamModel`](@ref) object implementing the StatsBase interface.

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

# Multiple smooths
df.x2 = randn(n)
m3 = gam(@formula(y ~ s(x) + s(x2)), df)
```
"""
function gam(f::FormulaTerm, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    optimizer::Symbol = :pirls,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

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
        return _fit_gam_extended(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, weights, control)
    else
        y, X, X_para, smooths, n_parametric = setup_gam(f, data; family = family)
        return _fit_gam(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, optimizer, weights, control)
    end
end

function gam(gf::GamFormula, data;
    family::Union{UnivariateDistribution, ExtendedFamily} = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    optimizer::Symbol = :pirls,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

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
        f = term(gf.response) ~ term(1)
        return _fit_gam_extended(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, weights, control)
    else
        y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = family)
        f = term(gf.response) ~ term(1)
        return _fit_gam(y, X, smooths, n_parametric, f, data, family, link_eff,
            method, optimizer, weights, control)
    end
end

function _fit_gam(y, X, smooths, n_parametric, f, data,
    family, link, method, optimizer, weights, control)
    n, p = size(X)

    # Weights
    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))

    # Setup penalties
    penalty = setup_penalties(smooths, n_parametric)

    # Initialize smoothing parameters (mgcv's initial.sp heuristic)
    _initial_sp(X, penalty)

    # Outer iteration: optimize smoothing parameters
    if optimizer == :general
        log_sp, result = outer_iteration_general(X, y, smooths, penalty, family, link;
            method = method, weights = wts, control = control)
    else
        log_sp, result = outer_iteration(X, y, smooths, penalty, family, link;
            method = method, weights = wts, control = control)
    end

    # Compute per-smooth EDF
    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    # Bayesian covariance (Vp) and frequentist covariance (Ve)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X' * Diagonal(result.working_weights) * X
    A = XtWX + S_total
    A_chol = cholesky(Symmetric(A))
    Vp = inv(A_chol)
    F = Vp * XtWX
    Ve = F * Vp  # sandwich: F * Vp * F' but since Vp is symmetric...
    Ve = Symmetric(F * Vp * F') |> Matrix

    # Scale parameter — estimated for Normal, Gamma, InverseGaussian (like mgcv)
    if _needs_scale_estimate(family)
        scale_est = result.pearson / (n - edf_total_val)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    # Null deviance
    null_dev = _null_deviance(family, y, wts)

    # REML score
    reml_val, _ = reml_score(X, y, penalty, log_sp, family, link,
        wts, result; method = method, gamma = control.gamma)

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
        result.converged,
        0,  # outer iterations tracked in outer_iteration
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end

# Convenience: fit(GamModel, formula, data; ...)
function StatsAPI.fit(::Type{GamModel}, f::FormulaTerm, data; kwargs...)
    return gam(f, data; kwargs...)
end

function StatsAPI.fit(::Type{GamModel}, gf::GamFormula, data; kwargs...)
    return gam(gf, data; kwargs...)
end

function _null_deviance(family::Normal, y, wt)
    mu = mean(y)
    return sum(wt .* (y .- mu) .^ 2)
end

function _null_deviance(family::BinomialLike, y, wt)
    mu = mean(y)
    mu = clamp(mu, eps(), 1 - eps())
    return _deviance(family, y, fill(mu, length(y)), wt)
end

function _null_deviance(family::Poisson, y, wt)
    mu = mean(y)
    mu = max(mu, eps())
    return _deviance(family, y, fill(mu, length(y)), wt)
end

function _null_deviance(family::UnivariateDistribution, y, wt)
    mu = mean(y)
    return _deviance(family, y, fill(mu, length(y)), wt)
end

# ============================================================================
# Extended family fitting
# ============================================================================

function _fit_gam_extended(y, X, smooths, n_parametric, f, data,
    family::ExtendedFamily, link::GLM.Link, method, weights, control)
    n, p = size(X)

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric)
    _initial_sp(X, penalty)

    log_sp, result = outer_iteration(X, y, smooths, penalty, family, link;
        method = method, weights = wts, control = control)

    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    S_total = total_penalty(penalty, log_sp, p)
    XtWX = X' * Diagonal(result.working_weights) * X
    A = XtWX + S_total
    A_chol = cholesky(Symmetric(A))
    Vp = inv(A_chol)
    F = Vp * XtWX
    Ve = Symmetric(F * Vp * F') |> Matrix

    # Scale parameter
    if _estimates_scale(family)
        scale_est = max(result.pearson / (n - edf_total_val), 1e-10)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    null_dev = _null_deviance(family, y, wts)

    # Simplified REML score for extended families
    reml_val = result.deviance / 2.0

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
        result.converged,
        0,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end
