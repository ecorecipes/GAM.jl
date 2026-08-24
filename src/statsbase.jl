# StatsBase / StatsAPI interface methods for GamModel

coef(m::GamModel) = m.coefficients
nobs(m::GamModel) = length(m.y)
deviance(m::GamModel) = m.deviance_val
nulldeviance(m::GamModel) = m.null_deviance
response(m::GamModel) = m.y
fitted(m::GamModel) = m.fitted_values
weights(m::GamModel) = m.weights

function _likelihood_extra_dof(family)
    extra = _needs_scale_estimate(family) ? 1.0 : 0.0
    if family isa NegBinFamily
        extra += family.estimate_theta ? 1.0 : 0.0
    elseif family isa BetaFamily
        extra += family.estimate_phi ? 1.0 : 0.0
    elseif family isa TweedieFamily
        extra += family.estimate_p ? 1.0 : 0.0
    end
    return extra
end

"""
    dof(m::GamModel)

Effective degrees of freedom for the model, including estimated nuisance
parameters such as scale or family-specific hyperparameters.
"""
dof(m::GamModel) = m.edf_total + _likelihood_extra_dof(m.family)

"""
    dof_residual(m::GamModel)

Residual degrees of freedom: n - edf_total.
"""
dof_residual(m::GamModel) = nobs(m) - m.edf_total

"""
    edf(m::GamModel)

Effective degrees of freedom per smooth term.
"""
edf(m::GamModel) = m.edf

"""
    loglikelihood(m::GamModel)

Log-likelihood of the fitted model, on the absolute scale (saturated-model
constants included), so `aic(m)` is comparable across families and
hyperparameters (e.g. two NegBin fits with different θ).

For families with a dispersion parameter (Normal, Gamma, InverseGaussian)
the plug-in φ is the **maximum-likelihood** estimate `deviance/n`, matching
R's `family$aic` convention used by both `stats::glm` and `mgcv`. This is
deliberately *not* `m.scale` (the Pearson/Fletcher estimate, `≈ pearson /
(n − edf)`), which is the right scale for *inference* — standard errors and
intervals — but not the maximiser of the likelihood the AIC is defined from.

Quasi families (QuasiPoisson, QuasiBinomial) have no true likelihood and
return `NaN` (R reports `NA` for their AIC).

See [`aic`](@ref) for how this relates to mgcv's `AIC.gam`.
"""
function loglikelihood(m::GamModel)
    dev = deviance(m)
    scale = m.scale
    y = m.y
    mu = m.fitted_values
    w = m.weights
    if m.family isa TweedieFamily
        return _tweedie_total_loglik(y, mu, w, m.family.p,
            clamp(scale, 1e-8, 1e8))
    elseif m.family isa Normal
        # R's gaussian()$aic convention: φ̂ = dev/nobs (the ML dispersion),
        # ℓ = -½[n·(log(2πφ̂) + 1) - Σ log wᵢ]
        nobs = length(y)
        return -0.5 * (nobs * (log(2π * dev / nobs) + 1) - sum(log.(w)))
    elseif m.family isa Gamma
        # R's Gamma()$aic convention: φ̂ = dev/Σwᵢ, shape 1/φ̂, scale μᵢφ̂,
        # each term weighted by wᵢ
        sw = sum(w)
        phi = max(dev / sw, 1e-10)
        return sum(w[i] * logpdf(Gamma(1 / phi, mu[i] * phi), y[i])
                   for i in eachindex(y))
    elseif m.family isa InverseGaussian
        # R's inverse.gaussian()$aic convention: φ̂ = dev/Σwᵢ,
        # ℓ = -½[Σwᵢ·(1 + log(2πφ̂)) + 3Σ wᵢ log yᵢ]
        sw = sum(w)
        phi = max(dev / sw, 1e-10)
        return -0.5 * (sw * (1 + log(2π * phi)) + 3 * sum(w .* log.(y)))
    elseif m.family isa Poisson
        ll = 0.0
        @inbounds for i in eachindex(y)
            mui = max(mu[i], eps())
            ll += w[i] * (y[i] * log(mui) - mui - logabsgamma(y[i] + 1.0)[1])
        end
        return ll
    elseif m.family isa BinomialLike
        # wᵢ = number of trials, yᵢ = observed proportion. The binomial
        # coefficient is included when the counts are integral (as in R's
        # dbinom); for unit weights it is 0 (Bernoulli).
        ll = 0.0
        @inbounds for i in eachindex(y)
            mui = clamp(mu[i], eps(), 1 - eps())
            ll += w[i] * (y[i] * log(mui) + (1 - y[i]) * log(1 - mui))
            ki = w[i] * y[i]
            if w[i] != 1.0 && isinteger(w[i]) && isinteger(ki)
                ll += logabsbinomial(round(Int, w[i]), round(Int, ki))[1]
            end
        end
        return ll
    elseif m.family isa NegBinFamily
        # Full NB log-likelihood including the θ-dependent constants, so AIC
        # is valid across fits with different (estimated) θ
        θ = m.family.theta
        ll = 0.0
        @inbounds for i in eachindex(y)
            mui = max(mu[i], eps())
            ll += w[i] * (logabsgamma(y[i] + θ)[1] - logabsgamma(θ)[1] -
                          logabsgamma(y[i] + 1.0)[1] +
                          θ * log(θ / (θ + mui)) + y[i] * log(mui / (θ + mui)))
        end
        return ll
    elseif m.family isa BetaFamily
        phi = max(m.family.phi, 1e-10)
        ll = 0.0
        @inbounds for i in eachindex(y)
            mui = clamp(mu[i], eps(), 1 - eps())
            yi = clamp(y[i], eps(), 1 - eps())
            ll += w[i] * logpdf(Beta(mui * phi, (1 - mui) * phi), yi)
        end
        return ll
    elseif m.family isa Union{QuasiPoissonFamily, QuasiBinomialFamily}
        # Quasi-likelihood families have no true likelihood
        return NaN
    elseif m.family isa ExtendedFamily
        return -dev / 2
    else
        # Fallback: -dev/2 (saturated model comparison)
        return -dev / 2
    end
end

"""
    aic(m::GamModel)

Akaike information criterion, `-2ℓ̂ + 2·dof(m)`, where `dof(m) = sum(edf) + 1`
if the scale is estimated (`sum(edf)` otherwise). The smoothing penalty enters
through the **effective** degrees of freedom: a heavily penalised smooth
contributes far less than its basis dimension.

# Relationship to mgcv
This is exactly mgcv's `m$aic` field:

    m$aic = family$aic(y, n, μ̂, w, dev) + 2·sum(m$edf)          # gam.outer

mgcv's `AIC(m)` is *not* `m$aic`. `logLik.gam` reports a df attribute based on
`edf2` — the Wood, Pya & Säfken (2016) correction for smoothing-parameter
uncertainty — when the outer optimiser supplies `dβ/dρ`, so

    AIC(m) = m\$aic + 2·(sum(m\$edf2) - sum(m\$edf))

with `edf2 ≥ edf` (capped at `edf1`). GAM.jl does not yet compute `edf2`
(it needs the corrected covariance `Vc`; see the roadmap), so `aic(m)` is the
**conditional** AIC, treating the smoothing parameters as known. Measured gaps
against `AIC(m)` on typical single-smooth fits: ≈0.2–1.5.

For `method="GCV.Cp"` fits mgcv leaves `edf2` unset and falls back to `edf`,
so `AIC(m)` and `aic(m)` then use the *same* convention.
"""
aic(m::GamModel) = -2loglikelihood(m) + 2dof(m)

function aicc(m::GamModel)
    k = dof(m)
    n = nobs(m)
    n - k - 1 > 0 || return Inf
    return -2loglikelihood(m) + 2k + 2k * (k + 1) / (n - k - 1)
end

bic(m::GamModel) = -2loglikelihood(m) + dof(m) * log(nobs(m))

"""
    vcov(m::GamModel)

Bayesian posterior covariance matrix of the parameters.
"""
vcov(m::GamModel) = m.Vp

stderror(m::GamModel) = sqrt.(max.(diag(vcov(m)), 0.0))

function confint(m::GamModel; level::Real = 0.95)
    cc = coef(m)
    se = stderror(m)
    z = quantile(Normal(), (1 + level) / 2)
    return hcat(cc .- z .* se, cc .+ z .* se)
end

"""
    coeftable(m::GamModel)

Coefficient table for parametric terms. For smooth terms, use `summary(m)`.
"""
function coeftable(m::GamModel; level::Real = 0.95)
    cc = coef(m)
    se = stderror(m)

    # Only show parametric coefficients in the coefficient table
    n_para = m.n_parametric
    cc_para = cc[1:n_para]
    se_para = se[1:n_para]

    z = cc_para ./ se_para

    if _needs_scale_estimate(m.family)
        dofr = dof_residual(m)
        p_vals = 2 .* ccdf.(Ref(TDist(dofr)), abs.(z))
        test_stat_name = "t"
    elseif m.family isa ExtendedFamily && _estimates_scale(m.family)
        dofr = dof_residual(m)
        p_vals = 2 .* ccdf.(Ref(TDist(dofr)), abs.(z))
        test_stat_name = "t"
    else
        p_vals = 2 .* ccdf.(Ref(Normal()), abs.(z))
        test_stat_name = "z"
    end

    # Parameter names
    names = _gam_parametric_names(m)
    if length(names) != n_para
        names = String["(Intercept)"]
        if n_para > 1
            for i in 2:n_para
                push!(names, "x$i")
            end
        end
    end

    return CoefTable(
        hcat(cc_para, se_para, z, p_vals),
        ["Coef.", "Std. Error", test_stat_name, "Pr(>|$test_stat_name|)"],
        names, 4, 3,
    )
end

function _gam_parametric_names(m::GamModel)
    # One name per dummy-coded design-matrix column. Needs the training data
    # to resolve factor levels; falls back to per-variable names without it.
    if (m.formula isa GamFormula || m.formula isa FormulaTerm) && m.data !== nothing
        colnames, _ = _parametric_term_groups(m.formula, m.data)
        if length(colnames) == m.n_parametric
            return colnames
        end
    end
    if m.formula isa GamFormula || m.formula isa FormulaTerm
        return _formula_parametric_names(m.formula)
    end

    if m.n_parametric == 0
        return String[]
    end

    names = String["(Intercept)"]
    for i in 2:(m.n_parametric)
        push!(names, "x$i")
    end
    return names
end

function coefnames(m::GamModel)
    names = _gam_parametric_names(m)
    # Smooth terms
    for sm in m.smooths
        k = size(sm.X, 2)
        for j in 1:k
            push!(names, "$(sm.spec.label).$j")
        end
    end
    return names
end

"""
    predict(m::GamModel; type=:link)

Return predictions from the fitted model.
- `type=:link`: predictions on the link scale (η)
- `type=:response`: predictions on the response scale (μ)
"""
function predict(m::GamModel; type::Symbol = :link)
    if type == :link
        return m.linear_predictor
    elseif type == :response
        return m.fitted_values
    else
        throw(ArgumentError("type must be :link or :response"))
    end
end

"""
    predict(m::GamModel, newdata; type=:link, se=false)

Predict at new data points.
"""
function _gam_parametric_matrix(m::GamModel, t)
    n_new = _table_nrows(t)

    if m.formula isa GamFormula
        # Reuse the categorical factor levels from the training data so dummy
        # coding is consistent even when newdata contains a subset of levels.
        ref_levels = _parametric_ref_levels(m.formula, m.data)
        X_para, _ = _build_parametric_matrix(m.formula, t; ref_levels = ref_levels)
        size(X_para, 2) == m.n_parametric || throw(DimensionMismatch(
            "Prediction parametric matrix has $(size(X_para, 2)) columns, expected $(m.n_parametric)"))
        return X_para
    elseif m.formula isa FormulaTerm
        # Build the schema from the training data so categorical levels (and
        # hence dummy columns) are consistent at prediction time.
        X_para, _ = _build_parametric_matrix(m.formula, t;
            schema_data = m.data === nothing ? t : m.data)
        size(X_para, 2) == m.n_parametric || throw(DimensionMismatch(
            "Prediction parametric matrix has $(size(X_para, 2)) columns, expected $(m.n_parametric)"))
        return X_para
    end

    if m.n_parametric == 0
        return Matrix{Float64}(undef, n_new, 0)
    elseif m.n_parametric == 1
        return ones(n_new, 1)
    end

    throw(ArgumentError(
        "Model does not retain enough formula information to predict $(m.n_parametric) parametric columns"))
end

function _gam_has_intercept(m::GamModel)
    if m.formula isa GamFormula || m.formula isa FormulaTerm
        return _formula_has_intercept(m.formula)
    end
    return m.n_parametric > 0
end

function _gam_prediction_matrix(m::GamModel, newdata)
    t = Tables.columntable(newdata)
    X_para = _gam_parametric_matrix(m, t)

    X_smooth_parts = Matrix{Float64}[]
    for sm in m.smooths
        X_sm = predict_matrix(sm, t)
        push!(X_smooth_parts, X_sm)
    end

    X_new = isempty(X_smooth_parts) ? X_para : hcat(X_para, X_smooth_parts...)
    size(X_new, 2) == length(m.coefficients) || throw(DimensionMismatch(
        "Prediction matrix has $(size(X_new, 2)) columns, expected $(length(m.coefficients))"))
    return X_new
end

function predict(m::GamModel, newdata; type::Symbol = :link, se::Bool = false,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing)
    if type == :terms
        return _predict_terms(m, newdata; se = se)
    end
    X_new = _gam_prediction_matrix(m, newdata)
    eta = X_new * m.coefficients
    # Models fit with an offset need the same offset supplied at prediction
    # (mgcv requires the offset in newdata).
    if offset !== nothing
        length(offset) == length(eta) || throw(DimensionMismatch(
            "offset length $(length(offset)) ≠ number of prediction rows $(length(eta))"))
        eta = eta .+ Float64.(offset)
    end

    if se
        # Standard errors of predictions
        se_eta = sqrt.(max.(vec(sum((X_new * m.Vp) .* X_new; dims = 2)), 0.0))
        if type == :response
            mu = _response_predictions(m.family, m.link, eta)
            dmu = GLM.mueta.(Ref(m.link), eta)
            return mu, abs.(dmu) .* se_eta
        else
            return eta, se_eta
        end
    end

    if type == :response
        return _response_predictions(m.family, m.link, eta)
    else
        return eta
    end
end

"""
Response-scale predictions with the family's mean-domain clamp applied,
matching the clamping used during fitting (mgcv's validmu convention).
Without this, a link whose inverse can leave the mean domain (e.g. the
canonical inverse link for Gamma when η crosses zero) returns invalid
values — negative means for a strictly positive family — that are
inconsistent with the clamped `fitted` values.
"""
function _response_predictions(family, link, eta)
    mu = GLM.linkinv.(Ref(link), eta)
    mu_c = _clamp_mu(family, mu)
    if mu_c !== mu && any(i -> mu_c[i] != mu[i], eachindex(mu))
        @warn "Response predictions outside the $(nameof(typeof(family))) mean " *
              "domain were clamped to the boundary (link $(nameof(typeof(link))) " *
              "inverse left the valid range); consider a different link." maxlog = 1
    end
    return mu_c
end

"""
    lpmatrix(m::GamModel, newdata) -> Matrix{Float64}

The linear-predictor (design) matrix `Xp` such that `Xp * coef(m)` gives the
link-scale predictions at `newdata`. Equivalent to mgcv's
`predict(m, newdata, type="lpmatrix")`; useful for building custom predictions
and posterior intervals (`Xp * Vp * Xp'`).
"""
lpmatrix(m::GamModel, newdata) = _gam_prediction_matrix(m, newdata)

"""
    _predict_terms(m, newdata; se=false)

Per-term contributions on the link scale (mgcv's `type="terms"`). Returns a
`NamedTuple` of vectors, one per parametric *term* (a categorical variable's
dummy columns are summed into a single entry, as in mgcv) and one per smooth
term (each already centered, as the smooths are sum-to-zero constrained). The
intercept is reported separately as `:Intercept`. When `se=true`, returns
`(terms, se_terms)`.
"""
function _predict_terms(m::GamModel, newdata; se::Bool = false)
    t = Tables.columntable(newdata)
    X_para = _gam_parametric_matrix(m, t)
    β = m.coefficients
    has_int = _gam_has_intercept(m)

    labels = Symbol[]
    cols = Vector{Float64}[]
    se_cols = Vector{Float64}[]
    Vp = m.Vp
    np = m.n_parametric

    # Group parametric columns by originating term (intercept, then one group
    # per variable — a categorical term spans several dummy columns).
    groups = if (m.formula isa GamFormula || m.formula isa FormulaTerm) &&
                m.data !== nothing
        _, grps = _parametric_term_groups(m.formula, m.data)
        # Defensive: groups must tile exactly the np parametric columns
        (isempty(grps) ? 0 : last(grps)[2][end]) == np ? grps :
            [(has_int && j == 1 ? "(Intercept)" : "x$j", j:j) for j in 1:np]
    else
        [(has_int && j == 1 ? "(Intercept)" : "x$j", j:j) for j in 1:np]
    end

    for (name, idx) in groups
        contrib = X_para[:, idx] * β[idx]
        nm = name == "(Intercept)" ? :Intercept : Symbol(name)
        push!(labels, nm)
        push!(cols, vec(contrib))
        if se
            Vp_blk = Vp[idx, idx]
            Xg = X_para[:, idx]
            push!(se_cols,
                sqrt.(max.(vec(sum((Xg * Vp_blk) .* Xg; dims = 2)), 0.0)))
        end
    end

    for sm in m.smooths
        X_sm = predict_matrix(sm, t)
        idx = sm.first_para:sm.last_para
        push!(labels, Symbol(sm.spec.label))
        push!(cols, X_sm * β[idx])
        if se
            Vp_blk = Vp[idx, idx]
            push!(se_cols,
                sqrt.(max.(vec(sum((X_sm * Vp_blk) .* X_sm; dims = 2)), 0.0)))
        end
    end

    terms = NamedTuple{Tuple(labels)}(Tuple(cols))
    se ? (terms, NamedTuple{Tuple(labels)}(Tuple(se_cols))) : terms
end

function _mp_link(link::Symbol)
    if link === :identity
        return IdentityLink()
    elseif link === :log
        return LogLink()
    elseif link === :logit
        return LogitLink()
    elseif link === :inverse
        return InverseLink()
    elseif link === :sqrt
        return SqrtLink()
    end
    throw(ArgumentError("Unsupported MultiParameterModel link $link in prediction"))
end

function _mp_num_parametric(m::MultiParameterModel, k::Int)
    n_smooth_cols = sum(size(sm.X, 2) for sm in m.smooths[k]; init = 0)
    n_parametric = size(m.X_list[k], 2) - n_smooth_cols
    n_parametric >= 0 || throw(ArgumentError(
        "Invalid MultiParameterModel design for parameter $k: parametric column count is negative"))
    return n_parametric
end

function _mp_prediction_matrix(m::MultiParameterModel, k::Int, t)
    n_parametric = _mp_num_parametric(m, k)
    X_parts = Matrix{Float64}[]

    if !isempty(m.formulas)
        X_para, _ = _build_parametric_matrix(m.formulas[k], t)
        size(X_para, 2) == n_parametric || throw(DimensionMismatch(
            "Prediction parametric matrix for parameter $k has $(size(X_para, 2)) columns, expected $n_parametric"))
        size(X_para, 2) > 0 && push!(X_parts, X_para)
    elseif n_parametric == 1
        push!(X_parts, ones(_table_nrows(t), 1))
    elseif n_parametric > 1
        throw(ArgumentError(
            "Model does not retain enough formula information to predict $n_parametric parametric columns for parameter $k"))
    end

    for sm in m.smooths[k]
        push!(X_parts, predict_matrix(sm, t))
    end

    if isempty(X_parts)
        return Matrix{Float64}(undef, _table_nrows(t), 0)
    end
    Xk = hcat(X_parts...)
    size(Xk, 2) == size(m.X_list[k], 2) || throw(DimensionMismatch(
        "Prediction matrix for parameter $k has $(size(Xk, 2)) columns, expected $(size(m.X_list[k], 2))"))
    return Xk
end

function _predict_multiparameter(m::MultiParameterModel, X_list::Vector{Matrix{Float64}};
                                 type::Symbol = :link, se::Bool = false,
                                 off_list = nothing)
    type in (:link, :response) || throw(ArgumentError("type must be :link or :response"))

    K = nparams(m)
    n = size(X_list[1], 1)
    fit = Matrix{Float64}(undef, n, K)
    se_fit = se ? Matrix{Float64}(undef, n, K) : nothing
    links = param_links(m.family)

    length(links) == K || throw(ArgumentError(
        "Expected $K parameter links for $(typeof(m.family)), got $(length(links))"))

    for k in 1:K
        Xk = X_list[k]
        s = m.param_offsets[k] + 1
        e = m.param_offsets[k + 1]
        pk = e - s + 1
        size(Xk, 2) == pk || throw(DimensionMismatch(
            "Prediction matrix for parameter $k has $(size(Xk, 2)) columns, expected $pk"))

        βk = @view m.coefficients[s:e]
        ηk = Xk * βk
        off_list === nothing || (ηk .+= off_list[k])

        if se
            Vk = @view m.Vp[s:e, s:e]
            se_eta = sqrt.(max.(vec(sum((Xk * Vk) .* Xk; dims = 2)), 0.0))
            if type == :response
                link = _mp_link(links[k])
                fit[:, k] = GLM.linkinv.(Ref(link), ηk)
                dmu = GLM.mueta.(Ref(link), ηk)
                se_fit[:, k] = abs.(dmu) .* se_eta
            else
                fit[:, k] = ηk
                se_fit[:, k] = se_eta
            end
        elseif type == :response
            link = _mp_link(links[k])
            fit[:, k] = GLM.linkinv.(Ref(link), ηk)
        else
            fit[:, k] = ηk
        end
    end

    return se ? (fit, se_fit) : fit
end

"""
    predict(m::MultiParameterModel; type=:link, se=false)
    predict(m::MultiParameterModel, newdata; type=:link, se=false, offset=nothing)

Predict each parameter of a fitted multi-parameter model. The returned matrix has
one column per parameter, ordered as `param_names(m.family)`.

- `type=:link`: linear predictors for each parameter
- `type=:response`: parameter values on the response scale

With `se=true`, returns `(fit, se_fit)` where `se_fit` contains pointwise
standard errors derived from `m.Vp`.

Offsets: training-data predictions include the offsets the model was fitted
with (stored on the model). For new data, supply `offset=` in the same
shapes accepted at fitting time — a single length-`n` vector (first linear
predictor) or a length-`K` per-parameter vector; the default is no offset,
mirroring `GamModel`'s supply-at-predict convention.
"""
function predict(m::MultiParameterModel; type::Symbol = :link, se::Bool = false)
    return _predict_multiparameter(m, m.X_list; type = type, se = se,
                                   off_list = m.offsets)
end

function predict(m::MultiParameterModel, newdata; type::Symbol = :link, se::Bool = false,
                 offset = nothing)
    t = Tables.columntable(newdata)
    X_list = Matrix{Float64}[]
    for k in 1:nparams(m)
        push!(X_list, _mp_prediction_matrix(m, k, t))
    end
    n_new = size(X_list[1], 1)
    off_list = _normalize_mp_offset(offset, nparams(m), n_new)
    return _predict_multiparameter(m, X_list; type = type, se = se,
                                   off_list = off_list)
end

fitted(m::MultiParameterModel) = predict(m; type = :response)

"""
    residuals(m::GamModel; type=:deviance)

Model residuals.
- `:deviance`: deviance residuals
- `:pearson`: Pearson residuals
- `:working`: working residuals from final IRLS iteration
- `:response`: response residuals (y - μ)
"""
function residuals(m::GamModel; type::Symbol = :deviance)
    y = m.y
    mu = m.fitted_values
    wt = m.weights

    if type == :response
        return y .- mu
    elseif type == :pearson
        var_mu = _variance(m.family, mu)
        return sqrt.(wt) .* (y .- mu) ./ sqrt.(max.(var_mu, eps()))
    elseif type == :deviance
        return _deviance_residuals(m.family, y, mu, wt)
    elseif type == :working
        link = m.link
        dmu = GLM.mueta.(Ref(link), m.linear_predictor)
        return (y .- mu) ./ dmu
    else
        throw(ArgumentError("type must be :deviance, :pearson, :working, or :response"))
    end
end

function _deviance_residuals(d::Normal, y, mu, wt)
    return sign.(y .- mu) .* sqrt.(wt .* (y .- mu) .^ 2)
end

function _deviance_residuals(d::Poisson, y, mu, wt)
    r = similar(y)
    for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        yi = y[i]
        if yi > 0
            di = 2 * (yi * log(yi / mui) - (yi - mui))
        else
            di = 2 * mui
        end
        r[i] = sign(yi - mui) * sqrt(max(wt[i] * di, 0))
    end
    return r
end

function _deviance_residuals(d::BinomialLike, y, mu, wt)
    r = similar(y)
    for i in eachindex(y, mu, wt)
        mui = clamp(mu[i], eps(), 1 - eps())
        yi = y[i]
        di = 0.0
        if yi > 0
            di += yi * log(yi / mui)
        end
        if yi < 1
            di += (1 - yi) * log((1 - yi) / (1 - mui))
        end
        r[i] = sign(yi - mui) * sqrt(max(2 * wt[i] * di, 0))
    end
    return r
end

function _deviance_residuals(d::Gamma, y, mu, wt)
    r = similar(y)
    for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        yi = max(y[i], eps())
        di = 2 * (-log(yi / mui) + (yi - mui) / mui)
        r[i] = sign(y[i] - mu[i]) * sqrt(max(wt[i] * di, 0))
    end
    return r
end

function _deviance_residuals(d::InverseGaussian, y, mu, wt)
    r = similar(y)
    for i in eachindex(y, mu, wt)
        mui = max(mu[i], eps())
        yi = max(y[i], eps())
        di = (yi - mui)^2 / (mui^2 * yi)
        r[i] = sign(y[i] - mu[i]) * sqrt(max(wt[i] * di, 0))
    end
    return r
end

function _deviance_residuals(d::UnivariateDistribution, y, mu, wt)
    return sign.(y .- mu) .* sqrt.(max.(wt .* (y .- mu) .^ 2, 0))
end

"""
    r2(m::GamModel)

R-squared based on working (response) residuals, matching mgcv's `summary.gam`:
  R² = 1 - Σ wᵢ(yᵢ - μᵢ)² / Σ wᵢ(yᵢ - ȳ)²
For Gaussian, this equals 1 - deviance/null_deviance.
"""
function r2(m::GamModel)
    y = m.y
    mu = m.fitted_values
    w = m.weights
    ymean = sum(w .* y) / sum(w)
    ss_res = sum(w .* (y .- mu) .^ 2)
    ss_tot = sum(w .* (y .- ymean) .^ 2)
    return ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
end

"""
    deviance_explained(m::GamModel)

Deviance explained: 1 - deviance/null_deviance. This is the quantity mgcv's
`summary.gam` reports as "Deviance explained"; it differs from R² for
non-Gaussian families, where [`r2`](@ref) is a response-scale quantity.

# Example
```julia
m = gam(@formula(y ~ s(x)), df; family = Poisson(), link = LogLink())
deviance_explained(m)   # e.g. 0.83
```
"""
deviance_explained(m::GamModel) = 1.0 - deviance(m) / nulldeviance(m)

"""
    adjr2(m::GamModel)

Adjusted R-squared accounting for effective degrees of freedom:
  R²(adj) = 1 - (1 - R²)(n-1)/(n - edf_total)
where edf_total = tr(hat matrix) already includes the intercept.
"""
function adjr2(m::GamModel)
    n = nobs(m)
    edf = m.edf_total  # tr(F), already includes intercept
    dof_res = n - edf
    dof_res > 0 || return NaN
    return 1.0 - (1.0 - r2(m)) * (n - 1) / dof_res
end
