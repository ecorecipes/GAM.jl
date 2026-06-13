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

Log-likelihood of the fitted model.
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
        # Weighted Gaussian: -1/2 Σ [log(2πφ/wᵢ) + wᵢ(yᵢ-μᵢ)²/φ]
        return -0.5 * (sum(log.(2π * scale ./ w)) + dev / scale)
    elseif m.family isa Gamma
        # shapeᵢ = wᵢ/φ, scale parameter μᵢφ/wᵢ (mean μᵢ, var μᵢ²φ/wᵢ)
        phi = max(scale, 1e-10)
        return sum(logpdf(Gamma(w[i] / phi, mu[i] * phi / w[i]), y[i])
                   for i in eachindex(y))
    elseif m.family isa InverseGaussian
        # λᵢ = wᵢ/φ (mean μᵢ, var μᵢ³φ/wᵢ)
        phi = max(scale, 1e-10)
        return sum(logpdf(InverseGaussian(mu[i], w[i] / phi), y[i])
                   for i in eachindex(y))
    elseif m.family isa ExtendedFamily
        return -dev / 2
    else
        # For non-Gaussian: -dev/2 (saturated model comparison)
        return -dev / 2
    end
end

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

    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    ci = quantile(Normal(), (1 - level) / 2) .* se_para

    return CoefTable(
        hcat(cc_para, se_para, z, p_vals),
        ["Coef.", "Std. Error", test_stat_name, "Pr(>|$test_stat_name|)"],
        names, 4, 3,
    )
end

function _gam_parametric_names(m::GamModel)
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
        X_para, _ = _build_parametric_matrix(m.formula, t)
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
            mu = GLM.linkinv.(Ref(m.link), eta)
            dmu = GLM.mueta.(Ref(m.link), eta)
            return mu, abs.(dmu) .* se_eta
        else
            return eta, se_eta
        end
    end

    if type == :response
        return GLM.linkinv.(Ref(m.link), eta)
    else
        return eta
    end
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
`NamedTuple` of vectors, one per parametric column and one per smooth term
(each already centered, as the smooths are sum-to-zero constrained). The
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

    para_names = _gam_parametric_names(m)
    for j in 1:np
        contrib = X_para[:, j] .* β[j]
        nm = (has_int && j == 1) ? :Intercept : Symbol(para_names[j])
        push!(labels, nm)
        push!(cols, contrib)
        if se
            push!(se_cols,
                sqrt.(max.(abs2.(X_para[:, j]) .* Vp[j, j], 0.0)))
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
                                 type::Symbol = :link, se::Bool = false)
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
    predict(m::MultiParameterModel, newdata; type=:link, se=false)

Predict each parameter of a fitted multi-parameter model. The returned matrix has
one column per parameter, ordered as `param_names(m.family)`.

- `type=:link`: linear predictors for each parameter
- `type=:response`: parameter values on the response scale

With `se=true`, returns `(fit, se_fit)` where `se_fit` contains pointwise
standard errors derived from `m.Vp`.
"""
function predict(m::MultiParameterModel; type::Symbol = :link, se::Bool = false)
    return _predict_multiparameter(m, m.X_list; type = type, se = se)
end

function predict(m::MultiParameterModel, newdata; type::Symbol = :link, se::Bool = false)
    t = Tables.columntable(newdata)
    X_list = Matrix{Float64}[]
    for k in 1:nparams(m)
        push!(X_list, _mp_prediction_matrix(m, k, t))
    end
    return _predict_multiparameter(m, X_list; type = type, se = se)
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

Deviance explained: 1 - deviance/null_deviance (different from R² for non-Gaussian).
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
