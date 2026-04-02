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
    n = nobs(m)
    dev = deviance(m)
    scale = m.scale
    if m.family isa TweedieFamily
        return _tweedie_total_loglik(m.y, m.fitted_values, m.weights, m.family.p,
            clamp(scale, 1e-8, 1e8))
    elseif _needs_scale_estimate(m.family)
        return -n / 2 * (log(2π * scale) + dev / (n * scale))
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
    names = String["(Intercept)"]
    if n_para > 1
        for i in 2:n_para
            push!(names, "x$i")
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

function coefnames(m::GamModel)
    names = String["(Intercept)"]
    # Parametric terms (beyond intercept)
    for i in 2:(m.n_parametric)
        push!(names, "x$i")
    end
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
function predict(m::GamModel, newdata; type::Symbol = :link, se::Bool = false)
    t = Tables.columntable(newdata)
    n_new = length(Tables.getcolumn(t, first(Tables.columnnames(t))))

    # Build prediction matrix
    X_para = ones(n_new, 1)  # intercept

    X_smooth_parts = Matrix{Float64}[]
    for sm in m.smooths
        X_sm = predict_matrix(sm, t)
        push!(X_smooth_parts, X_sm)
    end

    X_new = isempty(X_smooth_parts) ? X_para : hcat(X_para, X_smooth_parts...)

    eta = X_new * m.coefficients

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
    n_new = length(Tables.getcolumn(t, first(Tables.columnnames(t))))
    n_parametric = _mp_num_parametric(m, k)
    n_parametric <= 1 || throw(ArgumentError(
        "Prediction for MultiParameterModel currently supports at most an intercept-only parametric block per parameter; parameter $k has $n_parametric parametric columns"))

    X_parts = Matrix{Float64}[]
    if n_parametric == 1
        push!(X_parts, ones(n_new, 1))
    end
    for sm in m.smooths[k]
        push!(X_parts, predict_matrix(sm, t))
    end

    if isempty(X_parts)
        return Matrix{Float64}(undef, n_new, 0)
    end
    return hcat(X_parts...)
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
        if m.family isa ExtendedFamily
            var_mu = _variance(m.family, mu)
        else
            var_mu = _variance(m.family, mu)
        end
        return (y .- mu) ./ sqrt.(max.(var_mu, eps()))
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
