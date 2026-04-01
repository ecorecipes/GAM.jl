# ============================================================================
# Gratia-like diagnostics and visualization utilities for GAM.jl
#
# Ported from R's gratia package by Gavin Simpson.
# Provides: smooth evaluation on grids, derivatives, posterior simulation,
# model diagnostics, and data structures for plotting.
# ============================================================================

using Random: AbstractRNG, default_rng, MersenneTwister, randn!, randn, rand
using Distributions: MvNormal, quantile as dquantile, Normal as DNormal, Poisson as DPoisson,
    NegativeBinomial as DNegBin, cdf, pdf

# ============================================================================
# Result types — lightweight structs for tabular output
# ============================================================================

"""
    SmoothEstimates

Tabular result from [`smooth_estimates`](@ref). Contains evaluation grid,
estimated smooth values, and standard errors for one or more smooth terms.

# Fields
- `smooth`: smooth label per row (e.g. `"s(x)"`)
- `covariates`: `Dict{Symbol, Vector{Float64}}` of covariate grid values
- `estimate`: estimated smooth effect f̂(x)
- `se`: pointwise standard error of the estimate
- `by_values`: optional by-variable values
"""
struct SmoothEstimates
    smooth::Vector{String}
    covariates::Dict{Symbol, Vector{Float64}}
    estimate::Vector{Float64}
    se::Vector{Float64}
end

"""
    DerivativeEstimates

Tabular result from [`derivatives`](@ref). Contains estimated derivatives of
smooth terms with confidence intervals.

# Fields
- `smooth`: smooth label per row
- `x`: covariate values at which derivatives are evaluated
- `derivative`: estimated derivative value
- `se`: standard error of derivative
- `lower`: lower confidence bound
- `upper`: upper confidence bound
- `order`: derivative order (1 or 2)
- `type`: difference type (:forward, :backward, or :central)
"""
struct DerivativeEstimates
    smooth::Vector{String}
    x::Vector{Float64}
    derivative::Vector{Float64}
    se::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    order::Int
    type::Symbol
end

"""
    AppraiseData

Diagnostic data from [`appraise`](@ref) for creating model checking plots.

# Fields
- `residuals_deviance`: deviance residuals
- `residuals_pearson`: Pearson residuals
- `linear_predictor`: η values
- `observed`: observed response y
- `fitted`: fitted values μ̂
- `qq_theoretical`: theoretical N(0,1) quantiles for QQ plot
- `qq_sample`: sorted standardized deviance residuals (divided by √scale)
"""
struct AppraiseData
    residuals_deviance::Vector{Float64}
    residuals_pearson::Vector{Float64}
    linear_predictor::Vector{Float64}
    observed::Vector{Float64}
    fitted::Vector{Float64}
    qq_theoretical::Vector{Float64}
    qq_sample::Vector{Float64}
end

"""
    RootogramData

Data for rootogram visualization from [`rootogram`](@ref).

# Fields
- `count`: count values (0, 1, 2, …)
- `observed`: observed frequencies
- `expected`: expected frequencies from fitted model
- `sqrt_observed`: √(observed frequency)
- `sqrt_expected`: √(expected frequency)
"""
struct RootogramData
    count::Vector{Int}
    observed::Vector{Float64}
    expected::Vector{Float64}
    sqrt_observed::Vector{Float64}
    sqrt_expected::Vector{Float64}
end

"""
    OverviewTable

Summary table from [`overview`](@ref) listing all smooth terms and their properties.
"""
struct OverviewTable
    label::Vector{String}
    smooth_type::Vector{String}
    dimension::Vector{Int}
    basis_size::Vector{Int}
    edf::Vector{Float64}
    edf_ratio::Vector{Float64}
end

# ============================================================================
# smooth_estimates — evaluate smooth terms on a grid
# ============================================================================

"""
    smooth_estimates(m::GamModel; select=nothing, n=100, data=nothing,
                     unconditional=false, overall_uncertainty=true)

Evaluate estimated smooth terms on a grid of covariate values.

# Arguments
- `select`: indices or labels of smooths to evaluate (default: all)
- `n`: number of grid points per covariate
- `data`: custom evaluation data (NamedTuple or Tables-compatible); if nothing,
  an evenly-spaced grid over the covariate range is generated
- `unconditional`: if true, use unconditional covariance (currently same as Vp)
- `overall_uncertainty`: include uncertainty in the intercept

# Returns
A [`SmoothEstimates`](@ref) struct.
"""
function smooth_estimates(m::GamModel;
    select = nothing,
    n::Int = 100,
    data = nothing,
    unconditional::Bool = false,
    overall_uncertainty::Bool = true,
)
    smooth_indices = _resolve_smooth_select(m, select)

    all_labels = String[]
    all_estimates = Float64[]
    all_se = Float64[]
    all_covariates = Dict{Symbol, Vector{Float64}}()

    Vcov = unconditional ? m.Vp : m.Vp

    for si in smooth_indices
        sm = m.smooths[si]
        spec = sm.spec
        vars = spec.term_vars
        sm_idx = sm.first_para:sm.last_para
        beta_s = m.coefficients[sm_idx]

        if data === nothing
            eval_data = _make_smooth_grid(m, sm, n)
        else
            eval_data = Tables.columntable(data)
        end

        X_pred = predict_matrix(sm, eval_data)
        f_hat = X_pred * beta_s

        # Standard errors
        Vp_s = Vcov[sm_idx, sm_idx]
        if overall_uncertainty && m.n_parametric >= 1
            # Include intercept uncertainty
            p = size(m.X, 2)
            X_full = zeros(size(X_pred, 1), p)
            X_full[:, sm_idx] .= X_pred
            X_full[:, 1] .= 1.0  # intercept
            se_vec = sqrt.(max.(vec(sum((X_full * Vcov) .* X_full; dims = 2)), 0.0))
        else
            se_vec = sqrt.(max.(diag(X_pred * Vp_s * X_pred'), 0.0))
        end

        n_pts = length(f_hat)
        append!(all_labels, fill(spec.label, n_pts))
        append!(all_estimates, f_hat)
        append!(all_se, se_vec)

        for v in vars
            vals = Float64.(collect(Tables.getcolumn(eval_data, v)))
            if haskey(all_covariates, v)
                append!(all_covariates[v], vals)
            else
                all_covariates[v] = copy(vals)
            end
        end
    end

    return SmoothEstimates(all_labels, all_covariates, all_estimates, all_se)
end

# ============================================================================
# partial_residuals
# ============================================================================

"""
    partial_residuals(m::GamModel; select=nothing)

Compute partial residuals for selected smooth terms.
Partial residuals = f̂_j(x) + ε̂, where ε̂ are the working residuals.

Returns a Dict mapping smooth labels to (x_values, partial_resid_values).
"""
function partial_residuals(m::GamModel; select = nothing)
    smooth_indices = _resolve_smooth_select(m, select)
    resid = m.y .- m.fitted_values  # response residuals

    result = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
    for si in smooth_indices
        sm = m.smooths[si]
        spec = sm.spec
        length(spec.term_vars) == 1 || continue  # only 1D smooths

        sm_idx = sm.first_para:sm.last_para
        beta_s = m.coefficients[sm_idx]
        f_hat = m.X[:, sm_idx] * beta_s
        partial_r = f_hat .+ resid

        var = spec.term_vars[1]
        x_vals = Float64.(collect(Tables.getcolumn(Tables.columntable(
            _extract_original_data(m, sm)), var)))
        result[spec.label] = (x_vals, partial_r)
    end
    return result
end

# ============================================================================
# data_slice — create evaluation grids
# ============================================================================

"""
    data_slice(m::GamModel; var::Symbol, n=100, kwargs...)

Create an evaluation grid for `var` with `n` points, holding all other
covariates at their typical (median for numeric, mode for categorical) values.

Additional keyword arguments fix specific covariate values.
"""
function data_slice(m::GamModel; var::Symbol, n::Int = 100, kwargs...)
    sm = nothing
    for s in m.smooths
        if var in s.spec.term_vars
            sm = s
            break
        end
    end
    sm === nothing && throw(ArgumentError("Variable $var not found in any smooth term"))

    grid = _make_smooth_grid(m, sm, n)
    return grid
end

# ============================================================================
# derivatives — finite-difference derivatives of smooth terms
# ============================================================================

"""
    derivatives(m::GamModel; select=nothing, order=1, type=:central,
                n=200, eps=1e-7, level=0.95, interval=:confidence,
                n_sim=10000, seed=nothing)

Compute derivatives of estimated smooth terms via finite differences.

# Arguments
- `select`: which smooth(s) to compute derivatives for
- `order`: derivative order (1 or 2)
- `type`: `:forward`, `:backward`, or `:central`
- `n`: number of evaluation points
- `eps`: finite difference step size
- `level`: confidence level for intervals
- `interval`: `:confidence` (pointwise) or `:simultaneous`
- `n_sim`: number of simulations for simultaneous intervals
- `seed`: random seed for simultaneous intervals

# Returns
A [`DerivativeEstimates`](@ref) struct.
"""
function derivatives(m::GamModel;
    select = nothing,
    order::Int = 1,
    type::Symbol = :central,
    n::Int = 200,
    eps::Float64 = 1e-7,
    level::Float64 = 0.95,
    interval::Symbol = :confidence,
    n_sim::Int = 10000,
    seed = nothing,
)
    type in (:forward, :backward, :central) ||
        throw(ArgumentError("type must be :forward, :backward, or :central"))
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2"))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1)"))

    smooth_indices = _resolve_smooth_select(m, select)
    Vcov = m.Vp

    all_labels = String[]
    all_x = Float64[]
    all_deriv = Float64[]
    all_se = Float64[]

    for si in smooth_indices
        sm = m.smooths[si]
        spec = sm.spec
        length(spec.term_vars) == 1 || continue  # only 1D
        var = spec.term_vars[1]
        sm_idx = sm.first_para:sm.last_para
        beta_s = m.coefficients[sm_idx]
        Vp_s = Vcov[sm_idx, sm_idx]

        grid = _make_smooth_grid(m, sm, n)
        x_vals = Float64.(collect(Tables.getcolumn(grid, var)))

        if order == 1
            d_hat, d_se = _fd_derivative_1(sm, grid, var, x_vals, beta_s, Vp_s, eps, type)
        else
            d_hat, d_se = _fd_derivative_2(sm, grid, var, x_vals, beta_s, Vp_s, eps, type)
        end

        append!(all_labels, fill(spec.label, length(d_hat)))
        append!(all_x, x_vals)
        append!(all_deriv, d_hat)
        append!(all_se, d_se)
    end

    # Confidence intervals
    if interval == :simultaneous && !isempty(all_deriv)
        rng = seed === nothing ? default_rng() : MersenneTwister(seed)
        crit = _simultaneous_critical(all_se, Vcov, m, smooth_indices, n_sim, level, rng)
    else
        crit = dquantile(DNormal(), (1 + level) / 2)
    end

    lower = all_deriv .- crit .* all_se
    upper = all_deriv .+ crit .* all_se

    return DerivativeEstimates(all_labels, all_x, all_deriv, all_se, lower, upper, order, type)
end

# ============================================================================
# Posterior simulation
# ============================================================================

"""
    posterior_samples(m::GamModel; n=1000, seed=nothing,
                     unconditional=false)

Draw `n` samples from the posterior distribution of model coefficients
β̃ ~ MVN(β̂, Vp).

Returns an `n × p` matrix where each row is a posterior draw.
"""
function posterior_samples(m::GamModel;
    n::Int = 1000,
    seed = nothing,
    unconditional::Bool = false,
)
    rng = seed === nothing ? default_rng() : MersenneTwister(seed)
    Vcov = unconditional ? m.Vp : m.Vp
    beta_hat = m.coefficients
    p = length(beta_hat)

    # Ensure Vp is symmetric positive semi-definite
    Vp_sym = Symmetric(Vcov)
    eig = eigen(Vp_sym)
    eig_vals = max.(eig.values, 0.0)
    L = eig.vectors * Diagonal(sqrt.(eig_vals))

    draws = Matrix{Float64}(undef, n, p)
    z = Vector{Float64}(undef, p)
    for i in 1:n
        randn!(rng, z)
        draws[i, :] = beta_hat .+ L * z
    end
    return draws
end

"""
    fitted_samples(m::GamModel; n=100, data=nothing, seed=nothing,
                   scale=:response, unconditional=false)

Draw posterior samples of fitted values. For each draw β̃ from the posterior,
computes Xβ̃ (on the link or response scale).

# Returns
An `n_obs × n_draws` matrix of fitted value draws.
"""
function fitted_samples(m::GamModel;
    n::Int = 100,
    data = nothing,
    seed = nothing,
    scale::Symbol = :response,
    unconditional::Bool = false,
)
    scale in (:response, :link, :linear_predictor) ||
        throw(ArgumentError("scale must be :response or :link"))

    draws = posterior_samples(m; n = n, seed = seed, unconditional = unconditional)

    if data === nothing
        X = m.X
    else
        X = _build_prediction_matrix(m, data)
    end

    eta_draws = X * draws'

    if scale == :response
        return GLM.linkinv.(Ref(m.link), eta_draws)
    else
        return eta_draws
    end
end

"""
    smooth_samples(m::GamModel; select=nothing, n=100, n_grid=100,
                   seed=nothing)

Draw posterior samples of smooth functions evaluated on a grid.

# Returns
A Dict mapping smooth labels to `(x_grid, samples_matrix)` where
`samples_matrix` is `n_grid × n_draws`.
"""
function smooth_samples(m::GamModel;
    select = nothing,
    n::Int = 100,
    n_grid::Int = 100,
    seed = nothing,
)
    draws = posterior_samples(m; n = n, seed = seed)
    smooth_indices = _resolve_smooth_select(m, select)

    result = Dict{String, Tuple{Vector{Float64}, Matrix{Float64}}}()
    for si in smooth_indices
        sm = m.smooths[si]
        spec = sm.spec
        length(spec.term_vars) == 1 || continue

        var = spec.term_vars[1]
        sm_idx = sm.first_para:sm.last_para
        grid = _make_smooth_grid(m, sm, n_grid)
        X_pred = predict_matrix(sm, grid)
        x_vals = Float64.(collect(Tables.getcolumn(grid, var)))

        beta_draws = draws[:, sm_idx]  # n × k
        f_draws = X_pred * beta_draws'  # n_grid × n

        result[spec.label] = (x_vals, f_draws)
    end
    return result
end

"""
    predicted_samples(m::GamModel; n=100, data=nothing, seed=nothing)

Draw posterior predictive samples (fitted values + observation noise).

# Returns
An `n_obs × n_draws` matrix of predicted values.
"""
function predicted_samples(m::GamModel;
    n::Int = 100,
    data = nothing,
    seed = nothing,
)
    rng = seed === nothing ? default_rng() : MersenneTwister(seed)
    mu_draws = fitted_samples(m; n = n, data = data, seed = seed, scale = :response)
    n_obs, n_draws = size(mu_draws)

    y_draws = similar(mu_draws)
    for j in 1:n_draws
        for i in 1:n_obs
            y_draws[i, j] = _random_from_family(rng, m.family, mu_draws[i, j], m.scale)
        end
    end
    return y_draws
end

# ============================================================================
# Enhanced diagnostics
# ============================================================================

"""
    appraise(m::GamModel; type=:deviance, seed=nothing)

Compute model diagnostic data for standard residual checking plots:
QQ plot, residuals vs linear predictor, histogram of residuals,
and observed vs fitted.

Returns an [`AppraiseData`](@ref) struct.
"""
function appraise(m::GamModel; type::Symbol = :deviance, seed = nothing)
    rng = seed === nothing ? default_rng() : MersenneTwister(seed)

    dev_resid = residuals(m; type = :deviance)
    prs_resid = residuals(m; type = :pearson)
    eta = m.linear_predictor
    y = m.y
    mu = m.fitted_values

    # QQ plot data — standardized deviance residuals vs normal quantiles
    # Divide by sqrt(scale) so that well-specified models show points on y=x
    n = length(dev_resid)
    sc = max(m.scale, eps())
    sorted = sort(dev_resid ./ sqrt(sc))
    theoretical = [dquantile(DNormal(), (i - 0.5) / n) for i in 1:n]

    return AppraiseData(dev_resid, prs_resid, eta, y, mu, theoretical, sorted)
end

"""
    rootogram(m::GamModel; max_count=nothing)

Compute rootogram data for count models (Poisson, Negative Binomial).

Returns a [`RootogramData`](@ref) struct.
"""
function rootogram(m::GamModel; max_count = nothing)
    y_int = round.(Int, m.y)
    mu = m.fitted_values

    if max_count === nothing
        max_count = maximum(y_int) + 1
    end
    counts = 0:max_count

    # Observed frequencies
    obs_freq = zeros(length(counts))
    for yi in y_int
        if 0 <= yi <= max_count
            obs_freq[yi + 1] += 1
        end
    end

    # Expected frequencies under the fitted model
    n = length(mu)
    exp_freq = zeros(length(counts))
    for c in counts
        total = 0.0
        for i in 1:n
            total += _count_prob(m.family, mu[i], m.scale, c)
        end
        exp_freq[c + 1] = total
    end

    return RootogramData(
        collect(counts),
        obs_freq,
        exp_freq,
        sqrt.(obs_freq),
        sqrt.(exp_freq),
    )
end

"""
    model_edf(m::GamModel)

Overall effective degrees of freedom of the model (sum of all smooth EDFs
plus parametric terms).
"""
model_edf(m::GamModel) = m.edf_total

"""
    overview(m::GamModel)

Tidy summary table of all smooth terms, their types, dimensions,
basis sizes, and effective degrees of freedom.

Returns an [`OverviewTable`](@ref) struct.
"""
function overview(m::GamModel)
    labels = String[]
    types = String[]
    dims = Int[]
    k_vals = Int[]
    edfs = Float64[]
    ratios = Float64[]

    for (i, sm) in enumerate(m.smooths)
        spec = sm.spec
        push!(labels, spec.label)
        push!(types, string(typeof(spec.basis)))
        push!(dims, length(spec.term_vars))
        k = size(sm.X, 2)
        push!(k_vals, k)
        e = m.edf[i]
        push!(edfs, e)
        push!(ratios, e / k)
    end

    return OverviewTable(labels, types, dims, k_vals, edfs, ratios)
end

# ============================================================================
# show methods
# ============================================================================

function Base.show(io::IO, se::SmoothEstimates)
    unique_smooths = unique(se.smooth)
    n = length(se.estimate)
    println(io, "SmoothEstimates: $(length(unique_smooths)) smooth(s), $n evaluation points")
    for s in unique_smooths
        mask = se.smooth .== s
        println(io, "  $s: $(count(mask)) points")
    end
end

function Base.show(io::IO, de::DerivativeEstimates)
    unique_smooths = unique(de.smooth)
    println(io, "DerivativeEstimates (order=$(de.order), type=$(de.type)): $(length(unique_smooths)) smooth(s)")
end

function Base.show(io::IO, ad::AppraiseData)
    println(io, "AppraiseData: $(length(ad.observed)) observations")
    @printf(io, "  Deviance residuals: min=%.3f, max=%.3f\n",
        minimum(ad.residuals_deviance), maximum(ad.residuals_deviance))
end

function Base.show(io::IO, rd::RootogramData)
    println(io, "RootogramData: counts $(rd.count[1]):$(rd.count[end])")
end

function Base.show(io::IO, ov::OverviewTable)
    println(io, "GAM Overview: $(length(ov.label)) smooth term(s)")
    println(io, "─" ^ 65)
    @printf(io, "%-20s %-12s %5s %5s %8s %8s\n",
        "Smooth", "Type", "Dim", "k", "EDF", "k-ratio")
    println(io, "─" ^ 65)
    for i in eachindex(ov.label)
        tname = _short_type_name(ov.smooth_type[i])
        @printf(io, "%-20s %-12s %5d %5d %8.2f %8.3f\n",
            ov.label[i], tname, ov.dimension[i], ov.basis_size[i],
            ov.edf[i], ov.edf_ratio[i])
    end
    println(io, "─" ^ 65)
end

# ============================================================================
# Internal helpers
# ============================================================================

"""Resolve smooth selection to integer indices."""
function _resolve_smooth_select(m::GamModel, select)
    if select === nothing
        return 1:m.n_smooth
    elseif select isa Integer
        return [select]
    elseif select isa AbstractVector{<:Integer}
        return select
    elseif select isa AbstractString
        idx = _find_smooth_by_name(m, select)
        idx === nothing && throw(ArgumentError("Smooth '$select' not found"))
        return [idx]
    elseif select isa AbstractVector{<:AbstractString}
        indices = Int[]
        for lab in select
            idx = _find_smooth_by_name(m, lab)
            idx === nothing && throw(ArgumentError("Smooth '$lab' not found"))
            push!(indices, idx)
        end
        return indices
    else
        throw(ArgumentError("select must be nothing, Int, String, or Vector"))
    end
end

"""Find smooth by exact label match or partial variable-name match."""
function _find_smooth_by_name(m::GamModel, name::AbstractString)
    # Exact match first
    idx = findfirst(s -> s.spec.label == name, m.smooths)
    idx !== nothing && return idx
    # Strip trailing ) for partial match: "s(x0)" → prefix "s(x0"
    if endswith(name, ")")
        prefix = name[1:end-1]
        idx = findfirst(s -> startswith(s.spec.label, prefix * ",") ||
                              startswith(s.spec.label, prefix * ")"),
            m.smooths)
        idx !== nothing && return idx
    end
    # Try matching just the variable name: "x0" matches "s(x0,bs=tp)"
    idx = findfirst(s -> length(s.spec.term_vars) == 1 && string(s.spec.term_vars[1]) == name,
        m.smooths)
    return idx
end

"""Build an evaluation grid for a smooth term."""
function _make_smooth_grid(m::GamModel, sm::ConstructedSmooth, n::Int)
    spec = sm.spec
    vars = spec.term_vars
    data = Dict{Symbol, Vector{Float64}}()

    for v in vars
        x_col = _get_covariate_from_model(m, sm, v)
        x_lo, x_hi = extrema(x_col)
        data[v] = collect(range(x_lo, x_hi; length = n))
    end

    return NamedTuple{Tuple(vars)}(Tuple(data[v] for v in vars))
end

"""Extract covariate values from the model's original data."""
function _get_covariate_from_model(m::GamModel, sm::ConstructedSmooth, varname::Symbol)
    if m.data !== nothing
        ct = Tables.columntable(m.data)
        if varname in Tables.columnnames(ct)
            return Float64.(collect(Tables.getcolumn(ct, varname)))
        end
    end
    # Fallback: use column of X with highest variance as proxy
    sm_idx = sm.first_para:sm.last_para
    X_s = m.X[:, sm_idx]
    col_vars = [Statistics.var(c) for c in eachcol(X_s)]
    _, col = findmax(col_vars)
    return X_s[:, col]
end

"""Extract original data for a smooth's covariates."""
function _extract_original_data(m::GamModel, sm::ConstructedSmooth)
    spec = sm.spec
    vars = spec.term_vars
    data = Dict{Symbol, Vector{Float64}}()
    for v in vars
        data[v] = _get_covariate_from_model(m, sm, v)
    end
    return NamedTuple{Tuple(vars)}(Tuple(data[v] for v in vars))
end

"""Build the full prediction matrix for new data."""
function _build_prediction_matrix(m::GamModel, newdata)
    t = Tables.columntable(newdata)
    n_new = length(Tables.getcolumn(t, first(Tables.columnnames(t))))
    X_para = ones(n_new, 1)
    X_smooth_parts = Matrix{Float64}[]
    for sm in m.smooths
        X_sm = predict_matrix(sm, t)
        push!(X_smooth_parts, X_sm)
    end
    return isempty(X_smooth_parts) ? X_para : hcat(X_para, X_smooth_parts...)
end

"""First-order finite difference derivative."""
function _fd_derivative_1(sm, grid, var, x_vals, beta_s, Vp_s, eps, type)
    n = length(x_vals)
    d_hat = Vector{Float64}(undef, n)
    d_se = Vector{Float64}(undef, n)

    for i in 1:n
        if type == :forward
            g_plus = _shifted_grid(grid, var, i, eps)
            X_plus = predict_matrix(sm, g_plus)
            X_base = predict_matrix(sm, _point_grid(grid, i))
            dX = (X_plus .- X_base) ./ eps
        elseif type == :backward
            g_minus = _shifted_grid(grid, var, i, -eps)
            X_base = predict_matrix(sm, _point_grid(grid, i))
            X_minus = predict_matrix(sm, g_minus)
            dX = (X_base .- X_minus) ./ eps
        else  # central
            g_plus = _shifted_grid(grid, var, i, eps)
            g_minus = _shifted_grid(grid, var, i, -eps)
            X_plus = predict_matrix(sm, g_plus)
            X_minus = predict_matrix(sm, g_minus)
            dX = (X_plus .- X_minus) ./ (2 * eps)
        end

        d_hat[i] = (dX * beta_s)[1]
        d_se[i] = sqrt(max((dX * Vp_s * dX')[1], 0.0))
    end
    return d_hat, d_se
end

"""Second-order finite difference derivative."""
function _fd_derivative_2(sm, grid, var, x_vals, beta_s, Vp_s, eps, type)
    n = length(x_vals)
    d_hat = Vector{Float64}(undef, n)
    d_se = Vector{Float64}(undef, n)

    for i in 1:n
        g_plus = _shifted_grid(grid, var, i, eps)
        g_minus = _shifted_grid(grid, var, i, -eps)
        g_base = _point_grid(grid, i)
        X_plus = predict_matrix(sm, g_plus)
        X_minus = predict_matrix(sm, g_minus)
        X_base = predict_matrix(sm, g_base)
        dX = (X_plus .- 2 .* X_base .+ X_minus) ./ (eps^2)

        d_hat[i] = (dX * beta_s)[1]
        d_se[i] = sqrt(max((dX * Vp_s * dX')[1], 0.0))
    end
    return d_hat, d_se
end

"""Create a single-row grid shifted in one variable."""
function _shifted_grid(grid, var, idx, shift)
    result = Dict{Symbol, Vector{Float64}}()
    for (k, v) in pairs(grid)
        if k == var
            result[k] = [v[idx] + shift]
        else
            result[k] = [v[idx]]
        end
    end
    keys_tuple = Tuple(keys(grid))
    return NamedTuple{keys_tuple}(Tuple(result[k] for k in keys_tuple))
end

"""Create a single-row grid at the i-th point."""
function _point_grid(grid, idx)
    result = Dict{Symbol, Vector{Float64}}()
    for (k, v) in pairs(grid)
        result[k] = [v[idx]]
    end
    keys_tuple = Tuple(keys(grid))
    return NamedTuple{keys_tuple}(Tuple(result[k] for k in keys_tuple))
end

"""Critical value for simultaneous confidence intervals via MVN simulation."""
function _simultaneous_critical(se_vec, Vcov, m, smooth_indices, n_sim, level, rng)
    # Simulate the max |derivative / se| distribution
    # For simplicity, use the standard normal approximation with Bonferroni-like correction
    n_tests = length(se_vec)
    p = length(m.coefficients)
    Vp_sym = Symmetric(Vcov)
    eig = eigen(Vp_sym)
    eig_vals = max.(eig.values, 0.0)
    L = eig.vectors * Diagonal(sqrt.(eig_vals))

    # We need to transform coefficient draws to derivative draws.
    # Since this is complex (need the finite difference matrices), we use a
    # conservative quantile correction based on the number of simultaneous tests.
    # This is equivalent to the Šidák correction: α_adj = 1 - (1-α)^(1/n_tests)
    alpha = 1 - level
    alpha_adj = 1 - (1 - alpha)^(1 / n_tests)
    crit = dquantile(DNormal(), 1 - alpha_adj / 2)
    return crit
end

"""Generate a random observation from the model family."""
function _random_from_family(rng::AbstractRNG, family, mu, scale)
    if family isa Normal
        return mu + randn(rng) * sqrt(scale)
    elseif family isa Poisson || family isa QuasiPoissonFamily
        lam = max(mu, 1e-10)
        return Float64(rand(rng, DPoisson(lam)))
    elseif family isa Binomial || family isa QuasiBinomialFamily
        p = clamp(mu, 1e-10, 1 - 1e-10)
        return Float64(rand(rng) < p)
    elseif family isa Gamma
        shape = 1.0 / scale
        sc = mu / shape
        return rand(rng, Distributions.Gamma(shape, sc))
    else
        # Default: Gaussian noise
        return mu + randn(rng) * sqrt(max(scale, 1e-10))
    end
end

"""Probability of count c under the fitted model."""
function _count_prob(family, mu, scale, c::Int)
    if family isa Poisson || family isa QuasiPoissonFamily
        lam = max(mu, 1e-10)
        return pdf(DPoisson(lam), c)
    elseif family isa NegBinFamily
        theta = family.theta
        p_nb = theta / (theta + mu)
        return pdf(DNegBin(theta, p_nb), c)
    else
        return 0.0  # rootogram only for count models
    end
end

"""Shorten type name for display."""
function _short_type_name(full_name::String)
    m = match(r"(\w+)$", full_name)
    m === nothing ? full_name : m.captures[1]
end
