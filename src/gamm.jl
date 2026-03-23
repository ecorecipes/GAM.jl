# GAMM — Generalized Additive Mixed Models
#
# Extends GAM with explicit random effects (random intercepts, random slopes).
# Smooths are already implicitly random effects via smooth2random(); GAMM adds
# user-specified grouped random effects alongside.
#
# Three backends:
#   1. LAMS (default) — extends existing LAMS outer loop with σ²_re parameters
#   2. MixedModels.jl — optional package extension
#   3. Turing.jl — Bayesian, via existing GAMTuringExt
#
# References:
#   Wood, S.N. (2017). Generalized Additive Models, 2nd ed. §6.6
#   Pedersen et al. (2019). Hierarchical GAMs in ecology. PeerJ 7:e6876.

# ============================================================================
# RandomEffectSpec — specification of a grouped random effect
# ============================================================================

"""
    RandomEffectSpec

Specification for a grouped random effect term in a GAMM formula.

# Fields
- `grouping::Symbol`: name of the grouping factor variable
- `terms::Vector{Symbol}`: variables for random slopes (empty = intercept only)
- `has_intercept::Bool`: whether to include a random intercept
- `correlated::Bool`: if true, intercept+slopes share a correlation matrix (default)
- `label::String`: human-readable label
"""
struct RandomEffectSpec
    grouping::Symbol
    terms::Vector{Symbol}
    has_intercept::Bool
    correlated::Bool
    label::String
end

function Base.show(io::IO, re::RandomEffectSpec)
    lhs = String[]
    re.has_intercept && push!(lhs, "1")
    for t in re.terms
        push!(lhs, string(t))
    end
    expr = join(lhs, " + ")
    print(io, "(", expr, " | ", re.grouping, ")")
end

# ============================================================================
# ConstructedRandomEffect — after building Z matrix from data
# ============================================================================

"""
    ConstructedRandomEffect

A random effect term after construction from data. Contains the design matrix Z,
the grouping factor levels, and the penalty/precision structure.

# Fields
- `spec::RandomEffectSpec`: original specification
- `Z::Matrix{Float64}`: random-effect design matrix (n × q), possibly constrained
- `levels::Vector`: unique levels of the grouping factor
- `n_levels::Int`: number of grouping levels
- `n_terms::Int`: number of random effect terms per group (1 for intercept, more for slopes)
- `penalty::Matrix{Float64}`: precision structure (identity for independent, block for correlated)
- `block_dim::Int`: dimension of random effect vector (may be < n_levels*n_terms after constraints)
- `constraint_basis::Union{Matrix{Float64}, Nothing}`: basis for back-transforming to per-level effects
"""
struct ConstructedRandomEffect
    spec::RandomEffectSpec
    Z::Matrix{Float64}
    levels::Vector
    n_levels::Int
    n_terms::Int
    penalty::Matrix{Float64}
    block_dim::Int
    constraint_basis::Union{Matrix{Float64}, Nothing}
end

"""
    construct_random_effect(spec::RandomEffectSpec, data) → ConstructedRandomEffect

Build the random-effect design matrix Z from a `RandomEffectSpec` and data.

For `(1|group)`: Z is a n × n_levels indicator matrix.
For `(x|group)`: Z is n × (2 * n_levels) with intercept + slope columns per group.
For `(0 + x|group)`: Z is n × n_levels with slope columns only.
"""
function construct_random_effect(spec::RandomEffectSpec, data)
    t = Tables.columntable(data)
    group_col = Tables.getcolumn(t, spec.grouping)
    n = length(group_col)

    levels = sort(unique(group_col))
    n_levels = length(levels)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Number of terms per group
    n_re_terms = (spec.has_intercept ? 1 : 0) + length(spec.terms)
    n_re_terms > 0 || throw(ArgumentError(
        "Random effect $(spec.label) has no terms (need intercept or slopes)"))

    q = n_levels * n_re_terms  # total random effect dimension
    Z = zeros(n, q)

    for i in 1:n
        j = level_map[group_col[i]]  # group index (1-based)
        col_offset = (j - 1) * n_re_terms

        term_idx = 0
        if spec.has_intercept
            term_idx += 1
            Z[i, col_offset + term_idx] = 1.0
        end

        for s_var in spec.terms
            term_idx += 1
            x_col = Tables.getcolumn(t, s_var)
            Z[i, col_offset + term_idx] = Float64(x_col[i])
        end
    end

    # Penalty: identity (each random effect penalized equally)
    S_re = Matrix{Float64}(I, q, q)
    Zn_basis = nothing  # constraint basis for back-transformation

    # Apply sum-to-zero constraint for random intercepts (identifiability with fixed intercept)
    if spec.has_intercept && n_re_terms == 1
        C = ones(1, n_levels)
        Zn = nullspace(C)  # n_levels × (n_levels - 1)
        Z = Z * Zn
        S_re = Zn' * S_re * Zn
        Zn_basis = Zn
        q = size(Z, 2)
    elseif spec.has_intercept && n_re_terms > 1
        int_cols = 1:n_re_terms:q
        C = zeros(1, q)
        for ic in int_cols
            C[1, ic] = 1.0
        end
        Zn = nullspace(C)
        Z = Z * Zn
        S_re = Zn' * S_re * Zn
        Zn_basis = Zn
        q = size(Z, 2)
    end

    return ConstructedRandomEffect(spec, Z, levels, n_levels, n_re_terms, S_re, q, Zn_basis)
end

"""
    predict_re_matrix(cre::ConstructedRandomEffect, newdata) → Matrix{Float64}

Build Z matrix for new data. New levels get zero columns (no RE contribution).
If the grouping column is missing from newdata, returns all zeros (population average).
"""
function predict_re_matrix(cre::ConstructedRandomEffect, newdata)
    t = Tables.columntable(newdata)

    # If the grouping column is missing entirely, return zeros (population average)
    col_names = Tables.columnnames(t)
    if !(cre.spec.grouping in col_names)
        n_new = length(Tables.getcolumn(t, first(col_names)))
        q = cre.constraint_basis !== nothing ? size(cre.constraint_basis, 2) : cre.n_levels * cre.n_terms
        return zeros(n_new, q)
    end

    group_col = Tables.getcolumn(t, cre.spec.grouping)
    n_new = length(group_col)

    level_map = Dict(lev => i for (i, lev) in enumerate(cre.levels))
    n_levels = cre.n_levels
    n_terms = cre.n_terms
    q_full = n_levels * n_terms

    # Build unconstrained Z first
    Z_raw = zeros(n_new, q_full)
    for i in 1:n_new
        j = get(level_map, group_col[i], 0)
        j == 0 && continue
        col_offset = (j - 1) * n_terms
        term_idx = 0
        if cre.spec.has_intercept
            term_idx += 1
            Z_raw[i, col_offset + term_idx] = 1.0
        end
        for s_var in cre.spec.terms
            term_idx += 1
            x_col = Tables.getcolumn(t, s_var)
            Z_raw[i, col_offset + term_idx] = Float64(x_col[i])
        end
    end

    # Apply constraint basis if present
    if cre.constraint_basis !== nothing
        return Z_raw * cre.constraint_basis
    end
    return Z_raw
end

# ============================================================================
# GammModel — fitted GAMM result
# ============================================================================

"""
    GammModel{D, L}

A fitted Generalized Additive Mixed Model. Extends `GamModel` with grouped
random effects information.

# Additional fields beyond GamModel
- `random_effects`: constructed random effect terms
- `random_coefs`: estimated BLUPs for each random effect
- `random_vars`: estimated variance components (σ² per RE block)
- `gam_model`: the underlying GamModel with smooth + fixed effects
"""
mutable struct GammModel{D, L<:GLM.Link}
    gam_model::GamModel{D, L}
    random_effects::Vector{ConstructedRandomEffect}
    random_coefs::Vector{Vector{Float64}}
    random_vars::Vector{Float64}
end

# Forward StatsAPI methods to underlying gam_model
StatsAPI.coef(m::GammModel) = coef(m.gam_model)
StatsAPI.vcov(m::GammModel) = vcov(m.gam_model)
StatsAPI.fitted(m::GammModel) = fitted(m.gam_model)
StatsAPI.residuals(m::GammModel) = residuals(m.gam_model)
StatsAPI.nobs(m::GammModel) = nobs(m.gam_model)
StatsAPI.deviance(m::GammModel) = deviance(m.gam_model)
StatsAPI.loglikelihood(m::GammModel) = loglikelihood(m.gam_model)
StatsAPI.dof(m::GammModel) = dof(m.gam_model) + sum(length.(m.random_coefs))
StatsAPI.response(m::GammModel) = response(m.gam_model)

function StatsAPI.predict(m::GammModel, newdata)
    t = Tables.columntable(newdata)
    n_new = length(Tables.getcolumn(t, first(Tables.columnnames(t))))

    gm = m.gam_model
    n_re = length(m.random_effects)
    n_actual_smooths = gm.n_smooth - n_re

    # Build parametric part (intercept)
    X_para = ones(n_new, 1)

    # Build smooth parts (actual smooths only, not RE-as-smooth)
    X_smooth_parts = Matrix{Float64}[]
    for idx in 1:n_actual_smooths
        sm = gm.smooths[idx]
        X_sm = predict_matrix(sm, t)
        push!(X_smooth_parts, X_sm)
    end

    X_new = isempty(X_smooth_parts) ? X_para : hcat(X_para, X_smooth_parts...)

    # Fixed + smooth coefficients (exclude RE coefficients)
    n_fixed_smooth = size(X_new, 2)
    β_fs = gm.coefficients[1:n_fixed_smooth]
    ŷ = X_new * β_fs

    # Add random effect contributions via predict_re_matrix
    # (handles missing/unknown groups gracefully)
    for (i, cre) in enumerate(m.random_effects)
        Z_new = predict_re_matrix(cre, newdata)
        ŷ .+= Z_new * m.random_coefs[i]
    end
    return ŷ
end

"""
    ranef(m::GammModel) → NamedTuple

Extract random effect estimates (BLUPs) for each grouping factor.
Returns a NamedTuple mapping group name → DataFrame-like structure.
"""
function ranef(m::GammModel)
    result = Dict{Symbol, Any}()
    for (i, cre) in enumerate(m.random_effects)
        b = m.random_coefs[i]
        n_levels = cre.n_levels
        n_terms = cre.n_terms

        # Back-transform from constrained to per-level effects
        if cre.constraint_basis !== nothing
            b_full = cre.constraint_basis * b  # (n_levels*n_terms) × 1
        else
            b_full = b
        end

        # Reshape: each row = one group level, each col = one term
        B = reshape(b_full, n_terms, n_levels)'

        col_names = Symbol[]
        if cre.spec.has_intercept
            push!(col_names, :Intercept)
        end
        for sv in cre.spec.terms
            push!(col_names, sv)
        end

        result[cre.spec.grouping] = (
            levels = cre.levels,
            effects = B,
            names = col_names,
        )
    end
    return (; result...)
end

"""
    VarCorr(m::GammModel) → VarCorrResult

Variance component estimates for random effects. The returned object prints
a clean variance component table and supports indexing.
"""
function VarCorr(m::GammModel)
    vc = NamedTuple[]
    for (i, cre) in enumerate(m.random_effects)
        v = max(m.random_vars[i], 0.0)
        push!(vc, (
            group = cre.spec.grouping,
            label = cre.spec.label,
            variance = v,
            std = sqrt(v),
            n_levels = cre.n_levels,
            n_terms = cre.n_terms,
        ))
    end

    # Add residual variance
    scale = m.gam_model.scale
    push!(vc, (
        group = :Residual,
        label = "Residual",
        variance = scale,
        std = sqrt(max(scale, 0.0)),
        n_levels = Int(nobs(m)),
        n_terms = 0,
    ))
    return VarCorrResult(vc)
end

"""
    VarCorrResult

Container for variance component estimates with pretty-printing.
Supports indexing to access individual components.
"""
struct VarCorrResult
    components::Vector{<:NamedTuple}
end

Base.length(v::VarCorrResult) = length(v.components)
Base.getindex(v::VarCorrResult, i) = v.components[i]
Base.iterate(v::VarCorrResult, args...) = iterate(v.components, args...)
Base.lastindex(v::VarCorrResult) = lastindex(v.components)

function Base.show(io::IO, vc::VarCorrResult)
    println(io, "Variance Components:")
    println(io, " ", rpad("Group", 20), "  ", rpad("Term", 20), "  ",
        lpad("Variance", 12), "  ", lpad("Std.Dev.", 12), "  ",
        lpad("Levels", 8))
    println(io, " ", repeat("─", 78))
    for c in vc.components
        group_str = string(c.group)
        label_str = c.group == :Residual ? "" : c.label
        lev_str = c.group == :Residual ? "" : string(c.n_levels)
        @printf(io, " %-20s  %-20s  %12.6f  %12.6f  %8s\n",
            group_str, label_str, c.variance, c.std, lev_str)
    end
end

# ============================================================================
# GAMM fitting — LAMS backend
# ============================================================================

"""
    _fit_gamm_lams(y, X, smooths, n_parametric, random_effects, f, data,
                   family, link, method, weights, control) → GammModel

Fit a GAMM by converting random effects to ConstructedSmooth{RandomEffect}
objects and delegating to the existing GAM fitting machinery (_fit_gam).

Each random effect becomes an additional smooth with identity penalty,
so the smooth REML/GCV machinery automatically estimates the
RE variance (1/λ_re) alongside the smooth penalties.
"""
function _fit_gamm_lams(y, X_gam, smooths, n_parametric,
    random_effects::Vector{ConstructedRandomEffect},
    f, data, family, link, method, weights, control)

    n = length(y)
    n_re = length(random_effects)

    # Convert random effects to ConstructedSmooth{RandomEffect} objects
    re_smooths = ConstructedSmooth[]
    for cre in random_effects
        # Create a SmoothSpec for this RE
        re_spec = SmoothSpec(
            [cre.spec.grouping],
            RandomEffect(),
            cre.block_dim,
            nothing, nothing, nothing,
            false, nothing,
            cre.spec.label)

        # The Z matrix IS the design matrix, penalty is cre.penalty (≈I after constraint)
        re_sm = ConstructedSmooth(
            re_spec,
            cre.Z,                          # design matrix
            [cre.penalty],                  # identity penalty
            Float64.(1:cre.block_dim),      # dummy knots
            0,                              # no null space (all penalized)
            cre.block_dim,                  # full rank penalty
            nothing, nothing,               # no constraints (already absorbed)
            0, 0,                           # first/last_para — set below
            nothing, nothing, nothing,
            Int[])      # no SCAM metadata, no side constraints
        push!(re_smooths, re_sm)
    end

    # Combine GAM smooths + RE smooths
    all_smooths = vcat(smooths, re_smooths)

    # Set first_para/last_para for RE smooths (appended after GAM smooths)
    p_start = isempty(smooths) ? n_parametric + 1 :
              smooths[end].last_para + 1
    for re_sm in re_smooths
        k = size(re_sm.X, 2)
        re_sm.first_para = p_start
        re_sm.last_para = p_start + k - 1
        p_start += k
    end

    # Build augmented model matrix
    X_re_parts = [re_sm.X for re_sm in re_smooths]
    X_aug = isempty(X_re_parts) ? X_gam : hcat(X_gam, X_re_parts...)

    # Delegate to existing _fit_gam
    gam_model = _fit_gam(y, X_aug, all_smooths, n_parametric,
        f, data, family, link, method, :pirls, weights, control)

    # Extract RE information from the fitted model
    re_coefs = Vector{Float64}[]
    re_vars = Float64[]
    n_gam_smooths = length(smooths)
    β = StatsAPI.coef(gam_model)

    for (i, re_sm) in enumerate(re_smooths)
        b = β[re_sm.first_para:re_sm.last_para]
        push!(re_coefs, b)

        # Variance = scale / λ where λ = exp(log_sp)
        # GamModel.sp stores log-scale smoothing parameters
        # Find the sp index for this smooth
        sp_idx = 0
        for (j, sm) in enumerate(all_smooths)
            if sm === re_sm
                sp_idx = j
                break
            end
        end
        if sp_idx > 0 && sp_idx <= length(gam_model.sp)
            λ = exp(gam_model.sp[sp_idx])
            push!(re_vars, gam_model.scale / λ)
        else
            push!(re_vars, gam_model.scale)
        end
    end

    # Rebuild GamModel with only the GAM part (strip RE columns)
    p_gam = size(X_gam, 2)
    β_gam = β[1:p_gam]
    η_gam = X_gam * β_gam
    for (i, cre) in enumerate(random_effects)
        η_gam .+= cre.Z * re_coefs[i]
    end
    μ_gam = GLM.linkinv.(Ref(link), η_gam)

    # Keep the full model for now but wrap it
    return GammModel(gam_model, random_effects, re_coefs, re_vars)
end

# ============================================================================
# PQL backend — Penalized Quasi-Likelihood (matches R's gamm() for non-Gaussian)
# ============================================================================

"""
    _fit_gamm_pql(y, X_gam, smooths, n_parametric, random_effects,
                  f, data, family, link, method, weights, control) → GammModel

Fit a non-Gaussian GAMM via Penalized Quasi-Likelihood (PQL), matching the
algorithm used by R's `mgcv::gamm()` which calls `MASS::glmmPQL`.

PQL iterates between:
1. Computing GLM working response z and working weights W
2. Fitting a Gaussian GAMM (via LAMS) to the working model

This provides closer agreement with R's `gamm()` for non-Gaussian families
than the direct PIRLS+REML approach used by `_fit_gamm_lams`.

Reference: Breslow & Clayton (1993). JASA 88:9-25.
"""
function _fit_gamm_pql(y, X_gam, smooths, n_parametric,
    random_effects::Vector{ConstructedRandomEffect},
    f, data, family, link, method, weights, control)

    n = length(y)
    w_prior = weights !== nothing ? Float64.(weights) : ones(n)

    # Initialize from a GLM fit (ignoring random effects)
    η = zeros(n)
    μ = zeros(n)

    # Start with intercept-only: μ = mean(y), η = g(μ)
    y_mean = mean(y)
    # Clamp for safety with link functions
    if family isa Poisson
        y_mean = max(y_mean, 0.1)
    elseif family isa Binomial
        y_mean = clamp(y_mean, 0.01, 0.99)
    elseif family isa Gamma
        y_mean = max(y_mean, 0.01)
    end
    fill!(μ, y_mean)
    fill!(η, GLM.linkfun(link, y_mean))

    # PQL iteration settings
    max_pql_iter = 20
    pql_tol = 1e-6

    prev_coefs = Float64[]
    gamm_result = nothing

    for pql_iter in 1:max_pql_iter
        # Step 1: Compute working response and weights
        z = zeros(n)    # working response
        w = zeros(n)    # working weights

        @inbounds for i in 1:n
            dm = GLM.mueta(link, η[i])        # dμ/dη = g'⁻¹(η)
            dm = clamp(dm, 1e-10, 1e10)
            vm = _variance_scalar(family, μ[i])
            vm = max(vm, 1e-10)

            # Working response: z = η + (y - μ) / (dμ/dη)
            z[i] = η[i] + (y[i] - μ[i]) / dm

            # Working weights: w = prior_w * (dμ/dη)² / V(μ)
            w[i] = clamp(w_prior[i] * dm * dm / vm, 1e-10, 1e10)
        end

        # Step 2: Fit Gaussian GAMM to working response z with weights w
        gamm_result = _fit_gamm_lams(z, X_gam, smooths, n_parametric,
            random_effects, f, data, Normal(), IdentityLink(), method, w, control)

        # Step 3: Extract updated linear predictor
        gm = gamm_result.gam_model
        β_all = StatsAPI.coef(gm)
        p_gam = size(X_gam, 2)
        β_gam = β_all[1:p_gam]
        η_new = X_gam * β_gam

        # Add random effect contributions
        for (i, cre) in enumerate(random_effects)
            η_new .+= cre.Z * gamm_result.random_coefs[i]
        end

        # Update μ from η
        μ_new = GLM.linkinv.(Ref(link), η_new)

        # Check convergence
        current_coefs = β_all
        if !isempty(prev_coefs) && length(prev_coefs) == length(current_coefs)
            max_change = maximum(abs.(current_coefs .- prev_coefs) ./
                                 max.(abs.(prev_coefs), 1e-8))
            if max_change < pql_tol
                η .= η_new
                μ .= μ_new
                break
            end
        end

        prev_coefs = copy(current_coefs)
        η .= η_new
        μ .= μ_new
    end

    # Final result: update fitted values to response scale
    if gamm_result !== nothing
        gm = gamm_result.gam_model
        # Recompute fitted values on response scale
        β_all = StatsAPI.coef(gm)
        p_gam = size(X_gam, 2)
        η_final = X_gam * β_all[1:p_gam]
        for (i, cre) in enumerate(random_effects)
            η_final .+= cre.Z * gamm_result.random_coefs[i]
        end
        μ_final = GLM.linkinv.(Ref(link), η_final)

        # Create a new GamModel with correct family, link, and fitted values
        gm_pql = GamModel(
            gm.formula, y, gm.X, gm.coefficients,
            μ_final, η_final, gm.weights,
            family, link,
            gm.smooths, gm.penalty, gm.sp, gm.edf, gm.edf_total,
            gm.scale, gm.deviance_val, gm.null_deviance, gm.reml,
            gm.method, gm.Vp, gm.Ve, gm.hat_matrix_diag, gm.R,
            gm.converged, gm.iterations, gm.n_smooth, gm.n_parametric,
            gm.control, gm.data)

        return GammModel(gm_pql, random_effects,
                         gamm_result.random_coefs, gamm_result.random_vars)
    end

    # Fallback: return LAMS result
    return _fit_gamm_lams(y, X_gam, smooths, n_parametric,
        random_effects, f, data, family, link, method, weights, control)
end

# ============================================================================
# Formula parsing for random effects
# ============================================================================

"""
    _parse_random_effect(ex) → RandomEffectSpec

Parse an expression like `(1 | group)`, `(x | group)`, `(1 + x | group)`,
`(0 + x | group)` into a `RandomEffectSpec`.
"""
function _parse_random_effect(ex::Expr)
    # ex should be a call to | with two arguments
    # (1 | group) parses as Expr(:call, :|, 1, :group)
    ex.head == :call && ex.args[1] == :(|) ||
        throw(ArgumentError("Expected (... | group), got $ex"))

    lhs = ex.args[2]
    rhs_expr = ex.args[3]

    # RHS is the grouping factor
    grouping = if rhs_expr isa Symbol
        rhs_expr
    else
        throw(ArgumentError("Grouping factor must be a symbol, got $rhs_expr"))
    end

    # Parse LHS for intercept and slope terms
    has_intercept = true
    terms = Symbol[]

    if lhs isa Integer
        if lhs == 1
            has_intercept = true
        elseif lhs == 0
            has_intercept = false
        end
    elseif lhs isa Symbol
        push!(terms, lhs)
    elseif lhs isa Expr && lhs.head == :call
        fname = lhs.args[1]
        if fname == :+
            for i in 2:length(lhs.args)
                arg = lhs.args[i]
                if arg isa Integer
                    if arg == 1
                        has_intercept = true
                    elseif arg == 0
                        has_intercept = false
                    end
                elseif arg isa Symbol
                    push!(terms, arg)
                end
            end
        else
            throw(ArgumentError("Unsupported random effect LHS: $lhs"))
        end
    end

    # Build label
    lhs_parts = String[]
    has_intercept && push!(lhs_parts, "1")
    for t in terms
        push!(lhs_parts, string(t))
    end
    label = "(" * join(lhs_parts, " + ") * " | " * string(grouping) * ")"

    return RandomEffectSpec(grouping, terms, has_intercept, true, label)
end

"""
    _is_random_effect_expr(ex) → Bool

Check if an expression is a random-effect term `(... | group)`.
The key marker is a call to `|` wrapped in parentheses (which Julia
represents as a bare `:call` to `:|`).
"""
function _is_random_effect_expr(ex)
    ex isa Expr || return false
    ex.head == :call || return false
    ex.args[1] == :(|) || return false
    return true
end

"""
    _is_nested_grouping(ex) → Bool

Check if the RHS of a random effect uses the `/` (nesting) operator,
e.g. `(1 | a / b)`.
"""
function _is_nested_grouping(ex)
    ex isa Expr || return false
    ex.head == :call || return false
    ex.args[1] == :(/) || return false
    return true
end

"""
    _expand_nested_re(ex::Expr) → Vector{Expr}

Expand a nested random effect `(lhs | a / b)` into two RE expressions:
`(lhs | a)` and `(lhs | a_b)`, where `a_b` represents the interaction `a:b`.

Following lme4 convention: `(1|a/b)` expands to `(1|a) + (1|a:b)`.
"""
function _expand_nested_re(ex::Expr)
    lhs = ex.args[2]
    rhs_nested = ex.args[3]  # Expr(:call, :/, :a, :b)
    outer = rhs_nested.args[2]  # :a
    inner = rhs_nested.args[3]  # :b

    # Create interaction symbol a_b for the nested term
    interaction_sym = Symbol(string(outer), "_", string(inner))

    # (lhs | a)
    re1 = Expr(:call, :(|), lhs, outer)
    # (lhs | a_b)  — interaction grouping variable
    re2 = Expr(:call, :(|), lhs, interaction_sym)

    return [re1, re2]
end

# ─── Extend _parse_gam_rhs! to handle (1|group) ─────────────────────────────

# We don't modify _parse_gam_rhs! directly; instead, the gamm pathway
# calls _parse_gamm_rhs! which delegates non-RE terms to _parse_gam_rhs!.

function _parse_gamm_rhs!(ex, parametric, smooth_calls, re_calls, has_intercept)
    if _is_random_effect_expr(ex)
        rhs_expr = ex.args[3]
        if _is_nested_grouping(rhs_expr)
            # Expand (lhs | a/b) into (lhs|a) + (lhs|a_b)
            for expanded in _expand_nested_re(ex)
                push!(re_calls.args, _build_re_spec_call(expanded))
            end
        else
            push!(re_calls.args, _build_re_spec_call(ex))
        end
    elseif ex isa Expr && ex.head == :call && ex.args[1] == :+
        for i in 2:length(ex.args)
            _parse_gamm_rhs!(ex.args[i], parametric, smooth_calls, re_calls, has_intercept)
        end
    else
        # Delegate to standard GAM RHS parser
        _parse_gam_rhs!(ex, parametric, smooth_calls, has_intercept)
    end
end

function _build_re_spec_call(ex::Expr)
    # Convert (1 + x | group) to a RandomEffectSpec(...) expression for the macro
    lhs = ex.args[2]
    grouping = ex.args[3]

    has_intercept = true
    term_syms = Symbol[]

    if lhs isa Integer
        if lhs == 1
            has_intercept = true
        elseif lhs == 0
            has_intercept = false
        end
    elseif lhs isa Symbol
        push!(term_syms, lhs)
    elseif lhs isa Expr && lhs.head == :call && lhs.args[1] == :+
        for i in 2:length(lhs.args)
            arg = lhs.args[i]
            if arg isa Integer
                if arg == 0
                    has_intercept = false
                end
            elseif arg isa Symbol
                push!(term_syms, arg)
            end
        end
    end

    # Build label string at parse time
    lhs_parts = String[]
    has_intercept && push!(lhs_parts, "1")
    for t in term_syms
        push!(lhs_parts, string(t))
    end
    label = "(" * join(lhs_parts, " + ") * " | " * string(grouping) * ")"

    # Build Symbol[] expression for terms
    terms_expr = Expr(:ref, :Symbol, [QuoteNode(s) for s in term_syms]...)

    return Expr(:call, :(GAM.RandomEffectSpec),
        QuoteNode(grouping),
        terms_expr,
        has_intercept,
        true,
        label)
end

# ============================================================================
# GamFormula extension for GAMM — GammFormula
# ============================================================================

"""
    GammFormula

A GAMM formula: GAM formula + random effect specifications.
"""
struct GammFormula
    gam_formula::GamFormula
    random_effects::Vector{RandomEffectSpec}
end

function Base.show(io::IO, gf::GammFormula)
    show(io, gf.gam_formula)
    for re in gf.random_effects
        print(io, " + ", re)
    end
end

"""
    @gamm_formula(ex)

Parse a GAMM formula with smooth terms and random effects.

# Examples
```julia
gf = @gamm_formula(y ~ s(x, k=20) + (1|group))
gf = @gamm_formula(y ~ x1 + s(x2) + (1|site) + (x1|subject))
```
"""
macro gamm_formula(ex)
    ex.head == :call && ex.args[1] == :(~) ||
        error("Expected formula expression like `y ~ ...`, got $ex")

    lhs = ex.args[2]
    rhs = ex.args[3]

    response = QuoteNode(lhs)
    parametric = Expr(:vect)
    smooth_calls = Expr(:vect)
    re_calls = Expr(:vect)
    has_intercept = Ref(true)

    _parse_gamm_rhs!(rhs, parametric, smooth_calls, re_calls, has_intercept)

    return esc(quote
        GammFormula(
            GamFormula($(QuoteNode(lhs)),
                Symbol[$(parametric.args...)],
                $(has_intercept[]),
                SmoothSpec[$(smooth_calls.args...)]),
            RandomEffectSpec[$(re_calls.args...)])
    end)
end

# ============================================================================
# FunctionTerm{typeof(|)} detection for @formula path
# ============================================================================

"""
    _is_re_functionterm(ft) → Bool

Check if a FunctionTerm represents a random effect (using `|` operator).
StatsModels parses `@formula(y ~ (1|g))` as `FunctionTerm{typeof(|)}`.
"""
function _is_re_functionterm(ft)
    ft isa StatsModels.FunctionTerm || return false
    return ft.f === (|)
end

"""
    _functionterm_to_re_spec(ft::FunctionTerm) → RandomEffectSpec

Convert `FunctionTerm{typeof(|)}` from `@formula(y ~ (1|g))` to `RandomEffectSpec`.
"""
function _functionterm_to_re_spec(ft::StatsModels.FunctionTerm)
    # ft.args should be [lhs_term, group_term]
    length(ft.args) == 2 || throw(ArgumentError(
        "Random effect term needs exactly 2 arguments (lhs | group), got $(length(ft.args))"))

    group_term = ft.args[2]
    grouping = if group_term isa Term
        group_term.sym
    else
        throw(ArgumentError("Grouping factor must be a variable, got $group_term"))
    end

    lhs_term = ft.args[1]
    has_intercept = true
    terms = Symbol[]

    if lhs_term isa ConstantTerm
        val = round(Int, lhs_term.n)
        has_intercept = val == 1
    elseif lhs_term isa Term
        push!(terms, lhs_term.sym)
    elseif lhs_term isa Tuple
        for lt in lhs_term
            if lt isa ConstantTerm
                val = round(Int, lt.n)
                if val == 0
                    has_intercept = false
                end
            elseif lt isa Term
                push!(terms, lt.sym)
            end
        end
    end

    lhs_parts = String[]
    has_intercept && push!(lhs_parts, "1")
    for t in terms
        push!(lhs_parts, string(t))
    end
    label = "(" * join(lhs_parts, " + ") * " | " * string(grouping) * ")"

    return RandomEffectSpec(grouping, terms, has_intercept, true, label)
end

# ============================================================================
# re() convenience function for @formula compatibility
# ============================================================================

"""
    re(group)
    re(group, n_levels)

Convenience function for random intercepts in `@formula`:
```julia
@formula(y ~ s(x) + re(group))  # equivalent to (1|group)
```
"""
function re(vars...; kwargs...)
    length(vars) >= 1 || throw(ArgumentError("re() requires at least one variable"))

    grouping = vars[1]
    terms = Symbol[]
    has_intercept = true

    for v in vars[2:end]
        if v isa Symbol
            push!(terms, v)
        end
    end

    lhs_parts = String[]
    has_intercept && push!(lhs_parts, "1")
    for t in terms
        push!(lhs_parts, string(t))
    end
    label = "re(" * join([string(grouping)], ", ") * ")"

    return RandomEffectSpec(grouping, terms, has_intercept, true, label)
end

# ============================================================================
# User-facing gamm() function
# ============================================================================

"""
    gamm(formula, data; family, link, method, weights, control, priors, ...)

Fit a Generalized Additive Mixed Model (GAMM).

Extends `gam()` by supporting grouped random effects alongside smooth terms.

# Formula syntax

With `@gamm_formula` (supports kwargs on smooths):
```julia
gamm(@gamm_formula(y ~ s(x, k=20) + (1|subject)), data)
gamm(@gamm_formula(y ~ x1 + s(x2) + (1|site) + (x1|subject)), data)
```

With `@formula` (positional args only, use `re()` for random effects):
```julia
gamm(@formula(y ~ cr(x, 20) + re(subject)), data)
```

# Arguments
- `formula`: a `GammFormula` (from `@gamm_formula`), `FormulaTerm`, or `GamFormula`
- `data`: a Tables.jl-compatible data source
- `family=Normal()`: response distribution
- `link=nothing`: link function (default: canonical link for family)
- `method=:REML`: smoothing parameter estimation method
- `weights=nothing`: prior weights
- `control=gam_control()`: fitting control parameters
- `priors=nothing`: if provided, triggers Bayesian fitting via Turing.jl

# Returns
A `GammModel` containing the fitted model with random effect estimates.
"""
function gamm(gf::GammFormula, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    backend::Symbol = :LAMS,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    _validate_gam_family(family)
    link_eff = link === nothing ? GLM.canonicallink(family) : link
    _validate_link(link_eff, family)

    # Bayesian dispatch
    if priors !== nothing
        return _fit_gamm_bayes(gf, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains,
            weights = weights)
    end

    # Setup GAM part (smooths + parametric)
    y, X, X_para, smooths, n_parametric = setup_gam(gf.gam_formula, data; family = family)

    # Build random effects
    random_effects = [construct_random_effect(re, data) for re in gf.random_effects]

    f = term(gf.gam_formula.response) ~ term(1)

    # Backend dispatch
    if backend == :MixedModels
        return _fit_gamm_mm(y, X, smooths, n_parametric, random_effects,
            f, data, family, link_eff, method, weights, control)
    end

    return _fit_gamm_lams(y, X, smooths, n_parametric, random_effects,
        f, data, family, link_eff, method, weights, control)
end

# @formula dispatch: detect FunctionTerm{typeof(|)} as random effects
function gamm(f::FormulaTerm, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    backend::Symbol = :LAMS,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    control::GamControl = gam_control(),
    priors::Union{PriorSpec, Nothing} = nothing,
    sampler::Any = nothing,
    nsamples::Int = 2000,
    nchains::Int = 4)

    _validate_gam_family(family)
    link_eff = link === nothing ? GLM.canonicallink(family) : link
    _validate_link(link_eff, family)

    t = Tables.columntable(data)
    resp_col = f.lhs isa Term ? f.lhs.sym : error("LHS must be a single term")
    y = Float64.(Tables.getcolumn(t, resp_col))
    n = length(y)

    rhs_terms = _flatten_rhs(f.rhs)

    smooth_terms = AppliedSmoothTerm[]
    para_terms = StatsModels.AbstractTerm[]
    re_specs = RandomEffectSpec[]

    for term_i in rhs_terms
        if term_i isa AppliedSmoothTerm || term_i isa SmoothTerm
            ast = term_i isa SmoothTerm ? AppliedSmoothTerm(term_i.spec, nothing) : term_i
            push!(smooth_terms, ast)
        elseif term_i isa StatsModels.FunctionTerm && _is_smooth_function(term_i.f)
            spec = _functionterm_to_smoothspec(term_i)
            ast = AppliedSmoothTerm(spec, nothing)
            push!(smooth_terms, ast)
        elseif _is_re_functionterm(term_i)
            push!(re_specs, _functionterm_to_re_spec(term_i))
        elseif term_i isa StatsModels.FunctionTerm && term_i.f === re
            # re(group) function call
            push!(re_specs, _functionterm_to_re_from_call(term_i))
        else
            push!(para_terms, term_i)
        end
    end

    # If no RE terms found, fall back to regular gam()
    if isempty(re_specs)
        return gam(f, data; family, link, method, weights, control, priors,
            sampler, nsamples, nchains)
    end

    # Build parametric matrix
    X_para = ones(n, 1)
    n_parametric = 1
    for pt in para_terms
        if pt isa InterceptTerm{true} || pt isa InterceptTerm{false}
            continue
        elseif pt isa Term
            col = Float64.(Tables.getcolumn(t, pt.sym))
            X_para = hcat(X_para, col)
            n_parametric += 1
        elseif pt isa ContinuousTerm
            col = Float64.(modelcols(pt, t))
            X_para = hcat(X_para, col)
            n_parametric += 1
        end
    end

    # Construct smooths
    smooths = ConstructedSmooth[]
    for st in smooth_terms
        sm = smooth_construct(st.spec, t)
        st.smooth = sm
        push!(smooths, sm)
    end

    p_start = n_parametric + 1
    for sm in smooths
        k = size(sm.X, 2)
        sm.first_para = p_start
        sm.last_para = p_start + k - 1
        p_start += k
    end

    X_smooth_parts = [sm.X for sm in smooths]
    X = isempty(X_smooth_parts) ? X_para : hcat(X_para, X_smooth_parts...)

    # Build random effects
    random_effects = [construct_random_effect(re, data) for re in re_specs]

    # Bayesian dispatch
    if priors !== nothing
        return _fit_gamm_bayes_from_parts(y, X, smooths, n_parametric, random_effects,
            f, data, family, link_eff, priors;
            sampler = sampler, nsamples = nsamples, nchains = nchains, weights = weights)
    end

    # MixedModels.jl backend dispatch
    if backend == :MixedModels
        return _fit_gamm_mm(y, X, smooths, n_parametric, random_effects,
            f, data, family, link_eff, method, weights, control)
    end

    # PQL backend: use for non-Gaussian, or when explicitly requested
    if backend == :PQL || (backend == :LAMS && !(family isa Normal))
        return _fit_gamm_pql(y, X, smooths, n_parametric, random_effects,
            f, data, family, link_eff, method, weights, control)
    end

    return _fit_gamm_lams(y, X, smooths, n_parametric, random_effects,
        f, data, family, link_eff, method, weights, control)
end

# GamFormula dispatch: no RE support, suggest @gamm_formula
function gamm(gf::GamFormula, data; kwargs...)
    @warn "GamFormula does not support random effects. Use @gamm_formula or @formula instead."
    return gam(gf, data; kwargs...)
end

# Positional family argument dispatches (match gam() API)
gamm(gf::GammFormula, data, family::UnivariateDistribution; kwargs...) =
    gamm(gf, data; family=family, kwargs...)
gamm(f::FormulaTerm, data, family::UnivariateDistribution; kwargs...) =
    gamm(f, data; family=family, kwargs...)

"""
    _functionterm_to_re_from_call(ft::FunctionTerm) → RandomEffectSpec

Convert `FunctionTerm` wrapping `re(group)` call to `RandomEffectSpec`.
"""
function _functionterm_to_re_from_call(ft::StatsModels.FunctionTerm)
    # ft.args[1] = Term(:group)
    length(ft.args) >= 1 || throw(ArgumentError("re() requires at least one argument"))

    grouping = if ft.args[1] isa Term
        ft.args[1].sym
    else
        throw(ArgumentError("re() argument must be a variable name"))
    end

    return RandomEffectSpec(grouping, Symbol[], true, true, "re($(grouping))")
end

# Bayesian stubs (implemented in GAMTuringExt)
function _fit_gamm_bayes end
function _fit_gamm_bayes_from_parts end

# MixedModels.jl stubs (implemented in GAMMixedModelsExt)
function _fit_gamm_mm end

# ============================================================================
# Show methods
# ============================================================================

function Base.show(io::IO, m::GammModel)
    println(io, "Generalized Additive Mixed Model")
    println(io)

    gm = m.gam_model
    println(io, "Family: ", typeof(gm.family).name.name)
    println(io, "Link:   ", typeof(gm.link).name.name)
    println(io)

    # Fixed effects
    println(io, "Fixed Effects Coefficients:")
    β = coef(gm)
    for i in 1:gm.n_parametric
        @printf(io, "  β[%d] = %10.6f\n", i, β[i])
    end
    println(io)

    # Smooth terms (exclude RE-as-smooth terms)
    n_actual_smooths = gm.n_smooth - length(m.random_effects)
    if n_actual_smooths > 0
        println(io, "Smooth Terms:")
        for idx in 1:n_actual_smooths
            sm = gm.smooths[idx]
            edf_i = idx <= length(gm.edf) ? gm.edf[idx] : NaN
            @printf(io, "  %-20s  edf = %6.2f\n", sm.spec.label, edf_i)
        end
        println(io)
    end

    # Variance components table (Random Effects + Residual)
    vc = VarCorr(m)
    show(io, vc)
    println(io)

    # Summary stats
    @printf(io, "Deviance:     %12.4f\n", deviance(gm))
    @printf(io, "REML:         %12.4f\n", gm.reml)
    @printf(io, "Scale est.:   %12.6f\n", gm.scale)
    @printf(io, "n = %d\n", nobs(gm))
end
