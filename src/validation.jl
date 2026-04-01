# Input validation helpers for GAM.jl
#
# Catches common user mistakes early with actionable error messages,
# rather than letting them propagate to cryptic linear algebra errors.

# ============================================================================
# Response variable validation
# ============================================================================

"""
    _validate_response(y, family)

Validate the response vector `y` for the given distribution `family`.
Checks for NaN/Inf values and family-specific constraints.
"""
function _validate_response(y::AbstractVector, family)
    _validate_response_numeric(y)
    _validate_response_finite(y)
    _validate_response_family(y, family)
    return nothing
end

"""
    _validate_response_numeric(y)

Check that the response vector is numeric (not strings, booleans, etc.).
"""
function _validate_response_numeric(y::AbstractVector{<:Real})
    return nothing  # already numeric
end

function _validate_response_numeric(y::AbstractVector)
    T = eltype(y)
    throw(ArgumentError(
        "Response variable must be numeric, got element type $T. " *
        "Convert to numeric with `Float64.(y)` or check your data."))
end

"""
    _validate_response_finite(y)

Check that the response contains no NaN or Inf values.
"""
function _validate_response_finite(y::AbstractVector{<:Real})
    n_nan = count(isnan, y)
    n_inf = count(isinf, y)
    if n_nan > 0 || n_inf > 0
        parts = String[]
        n_nan > 0 && push!(parts, "$n_nan NaN")
        n_inf > 0 && push!(parts, "$n_inf Inf")
        throw(ArgumentError(
            "Response variable contains non-finite values ($(join(parts, " and "))). " *
            "Remove or impute missing/infinite values before fitting."))
    end
    return nothing
end

"""
    _validate_response_family(y, family)

Check family-specific constraints on the response.
"""
_validate_response_family(y, ::Any) = nothing

function _validate_response_family(y, ::Poisson)
    ymin = minimum(y)
    if ymin < 0
        throw(ArgumentError(
            "Response must be non-negative for Poisson family, but got " *
            "minimum value $(ymin). Use a different family or check your data."))
    end
    return nothing
end

function _validate_response_family(y, ::QuasiPoissonFamily)
    ymin = minimum(y)
    if ymin < 0
        throw(ArgumentError(
            "Response must be non-negative for QuasiPoisson family, but got " *
            "minimum value $(ymin). Use a different family or check your data."))
    end
    return nothing
end

function _validate_response_family(y, ::Union{Gamma, InverseGaussian})
    ymin = minimum(y)
    if ymin <= 0
        throw(ArgumentError(
            "Response must be strictly positive for $(nameof(typeof(y isa AbstractVector ? Gamma() : InverseGaussian()))) family, " *
            "but got minimum value $(ymin). " *
            "Use a different family or check your data."))
    end
    return nothing
end

# Explicit methods for Gamma and InverseGaussian to get name right
function _validate_response_family(y, ::Gamma)
    ymin = minimum(y)
    if ymin <= 0
        throw(ArgumentError(
            "Response must be strictly positive for Gamma family, " *
            "but got minimum value $(ymin). " *
            "Use a different family or check your data."))
    end
    return nothing
end

function _validate_response_family(y, ::InverseGaussian)
    ymin = minimum(y)
    if ymin <= 0
        throw(ArgumentError(
            "Response must be strictly positive for InverseGaussian family, " *
            "but got minimum value $(ymin). " *
            "Use a different family or check your data."))
    end
    return nothing
end

function _validate_response_family(y, ::BinomialLike)
    ymin, ymax = extrema(y)
    if ymin < 0 || ymax > 1
        throw(ArgumentError(
            "Response must be in [0, 1] for Binomial/Bernoulli family, " *
            "but got range [$ymin, $ymax]. " *
            "For count data use Poisson(); for proportions ensure y ∈ [0,1]."))
    end
    return nothing
end

function _validate_response_family(y, ::QuasiBinomialFamily)
    ymin, ymax = extrema(y)
    if ymin < 0 || ymax > 1
        throw(ArgumentError(
            "Response must be in [0, 1] for QuasiBinomial family, " *
            "but got range [$ymin, $ymax]. " *
            "For count data use QuasiPoissonFamily(); for proportions ensure y ∈ [0,1]."))
    end
    return nothing
end

# ============================================================================
# Smooth term validation
# ============================================================================

"""
    _validate_smooth_k(k::Int, n::Int, label::String)

Validate the basis dimension `k` against sample size `n`.
"""
function _validate_smooth_k(k::Int, n::Int, label::String)
    if k < 3
        throw(ArgumentError(
            "Basis dimension k=$k is too small for smooth $label. " *
            "k must be ≥ 3 (need at least 3 basis functions). " *
            "Increase k or remove this smooth term."))
    end
    if k >= n
        throw(ArgumentError(
            "Basis dimension k=$k ≥ sample size n=$n for smooth $label. " *
            "k must be less than n. Reduce k or add more data."))
    end
    if k > n ÷ 2
        @warn "Basis dimension k=$k is large relative to sample size n=$n for smooth $label. " *
              "This may lead to overfitting. Consider reducing k."
    end
    return nothing
end

"""
    _validate_smooth_data(x::AbstractVector, var::Symbol)

Validate that smooth term data is finite.
"""
function _validate_smooth_data(x::AbstractVector{<:Real}, var::Symbol)
    n_nan = count(isnan, x)
    n_inf = count(isinf, x)
    if n_nan > 0 || n_inf > 0
        parts = String[]
        n_nan > 0 && push!(parts, "$n_nan NaN")
        n_inf > 0 && push!(parts, "$n_inf Inf")
        throw(ArgumentError(
            "Smooth variable :$var contains non-finite values ($(join(parts, " and "))). " *
            "Remove or impute missing/infinite values before fitting."))
    end
    return nothing
end

"""
    _validate_smooth_vars_in_data(spec::SmoothSpec, data)

Check that all variables referenced by a smooth spec exist in the data.
"""
function _validate_smooth_vars_in_data(spec::SmoothSpec, data)
    col_names = Tables.columnnames(data)
    for var in spec.term_vars
        if !(var in col_names)
            throw(ArgumentError(
                "Smooth variable :$var not found in data. " *
                "Available columns: $(join(sort(collect(col_names)), ", ")). " *
                "Check for typos in your formula."))
        end
    end
    if spec.by !== nothing && !(spec.by in col_names)
        throw(ArgumentError(
            "By-variable :$(spec.by) not found in data. " *
            "Available columns: $(join(sort(collect(col_names)), ", ")). " *
            "Check for typos in your formula."))
    end
    return nothing
end

"""
    _validate_formula_smooths(smooth_specs, data)

Validate smooth terms in a formula: check variables exist, data is finite,
and k values are sensible.
"""
function _validate_formula_smooths(smooth_specs::Vector{<:SmoothSpec}, data)
    t = Tables.columntable(data)
    n = _nrow(t)
    for spec in smooth_specs
        _validate_smooth_vars_in_data(spec, t)
        # Validate data finiteness for each variable
        for var in spec.term_vars
            col = Tables.getcolumn(t, var)
            if eltype(col) <: Real
                _validate_smooth_data(Float64.(col), var)
            end
        end
        # Validate k vs n (skip RE and MRF — k is determined at construction)
        if !(spec.basis isa RandomEffect) && !(spec.basis isa MarkovRandomField) && spec.k > 0
            _validate_smooth_k(spec.k, n, spec.label)
        end
    end
    return nothing
end

"""
    _validate_has_smooths(smooth_specs)

Warn if formula has no smooth terms (user should use GLM.jl instead).
"""
function _validate_has_smooths(smooth_specs)
    if isempty(smooth_specs)
        @warn "Formula contains no smooth terms. " *
              "Consider using GLM.jl for purely parametric models."
    end
    return nothing
end

# Helper: get number of rows from columntable
function _nrow(t)
    names = Tables.columnnames(t)
    isempty(names) && return 0
    return length(Tables.getcolumn(t, first(names)))
end

# ============================================================================
# GAMM-specific validation
# ============================================================================

"""
    _validate_gamm_random_effects(re_specs, data)

Validate random effect specifications for GAMM.
"""
function _validate_gamm_random_effects(re_specs::Vector{RandomEffectSpec}, data)
    isempty(re_specs) && throw(ArgumentError(
        "gamm() requires at least one random effect term. " *
        "Use `(1|group)` syntax in @gamm_formula, or `re(group)` in @formula. " *
        "For models without random effects, use gam() instead."))

    t = Tables.columntable(data)
    col_names = Tables.columnnames(t)
    for spec in re_specs
        # Grouping variable must exist
        if !(spec.grouping in col_names)
            throw(ArgumentError(
                "Random effect grouping variable :$(spec.grouping) not found in data. " *
                "Available columns: $(join(sort(collect(col_names)), ", "))."))
        end
        # Warn if grouping variable is numeric (might be continuous)
        gcol = Tables.getcolumn(t, spec.grouping)
        if eltype(gcol) <: AbstractFloat
            @warn "Random effect grouping variable :$(spec.grouping) is numeric ($(eltype(gcol))). " *
                  "This will be treated as a categorical grouping variable. " *
                  "If this is intentional, convert to CategoricalArray or String first."
        end
        # Validate slope variables exist
        for v in spec.terms
            if !(v in col_names)
                throw(ArgumentError(
                    "Random slope variable :$v (in $(spec.label)) not found in data. " *
                    "Available columns: $(join(sort(collect(col_names)), ", "))."))
            end
        end
    end
    return nothing
end

# ============================================================================
# SCAM-specific validation
# ============================================================================

"""
    _validate_scam_has_constraints(smooth_specs)

Warn if scam() is called without any shape-constrained smooth terms.
"""
function _validate_scam_has_constraints(smooth_specs)
    has_constrained = any(spec -> spec.basis isa AbstractConstrainedBasis, smooth_specs)
    if !has_constrained
        @warn "scam() called without any shape-constrained smooth terms. " *
              "Consider using gam() instead, which is more efficient for unconstrained models. " *
              "Shape-constrained basis types: :mpi, :mpd, :cv, :cx, :micx, :micv, :mdcx, :mdcv."
    end
    return nothing
end

# ============================================================================
# GAMLSS-specific validation
# ============================================================================

"""
    _validate_gamlss_formulas(formulas, family)

Validate that the number of formulas matches the number of distribution parameters.
"""
function _validate_gamlss_formulas(formulas, family::MultiParameterFamily)
    K = nparams(family)
    if !(formulas isa FormulaTerm || formulas isa GamFormula)
        if length(formulas) != K
            throw(ArgumentError(
                "Expected $K formula(s) for $(typeof(family)) " *
                "(parameters: $(join(param_names(family), ", "))), " *
                "got $(length(formulas)). " *
                "Provide one formula per distribution parameter."))
        end
    end
    return nothing
end

"""
    _validate_gamlss_family_type(family)

Validate that the family is appropriate for gamlss() — must be a
MultiParameterFamily or a supported UnivariateDistribution.
"""
function _validate_gamlss_family_type(family)
    if !(family isa MultiParameterFamily) && !(family isa UnivariateDistribution)
        throw(ArgumentError(
            "gamlss() requires a MultiParameterFamily or supported UnivariateDistribution, " *
            "got $(typeof(family)). " *
            "Use GammaLocationScale(), BetaRegression(), Normal(), etc."))
    end
    return nothing
end

# ============================================================================
# Data length consistency
# ============================================================================

"""
    _validate_data_lengths(data)

Check that all columns in the data have the same length.
"""
function _validate_data_lengths(data)
    t = Tables.columntable(data)
    names = Tables.columnnames(t)
    isempty(names) && return nothing

    n = length(Tables.getcolumn(t, first(names)))
    for name in names
        col = Tables.getcolumn(t, name)
        if length(col) != n
            throw(ArgumentError(
                "Data columns have inconsistent lengths: :$(first(names)) has $n rows " *
                "but :$name has $(length(col)) rows. All columns must have the same length."))
        end
    end
    return nothing
end

"""
    _validate_response_in_data(response::Symbol, data)

Check that the response variable exists in the data.
"""
function _validate_response_in_data(response::Symbol, data)
    t = Tables.columntable(data)
    col_names = Tables.columnnames(t)
    if !(response in col_names)
        throw(ArgumentError(
            "Response variable :$response not found in data. " *
            "Available columns: $(join(sort(collect(col_names)), ", ")). " *
            "Check for typos in your formula."))
    end
    return nothing
end
