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

function _validate_response_family(y, ::TweedieFamily)
    ymin = minimum(y)
    if ymin < 0
        throw(ArgumentError(
            "Response must be non-negative for Tweedie family, but got " *
            "minimum value $(ymin). Use a different family or check your data."))
    end
    return nothing
end

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
        # sp= and fx=true are contradictory: fx removes the penalty entirely,
        # so the supplied smoothing parameter would be silently discarded.
        if spec.fx && spec.sp !== nothing
            throw(ArgumentError(
                "sp=$(spec.sp) and fx=true are incompatible in $(spec.label): " *
                "fx=true leaves the smooth unpenalized, so the smoothing " *
                "parameter would be ignored. Drop one of them."))
        end
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
        "Use `@formula(... + (1 | group))` or `@formula(... + re(group))`. " *
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
    _validate_response_column(Tables.getcolumn(t, response), response)
    return nothing
end

"""
    _validate_response_column(col, response::Symbol)

Check the raw response column *before* it is converted to `Float64`.

Without this, a `missing` or non-numeric entry surfaces as a bare
`MethodError: no method matching Float64(::Missing)` from deep inside the
setup path, naming neither the variable nor the remedy.
"""
function _validate_response_column(col::AbstractVector, response::Symbol)
    T = eltype(col)
    if Missing <: T
        n_missing = count(ismissing, col)
        if n_missing > 0
            throw(ArgumentError(
                "Response variable :$response contains $n_missing missing " *
                "$(n_missing == 1 ? "value" : "values"). Pass `na_action = :omit` " *
                "to drop the affected rows (mgcv's `na.omit` default), or remove " *
                "or impute them before fitting."))
        end
    elseif !(T <: Real)
        throw(ArgumentError(
            "Response variable :$response must be numeric, got element type $T. " *
            "Convert with `Float64.(y)`, or use a categorical response with " *
            "Bernoulli()/Binomial() after coding it as 0/1."))
    end
    return nothing
end

"""
    _validate_weights(weights, n::Int)

Validate prior weights: correct length, finite, and non-negative.

Without this, a negative weight surfaces as
`DomainError with -1.0: sqrt was called with a negative real argument`
from inside the P-IRLS working-weight computation.
"""
function _validate_weights(weights, n::Int)
    weights === nothing && return nothing
    if length(weights) != n
        throw(ArgumentError(
            "weights length $(length(weights)) ≠ number of observations $n"))
    end
    for (i, w) in enumerate(weights)
        if ismissing(w)
            throw(ArgumentError(
                "weights must not be missing, but weights[$i] is missing. " *
                "Use zero to exclude an observation, or na_action = :omit."))
        end
        if !isfinite(w)
            throw(ArgumentError(
                "weights must be finite, but weights[$i] = $w"))
        end
        if w < 0
            throw(ArgumentError(
                "weights must be non-negative, got minimum $(minimum(weights)) " *
                "at index $i. Use zero to exclude an observation."))
        end
    end
    return nothing
end

"""
    _validate_offset(offset, n::Int)

Validate a supplied offset: correct length and finite (an `Inf`/`NaN` offset
otherwise poisons the linear predictor and surfaces as a singular-system or
`NaN`-deviance failure far from its cause).
"""
function _validate_offset(offset, n::Int)
    offset === nothing && return nothing
    if length(offset) != n
        throw(ArgumentError(
            "offset length $(length(offset)) ≠ number of observations $n"))
    end
    for (i, o) in enumerate(offset)
        if ismissing(o)
            throw(ArgumentError(
                "offset must not be missing, but offset[$i] is missing. " *
                "Remove those rows, or use na_action = :omit."))
        end
        if !isfinite(o)
            throw(ArgumentError(
                "offset must be finite, but offset[$i] = $o"))
        end
    end
    return nothing
end

# ============================================================================
# Missing-data handling (na.action)
# ============================================================================

"""
    _check_na_action(na_action::Symbol) -> Symbol

Validate an `na_action` keyword. Supported values mirror the R actions GAM.jl
implements: `:fail` (R's `na.fail` — the GAM.jl default) and `:omit` (R's
`na.omit`, which is *mgcv's* default).
"""
function _check_na_action(na_action::Symbol)
    na_action in (:fail, :omit) || throw(ArgumentError(
        "na_action must be :fail or :omit, got :$na_action. " *
        ":fail (the default) errors on missing/non-finite data; " *
        ":omit drops those rows, as mgcv does by default."))
    return na_action
end

"""
    _mark_incomplete!(bad::BitVector, col) -> BitVector

Flag rows of `col` that cannot enter a fit. `missing` always counts; for
numeric columns `NaN` and `±Inf` count too. Non-numeric columns (strings,
categoricals used as `by=` or grouping factors) are only checked for
`missing` — a string is never "non-finite".
"""
function _mark_incomplete!(bad::BitVector, col)
    @inbounds for i in eachindex(bad, col)
        v = col[i]
        if ismissing(v)
            bad[i] = true
        elseif v isa Real && !isfinite(v)
            bad[i] = true
        end
    end
    return bad
end

"""
    _incomplete_rows(t, cols; weights = nothing, offset = nothing) -> BitVector

Rows of the column table `t` that contain `missing`, `NaN` or `Inf` in any of
`cols` (columns not present in `t` are skipped — a missing column is reported
by the variable-existence validators, which give a better message), or in the
supplied `weights`/`offset`.
"""
function _incomplete_rows(t, cols; weights = nothing, offset = nothing)
    n = _nrow(t)
    bad = falses(n)
    present = Tables.columnnames(t)
    for c in cols
        c in present || continue
        _mark_incomplete!(bad, Tables.getcolumn(t, c))
    end
    # A length mismatch is a separate error raised by the caller's own
    # length check; skip rather than throw a confusing indexing error here.
    if weights !== nothing && length(weights) == n
        _mark_incomplete!(bad, weights)
    end
    if offset !== nothing && length(offset) == n
        _mark_incomplete!(bad, offset)
    end
    return bad
end

"""
    _apply_na_action(data, response, covariates, na_action;
                     weights = nothing, offset = nothing)
        -> (filtered_table, kept_index)

Apply an `na.action` policy to `data` before a fit.

`response` is the response variable (or `nothing`), `covariates` an iterable of
the other variables the model needs. Rows carrying `missing`, `NaN` or `Inf` in
any of those columns — or in `weights`/`offset`, when supplied — are the
incomplete ones.

- `:fail` returns `data` untouched (as a column table) with `kept_index =
  1:nrow`. The downstream validators then raise their own, more specific
  errors, so this is exactly the pre-existing behaviour.
- `:omit` drops the incomplete rows, mirroring R's `na.omit` — which is what
  `mgcv::gam` does by default.

`kept_index` is a `Vector{Int}` of the surviving rows *in the original data's
numbering*, so `original[kept_index, :]` lines up row-for-row with the fitted
values, residuals and stored model data. It is returned rather than stored on
the model; [`na_omit_rows`](@ref) recomputes it for a fitted model's inputs.

`weights` and `offset` are **not** subset here — the caller holds them and
should apply `kept_index` itself.
"""
function _apply_na_action(data, response, covariates, na_action::Symbol;
    weights = nothing, offset = nothing)

    _check_na_action(na_action)
    t = Tables.columntable(data)
    n = _nrow(t)

    # :fail is the historical path — leave the data alone and let the
    # per-variable validators produce their specific messages.
    na_action === :fail && return (t, collect(1:n))

    cols = Symbol[]
    response === nothing || push!(cols, response)
    for c in covariates
        c === nothing || push!(cols, c)
    end
    unique!(cols)

    bad = _incomplete_rows(t, cols; weights = weights, offset = offset)
    any(bad) || return (t, collect(1:n))

    kept = findall(!, bad)
    isempty(kept) && throw(ArgumentError(
        "na_action = :omit removed every row: all $n observations contain " *
        "missing or non-finite values in " * join(string.(':', cols), ", ") *
        " (or in the supplied weights/offset)."))

    filtered = NamedTuple{Tables.columnnames(t)}(
        map(c -> Tables.getcolumn(t, c)[kept], Tables.columnnames(t)))
    return (filtered, kept)
end

"""
    _validate_model_columns(data, cols)

Under `na_action = :fail`, check that every model column is complete.

`gam` and `gamm` reach `_validate_formula_smooths`, which reports a `missing`
or `NaN` covariate by name. `bam` and `gam_nl` do not, so the same input used
to surface as a bare `MethodError: no method matching Float64(::Missing)` from
inside the design-matrix build (`gam_nl`) or as a silently poisoned basis
(`bam`). Give them the same class of message.
"""
function _validate_model_columns(data, cols)
    t = Tables.columntable(data)
    present = Tables.columnnames(t)
    for c in cols
        c in present || continue
        col = Tables.getcolumn(t, c)
        n_missing = count(ismissing, col)
        if n_missing > 0
            throw(ArgumentError(
                "Variable :$c contains $n_missing missing " *
                "$(n_missing == 1 ? "value" : "values"). Remove or impute " *
                "them before fitting, or pass na_action = :omit to drop " *
                "those rows (as mgcv does by default)."))
        end
        eltype(col) <: Real || continue
        n_nan = count(isnan, col)
        n_inf = count(isinf, col)
        if n_nan > 0 || n_inf > 0
            parts = String[]
            n_nan > 0 && push!(parts, "$n_nan NaN")
            n_inf > 0 && push!(parts, "$n_inf Inf")
            throw(ArgumentError(
                "Variable :$c contains non-finite values " *
                "($(join(parts, " and "))). Remove or impute them before " *
                "fitting, or pass na_action = :omit to drop those rows " *
                "(as mgcv does by default)."))
        end
    end
    return nothing
end

"""
    _na_prepare(data, response, covariates, na_action;
                weights = nothing, offset = nothing)
        -> (data_used, kept_index, weights_used, offset_used)

One-call front-end preamble: check column lengths, apply the `na_action`
policy, subset `weights`/`offset` to the surviving rows, then validate them.

Split this way because the order matters. Lengths are checked against the
*original* row count (that is what the user supplied), but `weights` and
`offset` are only validated for finiteness *after* row removal — under
`na_action = :omit` a `missing` weight is a reason to drop a row, not an
error.
"""
function _na_prepare(data, response, covariates, na_action::Symbol;
    weights = nothing, offset = nothing)

    _validate_data_lengths(data)
    t = Tables.columntable(data)
    n0 = _nrow(t)

    weights === nothing || length(weights) == n0 || throw(ArgumentError(
        "weights length $(length(weights)) ≠ number of observations $n0"))
    offset === nothing || length(offset) == n0 || throw(ArgumentError(
        "offset length $(length(offset)) ≠ number of observations $n0"))

    data_used, kept = _apply_na_action(t, response, covariates, na_action;
        weights = weights, offset = offset)

    w   = weights === nothing ? nothing : weights[kept]
    off = offset === nothing ? nothing : offset[kept]
    _validate_weights(w, length(kept))
    _validate_offset(off, length(kept))

    # Narrow back to a concrete element type. `weights = [1.0, missing, ...]`
    # is a legitimate input under :omit, but every fitter downstream wants a
    # plain Float64 vector — and validation above has just established that
    # what survives contains no `missing`.
    w   = w   === nothing ? nothing : convert(Vector{Float64}, w)
    off = off === nothing ? nothing : convert(Vector{Float64}, off)

    return (data_used, kept, w, off)
end

"""
    na_omit_rows(data, response, covariates; weights = nothing, offset = nothing)
        -> Vector{Int}

Indices of the rows of `data` that a `na_action = :omit` fit keeps — those with
no `missing`, `NaN` or `Inf` in the response, in any listed covariate, or in the
supplied `weights`/`offset`.

Fitting with `na_action = :omit` silently drops the complement (as `mgcv` does),
so a fitted model has fewer rows than the table it was given. Use this to line
results back up with the original data:

```julia
m    = gam(f, df; na_action = :omit)
keep = na_omit_rows(df, :y, [:x1, :x2])
df.fit = missing
df.fit[keep] = fitted(m)
```

The computation is deterministic and depends only on the columns listed, so the
result matches what the fit used as long as the same variables are named.
"""
function na_omit_rows(data, response::Union{Symbol, Nothing}, covariates;
    weights = nothing, offset = nothing)
    t = Tables.columntable(data)
    cols = Symbol[]
    response === nothing || push!(cols, response)
    for c in covariates
        c === nothing || push!(cols, c)
    end
    unique!(cols)
    bad = _incomplete_rows(t, cols; weights = weights, offset = offset)
    return findall(!, bad)
end

# ---------------------------------------------------------------------------
# Which variables a model actually needs
# ---------------------------------------------------------------------------

"""
    _model_covariates(formula) -> Vector{Symbol}

Every non-response variable a formula refers to: parametric terms, smooth
term variables, `by=` variables, and (for a GAMM formula) random-effect
grouping and slope variables. Used to decide which columns `na_action` should
consider — a row with a `missing` in an unused column is not incomplete.
"""
function _model_covariates(gf::GamFormula)
    cols = Symbol[]
    append!(cols, gf.parametric)
    for spec in gf.smooth_specs
        append!(cols, spec.term_vars)
        spec.by === nothing || push!(cols, spec.by)
    end
    return unique!(cols)
end

function _model_covariates(gf::GammFormula)
    cols = _model_covariates(gf.gam_formula)
    for re in gf.random_effects
        push!(cols, re.grouping)
        append!(cols, re.terms)
    end
    return unique!(cols)
end

function _model_covariates(f::FormulaTerm)
    # termvars covers Term/InteractionTerm/FunctionTerm arguments, which is
    # exactly the `s(x, by=g)`/`te(x, z)` variable set once the response is
    # dropped.
    resp = f.lhs isa Term ? f.lhs.sym : nothing
    cols = Symbol[v for v in StatsModels.termvars(f) if v !== resp]
    return unique!(cols)
end
