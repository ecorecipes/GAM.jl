# StatsModels formula integration — SmoothTerm for @formula / @formulak

# ─── @formula support: convert FunctionTerm{typeof(s)} to SmoothSpec ──────

"""
    _is_smooth_function(f)

Return `true` if `f` is a GAM smooth-constructing function (`s`, `te`, `ti`,
or a basis-specific alias like `cr`, `tp`, `ps`).
"""
function _is_smooth_function(f)
    f === s && return true
    f === te && return true
    f === ti && return true
    f === t2 && return true
    f in _SMOOTH_ALIASES && return true
    return false
end

# Populated after basis-alias definitions in smoothspec.jl
const _SMOOTH_ALIASES = Set{Function}()
const _RAW_SMOOTH_FUNCTION_NAMES = (:s, :te, :ti, :t2, :cr, :tp, :ts, :cs, :cc, :ps, :cps, :s_nest)

"""
    _functionterm_to_smoothspec(ft::FunctionTerm) → SmoothSpec

Convert a StatsModels `FunctionTerm` produced by `@formula(y ~ s(x, 10))` into
a [`SmoothSpec`](@ref). The positional-argument convention is:

| Argument type     | Interpretation          |
|:------------------|:------------------------|
| `Term`            | variable name           |
| `ConstantTerm{Int}` | basis dimension `k`  |

StatsModels' own `@formula` does not parse keyword arguments (`k=10`,
`bs=:cr`, etc.). When you import `@formula` from GAM, keyword smooth calls are
automatically diverted to [`@formulak`](@ref).
"""
function _functionterm_to_smoothspec(ft::StatsModels.FunctionTerm)
    var_syms = Symbol[]
    k_val = -1

    for arg in ft.args
        if arg isa Term
            push!(var_syms, arg.sym)
        elseif arg isa ConstantTerm
            k_val = round(Int, arg.n)
        end
    end

    isempty(var_syms) && throw(ArgumentError(
        "Smooth term $(ft.exorig) requires at least one variable"))

    return ft.f(var_syms...; k = k_val)
end

"""
    SmoothTerm <: AbstractTerm

A smooth term in a GAM formula. Wraps a [`SmoothSpec`](@ref) and integrates
with StatsModels.jl's formula machinery.
"""
struct SmoothTerm <: StatsModels.AbstractTerm
    spec::SmoothSpec
end

StatsModels.width(t::SmoothTerm) = t.spec.k - 1  # after constraint absorption

Base.show(io::IO, t::SmoothTerm) = print(io, t.spec.label)

"""
    AppliedSmoothTerm

A smooth term after schema application — knows its data types but not yet
evaluated. Contains the constructed smooth after `modelcols` is called.
"""
mutable struct AppliedSmoothTerm <: StatsModels.AbstractTerm
    spec::SmoothSpec
    smooth::Union{ConstructedSmooth, Nothing}
end

StatsModels.width(t::AppliedSmoothTerm) =
    t.smooth === nothing ? t.spec.k - 1 : size(t.smooth.X, 2)

Base.show(io::IO, t::AppliedSmoothTerm) = print(io, t.spec.label)

# Schema application: SmoothTerm → AppliedSmoothTerm
function StatsModels.apply_schema(t::SmoothTerm, sch, ::Type{<:Any})
    return AppliedSmoothTerm(t.spec, nothing)
end

# Disambiguation vs StatsModels' apply_schema(::AbstractTerm, ::FullRank, ::Type)
# (the untyped-sch method above is otherwise ambiguous with it for SmoothTerm)
function StatsModels.apply_schema(t::SmoothTerm, sch::StatsModels.FullRank,
    ::Type{<:Any})
    return AppliedSmoothTerm(t.spec, nothing)
end

# Schema application: FunctionTerm{typeof(smooth_f)} → AppliedSmoothTerm
# This makes the standard StatsModels pipeline (apply_schema → modelcols)
# work seamlessly for smooth terms created by @formula(y ~ s(x, 10)).
for _smooth_f in (s, te, ti, t2, cr, tp, ts, cs, cc, ps, cps)
    @eval function StatsModels.apply_schema(
        ft::StatsModels.FunctionTerm{typeof($_smooth_f)},
        sch::StatsModels.Schema,
        Mod::Type)
        spec = _functionterm_to_smoothspec(ft)
        return AppliedSmoothTerm(spec, nothing)
    end
end

# Model columns: construct basis and return matrix columns
function StatsModels.modelcols(t::AppliedSmoothTerm, d)
    if t.smooth === nothing
        t.smooth = smooth_construct(t.spec, d)
    end
    return t.smooth.X
end

StatsModels.coefnames(t::AppliedSmoothTerm) =
    [t.spec.label * ".$i" for i in 1:width(t)]

# ─── GamFormula: formula container with smooth terms ───────────────────────

"""
    GamFormula

A GAM formula containing both a parametric formula (for StatsModels) and
a vector of smooth term specifications. Created by `@formulak`.

# Example
```julia
gf = @formulak(y ~ 1 + x1 + s(x2, k=15, bs=:cr) + s(x3))
```
"""
struct GamFormula
    response::Symbol
    parametric::Vector{Symbol}            # parametric predictor symbols
    has_intercept::Bool
    smooth_specs::Vector{SmoothSpec}
end

function Base.show(io::IO, gf::GamFormula)
    parts = String[]
    if gf.has_intercept
        push!(parts, "1")
    end
    for p in gf.parametric
        push!(parts, string(p))
    end
    for sp in gf.smooth_specs
        push!(parts, sp.label)
    end
    rhs = isempty(parts) ? "1" : join(parts, " + ")
    print(io, gf.response, " ~ ", rhs)
end

function _formulak_expr(ex)
    ex.head == :call && ex.args[1] == :(~) ||
        error("Expected formula expression like `y ~ ...`, got $ex")

    lhs = ex.args[2]
    rhs = ex.args[3]

    response = QuoteNode(lhs)
    parametric = Expr(:vect)
    smooth_calls = Expr(:vect)
    has_intercept = Ref(true)

    _parse_gam_rhs!(rhs, parametric, smooth_calls, has_intercept)

    return esc(quote
        $GamFormula($response,
            Symbol[$(parametric.args...)],
            $(has_intercept[]),
            $SmoothSpec[$(smooth_calls.args...)])
    end)
end

function _has_keyword_smooth_syntax(ex)
    if ex isa Expr
        if ex.head == :call
            fname = ex.args[1]
            if fname isa Symbol && fname in _RAW_SMOOTH_FUNCTION_NAMES
                any(arg -> arg isa Expr && (arg.head == :parameters || arg.head == :kw),
                    ex.args[2:end]) && return true
            end
        end
        return any(_has_keyword_smooth_syntax, ex.args)
    end
    return false
end

function _has_gamm_syntax(ex)
    if ex isa Expr
        if ex.head == :call
            fname = ex.args[1]
            if fname == :(|) || fname == :re
                return true
            end
        end
        return any(_has_gamm_syntax, ex.args)
    end
    return false
end

"""
    @formulak(ex)

Create a [`GamFormula`](@ref) from an expression. Unlike StatsModels' `@formula`,
this macro supports `s()`, `te()`, and `ti()` smooth terms with keyword arguments.

# Examples
```julia
gf = @formulak(y ~ s(x))
gf = @formulak(y ~ 1 + s(x, k=15, bs=:cr))
gf = @formulak(y ~ x1 + s(x2) + s(x3, k=20))
```
"""
macro formulak(ex)
    return _formulak_expr(ex)
end

"""
    @formula(ex)

Public formula macro for GAM.jl models.

Use `@formula` for ordinary linear terms, smooth constructors with keyword
arguments such as `s(x, k=15, bs=:cr)`, and GAMM random-effect syntax such as
`(1 | group)` or `re(group)`.

# Examples
```julia
@formula(y ~ x1 + x2)
@formula(y ~ s(x, k=15, bs=:cr))
@formula(y ~ s(x, k=10) + (1 | subject))
@formula(y ~ s(x, k=10) + re(subject))
```
"""
macro formula(ex)
    if _has_gamm_syntax(ex)
        return _gamm_formula_expr(ex)
    elseif _has_keyword_smooth_syntax(ex)
        return _formulak_expr(ex)
    end
    return Expr(:macrocall, GlobalRef(StatsModels, Symbol("@formula")), __source__, ex)
end

function _parse_gam_rhs!(ex, parametric, smooth_calls, has_intercept)
    if ex isa Symbol
        push!(parametric.args, QuoteNode(ex))
    elseif ex isa Integer
        if ex == 1
            has_intercept[] = true
        elseif ex == 0
            has_intercept[] = false
        end
    elseif ex isa Expr
        if ex.head == :call
            fname = ex.args[1]
            if fname == :+
                for i in 2:length(ex.args)
                    _parse_gam_rhs!(ex.args[i], parametric, smooth_calls,
                        has_intercept)
                end
            elseif fname isa Symbol && fname in _RAW_SMOOTH_FUNCTION_NAMES
                # Extract cr(x, 15) / s(x, k=15, bs=:cr) → GAM.cr(:x; k=15) / GAM.s(:x; ...)
                push!(smooth_calls.args, _build_smooth_call(ex))
            else
                # Other function calls go to parametric as-is
                push!(parametric.args, ex)
            end
        elseif ex.head == :parameters
            # keyword args block — shouldn't appear at top level
            error("Unexpected keyword arguments at formula top level")
        else
            push!(parametric.args, ex)
        end
    end
end

function _table_nrows(t)
    names = collect(Tables.columnnames(t))
    isempty(names) && throw(ArgumentError("Cannot build a model matrix from a table with no columns"))
    return length(Tables.getcolumn(t, first(names)))
end

function _split_formula_terms(f::FormulaTerm)
    rhs_terms = _flatten_rhs(f.rhs)

    smooth_terms = AppliedSmoothTerm[]
    para_terms = StatsModels.AbstractTerm[]

    for term in rhs_terms
        if term isa AppliedSmoothTerm || term isa SmoothTerm
            ast = term isa SmoothTerm ? AppliedSmoothTerm(term.spec, nothing) : term
            push!(smooth_terms, ast)
        elseif term isa StatsModels.FunctionTerm && _is_smooth_function(term.f)
            spec = _functionterm_to_smoothspec(term)
            push!(smooth_terms, AppliedSmoothTerm(spec, nothing))
        else
            push!(para_terms, term)
        end
    end

    return smooth_terms, para_terms
end

function _formula_has_intercept(para_terms::AbstractVector{<:StatsModels.AbstractTerm})
    has_intercept = true
    for pt in para_terms
        if pt isa InterceptTerm{true}
            has_intercept = true
        elseif pt isa InterceptTerm{false}
            has_intercept = false
        elseif pt isa ConstantTerm
            has_intercept = getfield(pt, :n) == 1
        end
    end
    return has_intercept
end

_formula_has_intercept(gf::GamFormula) = gf.has_intercept

function _formula_has_intercept(f::FormulaTerm)
    _, para_terms = _split_formula_terms(f)
    return _formula_has_intercept(para_terms)
end

function _term_matrix(pt, t)
    if pt isa Term
        n = _table_nrows(t)
        return reshape(Float64.(Tables.getcolumn(t, pt.sym)), n, 1)
    elseif pt isa ContinuousTerm
        col = StatsModels.modelcols(pt, t)
        return reshape(Float64.(col), :, 1)
    else
        cols = StatsModels.modelcols(pt, t)
        if cols isa AbstractMatrix
            return Matrix{Float64}(cols)
        elseif cols isa AbstractVector
            return reshape(Float64.(cols), :, 1)
        end
        return reshape(Float64.(collect(cols)), :, 1)
    end
end

"""
    _apply_parametric_schema(para_terms, t) -> Vector{AbstractTerm}

Apply a StatsModels schema (built from `t`) to raw parametric terms so that
categorical/string columns are dummy-coded like any StatsModels model, instead
of failing the `Float64.(...)` conversion in `_term_matrix`. Intercept and
constant terms pass through unchanged. Pass the *training* data as `t` when
building prediction matrices so factor levels stay consistent.
"""
function _apply_parametric_schema(para_terms::AbstractVector{<:StatsModels.AbstractTerm}, t)
    isempty(para_terms) && return para_terms
    needs_schema = any(para_terms) do pt
        pt isa Term && !(eltype(Tables.getcolumn(t, pt.sym)) <: Real)
    end
    needs_schema || return para_terms
    sch = StatsModels.schema(t)
    return StatsModels.AbstractTerm[
        pt isa Term ?
            StatsModels.apply_schema(pt, sch, StatsModels.StatisticalModel) : pt
        for pt in para_terms]
end

function _term_names(pt)
    if pt isa Term
        return [string(pt.sym)]
    elseif pt isa ContinuousTerm
        return [string(pt.sym)]
    else
        names = StatsModels.coefnames(pt)
        return names isa AbstractVector ? String.(names) : [string(names)]
    end
end

function _formula_parametric_names(gf::GamFormula)
    names = gf.has_intercept ? String["(Intercept)"] : String[]
    append!(names, string.(gf.parametric))
    return names
end

function _formula_parametric_names(f::FormulaTerm)
    _, para_terms = _split_formula_terms(f)
    names = _formula_has_intercept(para_terms) ? String["(Intercept)"] : String[]
    for pt in para_terms
        if pt isa InterceptTerm{true} || pt isa InterceptTerm{false} || pt isa ConstantTerm
            continue
        end
        append!(names, _term_names(pt))
    end
    return names
end

"""
    _parametric_term_groups(formula, data)
        -> (colnames::Vector{String}, groups::Vector{Tuple{String, UnitRange{Int}}})

One name per dummy-coded parametric *column* (matching the design matrix built
by `_build_parametric_matrix`), plus the grouping of columns by originating
term (intercept and each variable get one group) for `type=:terms` prediction.
Needs the training `data` to know factor levels.
"""
function _parametric_term_groups(gf::GamFormula, data)
    t = Tables.columntable(data)
    colnames = String[]
    groups = Tuple{String, UnitRange{Int}}[]
    col = 0
    if gf.has_intercept
        push!(colnames, "(Intercept)")
        push!(groups, ("(Intercept)", 1:1))
        col = 1
    end
    for sym in gf.parametric
        c = Tables.getcolumn(t, sym)
        if eltype(c) <: Real
            push!(colnames, string(sym))
            push!(groups, (string(sym), (col + 1):(col + 1)))
            col += 1
        else
            levels = sort!(unique(collect(c)))
            ref = gf.has_intercept ? levels[2:end] : levels
            for lev in ref
                push!(colnames, string(sym, ": ", lev))
            end
            push!(groups, (string(sym), (col + 1):(col + length(ref))))
            col += length(ref)
        end
    end
    return colnames, groups
end

function _parametric_term_groups(f::FormulaTerm, data)
    t = Tables.columntable(data)
    _, para_terms = _split_formula_terms(f)
    para_terms = _apply_parametric_schema(para_terms, t)
    colnames = String[]
    groups = Tuple{String, UnitRange{Int}}[]
    col = 0
    if _formula_has_intercept(para_terms)
        push!(colnames, "(Intercept)")
        push!(groups, ("(Intercept)", 1:1))
        col = 1
    end
    for pt in para_terms
        if pt isa InterceptTerm{true} || pt isa InterceptTerm{false} || pt isa ConstantTerm
            continue
        end
        nms = _term_names(pt)
        append!(colnames, nms)
        label = pt isa Union{Term, ContinuousTerm, StatsModels.CategoricalTerm} ?
                string(pt.sym) : (length(nms) == 1 ? nms[1] : string(pt))
        push!(groups, (label, (col + 1):(col + length(nms))))
        col += length(nms)
    end
    return colnames, groups
end

function _build_parametric_matrix(gf::GamFormula, t;
    ref_levels::Union{Nothing, Dict{Symbol, Vector}} = nothing)
    n = _table_nrows(t)
    X_para = gf.has_intercept ? ones(n, 1) : Matrix{Float64}(undef, n, 0)
    para_names = gf.has_intercept ? String["(Intercept)"] : String[]

    for sym in gf.parametric
        col = Tables.getcolumn(t, sym)
        if eltype(col) <: Real
            X_para = hcat(X_para, reshape(Float64.(col), n, 1))
            push!(para_names, string(sym))
        else
            # Categorical / string parametric term: treatment (dummy) coding
            # against the first sorted level (mgcv's default factor contrast),
            # dropping the reference level when an intercept is present.
            # Use ref_levels (the training levels) when supplied so prediction
            # data with a subset of levels still produces the right columns.
            levels = ref_levels !== nothing && haskey(ref_levels, sym) ?
                     ref_levels[sym] : sort!(unique(collect(col)))
            # A level absent from the training data matches none of the dummy
            # columns. Silently returning the reference level's prediction (as
            # this did before) is wrong; mgcv warns and contributes nothing for
            # such rows — verified against mgcv 1.9.4, which returns a value
            # rather than erroring, and does not fall back to the reference
            # level. Match that, and mirror the factor-`by` convention.
            unseen_rows = falses(n)
            if ref_levels !== nothing && haskey(ref_levels, sym)
                new_levels = setdiff(unique(collect(col)), levels)
                if !isempty(new_levels)
                    @warn "Parametric term :$sym has level(s) " *
                          join(repr.(sort!(new_levels; by = string)), ", ") *
                          " that were not present when the model was fitted; " *
                          "these rows contribute nothing for this term " *
                          "(mgcv behaves the same way). Known levels: " *
                          join(repr.(levels), ", ") * "."
                    unseen_rows = [c in new_levels for c in col]
                end
            end
            ref = gf.has_intercept ? levels[2:end] : levels
            for lev in ref
                colvals = Float64.(col .== lev)
                # Rows whose level was never seen match no dummy column; they
                # are already zero here, which is the intended contribution.
                X_para = hcat(X_para, colvals)
                push!(para_names, string(sym, ": ", lev))
            end
            any(unseen_rows)  # documented above; rows stay all-zero for this term
        end
    end

    return X_para, para_names
end

"""
    _parametric_ref_levels(gf, data) -> Dict{Symbol, Vector}

Collect the sorted unique levels of each non-numeric parametric term from the
(training) `data`, so dummy coding can be reproduced consistently at
prediction time regardless of which levels appear in the new data.
"""
function _parametric_ref_levels(gf::GamFormula, data)
    t = Tables.columntable(data)
    levels = Dict{Symbol, Vector}()
    for sym in gf.parametric
        col = Tables.getcolumn(t, sym)
        if !(eltype(col) <: Real)
            levels[sym] = sort!(unique(collect(col)))
        end
    end
    return levels
end

function _build_parametric_matrix(para_terms::AbstractVector{<:StatsModels.AbstractTerm}, t)
    n = _table_nrows(t)
    has_intercept = _formula_has_intercept(para_terms)
    X_para = has_intercept ? ones(n, 1) : Matrix{Float64}(undef, n, 0)
    para_names = has_intercept ? String["(Intercept)"] : String[]

    for pt in para_terms
        if pt isa InterceptTerm{true} || pt isa InterceptTerm{false} || pt isa ConstantTerm
            continue
        end
        cols = _term_matrix(pt, t)
        X_para = hcat(X_para, cols)
        append!(para_names, _term_names(pt))
    end

    return X_para, para_names
end

function _build_parametric_matrix(f::FormulaTerm, t; schema_data = t)
    _, para_terms = _split_formula_terms(f)
    para_terms = _apply_parametric_schema(para_terms, Tables.columntable(schema_data))
    return _build_parametric_matrix(para_terms, t)
end

function _build_smooth_call(ex::Expr)
    fname = ex.args[1]
    pos_args = Any[]
    kw_args = Any[]
    k_pos = nothing
    has_k_kw = false

    # Keyword arguments that name a data column (so a bare identifier must be
    # quoted into a Symbol, e.g. `by=g` → `by=:g`), matching the positional
    # variable handling.
    _quote_kw(name, val) =
        (name in (:by, :id) && val isa Symbol) ? QuoteNode(val) : val

    for i in 2:length(ex.args)
        arg = ex.args[i]
        if arg isa Symbol
            push!(pos_args, QuoteNode(arg))
        elseif arg isa Expr && arg.head == :kw
            push!(kw_args, Expr(:kw, arg.args[1], _quote_kw(arg.args[1], arg.args[2])))
            has_k_kw |= arg.args[1] == :k
        elseif arg isa Expr && arg.head == :parameters
            for kw in arg.args
                if kw isa Expr && kw.head == :kw
                    push!(kw_args, Expr(:kw, kw.args[1], _quote_kw(kw.args[1], kw.args[2])))
                    has_k_kw |= kw.args[1] == :k
                end
            end
        elseif arg isa Integer
            # Positional integer is treated as k, matching StatsModels FunctionTerm parsing.
            k_pos = arg
        else
            push!(pos_args, arg)
        end
    end

    if k_pos !== nothing && !has_k_kw
        push!(kw_args, Expr(:kw, :k, k_pos))
    end

    # Qualify the smooth constructor (s, te, ti) with GAM module
    # to avoid scoping issues with @eval include and nested testsets
    qualified_fname = Expr(:., :GAM, QuoteNode(fname))

    if isempty(kw_args)
        return Expr(:call, qualified_fname, pos_args...)
    else
        return Expr(:call, qualified_fname,
            Expr(:parameters, kw_args...),
            pos_args...)
    end
end

# ─── setup_gam: build model matrix from GamFormula ────────────────────────

"""
    setup_gam(gf::GamFormula, data; family, contrasts)

Set up a GAM from a GamFormula and data. Returns all components needed for fitting:
- Response vector
- Full model matrix (parametric + smooth columns)
- Parametric model matrix
- Constructed smooths
- Number of parametric columns
"""
function setup_gam(gf::GamFormula, data;
    family::UnivariateDistribution = Normal(),
    contrasts::AbstractDict{Symbol} = Dict{Symbol, Any}())

    t = Tables.columntable(data)

    # Get response
    y = Float64.(Tables.getcolumn(t, gf.response))

    # Build parametric model matrix
    X_para, para_names = _build_parametric_matrix(gf, t)
    n_parametric = size(X_para, 2)

    # Construct smooth bases
    smooths = ConstructedSmooth[]
    for spec in gf.smooth_specs
        sm = smooth_construct(spec, t)
        push!(smooths, sm)
    end

    # Assign parameter indices to smooths
    _assign_smooth_indices!(smooths, n_parametric)

    # Apply side constraints for identifiability (mgcv's gam.side)
    if length(smooths) > 1
        modified = side_constrain!(smooths, X_para)
        if modified
            # Reassign parameter indices after column removal
            _assign_smooth_indices!(smooths, n_parametric)
        end
    end

    # Build full model matrix: [parametric | smooth1 | smooth2 | ...]
    X_smooth_parts = [sm.X for sm in smooths]
    X_full = isempty(X_smooth_parts) ? X_para :
             hcat(X_para, X_smooth_parts...)

    return y, X_full, X_para, smooths, n_parametric
end

"""
    _smooths_share_variables(specs) -> Bool

True when two smooths share a covariate, which is exactly when
`side_constrain!` (mgcv's `gam.side`) removes columns. That operates on the
`n`-row blocks, so reduced construction cannot support it and the caller must
fall back to dense.
"""
function _smooths_share_variables(specs)
    seen = Set{Symbol}()
    for spec in specs, v in spec.term_vars
        v in seen && return true
        push!(seen, v)
    end
    return false
end

"""
    setup_gam_discrete(gf, data, m_grid; family) -> (y, X, X_para, smooths, n_parametric)

Same contract as [`setup_gam`](@ref), but 1-D smooths are constructed at the
`m` unique covariate values and then scattered to `n` rows, rather than
evaluated at all `n` rows directly.

This exists for memory, not speed. Measured at n = 10⁵ with 4 × `s(k=20)`, a
thin-plate fit peaks at 4113 MB while retaining only 167 MB — roughly 3950 MB
is a construction transient, because TPRS with `max_knots = 2000` forms an
`n × 2000` dense matrix. Building at the grid makes that `m × 2000`.

The returned objects are indistinguishable from `setup_gam`'s: `sm.X` and the
model matrix both have `n` rows, since `ConstructedSmooth.X` and `GamModel.X`
are concretely-typed dense matrices. Only the transient is avoided.

Falls back to `setup_gam` wholesale when smooths share a covariate, since side
constraints need the `n`-row blocks.
"""
function setup_gam_discrete(gf::GamFormula, data, m_grid::Int;
    family::UnivariateDistribution = Normal())

    _smooths_share_variables(gf.smooth_specs) &&
        return setup_gam(gf, data; family = family)

    t = Tables.columntable(data)
    y = Float64.(Tables.getcolumn(t, gf.response))
    n = length(y)
    X_para, _ = _build_parametric_matrix(gf, t)
    n_parametric = size(X_para, 2)

    smooths = ConstructedSmooth[]
    idx = Vector{Union{Nothing, Vector{Int32}}}()
    for spec in gf.smooth_specs
        r = _reduced_smooth(spec, t, m_grid, n)
        if r === nothing
            push!(smooths, smooth_construct(spec, t))
            push!(idx, nothing)
        else
            sm, k, _ = r
            push!(smooths, sm)
            push!(idx, k)
        end
    end

    # Fill the model matrix column-block by column-block instead of `hcat`,
    # which would hold a second full copy while concatenating.
    p = n_parametric + sum(size(sm.X, 2) for sm in smooths; init = 0)
    X_full = Matrix{Float64}(undef, n, p)
    @inbounds X_full[:, 1:n_parametric] .= X_para
    c = n_parametric
    for (sm, k) in zip(smooths, idx)
        pb = size(sm.X, 2)
        cols = (c + 1):(c + pb)
        if k === nothing
            @inbounds X_full[:, cols] .= sm.X
        else
            Xd = sm.X
            @inbounds for j in 1:pb
                col = view(X_full, :, c + j)
                for i in 1:n
                    col[i] = Xd[k[i], j]
                end
            end
            # `ConstructedSmooth.X` is `Matrix{Float64}` and downstream code
            # (side constraints, gratia, concurvity) assumes `n` rows.
            sm.X = X_full[:, cols]
        end
        c += pb
    end

    _assign_smooth_indices!(smooths, n_parametric)
    return y, X_full, X_para, smooths, n_parametric
end

# Legacy: setup_gam from FormulaTerm (for @formula without smooth terms)
function setup_gam(f::FormulaTerm, data;
    family::UnivariateDistribution = Normal(),
    contrasts::AbstractDict{Symbol} = Dict{Symbol, Any}())

    t = Tables.columntable(data)
    resp_col = f.lhs isa Term ? f.lhs.sym : error("LHS must be a single term")
    y = Float64.(Tables.getcolumn(t, resp_col))
    smooth_terms, para_terms = _split_formula_terms(f)
    _validate_formula_smooths(SmoothSpec[st.spec for st in smooth_terms], t)
    para_terms = _apply_parametric_schema(para_terms, t)
    X_para, para_names = _build_parametric_matrix(para_terms, t)
    n_parametric = size(X_para, 2)

    smooths = ConstructedSmooth[]
    for st in smooth_terms
        sm = smooth_construct(st.spec, t)
        st.smooth = sm
        push!(smooths, sm)
    end

    _assign_smooth_indices!(smooths, n_parametric)

    # Apply side constraints for identifiability (mgcv's gam.side), exactly
    # as on the GamFormula path — without this, overlapping smooths (e.g.
    # s(x) + te(x, z)) leave an unconstrained direction in the design.
    if length(smooths) > 1
        modified = side_constrain!(smooths, X_para)
        if modified
            _assign_smooth_indices!(smooths, n_parametric)
        end
    end

    X_smooth_parts = [sm.X for sm in smooths]
    X_full = isempty(X_smooth_parts) ? X_para :
             hcat(X_para, X_smooth_parts...)

    return y, X_full, X_para, smooths, n_parametric
end

function _flatten_rhs(t::StatsModels.AbstractTerm)
    return [t]
end

function _flatten_rhs(t::Tuple)
    result = StatsModels.AbstractTerm[]
    for ti in t
        append!(result, _flatten_rhs(ti))
    end
    return result
end

function _flatten_rhs(t::FormulaTerm)
    return _flatten_rhs(t.rhs)
end

function _flatten_rhs(t::StatsModels.MatrixTerm)
    return _flatten_rhs(t.terms)
end
