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
const _RAW_SMOOTH_FUNCTION_NAMES = (:s, :te, :ti, :t2, :cr, :tp, :ts, :cs, :cc, :ps, :cps)

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

GAM-aware wrapper around StatsModels' `@formula`.

- Ordinary formulas continue to use the standard StatsModels path.
- Formulas containing keyword smooth terms such as `s(x, k=15, bs=:cr)` are
  diverted to [`@formulak`](@ref).
- Formulas containing GAMM random-effect syntax such as `(1 | group)` or
  `re(group)` are diverted to the GAMM parser, so keyword smooths and random
  effects work together under a single macro.

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

function _build_parametric_matrix(gf::GamFormula, t)
    n = _table_nrows(t)
    X_para = gf.has_intercept ? ones(n, 1) : Matrix{Float64}(undef, n, 0)
    para_names = gf.has_intercept ? String["(Intercept)"] : String[]

    for sym in gf.parametric
        X_para = hcat(X_para, reshape(Float64.(Tables.getcolumn(t, sym)), n, 1))
        push!(para_names, string(sym))
    end

    return X_para, para_names
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

function _build_parametric_matrix(f::FormulaTerm, t)
    _, para_terms = _split_formula_terms(f)
    return _build_parametric_matrix(para_terms, t)
end

function _build_smooth_call(ex::Expr)
    fname = ex.args[1]
    pos_args = Any[]
    kw_args = Any[]
    k_pos = nothing
    has_k_kw = false

    for i in 2:length(ex.args)
        arg = ex.args[i]
        if arg isa Symbol
            push!(pos_args, QuoteNode(arg))
        elseif arg isa Expr && arg.head == :kw
            push!(kw_args, Expr(:kw, arg.args[1], arg.args[2]))
            has_k_kw |= arg.args[1] == :k
        elseif arg isa Expr && arg.head == :parameters
            for kw in arg.args
                if kw isa Expr && kw.head == :kw
                    push!(kw_args, Expr(:kw, kw.args[1], kw.args[2]))
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
    p_start = n_parametric + 1
    for sm in smooths
        k = size(sm.X, 2)
        sm.first_para = p_start
        sm.last_para = p_start + k - 1
        p_start += k
    end

    # Apply side constraints for identifiability (mgcv's gam.side)
    if length(smooths) > 1
        modified = side_constrain!(smooths, X_para)
        if modified
            # Reassign parameter indices after column removal
            p_start = n_parametric + 1
            for sm in smooths
                k = size(sm.X, 2)
                sm.first_para = p_start
                sm.last_para = p_start + k - 1
                p_start += k
            end
        end
    end

    # Build full model matrix: [parametric | smooth1 | smooth2 | ...]
    X_smooth_parts = [sm.X for sm in smooths]
    X_full = isempty(X_smooth_parts) ? X_para :
             hcat(X_para, X_smooth_parts...)

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
    X_para, para_names = _build_parametric_matrix(para_terms, t)
    n_parametric = size(X_para, 2)

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
