# StatsModels formula integration — SmoothTerm for @formula / @gam_formula

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
    f in _SMOOTH_ALIASES && return true
    return false
end

# Populated after basis-alias definitions in smoothspec.jl
const _SMOOTH_ALIASES = Set{Function}()

"""
    _functionterm_to_smoothspec(ft::FunctionTerm) → SmoothSpec

Convert a StatsModels `FunctionTerm` produced by `@formula(y ~ s(x, 10))` into
a [`SmoothSpec`](@ref). The positional-argument convention is:

| Argument type     | Interpretation          |
|:------------------|:------------------------|
| `Term`            | variable name           |
| `ConstantTerm{Int}` | basis dimension `k`  |

Keyword arguments (`k=10`, `bs=:cr`, etc.) are **not** supported in `@formula`
because StatsModels does not parse them. Use `@gam_formula` for that syntax.
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
a vector of smooth term specifications. Created by `@gam_formula`.

# Example
```julia
gf = @gam_formula(y ~ 1 + x1 + s(x2, k=15, bs=:cr) + s(x3))
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

"""
    @gam_formula(ex)

Create a [`GamFormula`](@ref) from an expression. Unlike StatsModels' `@formula`,
this macro supports `s()`, `te()`, and `ti()` smooth terms with keyword arguments.

# Examples
```julia
gf = @gam_formula(y ~ s(x))
gf = @gam_formula(y ~ 1 + s(x, k=15, bs=:cr))
gf = @gam_formula(y ~ x1 + s(x2) + s(x3, k=20))
```
"""
macro gam_formula(ex)
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
        GamFormula($response,
            Symbol[$(parametric.args...)],
            $(has_intercept[]),
            SmoothSpec[$(smooth_calls.args...)])
    end)
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
            elseif fname in (:s, :te, :ti)
                # Extract s(x, k=15, bs=:cr) → s(:x; k=15, bs=:cr)
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

function _build_smooth_call(ex::Expr)
    fname = ex.args[1]  # :s, :te, or :ti
    pos_args = Any[]
    kw_args = Any[]

    for i in 2:length(ex.args)
        arg = ex.args[i]
        if arg isa Symbol
            push!(pos_args, QuoteNode(arg))
        elseif arg isa Expr && arg.head == :kw
            push!(kw_args, Expr(:kw, arg.args[1], arg.args[2]))
        elseif arg isa Expr && arg.head == :parameters
            for kw in arg.args
                if kw isa Expr && kw.head == :kw
                    push!(kw_args, Expr(:kw, kw.args[1], kw.args[2]))
                end
            end
        elseif arg isa Integer
            # Positional integer — could be k value in s(x, 15)
            push!(pos_args, arg)
        else
            push!(pos_args, arg)
        end
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
    n = length(y)

    # Build parametric model matrix
    X_para = gf.has_intercept ? ones(n, 1) : Matrix{Float64}(undef, n, 0)
    para_names = gf.has_intercept ? String["(Intercept)"] : String[]
    n_parametric = gf.has_intercept ? 1 : 0

    for sym in gf.parametric
        col = Float64.(Tables.getcolumn(t, sym))
        X_para = hcat(X_para, col)
        push!(para_names, string(sym))
        n_parametric += 1
    end

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
    n = length(y)

    rhs_terms = _flatten_rhs(f.rhs)

    smooth_terms = AppliedSmoothTerm[]
    para_terms = StatsModels.AbstractTerm[]

    for term in rhs_terms
        if term isa AppliedSmoothTerm || term isa SmoothTerm
            ast = term isa SmoothTerm ? AppliedSmoothTerm(term.spec, nothing) : term
            push!(smooth_terms, ast)
        elseif term isa StatsModels.FunctionTerm && _is_smooth_function(term.f)
            spec = _functionterm_to_smoothspec(term)
            ast = AppliedSmoothTerm(spec, nothing)
            push!(smooth_terms, ast)
        else
            push!(para_terms, term)
        end
    end

    X_para = ones(n, 1)
    para_names = String["(Intercept)"]
    n_parametric = 1

    for pt in para_terms
        if pt isa InterceptTerm{true}
            continue
        elseif pt isa InterceptTerm{false}
            continue
        elseif pt isa ContinuousTerm
            col = Float64.(modelcols(pt, t))
            X_para = hcat(X_para, col)
            push!(para_names, string(pt.sym))
            n_parametric += 1
        elseif pt isa Term
            col = Float64.(Tables.getcolumn(t, pt.sym))
            X_para = hcat(X_para, col)
            push!(para_names, string(pt.sym))
            n_parametric += 1
        end
    end

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
