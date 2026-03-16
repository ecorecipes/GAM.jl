# Smooth specification constructors — user-facing s(), te(), ti()

"""
    s(vars...; bs=:tp, k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Specify a smooth term for use in a GAM formula.

# Arguments
- `vars`: one or more variable names (as symbols or Term objects)
- `bs`: basis type (`:tp`, `:ts`, `:cr`, `:cs`, `:cc`, `:ps`, `:bs`, `:re`)
- `k`: basis dimension. `-1` (default) uses a sensible default based on basis type
- `by`: optional `by` variable for varying-coefficient models
- `id`: optional identifier for linking smooths sharing smoothing parameters
- `sp`: fixed smoothing parameter. `nothing` = estimate automatically
- `fx`: if `true`, smooth is unpenalized (fixed df)
- `m`: penalty order (meaning depends on basis type; `nothing` = default)

# Default basis dimensions
- 1d smooths: `k=10`
- 2d smooths: `k=30` for TP/TS, otherwise `k=25`
- random effects: `k` = number of levels (set during construction)

# Examples
```julia
s(:x)                      # TPRS smooth of x, default k=10
s(:x, bs=:cr)              # cubic regression spline
s(:x, :y)                  # 2d TPRS smooth
s(:x, bs=:cr, k=20)        # CR spline with 20 basis functions
s(:x, by=:group)           # varying coefficient by group
s(:x, fx=true, k=5)        # unpenalized with 5 basis functions
```
"""
function s(vars::Symbol...; bs::Symbol = :tp, k::Int = -1, by = nothing,
    id = nothing, sp = nothing, fx::Bool = false, m = nothing)
    length(vars) >= 1 || throw(ArgumentError("s() requires at least one variable"))

    basis = resolve_basis_type(bs)

    # Default k based on dimension and basis type
    if k == -1
        d = length(vars)
        if basis isa RandomEffect
            k = -1  # determined at construction time from data
        elseif d == 1
            k = 10
        elseif d == 2
            k = basis isa Union{ThinPlateSpline, ThinPlateShrink} ? 30 : 25
        else
            k = 10 * d  # rough default for higher dimensions
        end
    end

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = sp === nothing ? nothing : Float64(sp)
    m_val = m === nothing ? nothing : Int(m)

    label = _smooth_label(vars, by_sym, bs)

    return SmoothSpec(collect(vars), basis, k, by_sym, id_sym, sp_val, fx, m_val, label)
end

# Overload to accept Term objects from @formula context
function s(vars::Union{Symbol, StatsModels.AbstractTerm}...; kwargs...)
    syms = map(vars) do v
        v isa Symbol ? v : (v isa Term ? v.sym : throw(ArgumentError("expected Symbol or Term, got $(typeof(v))")))
    end
    return s(syms...; kwargs...)
end

"""
    te(vars...; k=-1, bs=:cr, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Specify a tensor product smooth. Each variable gets its own marginal basis and the
tensor product model matrix is the row-wise Kronecker product of marginals.
Produces one penalty matrix per marginal dimension.

# Arguments
- `vars`: two or more variable names (Symbols)
- `k`: total basis dimension hint. Marginal dimensions are `round(Int, k^(1/d))`.
       Default `-1` gives 5 per margin.
- `bs`: marginal basis type — a single Symbol applied to all margins, or a Vector{Symbol}
- `by`, `id`, `sp`, `fx`, `m`: as for `s()`

# Examples
```julia
te(:x1, :x2)              # tensor product with CR margins, default k
te(:x1, :x2, k=25)        # k^(1/2) ≈ 5 per margin
te(:x1, :x2, bs=:ps)      # P-spline margins
```
"""
function te(vars::Symbol...; k::Int=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing)
    length(vars) >= 2 || throw(ArgumentError("te() requires at least 2 variables"))
    d = length(vars)

    # Resolve per-margin basis types
    bs_vec = bs isa Symbol ? fill(bs, d) : bs
    length(bs_vec) == d || throw(ArgumentError("bs vector length must match number of variables"))

    # Marginal basis dimensions
    if k == -1
        k_marginal = fill(5, d)
    else
        km = max(3, round(Int, k^(1/d)))
        k_marginal = fill(km, d)
    end

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = sp === nothing ? nothing : Float64(sp)
    m_val = m === nothing ? nothing : Int(m)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_val, fx, m_val, label_i))
    end

    label = _te_label(vars, by_sym, false)
    total_k = prod(k_marginal)
    # Store as SmoothSpec{TensorProduct} with marginals accessible via _tensor_marginals
    spec = SmoothSpec(collect(vars), TensorProduct(), total_k,
                      by_sym, id_sym, sp_val, fx, m_val, label)
    _register_marginals(spec, marginals)
    return spec
end

"""
    ti(vars...; k=-1, bs=:cr, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Specify a tensor product interaction smooth (main effects removed).
Like `te()` but only includes interaction terms, useful in ANOVA-like decompositions.
"""
function ti(vars::Symbol...; k::Int=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing)
    length(vars) >= 2 || throw(ArgumentError("ti() requires at least 2 variables"))
    d = length(vars)

    bs_vec = bs isa Symbol ? fill(bs, d) : bs
    length(bs_vec) == d || throw(ArgumentError("bs vector length must match number of variables"))

    if k == -1
        k_marginal = fill(5, d)
    else
        km = max(3, round(Int, k^(1/d)))
        k_marginal = fill(km, d)
    end

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = sp === nothing ? nothing : Float64(sp)
    m_val = m === nothing ? nothing : Int(m)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_val, fx, m_val, label_i))
    end

    label = _te_label(vars, by_sym, true)
    total_k = prod(k_marginal)
    spec = SmoothSpec(collect(vars), TensorInteraction(), total_k,
                      by_sym, id_sym, sp_val, fx, m_val, label)
    _register_marginals(spec, marginals)
    return spec
end

# Accept Term objects from @formula context
function te(vars::Union{Symbol, StatsModels.AbstractTerm}...; kwargs...)
    syms = map(vars) do v
        v isa Symbol ? v : (v isa Term ? v.sym : throw(ArgumentError("expected Symbol or Term, got $(typeof(v))")))
    end
    return te(syms...; kwargs...)
end

function ti(vars::Union{Symbol, StatsModels.AbstractTerm}...; kwargs...)
    syms = map(vars) do v
        v isa Symbol ? v : (v isa Term ? v.sym : throw(ArgumentError("expected Symbol or Term, got $(typeof(v))")))
    end
    return ti(syms...; kwargs...)
end

# Module-level storage for marginal specs (keyed by objectid of SmoothSpec)
const _MARGINAL_SPECS = Dict{UInt, Vector{SmoothSpec}}()

function _register_marginals(spec::SmoothSpec, marginals::Vector{SmoothSpec})
    _MARGINAL_SPECS[objectid(spec)] = marginals
end

function _get_marginals(spec::SmoothSpec)
    return get(_MARGINAL_SPECS, objectid(spec), nothing)
end

function _smooth_label(vars::Tuple, by, bs)
    return _smooth_label(collect(vars), by, bs)
end

function _smooth_label(vars, by, bs)
    vstr = join(string.(vars), ",")
    bstr = by === nothing ? "" : ",by=$by"
    return "s($vstr$bstr,bs=$bs)"
end

function _te_label(vars, by, interaction_only::Bool)
    vstr = join(string.(vars), ",")
    bstr = by === nothing ? "" : ",by=$by"
    fname = interaction_only ? "ti" : "te"
    return "$fname($vstr$bstr)"
end
