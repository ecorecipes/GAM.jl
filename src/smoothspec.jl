# Smooth specification constructors — user-facing s(), te(), ti()

function _normalize_xt(xt; pc=nothing)
    xt_norm = Dict{Symbol,Any}()
    if xt === nothing
        # no-op
    elseif xt isa AbstractDict
        for (k, v) in pairs(xt)
            xt_norm[Symbol(k)] = v
        end
    elseif xt isa AbstractString
        xt_norm[:constraints] = [String(xt)]
    elseif xt isa Symbol
        xt_norm[:constraints] = [String(xt)]
    elseif xt isa AbstractVector
        if all(v -> v isa AbstractString || v isa Symbol, xt)
            xt_norm[:constraints] = String[string(v) for v in xt]
        else
            xt_norm[:raw] = xt
        end
    else
        xt_norm[:raw] = xt
    end

    if pc !== nothing
        xt_norm[:pc] = pc
    end
    return xt_norm
end

function _normalize_tensor_xt(xt, d::Int)
    if xt === nothing
        return [Dict{Symbol,Any}() for _ in 1:d]
    elseif xt isa AbstractVector && length(xt) == d
        return [_normalize_xt(xti) for xti in xt]
    else
        xt_one = _normalize_xt(xt)
        return [copy(xt_one) for _ in 1:d]
    end
end

"""
    s(vars...; bs=:tp, k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing,
      xt=Dict{Symbol,Any}())

Specify a smooth term for use in a GAM formula.

# Arguments
- `vars`: one or more variable names (as symbols or Term objects)
- `bs`: basis type (`:tp`, `:ts`, `:cr`, `:cs`, `:cc`, `:ps`, `:bs`, `:re`, `:mrf`)
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
    id = nothing, sp = nothing, fx::Bool = false, m = nothing,
    xt = nothing, pc = nothing)
    length(vars) >= 1 || throw(ArgumentError("s() requires at least one variable"))

    basis = resolve_basis_type(bs)

    # Default k based on dimension and basis type
    if k == -1
        d = length(vars)
        if basis isa Union{RandomEffect, MarkovRandomField}
            k = -1  # determined at construction time from data
        elseif basis isa Union{FactorSmooth, ConstrainedFactorSmooth}
            # k refers to the marginal basis dimension; last var is the factor
            d_cont = max(d - 1, 1)
            k = d_cont == 1 ? 10 : (d_cont == 2 ? 30 : 10 * d_cont)
        elseif basis isa SphericalSpline
            k = 50  # default for spherical splines (2D on sphere)
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
    xt_norm = _normalize_xt(xt; pc = pc)

    label = _smooth_label(vars, by_sym, bs)

    return SmoothSpec{typeof(basis)}(collect(vars), basis, k, by_sym, id_sym, sp_val,
        fx, m_val, label, xt_norm)
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
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
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
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_val, fx, m_val, label_i, xt_vec[i]))
    end

    label = _te_label(vars, by_sym, false)
    total_k = prod(k_marginal)
    # Store as SmoothSpec{TensorProduct} with marginals accessible via _tensor_marginals
    spec = SmoothSpec(collect(vars), TensorProduct(), total_k,
                      by_sym, id_sym, sp_val, fx, m_val, label,
                      _normalize_xt(nothing; pc = pc))
    _register_marginals(spec, marginals)
    return spec
end

"""
    ti(vars...; k=-1, bs=:cr, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Specify a tensor product interaction smooth (main effects removed).
Like `te()` but only includes interaction terms, useful in ANOVA-like decompositions.
"""
function ti(vars::Symbol...; k::Int=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
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
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_val, fx, m_val, label_i, xt_vec[i]))
    end

    label = _te_label(vars, by_sym, true)
    total_k = prod(k_marginal)
    spec = SmoothSpec(collect(vars), TensorInteraction(), total_k,
                      by_sym, id_sym, sp_val, fx, m_val, label,
                      _normalize_xt(nothing; pc = pc))
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

function _t2_label(vars, by)
    vstr = join(string.(vars), ",")
    bstr = by === nothing ? "" : ",by=$by"
    return "t2($vstr$bstr)"
end

"""
    t2(vars...; k=-1, bs=:cr, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Specify an alternative tensor product smooth (mgcv's `t2()`). Like `te()`, the basis
matrix is the row-wise Kronecker product of marginal bases. The penalties differ:
each marginal penalty acts independently in its own direction via
`I ⊗ ... ⊗ S_j ⊗ ... ⊗ I`, plus a full interaction penalty `S_1 ⊗ S_2 ⊗ ...`.

This gives more penalties than `te()` but each is "simpler", providing more separate
control over penalization in each marginal direction.

# Arguments
- `vars`: two or more variable names (Symbols)
- `k`: total basis dimension hint. Marginal dimensions are `round(Int, k^(1/d))`.
       Default `-1` gives 5 per margin.
- `bs`: marginal basis type — a single Symbol applied to all margins, or a Vector{Symbol}
- `by`, `id`, `sp`, `fx`, `m`: as for `s()`

# Examples
```julia
t2(:x1, :x2)              # t2 tensor product with CR margins
t2(:x1, :x2, k=25)        # k^(1/2) ≈ 5 per margin
t2(:x1, :x2, bs=:ps)      # P-spline margins
```
"""
function t2(vars::Symbol...; k::Int=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
    length(vars) >= 2 || throw(ArgumentError("t2() requires at least 2 variables"))
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
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_val, fx, m_val, label_i, xt_vec[i]))
    end

    label = _t2_label(vars, by_sym)
    total_k = prod(k_marginal)
    spec = SmoothSpec(collect(vars), T2TensorProduct(), total_k,
                      by_sym, id_sym, sp_val, fx, m_val, label,
                      _normalize_xt(nothing; pc = pc))
    _register_marginals(spec, marginals)
    return spec
end

# Accept Term objects from @formula context
function t2(vars::Union{Symbol, StatsModels.AbstractTerm}...; kwargs...)
    syms = map(vars) do v
        v isa Symbol ? v : (v isa Term ? v.sym : throw(ArgumentError("expected Symbol or Term, got $(typeof(v))")))
    end
    return t2(syms...; kwargs...)
end

# ─── Basis-type convenience functions for @formula ─────────────────────────
#
# These let users write `@formula(y ~ cr(x, 20))` instead of needing
# `@formulak(y ~ s(x, k=20, bs=:cr))` for the most common basis types.
# Each is a thin wrapper around `s()` with a fixed `bs` argument.

"""
    cr(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Cubic regression spline smooth term. Equivalent to `s(vars...; bs=:cr, ...)`.
Usable in `@formula`: `@formula(y ~ cr(x, 20))`.
"""
function cr(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :cr, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    tp(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Thin plate regression spline smooth term. Equivalent to `s(vars...; bs=:tp, ...)`.
"""
function tp(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :tp, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    ts(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Thin plate regression spline with shrinkage. Equivalent to `s(vars...; bs=:ts, ...)`.
"""
function ts(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :ts, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    cs(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Cubic regression spline with shrinkage. Equivalent to `s(vars...; bs=:cs, ...)`.
"""
function cs(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :cs, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    cc(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Cyclic cubic regression spline. Equivalent to `s(vars...; bs=:cc, ...)`.
"""
function cc(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :cc, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    ps(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

P-spline smooth term. Equivalent to `s(vars...; bs=:ps, ...)`.
"""
function ps(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :ps, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

"""
    cps(vars...; k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing)

Cyclic P-spline smooth term. Equivalent to `s(vars...; bs=:cps, ...)`.
"""
function cps(vars::Symbol...; k::Int = -1, by = nothing, id = nothing,
    sp = nothing, fx::Bool = false, m = nothing)
    return s(vars...; bs = :cps, k = k, by = by, id = id, sp = sp, fx = fx, m = m)
end

# Accept Term objects from @formula context
for fname in (:cr, :tp, :ts, :cs, :cc, :ps, :cps)
    @eval function $fname(vars::Union{Symbol, StatsModels.AbstractTerm}...; kwargs...)
        syms = map(vars) do v
            v isa Symbol ? v :
            (v isa Term ? v.sym :
             throw(ArgumentError("expected Symbol or Term, got $(typeof(v))")))
        end
        return $fname(syms...; kwargs...)
    end
end

# Register aliases so _is_smooth_function recognizes them
function _register_smooth_aliases()
    push!(_SMOOTH_ALIASES, cr, tp, ts, cs, cc, ps, cps)
end
