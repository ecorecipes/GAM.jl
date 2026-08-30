# Smooth specification constructors — user-facing s(), te(), ti()

"""
    _check_sp_fx(sp, fx, call, vars)

Reject `sp=` combined with `fx=true` at construction time.

`fx=true` drops the penalty entirely, so a supplied smoothing parameter can
never be used. Accepting the pair and letting `fx` win means the smooth the
user asked for is silently not the smooth they get; failing in `s()` itself
points at the term instead of at a fit that ran to completion with a
different model. (`_validate_formula_smooths` keeps the same check for specs
built by other routes.)
"""
function _check_sp_fx(sp, fx::Bool, call::String, vars)
    (fx && sp !== nothing) || return nothing
    varlist = join(string.(vars), ", ")
    throw(ArgumentError(
        "$call($varlist, sp=$sp, fx=true): sp= and fx=true are incompatible. " *
        "fx=true leaves the smooth unpenalized, so the smoothing parameter " *
        "would be ignored. Drop sp= to keep the term unpenalized, or drop " *
        "fx=true to fit at the given smoothing parameter."))
end

"""
    _normalize_sp(sp, call, vars) -> Union{Float64, Vector{Float64}, Nothing}

Coerce a user-supplied `sp=` to the stored representation, rejecting values
that can never be valid smoothing parameters.

A scalar fixes every penalty of the smooth at that value (mgcv allows this
too, and it is what most smooths — which have a single penalty — need). A
vector fixes them individually, one entry per penalty, matching mgcv's
per-penalty `sp` vector for multi-penalty smooths such as `bs="ad"`, `t2`
and `bs="fs"`.

Positivity and element type are checked here because they are properties of
the value alone, and failing in `s()`/`te()` points at the offending term.
The *length* is deliberately NOT checked here: for several bases the penalty
count is not known until the basis has been constructed against data (an
adaptive smooth's sub-penalty count depends on `k` and `m`, `fs`'s on the
marginal null-space dimension, `t2`'s on the marginal block structure), so
that check lives in `setup_penalties` where `S_list` is in hand.
"""
function _normalize_sp(sp, call::String, vars)
    sp === nothing && return nothing
    varlist = join(string.(vars), ", ")

    if sp isa AbstractVector
        isempty(sp) && throw(ArgumentError(
            "$call($varlist, sp=$sp): sp= must not be empty. Pass a scalar to " *
            "fix every penalty at one value, or one entry per penalty."))
        all(x -> x isa Real, sp) || throw(ArgumentError(
            "$call($varlist, sp=$sp): sp= must contain real numbers."))
        vals = Float64[Float64(x) for x in sp]
        for (i, v) in enumerate(vals)
            (isfinite(v) && v > 0) || throw(ArgumentError(
                "$call($varlist, sp=$sp): sp= entries must be finite and " *
                "positive, got $v at index $i."))
        end
        return vals
    end

    sp isa Real || throw(ArgumentError(
        "$call($varlist, sp=$sp): sp= must be a positive number, or a vector " *
        "of them with one entry per penalty, got $(typeof(sp))."))
    v = Float64(sp)
    (isfinite(v) && v > 0) || throw(ArgumentError(
        "$call($varlist, sp=$sp): sp= must be finite and positive, got $v."))
    return v
end

"""
    _check_sos_units(bs, xt, vars)

Reject an invalid `xt[:units]` on an `sos` term at construction time.

`sos` reads latitude/longitude in degrees by default (as mgcv does); the only
other accepted value is `:radians`. A typo such as `:radian` or `:deg` would
otherwise fall through to the default and silently rescale the fit by 57×.
"""
function _check_sos_units(bs::Symbol, xt::Dict{Symbol,Any}, vars)
    bs === :sos || return nothing
    haskey(xt, :units) || return nothing
    u = xt[:units]
    u = u isa AbstractString ? Symbol(u) : u
    u === :degrees || u === :radians || throw(ArgumentError(
        "s($(join(string.(vars), ", ")), bs=:sos, xt=Dict(:units => $(repr(xt[:units])))): " *
        "xt[:units] must be :degrees (the default, matching mgcv) or :radians."))
    return nothing
end

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
    _normalize_m(m, fname) -> Union{Int, Nothing}

Validate the `m` (penalty/basis order) argument.

`m` is a SCALAR in GAM.jl. mgcv spells some orders as a vector — most often
`bs = "bs"` with `m = c(spline_order, penalty_order)`, and `bs = "ds"` with
`m = c(m, s)` — so a vector arrives here from anyone porting R code
verbatim. It used to die with a bare `MethodError: no method matching
Int64(::Vector{Int64})`, which says nothing about the convention; this raises
an error that names the mapping instead.

Note `sp` DOES accept a vector (adaptive smooths and `t2`/`fs` terms carry
several smoothing parameters), so rejecting a vector here is a real
asymmetry rather than a blanket rule — hence spelling it out.
"""
function _normalize_m(m, fname::String)
    m === nothing && return nothing
    if m isa AbstractVector
        throw(ArgumentError(
            "$fname(...; m = $(m)): `m` must be a scalar in GAM.jl, not a " *
            "vector. mgcv writes some orders as a vector. For `bs=\"bs\"`, " *
            "`m = c(spline_order, penalty_order)`, GAM.jl takes the penalty " *
            "order alone, so mgcv's `m = c(3, 2)` is `m = 2` here. For " *
            "`bs=:ds` (Duchon), which genuinely needs both orders, the second " *
            "goes in `xt`: mgcv's `m = c(2, 0.5)` is " *
            "`m = 2, xt = Dict(:s => 0.5)`. (`sp` does accept a vector; `m` " *
            "does not.)"))
    end
    m isa Real || throw(ArgumentError(
        "$fname(...; m = $(m)): `m` must be an integer, got $(typeof(m))"))
    return Int(m)
end

"""
    s(vars...; bs=:tp, k=-1, by=nothing, id=nothing, sp=nothing, fx=false, m=nothing,
      xt=Dict{Symbol,Any}())

Specify a smooth term for use in a GAM formula.

# Arguments
- `vars`: one or more variable names (as symbols or Term objects)
- `bs`: basis type. Core types: `:tp`, `:ts`, `:cr`, `:cs`, `:cc`, `:ps`,
  `:cps`, `:bs`, `:re`, `:mrf`; further registered types include `:gp`,
  `:ds`, `:so`, `:fs`, `:sos`, `:sz`, `:spde`, `:lo`, `:ad`, `:fp`, the
  SCAM shape constraints (`:mpi`, `:mpd`, `:cv`, `:cx`, `:micx`, `:micv`,
  `:mdcx`, `:mdcv`) and SCASM (`:sc`, `:scad`) — see `GAM.BASIS_TYPES`
- `k`: basis dimension. `-1` (default) uses a sensible default based on basis type
- `by`: optional `by` variable for varying-coefficient models
- `id`: identifier for linking smooths sharing smoothing parameters
  (not yet supported — passing it errors at fit time)
- `sp`: fixed smoothing parameter, held at this value (excluded from
  optimization). `nothing` = estimate automatically. A scalar fixes every
  penalty of the smooth; for a multi-penalty smooth (`bs=:ad`, `bs=:fs`) pass a
  vector with one entry per penalty, matching mgcv's per-penalty `sp` vector.
  The length is checked once the basis is built, so a mismatch reports the
  smooth's actual penalty count
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
    _check_sp_fx(sp, fx, "s", vars)

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
    sp_val = _normalize_sp(sp, "s", vars)
    m_val = _normalize_m(m, "s")
    xt_norm = _normalize_xt(xt; pc = pc)
    _check_sos_units(bs, xt_norm, vars)

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
- `k`: **per-marginal** basis dimension, recycled across margins (mgcv's
       convention): `te(:x1, :x2, k=8)` builds an 8×8 tensor product. Pass a
       vector for unequal margins (`k = [4, 7]` ≙ mgcv's `k = c(4, 7)`).
       Default `-1` gives 5 per margin.
- `bs`: marginal basis type — a single Symbol applied to all margins, or a Vector{Symbol}
- `by`, `id`, `sp`, `fx`, `m`: as for `s()`

!!! warning "Breaking change in 0.2"
    A scalar `k` was previously a *total* dimension hint split as `k^(1/d)`
    per margin, so `te(:x1, :x2, k=25)` meant 5×5; it now means 25×25. Use
    the vector form (`k = [5, 5]`) to keep an old model's basis size.

# Examples
```julia
te(:x1, :x2)              # tensor product with CR margins, 5 per margin
te(:x1, :x2, k=8)         # 8 basis functions per margin (8×8), as in mgcv
te(:x1, :x2, k=[4, 7])    # unequal marginal dimensions
te(:x1, :x2, bs=:ps)      # P-spline margins
```
"""
function te(vars::Symbol...; k::Union{Int,AbstractVector{<:Integer}}=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
    length(vars) >= 2 || throw(ArgumentError("te() requires at least 2 variables"))
    _check_sp_fx(sp, fx, "te", vars)
    d = length(vars)

    # Resolve per-margin basis types
    bs_vec = bs isa Symbol ? fill(bs, d) : bs
    length(bs_vec) == d || throw(ArgumentError("bs vector length must match number of variables"))

    # Marginal basis dimensions. A scalar `k` is the PER-MARGINAL dimension,
    # recycled across margins as in mgcv; pass a vector to set them directly,
    # as in mgcv's `k = c(4, 7)`. See `_tensor_marginal_k`.
    k_marginal = _tensor_marginal_k(k, d)

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = _normalize_sp(sp, "te", vars)
    # Marginals must not inherit a VECTOR sp: its entries index the tensor's
    # own penalties, not a marginal's. A scalar still propagates, unchanged.
    sp_marg = sp_val isa AbstractVector ? nothing : sp_val
    m_val = _normalize_m(m, "te")
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_marg, fx, m_val, label_i, xt_vec[i]))
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

A scalar `k` is the **per-marginal** basis dimension, recycled across
margins (mgcv's convention); pass a vector for unequal margins, e.g.
`k = [4, 7]`. Before 0.2 a scalar was a *total* hint split as `k^(1/d)` —
use the vector form to reproduce an old model's basis size.

# Example
```julia
using GAM, DataFrames
df = DataFrame(x = rand(200), z = rand(200), y = randn(200))

# main effects plus a pure interaction (ANOVA decomposition)
m = gam(@formula(y ~ s(x) + s(z) + ti(x, z, k = 5)), df)

ti(:x, :z, k = [4, 7])        # per-marginal basis dimensions
```
"""
function ti(vars::Symbol...; k::Union{Int,AbstractVector{<:Integer}}=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
    length(vars) >= 2 || throw(ArgumentError("ti() requires at least 2 variables"))
    _check_sp_fx(sp, fx, "ti", vars)
    d = length(vars)

    bs_vec = bs isa Symbol ? fill(bs, d) : bs
    length(bs_vec) == d || throw(ArgumentError("bs vector length must match number of variables"))

    k_marginal = _tensor_marginal_k(k, d)

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = _normalize_sp(sp, "ti", vars)
    # Marginals must not inherit a VECTOR sp: its entries index the tensor's
    # own penalties, not a marginal's. A scalar still propagates, unchanged.
    sp_marg = sp_val isa AbstractVector ? nothing : sp_val
    m_val = _normalize_m(m, "ti")
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_marg, fx, m_val, label_i, xt_vec[i]))
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

"""
    _tensor_marginal_k(k, d) -> Vector{Int}

Resolve the per-marginal basis dimensions for a tensor smooth over `d`
variables, following mgcv's convention exactly.

A scalar `k` is the **per-marginal** basis dimension, recycled across margins:
`te(x, z, k = 8)` builds an 8×8 tensor product, as in mgcv's
`te(x, z, k = 8)` (`R/smooth.r`, `te`: `if (length(k)==1&&ok) k<-rep(k,n.bases)`).
A vector gives the marginal dimensions directly (`k = [4, 7]` ≙ mgcv's
`k = c(4, 7)`), and `k = -1` selects mgcv's default of 5 per margin
(mgcv's `k <- 5^d` with `d = 1` per basis).

!!! warning "Breaking change"
    Before GAM.jl 0.2.0 a scalar `k` was a *total* dimension hint, split as
    `k^(1/d)` per margin, so `te(x, z, k = 25)` meant 5×5. It now means
    25×25. Models ported from mgcv were silently getting a basis roughly
    `d`-th-root smaller; models written against the old GAM.jl behaviour
    should switch to the explicit vector form (`k = [5, 5]`) to keep their
    previous basis size.
"""
function _tensor_marginal_k(k, d::Int)
    if k isa AbstractVector
        length(k) == d || throw(ArgumentError(
            "k vector has length $(length(k)) but there are $d variables"))
        all(>=(3), k) || throw(ArgumentError(
            "each marginal k must be >= 3, got $(collect(k))"))
        return collect(Int, k)
    end
    k == -1 && return fill(5, d)
    k >= 3 || throw(ArgumentError(
        "marginal k must be >= 3, got $k (a scalar k is the per-marginal " *
        "basis dimension, as in mgcv; use a vector for unequal margins)"))
    return fill(Int(k), d)
end

# Marginal specs for tensor product smooths (te/ti/t2) are stored in the
# spec's own `xt` Dict so they travel with the spec through serialization.
function _register_marginals(spec::SmoothSpec, marginals::AbstractVector{<:SmoothSpec})
    spec.xt[:marginals] = marginals
    return spec
end

function _get_marginals(spec::SmoothSpec)
    return get(spec.xt, :marginals, nothing)
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
matrix is the row-wise Kronecker product of marginal bases. The penalties follow
Wood, Scheipl & Faraway (2013): each marginal is split into orthogonal null and
range parts, and the tensor columns partition into blocks that each carry their
own identity penalty on their own columns — diagonal penalties with
non-overlapping support.

That non-overlap is what lets a `t2()` smooth be written as independent
random-effect blocks (one variance component per penalty), the property `te()`
lacks; use `t2()` when you need the mixed-model decomposition (as gamm4 does).

# Arguments
- `vars`: two or more variable names (Symbols)
- `k`: **per-marginal** basis dimension, recycled across margins (mgcv's
       convention); pass a vector for unequal margins. Default `-1` gives 5
       per margin. (Before 0.2 a scalar was a *total* hint split as
       `k^(1/d)`; use the vector form to keep an old model's basis size.)
- `bs`: marginal basis type — a single Symbol applied to all margins, or a Vector{Symbol}
- `by`, `id`, `sp`, `fx`, `m`: as for `s()`

# Examples
```julia
t2(:x1, :x2)              # t2 tensor product with CR margins, 5 per margin
t2(:x1, :x2, k=8)         # 8 basis functions per margin, as in mgcv
t2(:x1, :x2, bs=:ps)      # P-spline margins
```
"""
function t2(vars::Symbol...; k::Union{Int,AbstractVector{<:Integer}}=-1, bs::Union{Symbol,Vector{Symbol}}=:cr,
            by=nothing, id=nothing, sp=nothing, fx::Bool=false, m=nothing,
            xt=nothing, pc=nothing)
    length(vars) >= 2 || throw(ArgumentError("t2() requires at least 2 variables"))
    _check_sp_fx(sp, fx, "t2", vars)
    d = length(vars)

    bs_vec = bs isa Symbol ? fill(bs, d) : bs
    length(bs_vec) == d || throw(ArgumentError("bs vector length must match number of variables"))

    k_marginal = _tensor_marginal_k(k, d)

    by_sym = by isa Symbol ? by : (by isa Term ? by.sym : nothing)
    id_sym = id isa Symbol ? id : nothing
    sp_val = _normalize_sp(sp, "t2", vars)
    # Marginals must not inherit a VECTOR sp: its entries index the tensor's
    # own penalties, not a marginal's. A scalar still propagates, unchanged.
    sp_marg = sp_val isa AbstractVector ? nothing : sp_val
    m_val = _normalize_m(m, "t2")
    xt_vec = _normalize_tensor_xt(xt, d)

    marginals = SmoothSpec[]
    for i in 1:d
        basis_i = resolve_basis_type(bs_vec[i])
        label_i = "s($(vars[i]),bs=$(bs_vec[i]))"
        push!(marginals, SmoothSpec([vars[i]], basis_i, k_marginal[i],
                                    nothing, id_sym, sp_marg, fx, m_val, label_i, xt_vec[i]))
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
    push!(_SMOOTH_ALIASES, cr, tp, ts, cs, cc, ps, cps, s_nest)
end
