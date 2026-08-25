# Random effect smooth — bs="re"
#
# The simplest smooth type: an identity penalty on the coefficients.
# This is equivalent to treating the smooth as a random effect with
# iid normal prior. Useful for random intercepts/slopes in mixed models.
#
# Semantics follow mgcv's `smooth.construct.re.smooth.spec`, which builds
#     model.matrix(~ v1:v2:...:vk - 1, data)
# from the smooth's variables. In R's model-matrix rules that means:
#   * every *factor* variable expands to one indicator column per level, and
#     several factors give the product of their level counts (first variable
#     varying fastest);
#   * every *numeric* variable contributes no columns of its own — its values
#     multiply the indicators (a random slope).
# So `s(g, bs="re")` with a factor `g` is a random intercept per level, and
# `s(g, x, bs="re")` is a random slope of `x` within `g`. The rule is
# symmetric in the variables: `s(x, g)` and `s(g, x)` agree, as in mgcv.

"""
Prediction cache for random-effect smooths. Stores which variables were
treated as categorical (and their training levels, which may be non-numeric)
and which were treated as numeric multipliers, so new data can be mapped onto
the original coding at prediction time.
"""
struct REPredictCache <: AbstractSmoothPredictCache
    cat_vars::Vector{Symbol}
    levels::Vector{Vector{Any}}
    num_vars::Vector{Symbol}
end

"""
    _re_is_categorical(col) -> Bool

Whether a column expands to indicator columns in a `bs="re"` smooth, matching
R's model-matrix rule: anything that is not a real number is a factor, and
`Bool` counts as a factor (R's `model.matrix(~b-1)` gives two columns).
"""
function _re_is_categorical(col)
    T = nonmissingtype(eltype(col))
    return T <: Bool || !(T <: Real)
end

# Numeric columns holding a handful of repeated integer codes are almost
# always a grouping variable that the user meant to be a factor. mgcv would
# silently fit a *linear* effect in the code values; warn rather than let that
# pass unnoticed.
function _re_warn_if_group_codes(spec, var, col)
    T = nonmissingtype(eltype(col))
    T <: Integer || (T <: Real && all(x -> x == round(x), col)) || return nothing
    nu = length(unique(col))
    n = length(col)
    (nu >= 2 && nu < n && nu <= max(2, n ÷ 2)) || return nothing
    @warn "Random-effect smooth $(spec.label): variable :$var is numeric with " *
          "$nu distinct values, so it enters as a linear (random-slope) term " *
          "on those values — matching mgcv, which treats non-factors as " *
          "numeric. If :$var identifies groups, convert it to a factor " *
          "(e.g. `categorical($var)` or `string.($var)`) to get one random " *
          "effect per group." maxlog = 1
    return nothing
end

"""
    _re_level_index(spec, data) -> (index, slope, k, levels_list, cat_idx, num_idx)

Per-row column index and random-slope multiplier for a `bs=:re` smooth.

This is the level map the dense constructor immediately spends into an `n x k`
indicator and then discards: row `i` of that indicator is zero except at column
`index[i]`, where it is `slope[i]`. Both `_smooth_construct` and
[`re_marginal_representation`](@ref) go through here, so the dense and compact
forms cannot drift apart.
"""
function _re_level_index(spec::SmoothSpec, data)
    cols = [Tables.getcolumn(data, v) for v in spec.term_vars]
    n = length(cols[1])

    cat_idx = [i for i in eachindex(cols) if _re_is_categorical(cols[i])]
    num_idx = [i for i in eachindex(cols) if !_re_is_categorical(cols[i])]

    for i in num_idx
        _re_warn_if_group_codes(spec, spec.term_vars[i], cols[i])
    end

    # Level sets for the categorical variables, sorted as R sorts factor levels
    levels_list = [sort!(unique(collect(cols[i]))) for i in cat_idx]
    nlev = [length(l) for l in levels_list]
    k = isempty(nlev) ? 1 : prod(nlev)

    # Strides so that the first categorical variable varies fastest, matching
    # R's `model.matrix(~ a:b - 1)` column order.
    strides = ones(Int, length(nlev))
    for d in 2:length(nlev)
        strides[d] = strides[d - 1] * nlev[d - 1]
    end

    maps = [Dict(lev => i for (i, lev) in enumerate(l)) for l in levels_list]

    # Numeric variables multiply the indicator (random slopes)
    slope = ones(n)
    for i in num_idx
        slope .*= Float64.(cols[i])
    end

    index = Vector{Int32}(undef, n)
    for row in 1:n
        idx = 1
        for (d, ci) in enumerate(cat_idx)
            idx += (maps[d][cols[ci][row]] - 1) * strides[d]
        end
        index[row] = idx
    end

    return index, slope, k, levels_list, cat_idx, num_idx
end

"""
    re_marginal_representation(spec, data) -> (rep, levels_list)

Level index and random slope for a `bs=:re` smooth WITHOUT materialising the
`n x k` indicator that `_smooth_construct(::RandomEffect, ...)` builds.

`reconstruct_dense(rep, nothing)` reproduces that indicator exactly — the
"basis" is the implicit single all-ones column, so `kbase == 1` and
`nblocks == k`. Storage is `O(n)` against the dense form's `O(n*k)`: at
`n = 5e5` with 200 levels, ~6 MB against ~800 MB.
"""
function re_marginal_representation(spec::SmoothSpec, data)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("Random effect requires at least one variable"))
    index, slope, k, levels_list, _, _ = _re_level_index(spec, data)
    return MarginalBlockIndex(index, slope, 1, k), levels_list
end

function _smooth_construct(::RandomEffect, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("Random effect requires at least one variable"))

    index, slope, k, levels_list, cat_idx, num_idx = _re_level_index(spec, data)
    n = length(index)

    X = zeros(n, k)
    for row in 1:n
        X[row, index[row]] = slope[row]
    end

    # Identity penalty — penalizes all coefficients equally
    S = Matrix{Float64}(I, k, k)
    penalties = Matrix{Float64}[S]
    null_dim = 0  # no null space for random effects
    pen_rank = k

    # No identifiability constraint for random effects: mgcv never centers
    # re smooths — the full-rank ridge penalty already makes the coefficients
    # identifiable alongside an intercept (they are shrunk toward zero).
    return ConstructedSmooth(
        spec, X, penalties,
        Float64.(1:k),
        null_dim, pen_rank,
        nothing, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = REPredictCache(
            Symbol[spec.term_vars[i] for i in cat_idx],
            Vector{Any}[collect(Any, l) for l in levels_list],
            Symbol[spec.term_vars[i] for i in num_idx]),
    )
end

function _predict_matrix(::RandomEffect, smooth::ConstructedSmooth, newdata)
    # Smooths built outside `_smooth_construct` carry no cache — notably the
    # random-effect blocks that `gamm()` assembles directly from its own Z
    # matrix. `gamm` predicts those through `predict_re_matrix` and never
    # reaches here, but predicting on the inner `GamModel` does; fall back to
    # the historical integer-level coding so that path keeps working.
    cache = smooth.predict_cache
    if !(cache isa REPredictCache)
        k = length(smooth.knots)
        cache = REPredictCache(
            [smooth.spec.term_vars[1]], Vector{Any}[collect(Any, 1:k)], Symbol[])
    end

    n_new = _table_nrows(newdata)
    nlev = [length(l) for l in cache.levels]
    k = isempty(nlev) ? 1 : prod(nlev)

    strides = ones(Int, length(nlev))
    for d in 2:length(nlev)
        strides[d] = strides[d - 1] * nlev[d - 1]
    end
    maps = [Dict(lev => i for (i, lev) in enumerate(l)) for l in cache.levels]

    # Rebuild the slope product from the numeric variables
    slope = ones(n_new)
    for v in cache.num_vars
        slope .*= Float64.(Tables.getcolumn(newdata, v))
    end

    cat_cols = [Tables.getcolumn(newdata, v) for v in cache.cat_vars]

    X = zeros(n_new, k)
    unseen = Set{Any}()
    for row in 1:n_new
        idx = 1
        ok = true
        for d in eachindex(cat_cols)
            j = get(maps[d], cat_cols[d][row], nothing)
            if j === nothing
                push!(unseen, cat_cols[d][row])
                ok = false
                break
            end
            idx += (j - 1) * strides[d]
        end
        ok && (X[row, idx] = slope[row])  # unseen level → population level (0)
    end
    if !isempty(unseen)
        @warn "Random-effect smooth $(smooth.spec.label): level(s) not seen " *
              "during fitting; predicting at the population level (zero) for " *
              "these rows." unseen_levels = sort!(collect(unseen); by = string)
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X, 2))
        return X * Z
    end
    return X
end
