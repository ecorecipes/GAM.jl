# Random effect smooth — bs="re"
#
# The simplest smooth type: an identity penalty on the coefficients.
# This is equivalent to treating the smooth as a random effect with
# iid normal prior. Useful for random intercepts/slopes in mixed models.

struct RandomEffectPredictCache{T} <: AbstractSmoothPredictCache
    levels::Vector{T}
    level_map::Dict{T, Int}
end

function _smooth_construct(::RandomEffect, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("Random effect requires at least one variable"))

    var = spec.term_vars[1]
    col = Tables.getcolumn(data, var)
    levels = collect(unique(col))
    k = length(levels)

    # Dummy coding: one column per level
    n = length(col)
    X = zeros(n, k)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))
    for i in 1:n
        j = level_map[col[i]]
        X[i, j] = 1.0
    end

    # Identity penalty — penalizes all coefficients equally
    S = Matrix{Float64}(I, k, k)
    penalties = Matrix{Float64}[S]
    null_dim = 0  # no null space for random effects
    pen_rank = k

    # No identifiability constraint for random effects (penalty ensures shrinkage)
    # But we need sum-to-zero if there's also a fixed intercept
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    predict_cache = RandomEffectPredictCache(levels, level_map)
    return ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64.(1:k),
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[];
        predict_cache = predict_cache,
    )
end

function _predict_matrix(::RandomEffect, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    col = Tables.getcolumn(newdata, var)
    n_new = length(col)

    cache = smooth.predict_cache
    cache isa RandomEffectPredictCache || throw(ArgumentError("random-effect smooth is missing training level metadata"))
    k = length(cache.levels)
    X = zeros(n_new, k)
    for i in 1:n_new
        j = get(cache.level_map, col[i], nothing)
        if j !== nothing
            X[i, j] = 1.0
        end
    end

    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X, 2))
        return X * Z
    end
    return X
end
