# Random effect smooth — bs="re"
#
# The simplest smooth type: an identity penalty on the coefficients.
# This is equivalent to treating the smooth as a random effect with
# iid normal prior. Useful for random intercepts/slopes in mixed models.

"""
Prediction cache for random-effect smooths: stores the training level values
(which may be non-numeric, e.g. strings or categorical levels) so that new
data can be matched against the original dummy coding at prediction time.
"""
struct REPredictCache <: AbstractSmoothPredictCache
    levels::Vector{Any}
end

function _smooth_construct(::RandomEffect, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("Random effect requires at least one variable"))

    var = spec.term_vars[1]
    col = Tables.getcolumn(data, var)
    levels = sort(unique(col))
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
        predict_cache = REPredictCache(collect(Any, levels)),
    )
end

function _predict_matrix(::RandomEffect, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    col = Tables.getcolumn(newdata, var)
    n_new = length(col)

    # Match new data against the training levels stored at construction time.
    cache = smooth.predict_cache
    levels = if cache isa REPredictCache
        cache.levels
    else
        # Fallback for smooths constructed without a cache (integer coding)
        collect(Any, 1:length(smooth.knots))
    end
    k = length(levels)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    X = zeros(n_new, k)
    unseen = Set{Any}()
    for i in 1:n_new
        j = get(level_map, col[i], nothing)
        if j === nothing
            push!(unseen, col[i])  # zero row → population-level prediction
        else
            X[i, j] = 1.0
        end
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
