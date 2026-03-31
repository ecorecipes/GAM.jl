# Random effect smooth — bs="re"
#
# The simplest smooth type: an identity penalty on the coefficients.
# This is equivalent to treating the smooth as a random effect with
# iid normal prior. Useful for random intercepts/slopes in mixed models.

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

    # No identifiability constraint for random effects (penalty ensures shrinkage)
    # But we need sum-to-zero if there's also a fixed intercept
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64.(1:k),
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

function _predict_matrix(::RandomEffect, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    col = Tables.getcolumn(newdata, var)
    n_new = length(col)

    # Reconstruct the level mapping from the knots
    k = length(smooth.knots)
    # For prediction, need to match levels — use the original dummy coding
    # This is a simplified version; full implementation would store level labels
    X = zeros(n_new, k)
    for i in 1:n_new
        j = findfirst(==(col[i]), 1:k)
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
