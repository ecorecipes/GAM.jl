# Constrained Factor Smooth — bs="sz"
#
# Fits separate smooths for each factor level with a sum-to-zero constraint
# *within each level's coefficients*. This differs from factor-smooth
# interactions (bs="fs") by applying a per-level centering constraint, allowing
# a factor main effect to also be present in the model.
#
# Reference: mgcv smooth.r lines 2187-2329 (factor.smooth.construct)
#
# Algorithm:
#   1. Identify factor variable and continuous variable(s)
#   2. Build base smooth (TPRS) on continuous variable(s)
#   3. For each factor level: replicate basis, apply sum-to-zero constraint
#   4. Stack into block-diagonal structure
#   5. Penalty: replicate base penalty for each level (post-constraint)

"""Constrained factor smooth basis (mgcv `bs=\"sz\"`)."""
struct ConstrainedFactorSmooth <: AbstractBasisType end

BASIS_TYPES[:sz] = ConstrainedFactorSmooth()

function _smooth_construct(::ConstrainedFactorSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 2 ||
        throw(ArgumentError("Constrained factor smooth (sz) requires at least 2 variables: " *
            "continuous variable(s) and a grouping factor. Got: $(spec.term_vars)"))

    # Determine factor variable: from xt[:factor], or last variable by default
    factor_var = get(spec.xt, :factor, spec.term_vars[end])::Symbol
    cont_vars = Symbol[v for v in spec.term_vars if v != factor_var]

    isempty(cont_vars) &&
        throw(ArgumentError("sz smooth requires at least one continuous variable"))

    factor_col = Tables.getcolumn(data, factor_var)
    levels = sort(unique(factor_col))
    L = length(levels)
    n = length(factor_col)

    # Build marginal spec for continuous variables (TPRS, NO constraint absorption yet)
    marginal_spec = SmoothSpec(
        cont_vars, ThinPlateSpline(), spec.k,
        nothing, spec.id, spec.sp, spec.fx, spec.m,
        "s($(join(cont_vars, ",")),bs=tp)",
    )

    # Construct marginal smooth (this applies sum-to-zero constraint internally)
    marginal_sm = _smooth_construct(ThinPlateSpline(), marginal_spec, data, user_knots)
    X_marginal = marginal_sm.X    # n × k_eff (after TPRS constraint)
    k_eff = size(X_marginal, 2)

    # Per-level sum-to-zero constraint via QR projection
    # For each level, the constraint is: sum of basis columns for observations
    # in that level should sum to zero (1'X_l β_l = 0)
    # This removes one dimension per level, giving k_c = k_eff - 1 per level
    k_constrained = k_eff - 1
    k_constrained >= 1 ||
        throw(ArgumentError("Basis dimension too small for sz smooth after constraints " *
            "(k_eff=$k_eff per level, need at least 2)"))

    total_cols = L * k_constrained
    X = zeros(n, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Compute per-level constraint projection matrices
    # For each level, C_l = colMeans(X_marginal[level_l, :])
    # Then project onto null space of C_l via QR
    Z_levels = Vector{Matrix{Float64}}(undef, L)
    for (l_idx, lev) in enumerate(levels)
        mask = factor_col .== lev
        X_l = X_marginal[mask, :]  # observations for this level
        # Sum-to-zero constraint: column sums
        C_l = sum(X_l; dims=1)  # 1 × k_eff
        # QR of C_l' to get null space
        qr_C = qr(Matrix(C_l)')  # k_eff × 1
        Q_full = qr_C.Q * Matrix(I, k_eff, k_eff)
        Z_l = Q_full[:, 2:k_eff]  # k_eff × (k_eff - 1)
        Z_levels[l_idx] = Z_l
    end

    # Build block-diagonal model matrix with per-level constraint
    for i in 1:n
        l = level_map[factor_col[i]]
        col_offset = (l - 1) * k_constrained
        # Apply constraint: X_constrained = X_marginal * Z_l
        x_row = view(X_marginal, i, :)
        z_row = Z_levels[l]' * x_row  # k_constrained vector
        @inbounds for j in 1:k_constrained
            X[i, col_offset + j] = z_row[j]
        end
    end

    # Build penalties: transform marginal penalties through per-level Z
    # Each marginal penalty S_j becomes block-diagonal with Z_l' S_j Z_l per level
    penalties = Matrix{Float64}[]
    for S_j in marginal_sm.S
        S_sz = zeros(total_cols, total_cols)
        for l in 1:L
            rng = ((l - 1) * k_constrained + 1):(l * k_constrained)
            S_sz[rng, rng] .= Z_levels[l]' * S_j * Z_levels[l]
        end
        # Ensure symmetry
        S_sz = (S_sz + S_sz') / 2
        push!(penalties, S_sz)
    end

    # Penalty rank: marginal rank per level, adjusted for constraint removal
    # The constraint removes one unpenalized direction per level
    pen_rank = L * marginal_sm.rank
    null_dim = total_cols - pen_rank

    sm = ConstructedSmooth(
        spec, X, penalties,
        marginal_sm.knots,
        null_dim, pen_rank,
        nothing, nothing, 0, 0,  # no additional global constraint
        nothing, nothing, nothing,
        Int[],
        predict_cache = SZPredictCache(
            collect(levels), marginal_sm, factor_var, k_constrained, Z_levels,
        ),
    )

    return sm
end

function _predict_matrix(::ConstrainedFactorSmooth, smooth::ConstructedSmooth, newdata)
    info = smooth.predict_cache
    info isa SZPredictCache ||
        throw(ArgumentError("Cannot find constrained factor smooth metadata for prediction"))

    factor_col = Tables.getcolumn(newdata, info.factor_var)
    n_new = length(factor_col)

    # Predict marginal at new data
    marginal_sm = info.marginal_smooth
    X_marginal = _predict_matrix(marginal_sm.spec.basis, marginal_sm, newdata)
    k_eff = size(X_marginal, 2)
    k_constrained = info.k_constrained
    L = length(info.levels)
    total_cols = L * k_constrained

    X = zeros(n_new, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(info.levels))

    # Use the SAME per-level Z matrices that were absorbed at construction time.
    Z_levels = info.Z_levels

    for i in 1:n_new
        l = get(level_map, factor_col[i], 0)
        if l > 0
            col_offset = (l - 1) * k_constrained
            x_row = view(X_marginal, i, :)
            z_row = Z_levels[l]' * x_row
            @inbounds for j in 1:k_constrained
                X[i, col_offset + j] = z_row[j]
            end
        end
        # Unknown levels get zero rows
    end

    return X
end
