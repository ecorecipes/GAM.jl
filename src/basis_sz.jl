# Constrained Factor Smooth — bs="sz"
#
# mgcv's sz smooth: one smooth per factor level, constrained so that the
# level smooths SUM TO ZERO ACROSS LEVELS at every covariate value
# (deviation-from-average smooths). Because every level uses the same
# marginal basis, the functional constraint Σ_l f_l(x) = 0 ∀x reduces to the
# coefficient constraint Σ_l β_l = 0, absorbed via an orthonormal contrast
# basis Q_L (null space of 1_L') applied on the level index:
# X = (level-indicator ⊗ marginal) · (Q_L ⊗ I_k).
# This permits a factor main effect and a global smooth alongside the sz term.

"""Constrained factor smooth basis (mgcv `bs=\"sz\"`)."""
struct ConstrainedFactorSmooth <: AbstractBasisType end

BASIS_TYPES[:sz] = ConstrainedFactorSmooth()

"""
Prediction cache for sz smooths: stores the factor levels, the (raw,
unconstrained) marginal smooth template, the factor variable, and the
orthonormal level-contrast basis `Q_L` absorbed at construction.
"""
struct SZContrastPredictCache <: AbstractSmoothPredictCache
    levels::Vector
    marginal_smooth  # ConstructedSmooth (raw marginal, no constraint)
    factor_var::Symbol
    Q_L::Matrix{Float64}
end

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
    L >= 2 || throw(ArgumentError("sz smooth requires a factor with ≥ 2 levels"))
    n = length(factor_col)

    # Raw (unconstrained) marginal smooth on the continuous variable(s):
    # per-level constants stay in the span, so each level's deviation smooth
    # can shift as well as bend. Identifiability comes from the
    # sum-over-levels constraint absorbed below.
    marginal_spec = SmoothSpec(
        cont_vars, ThinPlateSpline(), spec.k,
        nothing, spec.id, spec.sp, spec.fx, spec.m,
        "s($(join(cont_vars, ",")),bs=tp)",
    )
    marginal_sm = _construct_tprs(marginal_spec, data, user_knots;
        absorb_cons = false)
    X_marginal = marginal_sm.X    # n × k_eff (raw)
    k_eff = size(X_marginal, 2)

    # Orthonormal contrast basis on the level index: Q_L spans null(1_L'),
    # via the same deterministic QR used by absorb_constraints!.
    qr_ones = qr(reshape(ones(L), :, 1))
    Q_L = (qr_ones.Q * Matrix(I, L, L))[:, 2:L]   # L × (L-1)

    total_cols = (L - 1) * k_eff
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Row i of the full model matrix: kron(Q_L[l, :], X_marginal[i, :])
    X = zeros(n, total_cols)
    @inbounds for i in 1:n
        l = level_map[factor_col[i]]
        for c in 1:(L - 1)
            w = Q_L[l, c]
            w == 0.0 && continue
            off = (c - 1) * k_eff
            for j in 1:k_eff
                X[i, off + j] = w * X_marginal[i, j]
            end
        end
    end

    # Penalties: (Q_L ⊗ I)' (I_L ⊗ S_j) (Q_L ⊗ I) = (Q_L'Q_L) ⊗ S_j = I_{L-1} ⊗ S_j
    #
    # mgcv requires the base smooth to be singly penalized (`smooth.r:2240`:
    # `if (length(object$S)>1) stop("\"sz\" smooth cannot use a multiply
    # penalized basis")`). The marginal here is always TPRS, so this holds by
    # construction; assert it rather than silently producing L*n_marg
    # penalties if the marginal ever becomes configurable.
    length(marginal_sm.S) == 1 || throw(ArgumentError(
        "sz smooth cannot use a multiply penalized base basis (got " *
        "$(length(marginal_sm.S)) marginal penalties)"))
    S_marg = marginal_sm.S[1]

    # ONE PENALTY PER FACTOR LEVEL, matching mgcv (`smooth.r:2281-2286`):
    #
    #     if (is.null(object$id)) {   ## one penalty and one sp per smooth
    #       for (i in 1:prod(nf)) { S0 <- matrix(0,p,p)
    #                               S0[ind,ind] <- object$S[[1]]
    #                               S[[i]] <- S0; ind <- ind + p0 }
    #       object$rank <- rep(object$rank,prod(nf))
    #     } else { ... single summed penalty ... }
    #
    # mgcv builds those on the UNCONSTRAINED L*k basis and applies the
    # sum-to-zero contrast afterwards (`smooth.r:4139-4147`, the
    # `length(sm$C)>1` branch, which hits each penalty with `XZKr` twice —
    # i.e. Z'S_iZ with the same Z for every i). We absorb the contrast at
    # construction instead, so we apply that transform here directly. With
    # Z = Q_L ⊗ I and the i-th level's penalty S_i = (e_i e_i') ⊗ S_marg,
    #
    #     Z' S_i Z = (Q_L' e_i e_i' Q_L) ⊗ S_marg = (q_i q_i') ⊗ S_marg
    #
    # where q_i = Q_L[i, :]. These sum back to the old single penalty, since
    # Σ_i q_i q_i' = Q_L'Q_L = I_{L-1} — so this is a strict decomposition of
    # what was here before, not a different model.
    #
    # It matters because one λ per level lets a weakly-deviating level be
    # shrunk almost to zero while a strongly-deviating one stays loose; a
    # single shared λ must compromise, which showed up as a markedly larger
    # deviation edf than mgcv's (14.63 against 10.24 on a three-region
    # seasonal model).
    #
    # An explicit `id` selects mgcv's other branch: one summed penalty, one
    # smoothing parameter, levels forced to share smoothness.
    penalties = Matrix{Float64}[]
    if spec.id === nothing
        for i in 1:L
            q = Q_L[i, :]
            push!(penalties, Matrix(Symmetric(kron(q * q', S_marg))))
        end
    else
        S_sz = zeros(total_cols, total_cols)
        for c in 1:(L - 1)
            rng = ((c - 1) * k_eff + 1):(c * k_eff)
            S_sz[rng, rng] .= S_marg
        end
        push!(penalties, Matrix(Symmetric(S_sz)))
    end

    marg_nullity = _penalty_nullity(marginal_sm.S, k_eff)
    null_dim = (L - 1) * marg_nullity
    pen_rank = total_cols - null_dim

    sm = ConstructedSmooth(
        spec, X, penalties,
        marginal_sm.knots,
        null_dim, pen_rank,
        nothing, nothing, 0, 0,  # constraint already absorbed via Q_L
        nothing, nothing, nothing,
        Int[],
        predict_cache = SZContrastPredictCache(
            collect(levels), marginal_sm, factor_var, Q_L,
        ),
    )

    return sm
end

function _predict_matrix(::ConstrainedFactorSmooth, smooth::ConstructedSmooth, newdata)
    info = smooth.predict_cache
    info isa SZContrastPredictCache ||
        throw(ArgumentError("Cannot find constrained factor smooth metadata for prediction"))

    factor_col = Tables.getcolumn(newdata, info.factor_var)
    n_new = length(factor_col)

    # Predict raw marginal at new data (no constraint on the template)
    marginal_sm = info.marginal_smooth
    X_marginal = _predict_matrix(marginal_sm.spec.basis, marginal_sm, newdata)
    k_eff = size(X_marginal, 2)
    Q_L = info.Q_L
    L = length(info.levels)
    total_cols = (L - 1) * k_eff

    X = zeros(n_new, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(info.levels))

    unseen = Set{eltype(factor_col)}()
    @inbounds for i in 1:n_new
        l = get(level_map, factor_col[i], 0)
        if l == 0  # unknown levels get zero rows
            push!(unseen, factor_col[i])
            continue
        end
        for c in 1:(L - 1)
            w = Q_L[l, c]
            w == 0.0 && continue
            off = (c - 1) * k_eff
            for j in 1:k_eff
                X[i, off + j] = w * X_marginal[i, j]
            end
        end
    end
    if !isempty(unseen)
        # Same warn-and-zero convention as `by=` and `bs=:re`; this path used
        # to zero SILENTLY, unlike its siblings (mgcv errors here).
        @warn "Constrained factor smooth $(smooth.spec.label): level(s) not " *
              "seen during fitting get zero contribution." unseen_levels =
            sort!(collect(unseen); by = string)
    end

    return X
end
