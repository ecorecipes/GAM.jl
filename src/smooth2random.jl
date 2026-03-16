# smooth2random — Mixed Model Reparameterization
#
# Converts a smooth term (basis matrix X + penalty matrix S) into the
# mixed-model representation used for Bayesian inference:
#   X_original * β = Xf * β_fixed + Zs * b_random
# where b_random ~ N(0, σ²I) and σ² controls smoothness.
#
# This is the key bridge between frequentist GAMs (penalized likelihood)
# and Bayesian GAMs (prior on smooth SD). Equivalent to mgcv::smooth2random.
#
# Reference: Wood, S.N. (2004). Stable and efficient multiple smoothing
# parameter estimation for generalized additive models. JASA, 99(467).

"""
    SmoothMixedModel

Result of `smooth2random`: a smooth term decomposed into fixed (unpenalized)
and random (penalized) components in the mixed-model parameterization.

The original smooth `X * β` is equivalent to `Xf * β_f + Zs[1] * b_1 + ...`
where each `b_i ~ N(0, σ²_i I)`.

# Fields
- `Xf::Matrix{Float64}`: fixed-effect (null space) design matrix (n × n_fixed)
- `Zs::Vector{Matrix{Float64}}`: random-effect design matrices, one per penalty
- `trans_U::Union{Matrix{Float64}, Nothing}`: orthogonal matrix mapping back to
  original parameterization (b_original = trans_U * (trans_D .* b_fit))
- `trans_D::Vector{Float64}`: diagonal rescaling vector
- `pen_ind::Vector{Int}`: which penalty (1,2,...) penalizes each column (0 = fixed)
- `rind::Vector{Int}`: indices mapping random coefs to position in fit vector
- `label::String`: smooth term label
- `fixed::Bool`: true if this smooth is unpenalized (fx=true)
"""
struct SmoothMixedModel
    Xf::Matrix{Float64}
    Zs::Vector{Matrix{Float64}}
    trans_U::Union{Matrix{Float64}, Nothing}
    trans_D::Vector{Float64}
    pen_ind::Vector{Int}
    rind::Vector{Int}
    label::String
    fixed::Bool
end

"""
    smooth2random(sm::ConstructedSmooth) -> SmoothMixedModel

Convert a constructed smooth term to mixed-model form.

For a smooth with basis matrix X (n × k) and penalty S (k × k):
1. Eigendecompose S = U D U'
2. Split into penalized columns (D > 0 → random effects Zs) and
   null space columns (D = 0 → fixed effects Xf)
3. Rescale so the random effect prior becomes N(0, σ²I)

The transform back to original parameterization is:
  β_original = trans_U * (trans_D .* [b_random; β_fixed])

# Single-penalty smooths (tp, cr, ps, bs, re)
Eigendecomposition of the single penalty matrix separates penalized from
unpenalized components. The random effects have identity penalty.

# Shrinkage smooths (ts, cs)
All columns are penalized (null_dim = 0), so Xf is empty.

# Multi-penalty smooths (te, ti, t2)
Sum all penalties, eigendecompose to find null space. Each original penalty
is projected into the penalized subspace, giving one random effect block
per penalty with its own SD parameter.

# Examples
```julia
sm = smooth_construct(s(:x, bs=:cr, k=10), data)
smm = smooth2random(sm)
# smm.Xf is (n × 1) — the null space (linear trend)
# smm.Zs[1] is (n × 8) — the penalized wiggle
```
"""
function smooth2random(sm::ConstructedSmooth)
    if sm.spec.fx
        # Unpenalized smooth — everything is fixed
        return SmoothMixedModel(
            sm.X, Matrix{Float64}[], nothing, ones(size(sm.X, 2)),
            zeros(Int, size(sm.X, 2)), Int[], sm.spec.label, true
        )
    end

    n_penalties = length(sm.S)
    if n_penalties == 0
        return SmoothMixedModel(
            sm.X, Matrix{Float64}[], nothing, ones(size(sm.X, 2)),
            zeros(Int, size(sm.X, 2)), Int[], sm.spec.label, true
        )
    elseif n_penalties == 1
        return _smooth2random_single(sm)
    else
        return _smooth2random_multi(sm)
    end
end

# ============================================================================
# Single-penalty smooths (tp, ts, cr, cs, ps, bs, re, etc.)
# ============================================================================

"""
    _smooth2random_single(sm) -> SmoothMixedModel

Mixed-model reparameterization for single-penalty smooths.
Follows mgcv's `smooth2random.mgcv.smooth`.
"""
function _smooth2random_single(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    S = sm.S[1]

    # Eigendecompose penalty
    eig = eigen(Symmetric(S))
    # Ensure deterministic sign (same as mgcv's hack)
    if eig.vectors[1, 1] < 0
        eig = Eigen(eig.values, -eig.vectors)
    end

    null_rank = sm.null_dim
    p_rank = sm.rank
    if p_rank > k
        p_rank = k
    end

    U = eig.vectors  # k × k orthogonal matrix

    # Build rescaling vector D:
    # - penalized columns: 1/sqrt(eigenvalue) (reduces penalty to identity)
    # - null space columns: 1 (no change)
    D = Vector{Float64}(undef, k)
    for j in 1:p_rank
        D[j] = 1.0 / sqrt(max(eig.values[j], eps()))
    end
    for j in (p_rank + 1):k
        D[j] = 1.0
    end

    # Transform: X_new = X * U * diag(D)
    UD = U * Diagonal(D)  # k × k
    X_new = sm.X * UD     # n × k

    # Split: first p_rank columns are random (penalized), rest are fixed
    if p_rank < k
        Xf = X_new[:, (p_rank + 1):k]
    else
        Xf = Matrix{Float64}(undef, size(sm.X, 1), 0)
    end

    Zs = [X_new[:, 1:p_rank]]

    # Index tracking
    rind = collect(1:p_rank)
    pen_ind = zeros(Int, k)
    pen_ind[1:p_rank] .= 1

    return SmoothMixedModel(Xf, Zs, U, D, pen_ind, rind, sm.spec.label, false)
end

# ============================================================================
# Multi-penalty smooths (te, ti, t2)
# ============================================================================

"""
    _smooth2random_multi(sm) -> SmoothMixedModel

Mixed-model reparameterization for multi-penalty smooths.
Follows mgcv's `smooth2random.t2.smooth` for t2 terms and
`smooth2random.tensor.smooth` for te terms.

For t2 smooths: each penalty has non-overlapping penalized columns;
columns penalized by penalty i are rescaled to identity and become
random effect block i.

For te smooths: sum all penalties (normalized), eigendecompose to find
null space, then project each penalty into the penalized subspace.
"""
function _smooth2random_multi(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    n_pen = length(sm.S)

    # Check if penalties have non-overlapping diagonal support (t2-style)
    if _is_t2_style(sm)
        return _smooth2random_t2(sm)
    else
        return _smooth2random_tensor(sm)
    end
end

"""Check if penalties have non-overlapping diagonal supports (t2 pattern)."""
function _is_t2_style(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    # For each column, count how many penalties have nonzero diagonal entry
    pen_count = zeros(Int, k)
    for S in sm.S
        for j in 1:k
            if abs(S[j, j]) > eps() * maximum(abs.(diag(S)))
                pen_count[j] += 1
            end
        end
    end
    # t2-style: each penalized column belongs to at most 1 penalty
    return all(pen_count .<= 1)
end

"""
    _smooth2random_t2(sm) -> SmoothMixedModel

For t2 smooths: each penalty has its own set of penalized columns.
Follows mgcv's `smooth2random.t2.smooth`.
"""
function _smooth2random_t2(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    n_pen = length(sm.S)

    fixed = trues(k)
    diagU = ones(k)
    pen_ind = zeros(Int, k)

    Zs = Matrix{Float64}[]
    n_para = 0

    for (i, Si) in enumerate(sm.S)
        # Find columns penalized by this penalty
        d = diag(Si)
        thresh = eps() * maximum(abs.(d))
        indi = findall(abs.(d) .> thresh)

        pen_ind[indi] .= i
        D_i = d[indi]
        diagU[indi] .= 1.0 ./ sqrt.(D_i)

        # Rescaled random effect matrix
        Z_i = sm.X[:, indi] * Diagonal(diagU[indi])
        push!(Zs, Z_i)
        fixed[indi] .= false
        n_para += length(indi)
    end

    # Fixed effect columns: those not penalized by any penalty
    if any(fixed)
        Xf = sm.X[:, fixed]
    else
        Xf = Matrix{Float64}(undef, size(sm.X, 1), 0)
    end

    rind = collect(1:n_para)
    return SmoothMixedModel(Xf, Zs, nothing, diagU, pen_ind, rind, sm.spec.label, false)
end

"""
    _smooth2random_tensor(sm) -> SmoothMixedModel

For te smooths: sum all (normalized) penalties, eigendecompose,
split into null space (fixed) and range space (random).
Follows mgcv's `smooth2random.tensor.smooth`.
"""
function _smooth2random_tensor(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    n_pen = length(sm.S)

    # Sum penalties (each normalized by its mean absolute value)
    sum_S = zeros(k, k)
    for Si in sm.S
        m_abs = mean(abs.(Si))
        if m_abs > 0
            sum_S .+= Si ./ m_abs
        end
    end

    # Eigendecompose summed penalty
    eig = eigen(Symmetric(sum_S))
    if eig.vectors[1, 1] < 0
        eig = Eigen(eig.values, -eig.vectors)
    end

    null_rank = sm.null_dim
    p_rank = k - null_rank
    if p_rank > k
        p_rank = k
    end

    U = eig.vectors  # k × k

    # Transform model matrix
    X_new = sm.X * U

    # Fixed effect columns (null space)
    if p_rank < k
        Xf = X_new[:, (p_rank + 1):k]
    else
        Xf = Matrix{Float64}(undef, size(sm.X, 1), 0)
    end

    # Project each penalty into the penalized subspace
    # and build random effect matrices
    Zs = Matrix{Float64}[]
    U_pen = U[:, 1:p_rank]  # k × p_rank

    for Si in sm.S
        # Project penalty: S_proj = U_pen' * S * U_pen (p_rank × p_rank)
        S_proj = U_pen' * Si * U_pen
        S_proj = (S_proj + S_proj') / 2  # ensure symmetry

        # Eigendecompose projected penalty for rescaling
        eig_p = eigen(Symmetric(S_proj))
        # Columns with nonzero eigenvalues belong to this penalty's random effect
        pos = eig_p.values .> eps() * maximum(abs.(eig_p.values))
        if any(pos)
            D_inv = 1.0 ./ sqrt.(eig_p.values[pos])
            Z_i = X_new[:, 1:p_rank] * eig_p.vectors[:, pos] * Diagonal(D_inv)
            push!(Zs, Z_i)
        end
    end

    # If no individual random effects could be extracted,
    # fall back to single block with all penalized columns
    if isempty(Zs)
        Zs = [X_new[:, 1:p_rank]]
    end

    # Build pen_ind and rind
    pen_ind = zeros(Int, k)
    pen_ind[1:p_rank] .= 1  # simplified: all penalized columns in first group
    rind = collect(1:p_rank)
    D = ones(k)

    return SmoothMixedModel(Xf, Zs, U, D, pen_ind, rind, sm.spec.label, false)
end

# ============================================================================
# Prediction helper
# ============================================================================

"""
    s2r_predict(smm::SmoothMixedModel, sm::ConstructedSmooth, newdata) -> SmoothMixedModel

Compute the mixed-model design matrices for new data, using the
transformation computed from the training data.
"""
function s2r_predict(smm::SmoothMixedModel, sm::ConstructedSmooth, newdata)
    X_new = predict_matrix(sm, newdata)  # n_new × k

    if smm.fixed
        return SmoothMixedModel(
            X_new, Matrix{Float64}[], nothing, smm.trans_D,
            smm.pen_ind, smm.rind, smm.label, true
        )
    end

    k = size(X_new, 2)

    if smm.trans_U !== nothing
        # Apply same transformation: X * U * diag(D)
        X_trans = X_new * (smm.trans_U * Diagonal(smm.trans_D))

        p_rank = length(smm.rind)
        if p_rank < k
            Xf = X_trans[:, (p_rank + 1):k]
        else
            Xf = Matrix{Float64}(undef, size(X_new, 1), 0)
        end

        # For tensor smooths with multiple Zs, we need the projected matrices
        if length(smm.Zs) == 1
            Zs = [X_trans[:, 1:p_rank]]
        else
            # Re-derive from the transformation
            # For multi-penalty, we need the sub-decompositions
            # Fall back to single block for now
            Zs = [X_trans[:, 1:p_rank]]
        end
    else
        # t2-style: use pen_ind to split columns
        Xf_cols = findall(smm.pen_ind .== 0)
        if !isempty(Xf_cols)
            Xf = X_new[:, Xf_cols] * Diagonal(smm.trans_D[Xf_cols])
        else
            Xf = Matrix{Float64}(undef, size(X_new, 1), 0)
        end

        Zs = Matrix{Float64}[]
        for i in 1:length(smm.Zs)
            cols_i = findall(smm.pen_ind .== i)
            Z_i = X_new[:, cols_i] * Diagonal(smm.trans_D[cols_i])
            push!(Zs, Z_i)
        end
    end

    return SmoothMixedModel(
        Xf, Zs, smm.trans_U, smm.trans_D,
        smm.pen_ind, smm.rind, smm.label, false
    )
end
