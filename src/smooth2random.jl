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
Following mgcv: t2 smooths (diagonal, non-overlapping penalties) get one
INDEPENDENT random-effect block per penalty, each with its own variance
component — this is what makes t2 usable with lme4/gamm4. te/ti smooths have
overlapping penalties that cannot be written as independent i.i.d. blocks
(mgcv's `gamm` needs the `pdTens` class, and `mgcv:::smooth2random` refuses
te outright with "te smooths not useable with gamm4: use t2 instead"), so
they are decomposed into a single structured block (see
[`_smooth2random_tensor`](@ref)).

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
    # Reverse to DESCENDING order (largest eigenvalues first, matching R convention)
    idx = k:-1:1
    vals = eig.values[idx]
    vecs = eig.vectors[:, idx]
    # Ensure deterministic sign (same as mgcv's hack)
    if vecs[1, 1] < 0
        vecs = -vecs
    end

    null_rank = sm.null_dim
    p_rank = sm.rank
    if p_rank > k
        p_rank = k
    end

    U = vecs  # k × k orthogonal matrix

    # Build rescaling vector D:
    # - penalized columns (1:p_rank): 1/sqrt(eigenvalue) → identity covariance
    # - null space columns (p_rank+1:k): 1 (no change)
    D = Vector{Float64}(undef, k)
    for j in 1:p_rank
        D[j] = 1.0 / sqrt(max(vals[j], eps()))
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

"""Check if penalties are diagonal with non-overlapping supports (t2 pattern)."""
function _is_t2_style(sm::ConstructedSmooth)
    k = size(sm.X, 2)
    # For each column, count how many penalties have nonzero diagonal entry
    pen_count = zeros(Int, k)
    for S in sm.S
        dmax = max(maximum(abs.(diag(S))), eps())
        # The diag-rescaling in _smooth2random_t2 is only valid for genuinely
        # DIAGONAL penalties — diagonal support alone is not enough
        off_max = 0.0
        for j in 1:k, i in 1:k
            i == j && continue
            off_max = max(off_max, abs(S[i, j]))
        end
        off_max > 1e-10 * dmax && return false
        for j in 1:k
            if abs(S[j, j]) > eps() * dmax
                pen_count[j] += 1
            end
        end
    end
    # t2-style: each penalized column belongs to at most 1 penalty
    return all(pen_count .<= 1)
end

"""
    _smooth2random_t2(sm) -> SmoothMixedModel

For t2 smooths, whose penalties are diagonal with non-overlapping support:
each penalty becomes its OWN independent random-effect block with its own
variance component, exactly as in mgcv's `smooth2random.t2.smooth` (this is
what makes t2 usable with lme4/gamm4).

Columns are reordered as `[block 1 | block 2 | ... | unpenalized]` and the
penalized ones rescaled by `1/sqrt(diagonal)` so each block carries an
identity covariance. `trans_U` is the corresponding permutation, so the
reassembly `β_original = trans_U * (trans_D .* [b_1; …; b_m; β_f])` holds
uniformly with the other paths.
"""
function _smooth2random_t2(sm::ConstructedSmooth)
    k = size(sm.X, 2)

    order = Int[]
    scales = Float64[]
    pen_ind = Int[]
    assigned = falses(k)

    for (i, Si) in enumerate(sm.S)
        d = diag(Si)
        thresh = eps() * max(maximum(abs, d), 1.0)
        indi = findall(j -> abs(d[j]) > thresh, 1:k)
        for j in indi
            push!(order, j)
            push!(scales, 1.0 / sqrt(abs(d[j])))
            push!(pen_ind, i)
            assigned[j] = true
        end
    end

    n_para = length(order)
    fixed_cols = findall(!, assigned)
    for j in fixed_cols
        push!(order, j)
        push!(scales, 1.0)
        push!(pen_ind, 0)
    end

    U = Matrix{Float64}(I, k, k)[:, order]
    X_trans = sm.X[:, order] * Diagonal(scales)

    Zs = Matrix{Float64}[]
    for i in 1:length(sm.S)
        cols = findall(==(i), pen_ind)
        isempty(cols) || push!(Zs, X_trans[:, cols])
    end

    Xf = isempty(fixed_cols) ? Matrix{Float64}(undef, size(sm.X, 1), 0) :
         X_trans[:, (n_para + 1):end]

    rind = collect(1:n_para)
    return SmoothMixedModel(Xf, Zs, U, scales, pen_ind, rind, sm.spec.label, false)
end

"""
    _smooth2random_tensor(sm) -> SmoothMixedModel

For te-style smooths with overlapping penalties: sum all penalties (each
normalized by its mean absolute value, i.e. equal initial weights) and
decompose the SUMMED penalty exactly like a single-penalty smooth — the
range space becomes ONE random-effect block with identity covariance, the
null space becomes fixed effects.

Note: a te smooth has one smoothing parameter per margin, but overlapping
penalties cannot be represented as independent i.i.d. random-effect blocks.
mgcv handles this with the special `pdTens` class in `gamm`, and
`mgcv:::smooth2random` refuses te for the gamm4-style (independent-block)
form altogether — "te smooths not useable with gamm4: use t2 instead". Here
the per-margin smoothing is approximated by a SINGLE variance component
(isotropic smoothing of the summed penalty); use `t2()` when independent
blocks are required. The
transform metadata (trans_U, trans_D) exactly reassembles
`β_original = U * (D .* [b; β_f])`, and `s2r_predict` reproduces the
training decomposition at new data.
"""
function _smooth2random_tensor(sm::ConstructedSmooth)
    k = size(sm.X, 2)

    # Sum penalties (each normalized by its mean absolute value — equal weights)
    sum_S = zeros(k, k)
    for Si in sm.S
        m_abs = mean(abs.(Si))
        if m_abs > 0
            sum_S .+= Si ./ m_abs
        end
    end

    # Eigendecompose summed penalty
    eig = eigen(Symmetric(sum_S))
    # Reverse to DESCENDING order (largest eigenvalues first, matching R convention)
    idx = k:-1:1
    vals = eig.values[idx]
    vecs = eig.vectors[:, idx]
    if vecs[1, 1] < 0
        vecs = -vecs
    end

    null_rank = sm.null_dim
    p_rank = clamp(k - null_rank, 0, k)

    U = vecs  # k × k

    # Rescale range-space columns to identity covariance (as in the
    # single-penalty path)
    D = Vector{Float64}(undef, k)
    for j in 1:p_rank
        D[j] = 1.0 / sqrt(max(vals[j], eps()))
    end
    for j in (p_rank + 1):k
        D[j] = 1.0
    end

    UD = U * Diagonal(D)
    X_new = sm.X * UD

    # Fixed effect columns (null space)
    if p_rank < k
        Xf = X_new[:, (p_rank + 1):k]
    else
        Xf = Matrix{Float64}(undef, size(sm.X, 1), 0)
    end

    # ONE random-effect block spanning the whole range space
    Zs = [X_new[:, 1:p_rank]]

    pen_ind = zeros(Int, k)
    pen_ind[1:p_rank] .= 1
    rind = collect(1:p_rank)

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

    # Apply the stored transform, then split by pen_ind (which indexes the
    # TRANSFORMED columns). This reproduces the training decomposition for
    # every path: single-penalty and te (one block, pen_ind = 1 on 1:p_rank)
    # and t2 (one block per penalty).
    X_trans = smm.trans_U !== nothing ?
              X_new * (smm.trans_U * Diagonal(smm.trans_D)) :
              X_new * Diagonal(smm.trans_D)

    fixed_cols = findall(==(0), smm.pen_ind)
    Xf = isempty(fixed_cols) ? Matrix{Float64}(undef, size(X_new, 1), 0) :
         X_trans[:, fixed_cols]

    Zs = Matrix{Float64}[]
    for i in 1:length(smm.Zs)
        cols_i = findall(==(i), smm.pen_ind)
        push!(Zs, X_trans[:, cols_i])
    end

    return SmoothMixedModel(
        Xf, Zs, smm.trans_U, smm.trans_D,
        smm.pen_ind, smm.rind, smm.label, false
    )
end
