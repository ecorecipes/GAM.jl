# Neighbourhood Cross Validation (NCV) for smoothness selection.
#
# Port of mgcv's NCV (Wood, "On neighbourhood cross validation"), whose C
# implementation is `src/ncv.c` (`Rncv`, mgcv master) with the score assembled
# in `R/gam.fit3.r:713`.
#
# WHY NCV EXISTS. GCV, AIC/UBRE and REML all assume the observations are
# independent given the model. When they are not — autocorrelated time series,
# spatially correlated data — a left-out point is partly predictable from its
# immediate neighbours, so ordinary leave-one-out CV rewards a fit that
# interpolates local noise and the selected smooth is far too wiggly. NCV fixes
# this by leaving out a NEIGHBOURHOOD around each point rather than the point
# alone: if the neighbourhood is wide enough to break the local correlation,
# the criterion can no longer be gamed by fitting it.
#
# THE ALGORITHM (ncv.c:431-439). Refitting once per neighbourhood would cost
# O(n p^3). Instead a SINGLE NEWTON STEP is taken from the full-data fit, using
# the gradient and Hessian implied by omitting the neighbourhood. At the full
# fit the penalized score is zero, so on dropping the set δ the gradient at β̂
# is exactly minus the dropped points' contribution:
#
#     g_δ = Σ_{k∈δ} w1_k x_k,        H_δ = X'WX + S − Σ_{k∈δ} w2_k x_k x_k'
#     β^{-δ} = β̂ − H_δ⁻¹ g_δ                         (ncv.c:585-587)
#
# where, in ncv.c's naming, `w1_i = dℓ_i/dη_i` and `w2_i = −d²ℓ_i/dη_i²` (the
# IRLS weight). Note mgcv's *R* code passes these as `ww` and `w1`
# respectively, i.e. the R and C names are offset by one — see the `.Call` at
# `gam.fit3.r:691` against the signature at `ncv.c:410`.
#
# Both g and H scale with 1/φ, so the step — and therefore the criterion — is
# invariant to the scale parameter, which is why the unscaled IRLS quantities
# can be used directly.
#
# THE SCORE (gam.fit3.r:713) is the deviance of the held-out points evaluated
# at their cross-validated means:
#
#     NCV = Σ_i D(y_i, g⁻¹(η^cv_i))
#
# which reduces to ordinary leave-one-out CV when every neighbourhood is the
# single point itself.
#
# WHERE THIS PORT DIVERGES, stated plainly. mgcv performs a rank-one Cholesky
# update/downdate per dropped point (O(p²) each) and computes analytic
# derivatives of the score with respect to the log smoothing parameters, which
# lets it drive a Newton optimizer. This port computes the same criterion by
# the Woodbury identity against a factorization of the full Hessian — the same
# O(p²) per neighbourhood and the same answer — but supplies NO analytic
# derivatives, so smoothing parameters are selected by the derivative-free
# optimizer already used for `:GCV`/`:UBRE`. That is slower in iterations, not
# different in what it converges to. The non-positive-definite fallbacks
# (`minres`/`woodbury`, ncv.c:32-90) become a dense symmetric solve here, with
# the same detection and a warning.

"""
    NeighbourhoodStructure

Which points are dropped, and which predicted, for each fold of a
[`ncv_score`](@ref) evaluation.

The encoding matches mgcv's `nei` list (see `?mgcv::gam` and `ncv.c:423-424`)
translated to 1-based indexing: for fold `i`,

- `k[(m[i-1]+1):m[i]]` are the observations **dropped** from the fit
  (mgcv's `nei\$a`, with `nei\$ma` the cumulative ends), and
- `ind[(mi[i-1]+1):mi[i]]` are the observations whose linear predictors are
  **predicted** from that reduced fit (mgcv's `nei\$d`/`nei\$md`),

with `m[0] = mi[0] = 0` by convention. A point may be predicted in several
folds. Leave-one-out cross validation is the case where every fold drops and
predicts the single point `i`, which [`loo_neighbourhoods`](@ref) builds.

See also [`ncv_score`](@ref), [`interval_neighbourhoods`](@ref).
"""
struct NeighbourhoodStructure
    k::Vector{Int}
    m::Vector{Int}
    ind::Vector{Int}
    mi::Vector{Int}
end

"""
    loo_neighbourhoods(n) -> NeighbourhoodStructure

Leave-one-out neighbourhoods for `n` observations: fold `i` drops observation
`i` and predicts observation `i`. This is the default for `method = :NCV`, and
makes the criterion ordinary leave-one-out cross validation.
"""
function loo_neighbourhoods(n::Int)
    n >= 1 || throw(ArgumentError("loo_neighbourhoods requires n >= 1, got $n"))
    idx = collect(1:n)
    return NeighbourhoodStructure(idx, copy(idx), copy(idx), copy(idx))
end

"""
    interval_neighbourhoods(n, half_width) -> NeighbourhoodStructure

Neighbourhoods of contiguous indices: fold `i` drops observations
`max(1, i-half_width):min(n, i+half_width)` and predicts observation `i`.

This is the structure to use for serially correlated data ordered by time (or
by position along a transect), and is mgcv's recommended default in that
setting. `half_width = 0` reduces to [`loo_neighbourhoods`](@ref).

The half-width should be wide enough that the dropped block spans the range
over which the residuals are appreciably correlated; leaving out too little
lets the criterion reward interpolating the correlated noise, which is the
failure mode NCV exists to avoid.
"""
function interval_neighbourhoods(n::Int, half_width::Int)
    n >= 1 || throw(ArgumentError("interval_neighbourhoods requires n >= 1, got $n"))
    half_width >= 0 ||
        throw(ArgumentError("half_width must be >= 0, got $half_width"))
    k = Int[]
    m = Vector{Int}(undef, n)
    for i in 1:n
        for j in max(1, i - half_width):min(n, i + half_width)
            push!(k, j)
        end
        m[i] = length(k)
    end
    idx = collect(1:n)
    return NeighbourhoodStructure(k, m, idx, copy(idx))
end

"""
    validate_neighbourhoods(nei, n)

Check a [`NeighbourhoodStructure`](@ref) is internally consistent and indexes
only observations `1:n`. Throws `ArgumentError` describing the first problem
found. Called by the fitting path so a malformed `nei` fails with a usable
message rather than a `BoundsError` deep in the criterion.
"""
function validate_neighbourhoods(nei::NeighbourhoodStructure, n::Int)
    length(nei.m) == length(nei.mi) || throw(ArgumentError(
        "NeighbourhoodStructure: `m` and `mi` must have one entry per fold, " *
        "got $(length(nei.m)) and $(length(nei.mi))"))
    isempty(nei.m) && throw(ArgumentError(
        "NeighbourhoodStructure has no folds"))
    issorted(nei.m) || throw(ArgumentError(
        "NeighbourhoodStructure: `m` must be non-decreasing cumulative ends"))
    issorted(nei.mi) || throw(ArgumentError(
        "NeighbourhoodStructure: `mi` must be non-decreasing cumulative ends"))
    nei.m[end] == length(nei.k) || throw(ArgumentError(
        "NeighbourhoodStructure: `m[end]` ($(nei.m[end])) must equal " *
        "length(k) ($(length(nei.k)))"))
    nei.mi[end] == length(nei.ind) || throw(ArgumentError(
        "NeighbourhoodStructure: `mi[end]` ($(nei.mi[end])) must equal " *
        "length(ind) ($(length(nei.ind)))"))
    for (name, v) in (("k", nei.k), ("ind", nei.ind))
        for j in v
            (1 <= j <= n) || throw(ArgumentError(
                "NeighbourhoodStructure: `$name` contains index $j, outside " *
                "1:$n"))
        end
    end
    return nothing
end

"""
    ncv_eta(X, beta, w1, w2, H, nei) -> Vector{Float64}

Cross-validated linear predictors, one per entry of `nei.ind`.

`entry j` is the linear predictor for observation `nei.ind[j]` from a fit that
omits the observations in that entry's fold. Computed by the single-Newton-step
approximation of `ncv.c:431-439` rather than by refitting.

`w1[i] = dℓ_i/dη_i`, `w2[i] = −d²ℓ_i/dη_i²` (the IRLS weight), and `H` is the
penalized Hessian `X'diag(w2)X + S`. Returns the number of folds whose
downdated Hessian was not positive definite as the second element.
"""
function ncv_eta(X::AbstractMatrix{Float64}, beta::Vector{Float64},
    w1::Vector{Float64}, w2::Vector{Float64},
    H::Matrix{Float64}, nei::NeighbourhoodStructure)

    p = size(X, 2)
    nfold = length(nei.m)
    eta_cv = Vector{Float64}(undef, length(nei.ind))
    npd_fails = 0

    # One factorization of the FULL penalized Hessian, reused by every fold;
    # each fold is then an O(p^2) Woodbury solve against it, matching mgcv's
    # O(p^2)-per-fold Cholesky downdate.
    F = try
        cholesky(Symmetric(H))
    catch
        # A non-PD full Hessian is a problem with the fit, not with a fold.
        return (fill(NaN, length(nei.ind)), nfold)
    end

    g = Vector{Float64}(undef, p)
    d = Vector{Float64}(undef, p)

    kstart = 0
    istart = 0
    @inbounds for i in 1:nfold
        kstop = nei.m[i]
        istop = nei.mi[i]
        nd = kstop - kstart

        if nd == 0
            # No points dropped: the fold's prediction is the full-data one.
            for jj in (istart + 1):istop
                r = nei.ind[jj]
                acc = 0.0
                for c in 1:p
                    acc += X[r, c] * beta[c]
                end
                eta_cv[jj] = acc
            end
            kstart, istart = kstop, istop
            continue
        end

        # g = Σ_{k∈δ} w1_k x_k   (ncv.c:530, accumulating w1ki * X[ki,j])
        fill!(g, 0.0)
        for kk in (kstart + 1):kstop
            r = nei.k[kk]
            a = w1[r]
            for c in 1:p
                g[c] += a * X[r, c]
            end
        end

        # Woodbury for (H − U W U')⁻¹ g with U = X[δ,:]', W = diag(w2[δ]):
        #   (H − U W U')⁻¹ = H⁻¹ + H⁻¹U (W⁻¹ − U'H⁻¹U)⁻¹ U'H⁻¹
        # For nd == 1 this is the classic one-step LOO update
        #   d = w1_k H⁻¹x_k / (1 − h_kk).
        Hg = F \ g
        U = Matrix{Float64}(undef, p, nd)
        for (col, kk) in enumerate((kstart + 1):kstop)
            r = nei.k[kk]
            for c in 1:p
                U[c, col] = X[r, c]
            end
        end
        HU = F \ U                      # p × nd
        Winv_minus = Matrix{Float64}(undef, nd, nd)
        for a in 1:nd, b in 1:nd
            acc = 0.0
            for c in 1:p
                acc += U[c, a] * HU[c, b]
            end
            Winv_minus[a, b] = -acc     # −U'H⁻¹U
        end
        pdef = true
        for a in 1:nd
            r = nei.k[kstart + a]
            wa = w2[r]
            if wa <= 0
                pdef = false
            else
                Winv_minus[a, a] += 1.0 / wa
            end
        end

        # The downdated Hessian is positive definite iff this nd × nd matrix
        # is (ncv.c:541-548 detects the same condition via the Cholesky
        # update failing). For nd == 1 the entry is (1 − h_kk)/w2_k, so this
        # is exactly the h_kk ≥ 1 check.
        local sol
        if pdef
            try
                Fw = cholesky(Symmetric(Winv_minus))
                UtHg = Vector{Float64}(undef, nd)
                for a in 1:nd
                    acc = 0.0
                    for c in 1:p
                        acc += U[c, a] * Hg[c]
                    end
                    UtHg[a] = acc
                end
                sol = Fw \ UtHg
            catch
                pdef = false
            end
        end

        if pdef
            copyto!(d, Hg)
            for a in 1:nd
                s = sol[a]
                for c in 1:p
                    d[c] += HU[c, a] * s
                end
            end
        else
            # Fallback: form the downdated Hessian densely and solve
            # symmetrically. mgcv falls back to minres/Woodbury here
            # (ncv.c:576-583); a dense solve is the same answer at higher
            # cost, and this path is rare.
            npd_fails += 1
            Hd = copy(H)
            for kk in (kstart + 1):kstop
                r = nei.k[kk]
                wr = w2[r]
                for a in 1:p
                    xa = X[r, a]
                    xa == 0.0 && continue
                    for b in 1:p
                        Hd[a, b] -= wr * xa * X[r, b]
                    end
                end
            end
            dd = try
                Symmetric(Hd) \ g
            catch
                fill(NaN, p)
            end
            copyto!(d, dd)
        end

        # eta_cv = x'(β − d)   (ncv.c:585-587)
        for jj in (istart + 1):istop
            r = nei.ind[jj]
            acc = 0.0
            for c in 1:p
                acc += X[r, c] * (beta[c] - d[c])
            end
            eta_cv[jj] = acc
        end

        kstart, istart = kstop, istop
    end

    return (eta_cv, npd_fails)
end

"""
    ncv_score(X, y, beta, eta, family, link, S_total, weights, offset, nei;
              gamma = 1.0) -> (score, npd_fails)

The neighbourhood cross validation criterion: the deviance of the held-out
observations at their cross-validated means (mgcv `R/gam.fit3.r:713`),

```
NCV = γ Σ_j D(y_{ind_j}, g⁻¹(η^cv_j))
```

Lower is better, as for GCV/UBRE. With leave-one-out neighbourhoods this is
ordinary leave-one-out cross validation.

`npd_fails` counts folds whose downdated Hessian was not positive definite and
so took the dense fallback path.
"""
function ncv_score(X::AbstractMatrix{Float64}, y::Vector{Float64},
    beta::Vector{Float64}, eta::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    S_total::Matrix{Float64},
    weights::Vector{Float64}, offset::Vector{Float64},
    nei::NeighbourhoodStructure; gamma::Float64 = 1.0)

    n, p = size(X)

    # IRLS quantities at the current fit. `w2` is the Fisher weight
    # a_i (dμ/dη)² / V(μ) — the same quantity `_pirls_weights!` builds — and
    # `w1 = dℓ/dη = a_i (y−μ)(dμ/dη)/V(μ)`, obtained as w2·(y−μ)/(dμ/dη).
    w1 = Vector{Float64}(undef, n)
    w2 = Vector{Float64}(undef, n)
    mu = similar(eta)
    @inbounds for i in 1:n
        mu[i] = GLM.linkinv(link, eta[i])
        dm = GLM.mueta(link, eta[i])
        if !isfinite(dm) || abs(dm) < eps()
            dm = isfinite(dm) ? (dm < 0 ? -eps() : eps()) :
                 (dm < 0 ? -1 / eps() : 1 / eps())
        end
        vm = _variance_scalar(family, mu[i])
        wi = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
        w2[i] = wi
        w1[i] = wi * (y[i] - mu[i]) / dm
    end

    # Penalized Hessian X'diag(w2)X + S.
    Xw = similar(X, n, p)
    @inbounds for c in 1:p, r in 1:n
        Xw[r, c] = X[r, c] * w2[r]
    end
    H = Matrix(X' * Xw)
    @inbounds for a in 1:p, b in 1:p
        H[a, b] += S_total[a, b]
    end

    eta_cv, npd_fails = ncv_eta(X, beta, w1, w2, H, nei)

    # Deviance of the held-out points at their cross-validated means.
    # `ncv_eta` returns X(β−d), which carries no offset, so add it back per
    # predicted row — `eta` above already includes it, which is why `mu` there
    # needed no adjustment.
    idx = nei.ind
    mu_cv = similar(eta_cv)
    @inbounds for j in eachindex(eta_cv)
        mu_cv[j] = GLM.linkinv(link, eta_cv[j] + offset[idx[j]])
    end
    any(!isfinite, mu_cv) && return (Inf, npd_fails)

    score = gamma * _deviance(family, y[idx], mu_cv, weights[idx])
    return (isfinite(score) ? score : Inf, npd_fails)
end


# ═══════════════════════════════════════════════════════════════════════════
# Analytic derivatives of the NCV criterion
# ═══════════════════════════════════════════════════════════════════════════
#
# Port of the `deriv > 0` branch of mgcv's `Rncv`: `ncv.c:309-320` builds
# dH/dρ, `ncv.c:368-397` does the per-fold gradient. mgcv takes `db` (dβ/dρ)
# and `dw` (dw2/dρ) as INPUTS, computed on the R side in `R/gam.fit3.r`; both
# are reconstructed here, which is the only structural difference.
#
# The chain, for log smoothing parameter ρ_l (λ_l = exp ρ_l):
#
#   dβ/dρ_l    = −H⁻¹ λ_l S_l β            implicit function theorem at the
#                                          penalized MLE, where X'w1 − S_λβ = 0
#   dη_i/dρ_l  = x_i' dβ/dρ_l
#   dw2_i/dρ_l = (dw2_i/dη_i)(dη_i/dρ_l)
#   dg_δ/dρ_l  = −Σ_{k∈δ} w2_k x_k (dη_k/dρ_l)          ncv.c:371-380, using
#                                                        dw1/dη = −w2
#   dH_δ/dρ_l  = λ_l S_l + X'diag(dw2_l)X
#                        − Σ_{k∈δ} (dw2_k/dρ_l) x_k x_k'
#   dd/dρ_l    = H_δ⁻¹ [dg_δ/dρ_l − (dH_δ/dρ_l) d]      ncv.c:381-389
#   dη^cv_j/dρ_l = x_j' (dβ/dρ_l − dd/dρ_l)             ncv.c:391-394
#
# and since NCV = γ Σ_j D(y_j, g⁻¹(η^cv_j + o_j)),
#
#   dNCV/dρ_l  = γ Σ_j (∂D/∂μ_j)(dμ_j/dη)(dη^cv_j/dρ_l),
#   ∂D_i/∂μ_i  = −2 a_i (y_i − μ_i)/V(μ_i).
#
# mgcv solves the two `H_δ` systems by preconditioned CG (`ncv.c:389`); this
# port reuses the Woodbury identity against ONE Cholesky of the full `H`, as
# `ncv_eta` already does for the criterion — the same answer, and it lets the
# fold's factorization be shared between `d` and `dd/dρ`.

"""
    _ncv_apply_sp_component!(out, penalty, log_sp, l, v) -> out

Write `λ_l S_l v` into `out`, with `S_l` the `l`-th sub-penalty embedded in the
`p`-dimensional coefficient space and `λ_l = exp(log_sp[l])`.

Only that sub-penalty's own diagonal block is touched, so the cost is `O(m²)`
in the block size rather than `O(p²)`. The traversal mirrors
`total_penalty!` exactly — same order, same `start`/`offsets` arithmetic — so
component `l` here is the one `log_sp[l]` scales there.
"""
function _ncv_apply_sp_component!(out::Vector{Float64}, penalty::PenaltySetup,
    log_sp::Vector{Float64}, l::Int, v::AbstractVector{Float64})
    fill!(out, 0.0)
    sp_idx = 1
    for block in penalty.blocks
        for (i, Si) in enumerate(block.S)
            if sp_idx == l
                lam = exp(log_sp[l])
                base = block.start + block.offsets[i] - 1
                m = size(Si, 1)
                @inbounds for j in 1:m
                    acc = 0.0
                    for kk in 1:m
                        acc += Si[j, kk] * v[base + kk]
                    end
                    out[base + j] = lam * acc
                end
                return out
            end
            sp_idx += 1
        end
    end
    return out
end

"""
    _ncv_add_sp_component!(M, penalty, log_sp, l) -> M

Add `λ_l S_l` into the `p × p` matrix `M`, matching
[`_ncv_apply_sp_component!`](@ref)'s indexing. Used to assemble `dH/dρ_l`.
"""
function _ncv_add_sp_component!(M::Matrix{Float64}, penalty::PenaltySetup,
    log_sp::Vector{Float64}, l::Int)
    sp_idx = 1
    for block in penalty.blocks
        for (i, Si) in enumerate(block.S)
            if sp_idx == l
                lam = exp(log_sp[l])
                base = block.start + block.offsets[i] - 1
                m = size(Si, 1)
                @inbounds for j in 1:m, kk in 1:m
                    M[base + j, base + kk] += lam * Si[j, kk]
                end
                return M
            end
            sp_idx += 1
        end
    end
    return M
end

"""
    _ncv_irls_derivs(family, link, y, eta, weights) -> (mu, w1, w2, dw2de, clamped)

IRLS quantities and the η-derivative of the working weight, for the NCV
gradient.

`w2 = a (dμ/dη)²/V(μ)` is the Fisher weight [`ncv_score`](@ref) uses, so

```
dw2/dη = a (dμ/dη) [2 (d²μ/dη²) V(μ) − (dμ/dη)² V'(μ)] / V(μ)²
```

`clamped[i]` records that `w2[i]` hit the guard rails imposed by `ncv_score`;
the criterion is locally constant in `η` there, so `dw2de[i]` is set to zero
rather than to the unclamped slope. Reporting a nonzero derivative for a
quantity the criterion is not actually using is how a gradient check passes on
the smooth interior and silently fails at the boundary.
"""
function _ncv_irls_derivs(family::UnivariateDistribution, link::GLM.Link,
    y::Vector{Float64}, eta::Vector{Float64}, weights::Vector{Float64})
    n = length(eta)
    mu = Vector{Float64}(undef, n)
    w1 = Vector{Float64}(undef, n)
    w2 = Vector{Float64}(undef, n)
    dw2de = Vector{Float64}(undef, n)
    clamped = falses(n)
    @inbounds for i in 1:n
        mu[i] = GLM.linkinv(link, eta[i])
        dm = GLM.mueta(link, eta[i])
        if !isfinite(dm) || abs(dm) < eps()
            dm = isfinite(dm) ? (dm < 0 ? -eps() : eps()) :
                 (dm < 0 ? -1 / eps() : 1 / eps())
        end
        vm = max(_variance_scalar(family, mu[i]), eps())
        raw = weights[i] * dm * dm / vm
        wi = clamp(raw, eps(), 1e10)
        w2[i] = wi
        w1[i] = wi * (y[i] - mu[i]) / dm
        if raw != wi
            clamped[i] = true
            dw2de[i] = 0.0
        else
            d2m = _d2mu_deta2(link, mu[i], eta[i])
            vp = _dvariance_scalar(family, mu[i])
            dw2de[i] = weights[i] * dm * (2.0 * d2m * vm - dm * dm * vp) / (vm * vm)
            isfinite(dw2de[i]) || (dw2de[i] = 0.0)
        end
    end
    return (mu, w1, w2, dw2de, clamped)
end

"""
    ncv_score_grad(X, y, beta, eta, family, link, penalty, log_sp, S_total,
                   weights, offset, nei; gamma = 1.0)
        -> (score, grad, npd_fails)

The NCV criterion **and its analytic gradient** with respect to the log
smoothing parameters — mgcv's `Rncv` with `deriv > 0`.

`score` matches [`ncv_score`](@ref) exactly; `grad[l] = dNCV/dρ_l` where
`ρ_l = log_sp[l]`. `npd_fails` counts folds whose downdated Hessian was not
positive definite and so took the dense fallback.

The gradient assumes `beta` is the penalized MLE at `log_sp` — it is derived
from the stationarity condition `X'w1 = S_λ β` — so it is only valid at a
converged P-IRLS fit, which is where the outer optimizer evaluates it.
"""
function ncv_score_grad(X::AbstractMatrix{Float64}, y::Vector{Float64},
    beta::Vector{Float64}, eta::Vector{Float64},
    family::UnivariateDistribution, link::GLM.Link,
    penalty::PenaltySetup, log_sp::Vector{Float64},
    S_total::Matrix{Float64},
    weights::Vector{Float64}, offset::Vector{Float64},
    nei::NeighbourhoodStructure; gamma::Float64 = 1.0)

    n, p = size(X)
    nsp = length(log_sp)

    mu, w1, w2, dw2de, _ = _ncv_irls_derivs(family, link, y, eta, weights)

    # Penalized Hessian H = X'diag(w2)X + S_λ, factorized once.
    Xw = similar(X, n, p)
    @inbounds for c in 1:p, r in 1:n
        Xw[r, c] = X[r, c] * w2[r]
    end
    H = Matrix(X' * Xw)
    @inbounds for a in 1:p, b in 1:p
        H[a, b] += S_total[a, b]
    end
    F = try
        cholesky(Symmetric(H))
    catch
        return (Inf, fill(NaN, nsp), length(nei.m))
    end

    # dβ/dρ_l = −H⁻¹ λ_l S_l β, and dw2_i/dρ_l = (dw2_i/dη_i)(x_i' dβ/dρ_l).
    db = Matrix{Float64}(undef, p, nsp)
    dw2 = Matrix{Float64}(undef, n, nsp)
    tmp = Vector{Float64}(undef, p)
    for l in 1:nsp
        _ncv_apply_sp_component!(tmp, penalty, log_sp, l, beta)
        col = F \ tmp
        @inbounds for c in 1:p
            db[c, l] = -col[c]
        end
        @inbounds for i in 1:n
            acc = 0.0
            for c in 1:p
                acc += X[i, c] * db[c, l]
            end
            dw2[i, l] = dw2de[i] * acc
        end
    end

    # dH/dρ_l = X'diag(dw2_l)X + λ_l S_l   (ncv.c:311-320)
    dH = Vector{Matrix{Float64}}(undef, nsp)
    for l in 1:nsp
        @inbounds for c in 1:p, r in 1:n
            Xw[r, c] = X[r, c] * dw2[r, l]
        end
        M = Matrix(X' * Xw)
        _ncv_add_sp_component!(M, penalty, log_sp, l)
        dH[l] = M
    end

    nfold = length(nei.m)
    eta_cv = Vector{Float64}(undef, length(nei.ind))
    deta = Matrix{Float64}(undef, length(nei.ind), nsp)
    npd_fails = 0

    g = Vector{Float64}(undef, p)
    d = Vector{Float64}(undef, p)
    dg = Vector{Float64}(undef, p)

    kstart = 0
    istart = 0
    @inbounds for i in 1:nfold
        kstop = nei.m[i]
        istop = nei.mi[i]
        nd = kstop - kstart

        if nd == 0
            for jj in (istart + 1):istop
                r = nei.ind[jj]
                acc = 0.0
                for c in 1:p
                    acc += X[r, c] * beta[c]
                end
                eta_cv[jj] = acc
                for l in 1:nsp
                    a2 = 0.0
                    for c in 1:p
                        a2 += X[r, c] * db[c, l]
                    end
                    deta[jj, l] = a2
                end
            end
            kstart, istart = kstop, istop
            continue
        end

        # g = Σ_{k∈δ} w1_k x_k
        fill!(g, 0.0)
        for kk in (kstart + 1):kstop
            r = nei.k[kk]
            a = w1[r]
            for c in 1:p
                g[c] += a * X[r, c]
            end
        end

        U = Matrix{Float64}(undef, p, nd)
        for (col, kk) in enumerate((kstart + 1):kstop)
            r = nei.k[kk]
            for c in 1:p
                U[c, col] = X[r, c]
            end
        end
        HU = F \ U
        Winv_minus = Matrix{Float64}(undef, nd, nd)
        for a in 1:nd, b in 1:nd
            acc = 0.0
            for c in 1:p
                acc += U[c, a] * HU[c, b]
            end
            Winv_minus[a, b] = -acc
        end
        pdef = true
        for a in 1:nd
            r = nei.k[kstart + a]
            wa = w2[r]
            if wa <= 0
                pdef = false
            else
                Winv_minus[a, a] += 1.0 / wa
            end
        end
        Fw = nothing
        if pdef
            Fw = try
                cholesky(Symmetric(Winv_minus))
            catch
                pdef = false
                nothing
            end
        end

        # Dense downdated Hessian, only when Woodbury is unusable.
        Hd_dense = nothing
        if !pdef
            npd_fails += 1
            Hd_dense = copy(H)
            for kk in (kstart + 1):kstop
                r = nei.k[kk]
                wr = w2[r]
                for a in 1:p
                    xa = X[r, a]
                    xa == 0.0 && continue
                    for b in 1:p
                        Hd_dense[a, b] -= wr * xa * X[r, b]
                    end
                end
            end
        end

        # Solve H_δ x = b, sharing this fold's factorization between the
        # criterion step `d` and each gradient step `d1`.
        function solve_Hd(b::Vector{Float64})
            if pdef
                Hb = F \ b
                UtHb = Vector{Float64}(undef, nd)
                for a in 1:nd
                    acc = 0.0
                    for c in 1:p
                        acc += U[c, a] * Hb[c]
                    end
                    UtHb[a] = acc
                end
                sol = Fw \ UtHb
                out = copy(Hb)
                for a in 1:nd
                    s = sol[a]
                    for c in 1:p
                        out[c] += HU[c, a] * s
                    end
                end
                return out
            else
                return try
                    Symmetric(Hd_dense) \ b
                catch
                    fill(NaN, p)
                end
            end
        end

        copyto!(d, solve_Hd(g))

        for jj in (istart + 1):istop
            r = nei.ind[jj]
            acc = 0.0
            for c in 1:p
                acc += X[r, c] * (beta[c] - d[c])
            end
            eta_cv[jj] = acc
        end

        # Per-smoothing-parameter gradient (ncv.c:368-397).
        for l in 1:nsp
            # dg = −Σ_{k∈δ} w2_k x_k (x_k' dβ/dρ_l)  +  Σ_δ (dw2_k/dρ_l) x_k x_k' d
            fill!(dg, 0.0)
            for kk in (kstart + 1):kstop
                r = nei.k[kk]
                dek = 0.0
                xd = 0.0
                for c in 1:p
                    dek += X[r, c] * db[c, l]
                    xd += X[r, c] * d[c]
                end
                coef = -w2[r] * dek + dw2[r, l] * xd
                for c in 1:p
                    dg[c] += coef * X[r, c]
                end
            end
            # − dH_l d
            Ml = dH[l]
            for a in 1:p
                acc = 0.0
                for b in 1:p
                    acc += Ml[a, b] * d[b]
                end
                dg[a] -= acc
            end
            d1 = solve_Hd(dg)
            for jj in (istart + 1):istop
                r = nei.ind[jj]
                acc = 0.0
                for c in 1:p
                    acc += X[r, c] * (db[c, l] - d1[c])
                end
                deta[jj, l] = acc
            end
        end

        kstart, istart = kstop, istop
    end

    # Score, and the chain rule onto it.
    idx = nei.ind
    mu_cv = similar(eta_cv)
    @inbounds for j in eachindex(eta_cv)
        mu_cv[j] = GLM.linkinv(link, eta_cv[j] + offset[idx[j]])
    end
    any(!isfinite, mu_cv) && return (Inf, fill(NaN, nsp), npd_fails)

    score = gamma * _deviance(family, y[idx], mu_cv, weights[idx])
    isfinite(score) || return (Inf, fill(NaN, nsp), npd_fails)

    grad = zeros(nsp)
    @inbounds for j in eachindex(eta_cv)
        r = idx[j]
        ecv = eta_cv[j] + offset[r]
        dm = GLM.mueta(link, ecv)
        vm = max(_variance_scalar(family, mu_cv[j]), eps())
        # dD_j/dμ = −2 a (y − μ)/V(μ)
        dDdmu = -2.0 * weights[r] * (y[r] - mu_cv[j]) / vm
        fac = gamma * dDdmu * dm
        for l in 1:nsp
            grad[l] += fac * deta[j, l]
        end
    end
    all(isfinite, grad) || return (score, fill(NaN, nsp), npd_fails)
    return (score, grad, npd_fails)
end
