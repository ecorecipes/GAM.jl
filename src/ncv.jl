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
