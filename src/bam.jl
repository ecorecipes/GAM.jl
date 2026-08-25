# bam() — Big Additive Models for large datasets
#
# Inspired by mgcv's bam() function. Fits the same penalized model as gam()
# but accumulates the normal equations X'WX and X'Wz in row chunks, so peak
# memory beyond the design matrix is bounded by the chunk size.
#
# Key differences from gam():
# 1. X'WX accumulated in chunks (never forms the full n×p weighted product)
# 2. Same EFS outer iteration for smoothing parameter estimation (REML/ML)
#
# `bam(...; discrete=true)` mirrors mgcv's `bam(discrete=TRUE)` for 1-D
# smooths: the basis is evaluated once per distinct covariate value and
# combined through an index vector, so `X'WX` costs O(n) rather than O(n*p^2)
# in the row dimension. Tensor, `by=`, random-effect and factor smooths stay
# dense (M2/M3). The dense design matrix is still materialised first, so this
# is currently a speed win and not yet a memory one — see `bam_design.jl`.
# Note also that solving the normal equations by Cholesky squares the
# condition number relative to mgcv's QR updating; poorly scaled bases are
# less stable here than in mgcv.
#
# Reference: Wood, Goude & Shaw (2015) JASA 110(512):1321-1331

# ============================================================================
# Discretization
# ============================================================================

"""
    DiscretizedData

Compressed representation of covariates after discretization.

# Fields
- `unique_values`: Dict mapping variable name → sorted unique discretized values
- `indices`: Dict mapping variable name → index vector (length n) into unique_values
- `n`: original number of observations
"""
struct DiscretizedData
    unique_values::Dict{Symbol, Vector{Float64}}
    indices::Dict{Symbol, Vector{Int}}
    n::Int
end

"""
    discretize_covariates(data, vars; max_unique=1000) -> DiscretizedData

Discretize continuous covariates onto `max_unique` bins, returning a
`DiscretizedData` struct with unique values and index mappings.

BREAKING (0.3): this now uses mgcv's binning rule rather than quantile bins.
A variable with at most `max_unique` distinct values keeps them exactly;
otherwise values are rounded onto an equally spaced `max_unique`-point grid
spanning the observed range, as `compress.df` does (`R/bam.r:153-159`). The
previous quantile-midpoint scheme placed bins where the data was dense, which
is not what `bam(...; discrete=true)` fits against, so the utility and the
fitter now share one implementation (`_bin_covariate`) and agree by
construction.

Values come back sorted; the fitter applies mgcv's permutation internally,
which is numerically inert there — see [`_bin_covariate`](@ref).
"""
function discretize_covariates(data, vars::Vector{Symbol}; max_unique::Int = 1000)
    n = length(Tables.getcolumn(data, first(vars)))
    unique_vals = Dict{Symbol, Vector{Float64}}()
    idx_map = Dict{Symbol, Vector{Int}}()

    for v in vars
        xu, k, _ = _bin_covariate(Tables.getcolumn(data, v), max_unique;
            shuffle = false)
        unique_vals[v] = xu
        idx_map[v] = Int.(k)
    end

    return DiscretizedData(unique_vals, idx_map, n)
end

# ============================================================================
# Chunk-wise accumulation
# ============================================================================

"""
    BamControl

Control parameters specific to bam() fitting.

# Fields
- `chunk_size`: number of observations per accumulation chunk
"""
struct BamControl
    chunk_size::Int
end

"""
    bam_control(; chunk_size=10000)

Construct a [`BamControl`](@ref) with the given parameters.

The former `discrete`, `max_unique`, and `nthreads` keywords are deprecated
and have no effect here. Discretization is now a keyword on [`bam`](@ref)
itself (`bam(...; discrete=true)`), matching mgcv; threading the accumulation
is still not implemented. Passing them warns.
"""
function bam_control(;
    chunk_size::Int = 10000,
    discrete::Union{Bool, Nothing} = nothing,
    max_unique::Union{Int, Nothing} = nothing,
    nthreads::Union{Int, Nothing} = nothing,
)
    if discrete !== nothing || max_unique !== nothing || nthreads !== nothing
        @warn "bam_control: `discrete`, `max_unique`, and `nthreads` are deprecated " *
              "here and ignored. Discretization moved to `bam(...; discrete=true)`, " *
              "matching mgcv; threading the accumulation is still not implemented."
    end
    return BamControl(chunk_size)
end

"""
    _accumulate_XtWX_XtWz_chunked!(XtWX, XtWz, X, w, z, chunk_size)

Accumulate X'WX and X'Wz in chunks of `chunk_size` rows to limit memory usage.
Uses BLAS syrk for each chunk.
"""
function _accumulate_XtWX_XtWz_chunked!(
    XtWX::Matrix{Float64}, XtWz::Vector{Float64},
    X::Matrix{Float64}, w::Vector{Float64}, z::Vector{Float64},
    chunk_size::Int)

    n, p = size(X)
    fill!(XtWX, 0.0)
    fill!(XtWz, 0.0)

    Xw_chunk = zeros(min(chunk_size, n), p)
    wz_chunk = zeros(min(chunk_size, n))

    for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        nc = stop - start + 1

        # Scale rows by sqrt(w) for this chunk
        Xw_view = view(Xw_chunk, 1:nc, :)
        wz_view = view(wz_chunk, 1:nc)

        @inbounds for i in 1:nc
            row = start + i - 1
            sw = sqrt(w[row])
            wz_view[i] = w[row] * z[row]
            for j in 1:p
                Xw_view[i, j] = X[row, j] * sw
            end
        end

        # Accumulate XtWX += Xw_chunk' * Xw_chunk
        BLAS.syrk!('U', 'T', 1.0, Xw_view, 1.0, XtWX)

        # Accumulate XtWz += X_chunk' * wz_chunk
        BLAS.gemv!('T', 1.0, view(X, start:stop, :), wz_view, 1.0, XtWz)
    end

    # Fill lower triangle
    @inbounds for j in 1:p
        for k in (j + 1):p
            XtWX[k, j] = XtWX[j, k]
        end
    end
end

"""
    _accumulate_XtWX_chunked!(XtWX, X, w, chunk_size)

Accumulate X'WX only (no rhs) in chunks.
"""
function _accumulate_XtWX_chunked!(
    XtWX::Matrix{Float64},
    X::Matrix{Float64}, w::Vector{Float64},
    chunk_size::Int)

    n, p = size(X)
    fill!(XtWX, 0.0)

    Xw_chunk = zeros(min(chunk_size, n), p)

    for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        nc = stop - start + 1

        Xw_view = view(Xw_chunk, 1:nc, :)
        @inbounds for i in 1:nc
            row = start + i - 1
            sw = sqrt(w[row])
            for j in 1:p
                Xw_view[i, j] = X[row, j] * sw
            end
        end

        BLAS.syrk!('U', 'T', 1.0, Xw_view, 1.0, XtWX)
    end

    @inbounds for j in 1:p
        for k in (j + 1):p
            XtWX[k, j] = XtWX[j, k]
        end
    end
end

# ============================================================================
# BAM P-IRLS with chunked accumulation
# ============================================================================

"""
    _bam_mustart(family, yi, wi) -> Float64

Family-appropriate initial value for μ (following mgcv's `mustart`):
- Binomial/Bernoulli: `(w*y + 0.5) / (w + 1)` (kept inside (0,1))
- Poisson: `y + 0.1`
- Gamma / InverseGaussian: `max(y, small positive)`
- Gaussian (and default): `y`
"""
function _bam_mustart(family::UnivariateDistribution, yi::Real, wi::Real)
    if family isa BinomialLike
        return clamp((wi * yi + 0.5) / (wi + 1.0), 1e-4, 1.0 - 1e-4)
    elseif family isa Poisson
        return yi + 0.1
    elseif family isa Gamma || family isa InverseGaussian
        return max(yi, 1e-3)
    else
        return float(yi)
    end
end

"""
    pirls_bam(X, y, S_total, family, link; weights, offset, start, control, chunk_size)

Penalized IRLS using chunk-wise X'WX accumulation for large datasets.
Functionally identical to `pirls()` but memory-efficient for large n.
"""
function pirls_bam(X::Matrix{Float64}, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution, link::GLM.Link; kwargs...)
    return pirls_bam(bam_design(X), y, S_total, family, link; kwargs...)
end

function pirls_bam(D::BamDesign, y::Vector{Float64},
    S_total::Matrix{Float64},
    family::UnivariateDistribution, link::GLM.Link;
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::GamControl = gam_control(),
    chunk_size::Int = 10000,
    compute_hat_diag::Bool = true,
    XtWX_out::Union{Matrix{Float64}, Nothing} = nothing)

    n, p = nrows(D), ncols(D)

    # Pre-allocate working buffers
    beta = zeros(p)
    beta_new = zeros(p)
    eta = zeros(n)
    eta_new = zeros(n)
    mu = zeros(n)
    mu_new = zeros(n)
    w = zeros(n)
    z = zeros(n)
    XtWz = zeros(p)
    A = zeros(p, p)
    XtWX = zeros(p, p)

    # Initialize
    if start !== nothing
        copyto!(beta, start)
        mul_eta!(eta, D, beta)
        eta .+= offset
    else
        # Family-appropriate initial μ (mgcv mustart)
        @inbounds for i in 1:n
            eta[i] = GLM.linkfun(link, _bam_mustart(family, y[i], weights[i]))
        end
        # Start from the constant fit on the link scale; locate the intercept
        # column rather than assuming it is column 1. `intercept_col` returns 0
        # when there is none, and memoises the O(n·p) scan the dense design
        # needs, so the outer loop's inner solves do not repeat it.
        icpt = intercept_col(D)
        if icpt != 0
            beta[icpt] = mean(eta)
            mul_eta!(eta, D, beta)
            eta .+= offset
        end
        # (no intercept column: keep the mustart η as the starting point)
    end

    @inbounds for i in 1:n
        mu[i] = GLM.linkinv(link, eta[i])
    end
    dev_old = _deviance(family, y, mu, weights)

    converged = false
    n_iter = 0

    # bam's step-acceptance policy: its own relative tolerance and 25-halving
    # cap, plus the shared family-domain guard.
    step_spec = PirlsStepControl(;
        threshold = _pirls_relative_threshold(control.epsilon),
        max_halvings = 25)
    prev_valid = all(_valid_mu_scalar(family, GLM.linkinv(link, e)) for e in eta)

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response (scalar ops)
        _pirls_working!(w, z, y, mu, eta, offset, weights, family, link)

        # Chunk-wise accumulation of X'WX and X'Wz
        accumulate_XtWX_XtWz!(XtWX, XtWz, D, w, z; chunk_size = chunk_size)

        # Add penalty: A = XtWX + S_total
        @inbounds for j in 1:p, k in 1:p
            A[j, k] = XtWX[j, k] + S_total[j, k]
        end

        # Solve via Cholesky (escalating-ridge recovery on indefiniteness,
        # matching the other fitters)
        A_chol = _protected_cholesky!(A)
        ldiv!(beta_new, A_chol, XtWz)

        # η, μ and the deviance for the full step are NOT computed here:
        # `pirls_halve!` evaluates `recompute!(beta_new)` unconditionally before
        # it halves anything (src/pirls.jl), so doing it here too cost an extra
        # O(n·p) `mul!` plus a link and deviance pass on every iteration, for
        # values that were overwritten microseconds later. `dev_new` is only
        # seeded here so the binding lives in this scope rather than being
        # captured as a local of the closure below.
        dev_new = dev_old

        # Step halving on the penalized deviance, using the acceptance policy
        # shared with pirls/scasm/scam. bam's own tolerance (relative, with 25
        # halvings) is preserved; what it gains is the family-domain check, so
        # a chunked fit can no longer walk outside the mean domain the way the
        # hand-rolled loop could.
        pdev_old_iter = dev_old + dot(beta, S_total, beta)
        recompute! = function (b)
            mul_eta!(eta_new, D, b)
            eta_new .+= offset
            ok = true
            @inbounds for i in 1:n
                li = GLM.linkinv(link, eta_new[i])
                ok &= _valid_mu_scalar(family, li)
                mu_new[i] = _clamp_mu_scalar(family, li)
            end
            dev_new = _deviance(family, y, mu_new, weights)
            return (dev_new + dot(b, S_total, b), ok)
        end

        # `pirls_halve!` returns immediately after the `recompute!` call that
        # accepted (or the last one it tried, on failure), so `dev_new` and
        # `mu_new` already describe that iterate — recomputing the deviance
        # here reproduced a value the closure had just stored.
        _, valid_new, step_ok, n_halvings =
            pirls_halve!(beta_new, beta, recompute!, step_spec,
                pdev_old_iter, prev_valid)

        if !step_ok
            # Same policy as pirls/gam: mgcv raises "step failure" here.
            # `beta`/`eta`/`mu` still hold the previous (best) iterate, and
            # `mu_new` holds the rejected one — keep the former and stop
            # rather than copying divergence in below.
            @inbounds for i in 1:n
                mu[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta[i]))
            end
            @warn "bam P-IRLS step failure: penalized deviance could not be " *
                  "reduced after $(step_spec.max_halvings) step halvings; " *
                  "returning last stable iterate" maxlog = 1
            converged = false
            break
        end
        prev_valid = valid_new

        # Convergence check
        crit = abs(dev_new - dev_old) / (abs(dev_new) + 0.1)
        copyto!(beta, beta_new)
        copyto!(eta, eta_new)
        copyto!(mu, mu_new)
        dev_old = dev_new

        # Mirrors the guard in `pirls` (see src/pirls.jl): a step rescued by
        # heavy halving says the search DIRECTION was poor, not that we are at
        # an optimum, so the deviance criterion is not evidence of convergence
        # there. Note `pirls_halve!` returns `max_halvings` when the step fails
        # outright, so this also stops a failed step being read as convergence.
        #
        # Measured as defence in depth here rather than an observed fix: across
        # 60 configurations spanning InverseGaussian+log/identity, Gamma+log,
        # and Bernoulli+cloglog/probit at high dispersion and sp from 0.01 to
        # 500, bam accepted at ZERO halvings every time, so the branch this
        # guards was never reached. bam's chunked accumulation reaches mgcv's
        # answer on the fit that broke `pirls` (deviance 299.63954 vs mgcv's
        # 299.6395385772956) without halving at all.
        if crit < control.epsilon && n_halvings <= 1
            converged = true
            break
        end
    end

    # Final quantities
    @inbounds for i in 1:n
        dm = GLM.mueta(link, eta[i])
        vm = _variance_scalar(family, mu[i])
        w[i] = clamp(weights[i] * dm * dm / max(vm, eps()), eps(), 1e10)
    end

    # Pearson statistic
    pearson = 0.0
    @inbounds for i in 1:n
        vm = _variance_scalar(family, mu[i])
        pearson += weights[i] * (y[i] - mu[i])^2 / max(vm, eps())
    end

    # EDF and hat matrix via chunked accumulation
    accumulate_XtWX!(XtWX, D, w; chunk_size = chunk_size)
    @inbounds for j in 1:p, k in 1:p
        A[j, k] = XtWX[j, k] + S_total[j, k]
    end
    # Hand this XᵀWX back before the Cholesky consumes `A`: it is formed from
    # the same `w` returned in the result, so an outer smoothing-parameter loop
    # can reuse it for its EFS traces instead of repeating the O(n·p²) sweep.
    XtWX_out === nothing || copyto!(XtWX_out, XtWX)
    A_chol = _protected_cholesky!(A)

    # Shared finalization: EDF, the weighted leverage h_i = w_i·x_i'A⁻¹x_i,
    # and R. Chunked internally, so memory stays bounded as it was here.
    edf_vec, hat_diag, R = design_finalize(D, w, XtWX, A_chol;
        chunk_size = chunk_size, compute_hat_diag = compute_hat_diag)

    return PirlsResult(
        beta, mu, eta, w, dev_old, pearson,
        converged, n_iter, R, hat_diag, edf_vec,
    )
end

# ============================================================================
# BAM outer iteration
# ============================================================================

"""
    outer_iteration_bam(X, y, smooths, penalty, family, link;
                        method, weights, offset, control, chunk_size)

Outer iteration for bam() — same EFS updates but using chunked P-IRLS.
"""
function outer_iteration_bam(X::Matrix{Float64}, y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth}, penalty::PenaltySetup,
    family::UnivariateDistribution, link::GLM.Link; kwargs...)
    return outer_iteration_bam(bam_design(X), y, smooths, penalty,
        family, link; kwargs...)
end

function outer_iteration_bam(D::BamDesign, y::Vector{Float64},
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::UnivariateDistribution, link::GLM.Link;
    method::Symbol = :REML,
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    control::GamControl = gam_control(),
    chunk_size::Int = 10000)

    n, p = nrows(D), ncols(D)
    n_sp = length(penalty.sp)

    if n_sp == 0
        S_total = zeros(p, p)
        result = pirls_bam(D, y, S_total, family, link;
            weights = weights, offset = offset, control = control,
            chunk_size = chunk_size)
        return penalty.sp, result
    end

    log_sp = copy(penalty.sp)
    prev_result = nothing

    # For Gaussian identity link, W is constant (= prior weights), so X'WX is
    # constant across iterations. Precompute it once to avoid O(n·p²) per outer step.
    is_gaussian = family isa Normal && link isa IdentityLink
    XtWX_cached = zeros(p, p)
    Xty_cached = zeros(p)
    yWy_cached = 0.0
    # For Gaussian identity the offset is absorbed by fitting to y - offset
    y_adj = is_gaussian ? y .- offset : y
    if is_gaussian
        accumulate_XtWX_XtWz!(XtWX_cached, Xty_cached,
            D, weights, y_adj; chunk_size = chunk_size)
        # y'Wy for O(p²) deviance formula
        @inbounds for i in 1:n
            yWy_cached += weights[i] * y_adj[i]^2
        end
    end

    for outer_iter in 1:(control.outer_maxit)
        S_total = total_penalty(penalty, log_sp, p)

        if is_gaussian
            # Fast Gaussian path: solve (X'X + S) β = X'y directly
            A = XtWX_cached + S_total
            A_chol = _protected_cholesky!(A)
            beta = A_chol \ Xty_cached

            # Deviance via O(p²) formula: ||y-Xβ||² = y'Wy - 2β'X'Wy + β'X'WXβ
            dev = yWy_cached - 2 * dot(beta, Xty_cached) + dot(beta, XtWX_cached * beta)
            pearson = dev

            # Lightweight EDF: only compute edf_total (tr(F)), skip hat_diag
            F = A_chol \ XtWX_cached
            edf_vec = diag(F)
            edf_total = sum(edf_vec)

            # Construct minimal result (hat_diag and eta/mu deferred to final solve)
            R = Matrix(A_chol.U)
            result = PirlsResult(beta, Float64[], Float64[], weights, dev, pearson,
                true, 1, R, Float64[], edf_vec)
        else
            start_coef = prev_result === nothing ? nothing : prev_result.coefficients
            # Two things this inner solve does NOT need to redo:
            #  * `hat_diag` — an O(n·p²) leverage sweep that nothing below
            #    reads off an inner result (only the final fit reports it), so
            #    7-of-8 to 33-of-34 of those passes were discarded outright.
            #  * its final XᵀWX — accumulated from the same working weights the
            #    EFS update needs, so it is handed back rather than swept again.
            XtWX_inner = zeros(p, p)
            result = pirls_bam(D, y, S_total, family, link;
                weights = weights, offset = offset, start = start_coef,
                control = control, chunk_size = chunk_size,
                compute_hat_diag = false, XtWX_out = XtWX_inner)
            edf_total = sum(result.edf_vec)
        end

        if !result.converged && control.trace
            @warn "P-IRLS did not converge at outer iteration $outer_iter"
        end

        beta = result.coefficients
        w = result.working_weights

        # Scale estimate (response-relative floor, matching the core loops)
        scale_est = _needs_scale_estimate(family) ?
            max(result.pearson / (n - edf_total), _scale_floor(y)) : 1.0

        # EFS update — reuse Cholesky from inner solve for Gaussian.
        # Only the per-block diagonal blocks of A⁻¹ are needed for the EFS
        # traces, so solve for those columns instead of forming the full
        # p×p inverse (O(p²·k) per block vs O(p³)).
        if is_gaussian
            # A_chol is already available from inner solve (same XtWX + S_total)
            A_fact = A_chol
            XtWX_cur = XtWX_cached
        else
            # `XtWX_inner` came back from the inner P-IRLS solve above,
            # accumulated from exactly this `w` (`result.working_weights`).
            XtWX_cur = XtWX_inner
            A_efs = XtWX_cur + S_total
            A_fact = _protected_cholesky!(A_efs)
        end

        log_sp_new = copy(log_sp)
        sp_idx = 1
        max_change = 0.0

        E_blk = zeros(p, 0)
        for block in penalty.blocks
            idx = block.start:block.stop
            beta_block = beta[idx]
            kb = length(idx)
            E_blk = zeros(p, kb)
            @inbounds for (c, j) in enumerate(idx)
                E_blk[j, c] = 1.0
            end
            Ainv_block = (A_fact \ E_blk)[idx, :]

            # Per-penalty λⱼ·tr(S_λ⁺Sⱼ): equals block.rank only for
            # single-penalty blocks; using block.rank per margin of a tensor
            # smooth systematically oversmooths (see _efs_sp_update)
            nSb = length(block.S)
            ldet_derivs = _block_logdet_derivs(block,
                view(log_sp, sp_idx:(sp_idx + nSb - 1)))

            for (jS, Si) in enumerate(block.S)
                if penalty.fixed[sp_idx]
                    sp_idx += 1
                    continue
                end
                λ = exp(log_sp[sp_idx])

                bSb = dot(beta_block, Si * beta_block)
                # tr(A⁻¹S) = Σᵢⱼ A⁻¹ᵢⱼSᵢⱼ for symmetric S — O(k²), not O(k³)
                tr_AinvS = λ * sum(Ainv_block .* Si)

                numerator = ldet_derivs[jS] - tr_AinvS
                denominator = bSb / scale_est

                if denominator > eps() && numerator > 0
                    λ_new = scale_est * numerator / (bSb + eps())
                    log_sp_new[sp_idx] = log(λ) + 0.5 * (log(max(λ_new, 1e-15)) - log(λ))
                end

                log_sp_new[sp_idx] = clamp(log_sp_new[sp_idx], -15.0, 15.0)
                max_change = max(max_change, abs(log_sp_new[sp_idx] - log_sp[sp_idx]))
                sp_idx += 1
            end
        end

        if control.trace
            println("BAM outer iter $outer_iter: " *
                    "sp=[$(join([@sprintf("%.4f", exp(s)) for s in log_sp], ", "))]" *
                    ", edf=$(round(edf_total; digits=2))")
        end

        # Score-based convergence (as in the core EFS loop): once the
        # conditional criterion stops moving, declare convergence even if the
        # smoothing parameters still wander along a flat ridge — otherwise a
        # flat λ→∞ direction walks to the log-sp clamp.
        if outer_iter > 1 && max_change > control.epsilon
            ls = _log_saturated_likelihood(family, y, weights, scale_est)
            reml_old = _conditional_reml(log_sp, XtWX_cur, beta, result.deviance,
                penalty, scale_est, n, p, edf_total, method,
                control.gamma, ls)
            reml_new = _conditional_reml(log_sp_new, XtWX_cur, beta, result.deviance,
                penalty, scale_est, n, p, edf_total, method,
                control.gamma, ls)
            if abs(reml_new - reml_old) <
               control.epsilon * (abs(reml_old) + 0.1)
                max_change = 0.0
            end
        end

        log_sp .= log_sp_new
        prev_result = result

        if max_change < control.epsilon * 10
            if control.trace
                println("BAM outer iteration converged at iteration $outer_iter")
            end
            break
        end
    end

    # Final solve with converged parameters
    penalty.sp .= log_sp
    S_total = total_penalty(penalty, log_sp, p)

    if is_gaussian
        A = XtWX_cached + S_total
        A_chol = _protected_cholesky!(A)
        beta = A_chol \ Xty_cached
        eta = Vector{Float64}(undef, n)
        mul_eta!(eta, D, beta)
        eta .+= offset
        mu = copy(eta)
        dev = _deviance(family, y, mu, weights)
        # Gaussian/identity: the working weights are the prior weights
        edf_vec, hat_diag, R = design_finalize(D, weights, XtWX_cached, A_chol;
            chunk_size = chunk_size)
        final_result = PirlsResult(beta, mu, eta, weights, dev, dev,
            true, 1, R, hat_diag, edf_vec)
    else
        final_result = pirls_bam(D, y, S_total, family, link;
            weights = weights, offset = offset,
            start = prev_result === nothing ? nothing : prev_result.coefficients,
            control = control, chunk_size = chunk_size)
    end

    return log_sp, final_result
end

# ============================================================================
# Discretized model matrix expansion
# ============================================================================

"""
    expand_discretized_X(X_unique, indices, n) -> Matrix{Float64}

Expand a basis evaluated at unique values back to full n observations
using index mapping. This is the "decompression" step.

# Arguments
- `X_unique`: basis matrix evaluated at unique values (n_unique × p)
- `indices`: index vector mapping observations to unique values (length n)
- `n`: number of observations
"""
function expand_discretized_X(X_unique::Matrix{Float64}, indices::Vector{Int}, n::Int)
    p = size(X_unique, 2)
    X_full = zeros(n, p)
    @inbounds for i in 1:n
        idx = indices[i]
        for j in 1:p
            X_full[i, j] = X_unique[idx, j]
        end
    end
    return X_full
end

# ============================================================================
# Main bam() function
# ============================================================================

function _bam_check_method(method::Symbol)
    method in (:GCV, :UBRE) && throw(ArgumentError(
        "bam() estimates smoothing parameters by EFS on the REML/ML criterion; " *
        "method :$method is not implemented for bam — use gam() for GCV/UBRE"))
    method in (:REML, :ML) ||
        throw(ArgumentError("method must be :REML or :ML, got :$method"))
    return method
end

"""
    bam(formula, data; family=Normal(), link=nothing, method=:REML,
        weights=nothing, offset=nothing, select=false,
        control=gam_control(), bam_ctrl=bam_control())

Fit a Generalized Additive Model to large datasets using chunk-wise
accumulation of the normal equations X'WX and X'Wz.

This is the large-dataset counterpart of [`gam`](@ref). Uses the same
smoothing parameter estimation (EFS, optimizing the REML/ML criterion)
but never forms the full weighted design product, so peak memory beyond
the design matrix is bounded by the chunk size. Note that solving the
normal equations squares the condition number relative to a QR approach;
for badly scaled bases prefer `gam()`.

# Arguments
- `formula`: a formula with smooth terms (via `@formulak` or `@formula`)
- `data`: a table (DataFrame, NamedTuple of vectors, etc.)
- `family`: distribution family (default: `Normal()`)
- `link`: link function (default: canonical link for family)
- `method`: smoothing parameter estimation (`:REML` or `:ML`; the
  criterion-optimizing `:GCV`/`:UBRE` methods of `gam()` are not
  implemented for `bam` and throw an error)
- `weights`: optional observation weights — must be finite and non-negative
  (zero excludes an observation)
- `offset`: optional known additive term on the link scale
- `na_action`: how to treat rows carrying `missing`, `NaN` or `Inf` in the
  response, in a model variable, or in `weights`/`offset`. `:fail` (default)
  errors; `:omit` drops them, as `mgcv` does by default. Dropping is silent,
  so a fit can use fewer rows than the table supplied — recover the surviving
  row numbers with [`na_omit_rows`](@ref) to line results back up with the
  original data
- `select`: if `true`, add a null-space penalty to every smooth
  (Marra & Wood 2011 term selection), as in `gam(...; select=true)`
- `discrete`: if `true`, store each 1-D smooth as its basis at the unique
  covariate values plus an index vector, as `mgcv::bam(..., discrete=TRUE)`
  does, instead of an `n x p` block. Pass an integer to set the grid
  resolution (default 1000). Covariates with at most that many distinct
  values are represented exactly; otherwise they are rounded onto an equally
  spaced grid, which makes the fit an approximation — smoothing parameters
  can move noticeably even where fitted values agree closely. Tensor, `by=`,
  random-effect and factor smooths are unaffected and stay dense
- `control`: GAM fitting control parameters
- `bam_ctrl`: BAM-specific control parameters (chunk size)

# Returns
A [`GamModel`](@ref) object — identical output type to `gam()`.

# Example
```julia
using GAM, DataFrames

# Large dataset
n = 100_000
x = randn(n)
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

# bam is faster than gam for large n
m = bam(@formulak(y ~ s(x, k=20, bs=:cr)), df)
```
"""
function bam(f::FormulaTerm, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    weights::Union{AbstractVector{<:Union{Real, Missing}}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Union{Real, Missing}}, Nothing} = nothing,
    na_action::Symbol = :fail,
    select::Bool = false,
    discrete::Union{Bool, Integer} = false,
    control::GamControl = gam_control(),
    bam_ctrl::BamControl = bam_control())

    _bam_check_method(method)

    if link === nothing
        link = GLM.canonicallink(family)
    end

    resp = f.lhs isa Term ? f.lhs.sym : nothing
    data, _, weights, offset = _na_prepare(data, resp, _model_covariates(f),
        na_action; weights = weights, offset = offset)
    na_action === :fail && _validate_model_columns(data, _model_covariates(f))

    y, X, X_para, smooths, n_parametric = setup_gam(f, data; family = family)
    return _fit_bam(y, X, smooths, n_parametric, f, data, family, link,
        method, weights, offset, select, control, bam_ctrl, discrete)
end

function bam(gf::GamFormula, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    weights::Union{AbstractVector{<:Union{Real, Missing}}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Union{Real, Missing}}, Nothing} = nothing,
    na_action::Symbol = :fail,
    select::Bool = false,
    discrete::Union{Bool, Integer} = false,
    control::GamControl = gam_control(),
    bam_ctrl::BamControl = bam_control())

    _bam_check_method(method)

    if link === nothing
        link = GLM.canonicallink(family)
    end

    data, _, weights, offset = _na_prepare(data, gf.response,
        _model_covariates(gf), na_action; weights = weights, offset = offset)
    _validate_response_in_data(gf.response, data)
    na_action === :fail && _validate_model_columns(data, _model_covariates(gf))

    # Under `discrete`, build each 1-D smooth at the unique covariate values
    # and scatter to `n` rows, rather than evaluating the basis at all `n`.
    # The result is indistinguishable downstream; what it avoids is the
    # construction transient, which for thin-plate at `max_knots = 2000` is an
    # `n × 2000` dense matrix and dominates peak RSS.
    y, X, X_para, smooths, n_parametric = if discrete === false
        setup_gam(gf, data; family = family)
    else
        setup_gam_discrete(gf, data, discrete === true ? 1000 : Int(discrete);
            family = family)
    end
    f = term(gf.response) ~ term(1)
    return _fit_bam(y, X, smooths, n_parametric, f, data, family, link,
        method, weights, offset, select, control, bam_ctrl, discrete)
end

function _fit_bam(y, X, smooths, n_parametric, f, data,
    family, link, method, weights, offset, select, control, bam_ctrl,
    discrete = false)
    # Construct the design once for the whole fit: `DenseDesign` memoises the
    # O(n·p) intercept scan, so the outer loop's inner P-IRLS solves inherit it
    # rather than repeating it. Under `discrete`, 1-D smooths are replaced by
    # their basis at the unique covariate values plus an index vector.
    D = bam_design(X, smooths, data, discrete)
    n, p = nrows(D), ncols(D)

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))

    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch(
        "offset length $(length(off)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric; select = select)

    # Use BAM outer iteration with chunked accumulation
    log_sp, result = outer_iteration_bam(D, y, smooths, penalty, family, link;
        method = method, weights = wts, offset = off, control = control,
        chunk_size = bam_ctrl.chunk_size)

    # Post-processing — use the R factor from pirls result to avoid O(n) passes
    edf_per_smooth = smooth_edf(result.edf_vec, smooths)
    edf_total_val = sum(result.edf_vec)

    # Reconstruct A from R factor: A = R'R, so A_chol.U = R
    # Use Vp = inv(A) = inv(R'R) = inv(R) * inv(R')
    R_upper = UpperTriangular(result.R)
    Vp = inv(R_upper) * inv(R_upper')

    # F = Vp * XtWX = Vp * (A - S_total)
    S_total = total_penalty(penalty, log_sp, p)
    XtWX_from_R = R_upper' * R_upper - S_total
    F = Vp * XtWX_from_R
    # Frequentist covariance: Ve = F·Vp·φ (mgcv; matches gamfit.jl)
    Ve = Symmetric(F * Vp) |> Matrix

    if _needs_scale_estimate(family)
        scale_est = result.pearson / (n - edf_total_val)
        Vp .*= scale_est
        Ve .*= scale_est
    else
        scale_est = 1.0
    end

    null_dev = _null_deviance(family, y, wts)

    # REML/ML score from the R factor, using the same criterion as gam()
    # (reml.jl / mgcv gam.fit3): (Dp/(2φ) - ls) + ½log|A| - ½log|S|₊
    # - (Mp/2)·log(2πφ), with Dp = dev + β'Sβ
    log_det_A = 2.0 * sum(log(abs(R_upper[i, i])) for i in 1:p)
    log_det_S = _log_penalty_det(penalty, log_sp)
    phi = scale_est
    Mp = p - sum(b.rank for b in penalty.blocks; init = 0)
    Dp = result.deviance + dot(result.coefficients, S_total, result.coefficients)
    ls = _log_saturated_likelihood(family, y, wts, phi)
    reml_val = (Dp / (2 * phi) - ls) + 0.5 * log_det_A - 0.5 * log_det_S
    if method == :REML
        reml_val -= 0.5 * Mp * log(2π * phi)
    end

    return GamModel(
        f,
        y, X,
        result.coefficients,
        result.fitted_values,
        result.linear_predictor,
        wts,
        family, link,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total_val,
        scale_est,
        result.deviance,
        null_dev,
        reml_val,
        NaN,
        method,
        Vp, Ve,
        result.hat_diag,
        result.R,
        result.converged,
        0,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )
end
