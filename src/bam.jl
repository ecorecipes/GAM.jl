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
# Unlike mgcv's bam(discrete=TRUE), covariates are NOT discretized: the full
# dense design matrix is built once. The standalone discretization utilities
# below (`discretize_covariates` etc.) are provided but not used by `bam()`.
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

Discretize continuous covariates by binning into `max_unique` quantile-based bins.
Returns a `DiscretizedData` struct with unique values and index mappings.

Standalone utility: `bam()` does not currently use discretization.
"""
function discretize_covariates(data, vars::Vector{Symbol}; max_unique::Int = 1000)
    n = length(Tables.getcolumn(data, first(vars)))
    unique_vals = Dict{Symbol, Vector{Float64}}()
    idx_map = Dict{Symbol, Vector{Int}}()

    for v in vars
        x = Float64.(Tables.getcolumn(data, v))
        ux = sort(unique(x))

        if length(ux) <= max_unique
            # Already few enough unique values
            unique_vals[v] = ux
            # Map each observation to index in ux
            val_to_idx = Dict(val => i for (i, val) in enumerate(ux))
            idx_map[v] = [val_to_idx[xi] for xi in x]
        else
            # Quantile-based binning
            probs = range(0, 1; length = max_unique + 1)
            breaks = quantile(x, probs)
            # Make breaks unique
            breaks = sort(unique(breaks))
            midpoints = [(breaks[i] + breaks[i + 1]) / 2 for i in 1:(length(breaks) - 1)]
            # First and last midpoints are the boundary values
            if !isempty(midpoints)
                midpoints[1] = breaks[1]
                midpoints[end] = breaks[end]
            end
            unique_vals[v] = midpoints

            # Assign each observation to nearest midpoint via searchsorted on breaks
            indices = zeros(Int, length(x))
            @inbounds for i in eachindex(x)
                # Find which bin x[i] falls in
                j = searchsortedlast(breaks, x[i])
                j = clamp(j, 1, length(midpoints))
                indices[i] = j
            end
            idx_map[v] = indices
        end
    end

    return DiscretizedData(unique_vals, idx_map, n)
end

"""
    discretized_smooth_construct(spec, disc_data, full_data)

Construct a smooth basis using discretized unique values, returning both
the compact basis (evaluated at unique values) and index mapping.

Standalone utility: `bam()` does not currently use discretization.
"""
function discretized_smooth_construct(spec::SmoothSpec, disc::DiscretizedData, full_data)
    # Build a "unique data" table for basis construction
    var = spec.term_vars[1]  # primary variable for single smooths
    uvals = disc.unique_values[var]
    unique_data = NamedTuple{(var,)}((uvals,))

    # Construct basis on unique values
    sm = smooth_construct(spec, unique_data)
    return sm, disc.indices[var]
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
and have no effect: `bam()` does not discretize covariates or thread the
accumulation. Passing them warns.
"""
function bam_control(;
    chunk_size::Int = 10000,
    discrete::Union{Bool, Nothing} = nothing,
    max_unique::Union{Int, Nothing} = nothing,
    nthreads::Union{Int, Nothing} = nothing,
)
    if discrete !== nothing || max_unique !== nothing || nthreads !== nothing
        @warn "bam_control: `discrete`, `max_unique`, and `nthreads` are deprecated " *
              "and ignored — bam() does not discretize covariates or thread the accumulation"
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
    family::UnivariateDistribution, link::GLM.Link;
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    start::Union{Vector{Float64}, Nothing} = nothing,
    control::GamControl = gam_control(),
    chunk_size::Int = 10000)

    n, p = size(X)

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
        mul!(eta, X, beta)
        eta .+= offset
    else
        # Family-appropriate initial μ (mgcv mustart)
        @inbounds for i in 1:n
            eta[i] = GLM.linkfun(link, _bam_mustart(family, y[i], weights[i]))
        end
        # Start from the constant fit on the link scale; locate the intercept
        # column rather than assuming it is column 1
        icpt = findfirst(j -> all(==(1.0), view(X, :, j)), 1:p)
        if icpt !== nothing
            beta[icpt] = mean(eta)
            mul!(eta, X, beta)
            eta .+= offset
        end
        # (no intercept column: keep the mustart η as the starting point)
    end

    @inbounds for i in 1:n
        mu[i] = GLM.linkinv(link, eta[i])
    end
    dev_old = _deviance(family, y, mu, weights)

    beta_prop = zeros(p)
    converged = false
    n_iter = 0

    for iter in 1:(control.maxit)
        n_iter = iter

        # Working weights and working response (scalar ops)
        _pirls_working!(w, z, y, mu, eta, offset, weights, family, link)

        # Chunk-wise accumulation of X'WX and X'Wz
        _accumulate_XtWX_XtWz_chunked!(XtWX, XtWz, X, w, z, chunk_size)

        # Add penalty: A = XtWX + S_total
        @inbounds for j in 1:p, k in 1:p
            A[j, k] = XtWX[j, k] + S_total[j, k]
        end

        # Solve via Cholesky (escalating-ridge recovery on indefiniteness,
        # matching the other fitters)
        A_chol = _protected_cholesky!(A)
        ldiv!(beta_new, A_chol, XtWz)

        # Update eta, mu
        mul!(eta_new, X, beta_new)
        eta_new .+= offset
        @inbounds for i in 1:n
            mu_new[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_new[i]))
        end
        dev_new = _deviance(family, y, mu_new, weights)

        # Step halving on the penalized deviance (matching pirls.jl/mgcv):
        # each trial interpolates from the ORIGINAL proposal, so factor 0.5^k
        # really is 0.5^k of the Newton step
        pdev_old_iter = dev_old + dot(beta, S_total, beta)
        pdev_new = dev_new + dot(beta_new, S_total, beta_new)
        if !(isfinite(pdev_new) &&
             pdev_new <= pdev_old_iter + control.epsilon * abs(pdev_old_iter))
            copyto!(beta_prop, beta_new)
            step_factor = 1.0
            for _ in 1:25
                step_factor *= 0.5
                @inbounds for j in 1:p
                    beta_new[j] = beta[j] + step_factor * (beta_prop[j] - beta[j])
                end
                mul!(eta_new, X, beta_new)
                eta_new .+= offset
                @inbounds for i in 1:n
                    mu_new[i] = _clamp_mu_scalar(family, GLM.linkinv(link, eta_new[i]))
                end
                dev_new = _deviance(family, y, mu_new, weights)
                pdev_new = dev_new + dot(beta_new, S_total, beta_new)
                if isfinite(pdev_new) &&
                   pdev_new <= pdev_old_iter + control.epsilon * abs(pdev_old_iter)
                    break
                end
                if step_factor < 1e-8
                    break
                end
            end
        end

        # Convergence check
        crit = abs(dev_new - dev_old) / (abs(dev_new) + 0.1)
        copyto!(beta, beta_new)
        copyto!(eta, eta_new)
        copyto!(mu, mu_new)
        dev_old = dev_new

        if crit < control.epsilon
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
    _accumulate_XtWX_chunked!(XtWX, X, w, chunk_size)
    @inbounds for j in 1:p, k in 1:p
        A[j, k] = XtWX[j, k] + S_total[j, k]
    end
    A_chol = _protected_cholesky!(A)
    F = A_chol \ XtWX
    edf_vec = diag(F)

    # Chunked hat diagonal: hat[i] = sum_j (X[i,:] / U)_j^2
    hat_diag = zeros(n)
    Uinv = inv(A_chol.U)  # p×p — small
    for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        X_chunk = view(X, start:stop, :)
        H_chunk = X_chunk * Uinv
        @inbounds for i in 1:(stop - start + 1)
            s = 0.0
            for j in 1:p
                s += H_chunk[i, j]^2
            end
            hat_diag[start + i - 1] = s
        end
    end

    R = Matrix(A_chol.U)

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
    smooths::Vector{<:ConstructedSmooth},
    penalty::PenaltySetup,
    family::UnivariateDistribution, link::GLM.Link;
    method::Symbol = :REML,
    weights::Vector{Float64} = ones(length(y)),
    offset::Vector{Float64} = zeros(length(y)),
    control::GamControl = gam_control(),
    chunk_size::Int = 10000)

    n, p = size(X)
    n_sp = length(penalty.sp)

    if n_sp == 0
        S_total = zeros(p, p)
        result = pirls_bam(X, y, S_total, family, link;
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
        _accumulate_XtWX_XtWz_chunked!(XtWX_cached, Xty_cached,
            X, weights, y_adj, chunk_size)
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
            result = pirls_bam(X, y, S_total, family, link;
                weights = weights, offset = offset, start = start_coef,
                control = control, chunk_size = chunk_size)
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
            XtWX_cur = zeros(p, p)
            _accumulate_XtWX_chunked!(XtWX_cur, X, w, chunk_size)
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
        eta = X * beta
        eta .+= offset
        mu = copy(eta)
        dev = _deviance(family, y, mu, weights)
        F = A_chol \ XtWX_cached
        edf_vec = diag(F)
        hat_diag = zeros(n)
        Uinv = inv(A_chol.U)
        for start_i in 1:chunk_size:n
            stop_i = min(start_i + chunk_size - 1, n)
            X_chunk = view(X, start_i:stop_i, :)
            H_chunk = X_chunk * Uinv
            @inbounds for i in 1:(stop_i - start_i + 1)
                s = 0.0
                for j in 1:p
                    s += H_chunk[i, j]^2
                end
                hat_diag[start_i + i - 1] = s
            end
        end
        R = Matrix(A_chol.U)
        final_result = PirlsResult(beta, mu, eta, weights, dev, dev,
            true, 1, R, hat_diag, edf_vec)
    else
        final_result = pirls_bam(X, y, S_total, family, link;
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

"""
    _discretized_XtWX_XtWz!(XtWX, XtWz, X_unique, indices, w, z, S, p, n)

Compute X'WX and X'Wz efficiently using discretized representation.
Instead of expanding X to full n×p, accumulates using index lookups.
"""
function _discretized_XtWX_XtWz!(
    XtWX::Matrix{Float64}, XtWz::Vector{Float64},
    X_unique::Matrix{Float64}, indices::Vector{Int},
    w::Vector{Float64}, z::Vector{Float64},
    S::Matrix{Float64}, p_smooth::Int, offset_col::Int, p_total::Int, n::Int)

    # Accumulate contributions from discretized smooth columns
    @inbounds for i in 1:n
        idx = indices[i]
        wi = w[i]
        wz = wi * z[i]
        for j in 1:p_smooth
            col_j = offset_col + j
            xij = X_unique[idx, j]
            xij_wi = xij * wi
            XtWz[col_j] += xij * wz
            for k in j:p_smooth
                col_k = offset_col + k
                XtWX[col_j, col_k] += xij_wi * X_unique[idx, k]
            end
        end
    end
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
- `weights`: optional observation weights
- `offset`: optional known additive term on the link scale
- `select`: if `true`, add a null-space penalty to every smooth
  (Marra & Wood 2011 term selection), as in `gam(...; select=true)`
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
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing,
    select::Bool = false,
    control::GamControl = gam_control(),
    bam_ctrl::BamControl = bam_control())

    _bam_check_method(method)

    if link === nothing
        link = GLM.canonicallink(family)
    end

    y, X, X_para, smooths, n_parametric = setup_gam(f, data; family = family)
    return _fit_bam(y, X, smooths, n_parametric, f, data, family, link,
        method, weights, offset, select, control, bam_ctrl)
end

function bam(gf::GamFormula, data;
    family::UnivariateDistribution = Normal(),
    link::Union{GLM.Link, Nothing} = nothing,
    method::Symbol = :REML,
    weights::Union{AbstractVector{<:Real}, Nothing} = nothing,
    offset::Union{AbstractVector{<:Real}, Nothing} = nothing,
    select::Bool = false,
    control::GamControl = gam_control(),
    bam_ctrl::BamControl = bam_control())

    _bam_check_method(method)

    if link === nothing
        link = GLM.canonicallink(family)
    end

    y, X, X_para, smooths, n_parametric = setup_gam(gf, data; family = family)
    f = term(gf.response) ~ term(1)
    return _fit_bam(y, X, smooths, n_parametric, f, data, family, link,
        method, weights, offset, select, control, bam_ctrl)
end

function _fit_bam(y, X, smooths, n_parametric, f, data,
    family, link, method, weights, offset, select, control, bam_ctrl)
    n, p = size(X)

    wts = weights === nothing ? ones(n) : Float64.(weights)
    length(wts) == n || throw(DimensionMismatch(
        "weights length $(length(wts)) ≠ data length $n"))

    off = offset === nothing ? zeros(n) : Float64.(offset)
    length(off) == n || throw(DimensionMismatch(
        "offset length $(length(off)) ≠ data length $n"))

    penalty = setup_penalties(smooths, n_parametric; select = select)

    # Use BAM outer iteration with chunked accumulation
    log_sp, result = outer_iteration_bam(X, y, smooths, penalty, family, link;
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
