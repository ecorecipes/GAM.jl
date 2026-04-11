# GINLA — GAM Integrated Nested Laplace Approximation
#
# Computes marginal posterior densities for GAM coefficients using the
# Integrated Nested Laplace Approximation (INLA) method, with Newton
# enhancement from Wood (2020).
#
# Unlike the standard Gaussian approximation from gam(), GINLA can reveal
# asymmetry and non-normality in the posterior, especially for smooth terms.
#
# Reference: Wood, S.N. (2020) "Simplified Integrated Nested Laplace
# Approximation." Biometrika, 107(1): 223-230.

"""
    GinlaResult

Result of `ginla()` — marginal posterior densities for GAM coefficients.

# Fields
- `beta`: matrix of β values (pa × nb), rows = parameters, cols = grid points
- `density`: matrix of posterior densities (pa × nb)
- `indices`: which coefficient indices were computed
"""
struct GinlaResult
    beta::Matrix{Float64}
    density::Matrix{Float64}
    indices::Vector{Int}
end

# ============================================================================
# Cholesky drop — update R when dropping row/col k from R'R = A
# ============================================================================

"""
    choldrop(R, k) -> Matrix{Float64}

Update upper-triangular Cholesky factor `R` (where R'R = A) when dropping
row/column `k` from `A`. Uses Givens rotations.
Returns the (p-1) × (p-1) updated factor.
"""
function choldrop(R::Matrix{Float64}, k::Int)
    p = size(R, 1)
    @assert 1 <= k <= p "k=$k out of range 1:$p"

    # Delete column k only → p × (p-1) upper Hessenberg matrix
    cols = [1:(k - 1); (k + 1):p]
    R1 = R[:, cols]  # p × (p-1)

    n1 = p - 1
    # For columns j >= k, R1[j+1, j] may be non-zero (the sub-diagonal "bulge").
    # Apply Givens rotations to zero out each sub-diagonal entry.
    for j in k:n1
        a = R1[j, j]
        b = R1[j + 1, j]
        if abs(b) < eps() * (abs(a) + abs(b) + 1.0)
            continue
        end
        r = hypot(a, b)
        c = a / r
        s = b / r

        for col in j:n1
            t1 = R1[j, col]
            t2 = R1[j + 1, col]
            R1[j, col] = c * t1 + s * t2
            R1[j + 1, col] = -s * t1 + c * t2
        end
        R1[j + 1, j] = 0.0
    end

    # After rotations, last row is zero; return (p-1) × (p-1) upper triangular
    return R1[1:n1, :]
end

# ============================================================================
# Log joint density and gradient
# ============================================================================

"""
    _logf(beta, model, X; deriv=false)

Compute the negative log joint density (deviance/2σ² + penalty/2σ²) and
optionally its gradient w.r.t. β for a fitted GamModel.

Returns `(nll, grad)` where `nll` is on the negative log-likelihood scale.
"""
function _logf(beta::Vector{Float64}, model::GamModel, X::Matrix{Float64};
    deriv::Bool = false)

    n = length(model.y)

    # Linear predictor and mean
    eta = X * beta
    mu = GLM.linkinv.(Ref(model.link), eta)
    mu .= _clamp_mu(model.family, mu)

    # Deviance
    dev = _deviance(model.family, model.y, mu, model.weights)

    # Gradient of deviance w.r.t. beta
    dd = nothing
    if deriv
        dmu_deta = GLM.mueta.(Ref(model.link), eta)
        var_mu = _variance(model.family, mu)
        # d(dev)/d(beta) = -2 * X' * [w * dmu/deta * (y-mu)/V(mu)]
        resid_w = model.weights .* dmu_deta .* (model.y .- mu) ./ max.(var_mu, eps())
        dd = -X' * resid_w  # gradient of deviance/2 w.r.t. beta
    end

    # Penalty contribution: sum_j λ_j * β'S_j β
    pen = 0.0
    sp_idx = 1
    for block in model.penalty.blocks
        idx = block.start:block.stop
        b_block = beta[idx]
        for Si in block.S
            λ = exp(model.sp[sp_idx])
            Sb = λ .* (Si * b_block)
            pen += dot(b_block, Sb)
            if deriv
                dd[idx] .+= Sb
            end
            sp_idx += 1
        end
    end

    # Total on neg log-lik scale: (dev + pen) / (2 * scale)
    scale = model.scale
    nll = (dev + pen) / (2 * scale)
    if deriv
        dd ./= scale
    end

    return nll, dd
end

# ============================================================================
# Preconditioned Cholesky solve
# ============================================================================

"""
    _rsolve(R, b, piv, dpc)

Solve R'R a = b with pivoting and diagonal pre-conditioning.
`R` is upper-triangular, `piv` is pivot order, `dpc` = 1/sqrt(diag(A)).
"""
function _rsolve(R::Matrix{Float64}, b::AbstractVector{Float64},
    piv::Vector{Int}, dpc::Vector{Float64})

    n = length(b)
    # Pre-condition: b̃ = dpc .* b
    b_pc = dpc .* b

    # Pivot
    b_piv = b_pc[piv]

    # Solve R'R a_piv = b_piv
    # Forward solve: R' z = b_piv
    Rt = LowerTriangular(R')
    z = Rt \ b_piv

    # Back solve: R a_piv = z
    Ru = UpperTriangular(R)
    a_piv = Ru \ z

    # Un-pivot
    ipiv = similar(piv)
    ipiv[piv] .= 1:n
    a = a_piv[ipiv]

    # Post-condition
    a .*= dpc
    return a
end

function _rsolve(R::Matrix{Float64}, B::AbstractMatrix{Float64},
    piv::Vector{Int}, dpc::Vector{Float64})
    # Matrix version
    result = similar(B)
    for j in axes(B, 2)
        result[:, j] = _rsolve(R, B[:, j], piv, dpc)
    end
    return result
end

# ============================================================================
# Main ginla function
# ============================================================================

"""
    ginla(model::GamModel; A=nothing, nk=16, nb=100, J=1, approx=0)

Compute marginal posterior densities for GAM coefficients using the
GAM Integrated Nested Laplace Approximation (GINLA).

# Arguments
- `model`: a fitted `GamModel` from `gam()` or `bam()`
- `A`: optional matrix of linear transforms (rows) or vector of coefficient
  indices. If `nothing`, computes posteriors for all coefficients.
- `nk`: number of evaluation points for log posterior density (default: 16)
- `nb`: number of points in the returned gridded density (default: 100)
- `J`: number of determinant update steps (default: 1)
- `approx`: approximation level: 0 = full Newton refinement of modes,
  1 = use Gaussian conditional modes, 2 = also assume constant Hessian

# Returns
A [`GinlaResult`](@ref) with fields `beta` (pa × nb) and `density` (pa × nb).

# Example
```julia
using GAM, DataFrames

n = 200
x = range(0, 2π; length=n) |> collect
y = sin.(x) .+ 0.3 .* randn(n)
df = DataFrame(x=x, y=y)

m = gam(@formulak(y ~ s(x, k=15, bs=:cr)), df)
inla = ginla(m)

# inla.beta[k,:] gives grid points for coefficient k
# inla.density[k,:] gives posterior density values
```

# Reference
Wood, S.N. (2020) "Simplified Integrated Nested Laplace Approximation."
*Biometrika*, 107(1): 223-230.
"""
function ginla(model::GamModel;
    A::Union{Matrix{Float64}, Vector{Int}, Nothing} = nothing,
    nk::Int = 16,
    nb::Int = 100,
    J::Int = 1,
    approx::Int = 0)

    @assert approx in (0, 1, 2) "approx must be 0, 1, or 2"

    X = model.X
    p = size(X, 2)
    beta = model.coefficients

    # Determine which coefficients to compute
    if A isa Matrix{Float64}
        pa = size(A, 1)
        kind = 1:pa
        # Transform to new parameterization: B*β
        # Complete A to a full-rank p×p matrix
        B, Bi = _acomp(A)
        # Transform Vp and beta
        Vp = B * model.Vp * B'
        beta_t = B * beta
        use_transform = true
    elseif A isa Vector{Int}
        pa = length(A)
        kind = A
        Vp = copy(model.Vp)
        beta_t = copy(beta)
        use_transform = false
        B = nothing
        Bi = nothing
    else
        pa = p
        kind = 1:p
        Vp = copy(model.Vp)
        beta_t = copy(beta)
        use_transform = false
        B = nothing
        Bi = nothing
    end

    # Hessian = inv(Vp) with diagonal preconditioning
    if approx < 2
        H = inv(Symmetric(Vp))
        dpc = 1.0 ./ sqrt.(diag(H))
        # Preconditioned H: D * H * D where D = Diagonal(dpc)
        H_pc = dpc .* H .* dpc'
        # Pivoted Cholesky
        R1_chol = cholesky(Symmetric(H_pc), RowMaximum())
        R1 = Matrix(R1_chol.U)
        piv = R1_chol.p
    end

    sd_gauss = sqrt.(diag(Vp))

    # Output storage
    out_beta = zeros(pa, nb)
    out_density = zeros(pa, nb)

    # Grid quantiles
    eps_q = 0.0001
    qn = [quantile(Normal(), q) for q in range(eps_q, 1 - eps_q; length = nk)]

    kk = 0
    for k in kind
        kk += 1

        # Cholesky of H[-k,-k] via choldrop
        if approx < 2
            kd = findfirst(==(k), piv)
            R = choldrop(R1, kd)
            # Update pivot indices
            pivk = copy(piv)
            deleteat!(pivk, kd)
            for i in eachindex(pivk)
                if pivk[i] > k
                    pivk[i] -= 1
                end
            end
            dpc_k = dpc[[1:(k - 1); (k + 1):p]]
            ldetH = 2 * (sum(log.(abs.(diag(R)))) - sum(log.(dpc_k)))
        end

        # Evaluation grid: β_k values spread around posterior mode
        bg = qn .* sd_gauss[k] .+ beta_t[k]
        BM = zeros(p, nk)
        BM[k, :] .= bg
        # Gaussian conditional modes: β_{-k}|β_k
        ik = [1:(k - 1); (k + 1):p]
        for i in 1:nk
            BM[ik, i] .= beta_t[ik] .+ Vp[ik, k] .* ((bg[i] - beta_t[k]) / Vp[k, k])
        end

        dens0 = zeros(nk)
        ldet = zeros(nk)

        if approx == 0
            # Newton refinement of conditional modes
            db0 = zeros(p)
            for i in [div(nk, 2):-1:1; div(nk, 2):nk]
                beta0 = BM[:, i] .+ db0
                nll, grad = _logf_for_ginla(beta0, model, X, Bi, use_transform; deriv = true)

                if isfinite(nll)
                    for j_newton in 1:20
                        # Check convergence
                        grad_k = grad[ik]
                        if maximum(abs.(grad_k)) < 1e-4 * abs(nll)
                            break
                        end

                        # Newton step: solve H[-k,-k] * db[-k] = -grad[-k]
                        db = zeros(p)
                        db[ik] .= -_rsolve(R, grad_k, pivk, dpc_k)

                        beta1 = beta0 .+ db
                        nll1, grad1 = _logf_for_ginla(beta1, model, X, Bi, use_transform; deriv = true)

                        # Step halving
                        hstep = 0
                        while !isfinite(nll1) || nll1 > nll
                            db .*= 0.5
                            hstep += 1
                            beta1 = beta0 .+ db
                            nll1, grad1 = _logf_for_ginla(beta1, model, X, Bi, use_transform; deriv = hstep < 10)
                            if hstep > 20
                                break
                            end
                        end

                        nll = nll1
                        grad = grad1
                        beta0 = beta1
                    end
                end

                db0 = i == 1 ? zeros(p) : beta0 .- BM[:, i]
                BM[:, i] .= beta0
                dens0[i] = nll
            end
        else
            # Just evaluate at Gaussian modes
            for i in 1:nk
                nll, _ = _logf_for_ginla(BM[:, i], model, X, Bi, use_transform; deriv = false)
                dens0[i] = nll
            end
        end

        # Log-determinant correction
        if approx < 2
            step_length = mean(sqrt.(sum((BM .- beta_t) .^ 2; dims = 1))) / 20
            for i in 1:nk
                if !isfinite(dens0[i])
                    continue
                end

                if J == 1
                    # Simple: use constant Hessian determinant
                    ldet[i] = ldetH
                else
                    # J-step correction with rank-2 updates
                    bm = BM[:, i]
                    db = beta_t .- bm
                    db[k] = 0.0
                    nrm = sqrt(sum(db .^ 2))
                    if nrm > eps()
                        db .*= step_length / nrm
                    end

                    u_cols = Matrix{Float64}(undef, p - 1, 0)
                    D_signs = Float64[]

                    for jj in 1:J
                        h = H[ik, ik] * db[ik]
                        if size(u_cols, 2) > 0
                            Dv = D_signs .* (u_cols' * db[ik])
                            h .+= u_cols * Dv
                        end

                        _, g1 = _logf_for_ginla(bm .+ db ./ 2, model, X, Bi, use_transform; deriv = true)
                        _, g0 = _logf_for_ginla(bm .- db ./ 2, model, X, Bi, use_transform; deriv = true)
                        g_diff = g1 .- g0

                        h_norm = sqrt(sum(db[ik] .* h))
                        g_norm = sqrt(sum(db .* g_diff))
                        v1 = h_norm > eps() ? h ./ h_norm : h
                        v2 = g_norm > eps() ? g_diff[ik] ./ g_norm : g_diff[ik]

                        u_cols = hcat(u_cols, v1, v2)
                        push!(D_signs, -1.0, 1.0)

                        db .*= -1.0
                    end

                    Hu = _rsolve(R, u_cols, pivk, dpc_k) .* D_signs'
                    det_corr = det(I(2 * J) + u_cols' * Hu)
                    ldet[i] = ldetH + log(abs(det_corr))
                end
            end
        end

        # Combine: posterior density = exp(-nll - ldet/2)
        dens0 .= -dens0 .- ldet ./ 2
        # Replace non-finite with minimum - 10
        min_d = minimum(d for d in dens0 if isfinite(d); init = 0.0)
        for i in eachindex(dens0)
            if !isfinite(dens0[i])
                dens0[i] = min_d - 10.0
            end
        end
        dens0 .-= maximum(dens0)  # overflow-proof

        # Spline interpolation of log-density and normalize
        bg0 = minimum(bg)
        bg1 = maximum(bg)
        ok = false
        while !ok
            beta_grid = range(bg0, bg1; length = nb) |> collect
            # Cubic spline interpolation of log density
            log_dens_grid = _cubic_interp(bg, dens0, beta_grid)
            dens_grid = exp.(log_dens_grid)

            # Normalize
            dx = beta_grid[2] - beta_grid[1]
            n_const = sum(dens_grid) * dx
            if n_const > 0
                dens_grid ./= n_const
            end

            # Check tails
            maxd = maximum(dens_grid)
            ok = true
            if dens_grid[1] > maxd * 5e-3
                bg0 -= sd_gauss[k]
                ok = false
            end
            if dens_grid[end] > maxd * 5e-3
                bg1 += sd_gauss[k]
                ok = false
            end

            if ok
                out_beta[kk, :] .= beta_grid
                out_density[kk, :] .= dens_grid
            end
        end
    end

    return GinlaResult(out_beta, out_density, collect(kind))
end

# ============================================================================
# Helpers
# ============================================================================

"""Compute logf with optional transform B."""
function _logf_for_ginla(beta_t::Vector{Float64}, model::GamModel,
    X::Matrix{Float64}, Bi, use_transform::Bool; deriv::Bool = false)
    if use_transform && Bi !== nothing
        beta_orig = Bi * beta_t
    else
        beta_orig = beta_t
    end
    nll, grad = _logf(beta_orig, model, X; deriv = deriv)
    if deriv && use_transform && Bi !== nothing
        grad = Bi' * grad
    end
    return nll, grad
end

"""Complete A (pa × p) to a full-rank p × p matrix B with its inverse Bi."""
function _acomp(A::Matrix{Float64})
    pa, p = size(A)
    @assert pa <= p "A cannot have more rows than columns"

    if pa == p
        return copy(A), inv(A)
    end

    # Null space of A provides the orthogonal complement rows
    N = nullspace(A)  # p × (p - pa)
    B = vcat(A, N')   # p × p
    Bi = inv(B)
    return B, Bi
end

"""Simple cubic spline interpolation of (x, y) evaluated at xnew."""
function _cubic_interp(x::Vector{Float64}, y::Vector{Float64}, xnew::Vector{Float64})
    n = length(x)
    @assert n >= 2

    # Natural cubic spline via tridiagonal system
    h = diff(x)
    dy = diff(y)
    slopes = dy ./ h

    if n == 2
        # Linear interpolation
        return [y[1] + slopes[1] * (xi - x[1]) for xi in xnew]
    end

    # Tridiagonal system for second derivatives
    m = n - 2
    A_diag = zeros(m)
    A_sub = zeros(m - 1)
    A_sup = zeros(m - 1)
    rhs = zeros(m)

    for i in 1:m
        A_diag[i] = 2 * (h[i] + h[i + 1])
        rhs[i] = 6 * (slopes[i + 1] - slopes[i])
    end
    for i in 1:(m - 1)
        A_sub[i] = h[i + 1]
        A_sup[i] = h[i + 1]
    end

    # Solve tridiagonal system
    T = Tridiagonal(A_sub, A_diag, A_sup)
    sigma_inner = T \ rhs

    # Full second derivatives (with natural BC: σ₀ = σₙ = 0)
    sigma = [0.0; sigma_inner; 0.0]

    # Evaluate at new points
    result = zeros(length(xnew))
    for (idx, xi) in enumerate(xnew)
        # Find interval
        j = searchsortedlast(x, xi)
        j = clamp(j, 1, n - 1)

        t = xi - x[j]
        ht = h[j]
        a = (sigma[j + 1] - sigma[j]) / (6 * ht)
        b = sigma[j] / 2
        c = slopes[j] - ht * (2 * sigma[j] + sigma[j + 1]) / 6
        d = y[j]
        result[idx] = d + t * (c + t * (b + t * a))
    end
    return result
end
