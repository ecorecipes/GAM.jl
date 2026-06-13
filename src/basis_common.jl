# Common basis construction utilities

"""
    smooth_construct(spec::SmoothSpec, data, knots=nothing)

Construct a smooth basis from the specification and data. Returns a
[`ConstructedSmooth`](@ref). Dispatches on the basis type in `spec`.

# Arguments
- `spec`: smooth specification
- `data`: column table or dict with data columns
- `knots`: optional pre-specified knot locations
"""
function smooth_construct(spec::SmoothSpec{B}, data, knots = nothing) where {B}
    # Validate smooth term data before construction
    t = Tables.columntable(data)
    for var in spec.term_vars
        if var in Tables.columnnames(t)
            col = Tables.getcolumn(t, var)
            if eltype(col) <: Real
                _validate_smooth_data(Float64.(col), var)
            end
        end
    end
    sm = _smooth_construct(spec.basis, spec, data, knots)
    _append_pc_constraints!(sm, data)
    if spec.by !== nothing
        _apply_by_variable!(sm, t)
    end
    return sm
end

"""
    _apply_by_variable!(sm, t)

Apply a `by=` variable to a constructed smooth (mgcv's varying-coefficient
and factor-smooth machinery).

- **Numeric by**: each row of the (constrained) basis is multiplied by the
  by-variable, giving a varying-coefficient term `by · f(x)`. Note that the
  smooth is centered, so include the by variable as a parametric main effect
  if the model should contain `by · const`.
- **Factor by**: the smooth is replicated once per level, each copy active
  only on that level's rows and carrying its own penalty (and smoothing
  parameter), as in mgcv. Include the factor main effect in the model for
  level offsets.
"""
function _apply_by_variable!(sm::ConstructedSmooth, t)
    by_sym = sm.spec.by
    by_col = Tables.getcolumn(t, by_sym)
    if eltype(by_col) <: Real
        sm.X = sm.X .* Float64.(by_col)
        return sm
    end

    # Factor by: replicate the smooth per level
    if sm.Ain !== nothing || sm.Aeq !== nothing
        throw(ArgumentError(
            "factor by= is not supported for linear-constraint (scasm) smooths"))
    end
    levels = sort!(unique(collect(by_col)))
    L = length(levels)
    n, k = size(sm.X)
    Xb = zeros(n, k * L)
    S_new = Matrix{Float64}[]
    for (l, lev) in enumerate(levels)
        mask = by_col .== lev
        cols = ((l - 1) * k + 1):(l * k)
        Xb[mask, cols] .= sm.X[mask, :]
        for Si in sm.S
            Sfull = zeros(k * L, k * L)
            Sfull[cols, cols] .= Si
            push!(S_new, Sfull)
        end
    end
    sm.X = Xb
    sm.S = S_new
    sm.null_dim *= L
    sm.rank *= L
    # Replicate the shape-constraint pattern per level so each level's smooth
    # carries the same monotonicity/convexity constraint (SCAM factor-by).
    # p_ident is read at fit time across the full k·L coefficient block; cmX
    # and Sigma are left at the original width because the SCAM
    # `_predict_matrix` rebuilds the per-level (k-wide) basis and the by-tiling
    # in `predict_matrix` then replicates it.
    if sm.p_ident !== nothing
        sm.p_ident = repeat(sm.p_ident, L)
    end
    sm.spec.xt[:_by_levels] = levels
    return sm
end

"""
    predict_matrix(smooth::ConstructedSmooth, newdata) -> Matrix{Float64}

Construct the prediction matrix for `smooth` at new data points.
"""
function predict_matrix(smooth::ConstructedSmooth{B}, newdata) where {B}
    Xp = _predict_matrix(smooth.spec.basis, smooth, newdata)
    # Apply the by= transform with the SAME convention as fitting
    # (before side-constraint column removal, which was determined on the
    # post-by matrix)
    if smooth.spec.by !== nothing
        t = Tables.columntable(newdata)
        by_col = Tables.getcolumn(t, smooth.spec.by)
        if eltype(by_col) <: Real
            Xp = Xp .* Float64.(by_col)
        else
            levels = smooth.spec.xt[:_by_levels]
            k = size(Xp, 2)
            Xb = zeros(size(Xp, 1), k * length(levels))
            seen = falses(length(by_col))
            for (l, lev) in enumerate(levels)
                mask = by_col .== lev
                seen .|= mask
                Xb[mask, ((l - 1) * k + 1):(l * k)] .= Xp[mask, :]
            end
            if !all(seen)
                unseen = unique(collect(by_col)[.!seen])
                @warn "by= factor levels not seen during fitting get zero " *
                      "contribution from $(smooth.spec.label): $(unseen)"
            end
            Xp = Xb
        end
    end
    # Apply side constraint column removal if needed
    if !isempty(smooth.del_index)
        keep = setdiff(1:size(Xp, 2), smooth.del_index)
        Xp = Xp[:, keep]
    end
    return Xp
end

"""
    penalty_matrix(smooth::ConstructedSmooth) -> Vector{Matrix{Float64}}

Return the penalty matrices for this smooth.
"""
penalty_matrix(smooth::ConstructedSmooth) = smooth.S

"""
    null_space_dim(smooth::ConstructedSmooth) -> Int

Return the dimension of the penalty null space.
"""
null_space_dim(smooth::ConstructedSmooth) = smooth.null_dim

# Internal dispatch — each basis_*.jl file implements _smooth_construct and _predict_matrix

function _constraint_basis(C::Union{Matrix{Float64}, Nothing}, p::Int)
    if C === nothing || size(C, 1) == 0
        return Matrix{Float64}(I, p, p)
    end
    qr_C = qr(Matrix(C)')
    return (qr_C.Q * Matrix{Float64}(I, p, p))[:, (size(C, 1) + 1):p]
end

"""
    absorb_constraints!(X, S; constraint=:sum_to_zero)

Apply identifiability constraint to smooth basis matrix and penalty.
Default is sum-to-zero: the smooth sums to zero over the observed data.

Returns `(X_new, S_new, C, qrc)` where:
- `X_new`: constrained model matrix (n × (k-1))
- `S_new`: constrained penalty matrices
- `C`: constraint matrix
- `qrc`: QR factorization used for absorption
"""
function absorb_constraints!(X::Matrix{Float64}, S::Vector{Matrix{Float64}};
    constraint::Symbol = :sum_to_zero,
    scale_penalty::Bool = true)
    n, k = size(X)

    # mgcv-style penalty rescaling (smoothCon, lines 3879-3886 of smooth.r).
    # Applied BEFORE constraint absorption, using the pre-absorption X and S.
    if scale_penalty && !isempty(S)
        maXX = opnorm(X, Inf)^2
        if maXX > 0
            for i in eachindex(S)
                nS = opnorm(S[i], 1)  # R's default norm() for matrices = "O" = 1-norm
                if nS > 0
                    S[i] = S[i] * (maXX / nS)
                end
            end
        end
    end

    if constraint == :sum_to_zero
        # R's smoothCon uses C = colSums(X) (not divided by n)
        C = sum(X; dims = 1)  # 1 × k
    else
        throw(ArgumentError("Unknown constraint type: $constraint"))
    end

    # R's absorb.cons uses: qrc = qr(t(C)), Z = qr.Q(qrc, complete=TRUE)[, -1]
    # This ensures the specific rotation matches R's parameterization.
    qr_C = qr(Matrix(C)')  # QR of C' (k × 1 matrix)
    Z = (qr_C.Q * Matrix(I, k, k))[:, 2:k]  # k × (k-1), drop first column

    X_new = X * Z
    S_new = [Z' * Si * Z for Si in S]
    # Ensure exact symmetry after QR rotation (floating-point round-off)
    for i in eachindex(S_new)
        Si = S_new[i]
        for a in 1:size(Si, 1), b in (a + 1):size(Si, 2)
            v = (Si[a, b] + Si[b, a]) / 2
            Si[a, b] = v
            Si[b, a] = v
        end
    end

    return X_new, S_new, Matrix(C), qr(X_new)
end

"""
    side_constrain!(smooths::Vector{<:ConstructedSmooth}, X_para::Matrix{Float64};
                    tol=sqrt(eps()), with_pen=true)

Apply side constraints when multiple smooths share variables.
Ensures identifiability by removing columns from higher-dimensional smooths
that are linearly dependent on lower-dimensional smooths with shared variables.

Equivalent to mgcv's `gam.side`. Modifies smooths in place and returns
the indices of columns removed from the full model matrix (for rebuilding X).

Returns `true` if any modifications were made.
"""
function side_constrain!(smooths::Vector{<:ConstructedSmooth}, X_para::Matrix{Float64};
                         tol::Float64=sqrt(eps()), with_pen::Bool=true)
    m = length(smooths)
    m <= 1 && return false

    # Stage 1: Collect variable names per smooth (including by variable)
    vn_list = Vector{Vector{Symbol}}(undef, m)
    max_dim = 1
    for i in 1:m
        vn = copy(smooths[i].spec.term_vars)
        by = smooths[i].spec.by
        if by !== nothing
            vn = [Symbol(string(v, by)) for v in vn]
        end
        vn_list[i] = vn
        dim_i = length(smooths[i].spec.term_vars)
        if dim_i > max_dim
            max_dim = dim_i
        end
    end

    # Collect all variable names, check for repeats
    all_names = Symbol[]
    for vn in vn_list
        append!(all_names, vn)
    end
    if length(unique(all_names)) == length(all_names)
        return false  # no shared variables → no nesting
    end

    # Stage 2: Check if parametric matrix contains intercept
    has_intercept = false
    n_para = size(X_para, 2)
    nobs = size(smooths[1].X, 1)
    if n_para > 0
        for j in 1:n_para
            col = @view X_para[:, j]
            if maximum(abs, col .- col[1]) < eps()^0.75
                has_intercept = true
                break
            end
        end
        if !has_intercept
            # Check if 1-vector is in the span of X_para
            f = ones(nobs)
            qrp = qr(X_para)
            ff = qrp \ f
            ff_proj = X_para * ff
            if maximum(abs, ff_proj .- f) < eps()^0.75
                has_intercept = true
            end
        end
    end

    # Stage 3: Build dependency map — which smooths share each variable
    # Map variable → list of (smooth_index, smooth_dim)
    var_to_smooths = Dict{Symbol, Vector{Tuple{Int,Int}}}()
    for d in 1:max_dim
        for i in 1:m
            dim_i = length(smooths[i].spec.term_vars)
            dim_i != d && continue
            # Skip smooths that shouldn't be side-constrained (re, fs)
            _should_side_constrain(smooths[i]) || continue
            for v in vn_list[i]
                if !haskey(var_to_smooths, v)
                    var_to_smooths[v] = Tuple{Int,Int}[]
                end
                push!(var_to_smooths[v], (i, d))
            end
        end
    end

    # Stage 4: For each dimension, constrain higher-dim smooths
    modified = false
    np = sum(size(sm.X, 2) for sm in smooths)  # total penalized params

    for d in 1:max_dim
        for i in 1:m
            dim_i = length(smooths[i].spec.term_vars)
            dim_i != d && continue
            _should_side_constrain(smooths[i]) || continue

            # Build X1: columns from all lower-dimensional smooths sharing variables
            X1_parts = Matrix{Float64}[]
            if has_intercept
                if with_pen
                    push!(X1_parts, vcat(ones(nobs, 1), zeros(np, 1)))
                else
                    push!(X1_parts, ones(nobs, 1))
                end
            end

            seen = Set{Int}()  # avoid adding same smooth twice
            for v in vn_list[i]
                !haskey(var_to_smooths, v) && continue
                for (j, dj) in var_to_smooths[v]
                    j == i && continue
                    j in seen && continue
                    dj >= d && continue  # only lower-dimensional terms
                    push!(seen, j)
                    if with_pen
                        push!(X1_parts, _augment_smooth_X(smooths[j], nobs, np))
                    else
                        push!(X1_parts, smooths[j].X)
                    end
                end
            end

            # If X1 is only the intercept (or empty), skip
            n_intercept = has_intercept ? 1 : 0
            length(X1_parts) <= n_intercept && continue

            X1 = hcat(X1_parts...)

            # Build X2 (augmented if with_pen)
            if with_pen
                X2 = _augment_smooth_X(smooths[i], nobs, np)
            else
                X2 = smooths[i].X
            end

            # Stage 5: Find dependent columns
            ind = _fix_dependence(X1, X2; tol=tol)
            isnothing(ind) && continue

            # Stage 6: Remove dependent columns
            keep = setdiff(1:size(smooths[i].X, 2), ind)
            smooths[i].X = smooths[i].X[:, keep]

            if smooths[i].Ain !== nothing
                smooths[i].Ain = smooths[i].Ain[:, keep]
            end
            if smooths[i].Aeq !== nothing
                smooths[i].Aeq = smooths[i].Aeq[:, keep]
            end
            if smooths[i].p_ident !== nothing
                smooths[i].p_ident = smooths[i].p_ident[keep]
            end

            # Update penalty matrices
            j_pen = length(smooths[i].S)
            while j_pen >= 1
                S_j = smooths[i].S[j_pen][keep, keep]
                smooths[i].S[j_pen] = S_j
                # Check if penalty is now zero
                if maximum(abs, S_j) < tol
                    deleteat!(smooths[i].S, j_pen)
                end
                j_pen -= 1
            end

            # Recalculate null space dimension
            if !isempty(smooths[i].S)
                St = smooths[i].S[1] / max(opnorm(smooths[i].S[1]), 1e-20)
                for j in 2:length(smooths[i].S)
                    St .+= smooths[i].S[j] / max(opnorm(smooths[i].S[j]), 1e-20)
                end
                eigs = eigvals(Symmetric(St))
                smooths[i].null_dim = count(e -> e < maximum(eigs) * eps()^0.75, eigs)
            end

            smooths[i].rank = size(smooths[i].X, 2) - smooths[i].null_dim
            smooths[i].del_index = ind
            modified = true
        end
    end

    return modified
end

"""Whether a smooth should have side constraints applied."""
function _should_side_constrain(sm::ConstructedSmooth)
    b = sm.spec.basis
    # Random effects and factor-smooth interactions handle identifiability differently
    b isa RandomEffect && return false
    b isa FactorSmooth && return false
    b isa ConstrainedFactorSmooth && return false
    return true
end

"""
Augment smooth model matrix with scaled penalty sqrt for side constraint testing.
Returns matrix of size (nobs + np) × k.
"""
function _augment_smooth_X(sm::ConstructedSmooth, nobs::Int, np::Int)
    k = size(sm.X, 2)
    X_aug = zeros(nobs + np, k)
    X_aug[1:nobs, :] = sm.X

    if !isempty(sm.S)
        # Combine all penalties, scaled by data magnitude
        ind = vec(any(abs.(sm.S[1]) .> 0, dims=1))
        sqrmaX = mean(abs2, @view(sm.X[:, ind]))
        St = sm.S[1] * (sqrmaX / max(mean(abs, @view(sm.S[1][ind, ind])), 1e-20))
        for j in 2:length(sm.S)
            ind_j = vec(any(abs.(sm.S[j]) .> 0, dims=1))
            if any(ind_j)
                alpha = sqrmaX / max(mean(abs, @view(sm.S[j][ind_j, ind_j])), 1e-20)
                St .+= sm.S[j] * alpha
            end
        end
        # Matrix square root
        eig = eigen(Symmetric(St))
        pos = eig.values .> max(maximum(eig.values), 1e-20) * 1e-10
        if any(pos)
            rS = eig.vectors[:, pos] * Diagonal(sqrt.(eig.values[pos]))
            # Place in augmented rows (offset by smooth's position in param vector)
            # Use first np rows after nobs, placing at columns 1:k
            rows_start = nobs + sm.first_para
            rows_end = min(nobs + sm.first_para + size(rS, 2) - 1, nobs + np)
            n_rows = rows_end - rows_start + 1
            if n_rows > 0 && rows_start >= nobs + 1 && rows_end <= nobs + np
                X_aug[rows_start:rows_end, :] = rS[:, 1:n_rows]'
            end
        end
    end

    return X_aug
end

"""
    _fix_dependence(X1, X2; tol, rank_def=0) → Union{Vector{Int}, Nothing}

Find columns of X2 that are linearly dependent on X1.
Returns column indices to remove from X2, or nothing if independent.
"""
function _fix_dependence(X1::Matrix{Float64}, X2::Matrix{Float64};
                         tol::Float64=sqrt(eps()), rank_def::Int=0)
    n = size(X1, 1)
    r = size(X1, 2)

    qr1 = qr(X1)
    R11_abs = abs(Matrix(qr1.R)[1, 1])

    # Project X2 into orthogonal complement of X1
    QtX2 = qr1.Q' * X2
    QtX2_comp = QtX2[(r+1):n, :]  # rows beyond the rank of X1

    qr2 = qr(QtX2_comp, ColumnNorm())
    R2 = Matrix(qr2.R)
    r2 = size(R2, 1)

    # Find rank deficiency
    r0 = r2
    if rank_def > 0 && rank_def <= r2
        r0 = r2 - rank_def
    else
        while r0 > 0 && mean(abs, @view(R2[r0:r2, r0:r2])) < R11_abs * tol
            r0 -= 1
        end
    end
    r0 += 1

    if r0 > r2
        return nothing  # fully independent
    end

    # Return the pivot indices of the dependent columns
    return sort(qr2.p[r0:r2])
end
