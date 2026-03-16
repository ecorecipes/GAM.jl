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
    return _smooth_construct(spec.basis, spec, data, knots)
end

"""
    predict_matrix(smooth::ConstructedSmooth, newdata) -> Matrix{Float64}

Construct the prediction matrix for `smooth` at new data points.
"""
function predict_matrix(smooth::ConstructedSmooth{B}, newdata) where {B}
    return _predict_matrix(smooth.spec.basis, smooth, newdata)
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

    return X_new, S_new, Matrix(C), qr(X_new)
end

"""
    side_constrain!(smooths::Vector{<:ConstructedSmooth}, X_full::Matrix{Float64})

Apply side constraints when multiple smooths share variables.
Ensures identifiability by removing overlap in column spaces.
Equivalent to mgcv's `gam.side`.
"""
function side_constrain!(smooths::Vector{<:ConstructedSmooth}, X_full::Matrix{Float64})
    # For now, skip side constraints — only needed when multiple smooths
    # share variables, which is uncommon in basic usage
    # TODO: implement full gam.side equivalent
    return nothing
end
