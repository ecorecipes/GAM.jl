# Splines on the Sphere (SOS) — bs="sos"
#
# Smooths on spherical surfaces using geodesic distance and a radial
# basis built from the TPS-like kernel η(d) = d^(2m-2) log(d).
# Default penalty order m=2 gives η(d) = d² log(d).
#
# Reference: mgcv smooth.r lines 2990-3112 (Sos.smooth.construct)
#
# Algorithm:
#   1. Input: latitude (radians) and longitude (radians)
#   2. Compute pairwise geodesic distances via great-circle formula
#   3. Build radial basis R_ij = η(d_ij)
#   4. Eigendecompose R → top k eigenvectors as basis
#   5. Penalty D = diag(1/eigenvalues)
#   6. Null space: constant (null_dim = 1)

"""Spline on the sphere basis (mgcv `bs=\"sos\"`)."""
struct SphericalSpline <: AbstractBasisType end

BASIS_TYPES[:sos] = SphericalSpline()

"""
    _geodesic_distance(lat1, lon1, lat2, lon2) -> Float64

Great-circle distance between two points on the unit sphere (inputs in radians).
"""
function _geodesic_distance(lat1::Real, lon1::Real, lat2::Real, lon2::Real)
    # Clamp to [-1,1] for numerical safety
    arg = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)
    return acos(clamp(arg, -1.0, 1.0))
end

"""
    _geodesic_distance_matrix(lat, lon) -> Matrix{Float64}

Pairwise geodesic distance matrix for vectors of lat/lon in radians.
"""
function _geodesic_distance_matrix(lat::AbstractVector, lon::AbstractVector)
    n = length(lat)
    D = zeros(n, n)
    @inbounds for j in 1:n, i in (j+1):n
        d = _geodesic_distance(lat[i], lon[i], lat[j], lon[j])
        D[i, j] = d
        D[j, i] = d
    end
    return D
end

"""
    _geodesic_distance_matrix(lat1, lon1, lat2, lon2) -> Matrix{Float64}

Geodesic distances from each point in (lat1, lon1) to each point in (lat2, lon2).
Returns n1 × n2 matrix.
"""
function _geodesic_distance_matrix(lat1::AbstractVector, lon1::AbstractVector,
                                    lat2::AbstractVector, lon2::AbstractVector)
    n1, n2 = length(lat1), length(lat2)
    D = zeros(n1, n2)
    @inbounds for j in 1:n2, i in 1:n1
        D[i, j] = _geodesic_distance(lat1[i], lon1[i], lat2[j], lon2[j])
    end
    return D
end

"""
    _sos_kernel(d, m) -> Float64

Spherical radial basis kernel: η(d) = d^(2m-2) * log(d), with η(0) = 0.
Default m=2 gives η(d) = d² log(d).
"""
function _sos_kernel(d::Real, m::Int)
    d <= 0.0 && return 0.0
    power = 2m - 2
    return d^power * log(d)
end

"""
    _sos_kernel_matrix(D, m) -> Matrix{Float64}

Apply the SOS kernel element-wise to a distance matrix.
"""
function _sos_kernel_matrix(D::Matrix{Float64}, m::Int)
    R = similar(D)
    @inbounds for j in axes(D, 2), i in axes(D, 1)
        R[i, j] = _sos_kernel(D[i, j], m)
    end
    return R
end

function _smooth_construct(::SphericalSpline, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 2 ||
        throw(ArgumentError("Spherical spline (sos) requires exactly 2 variables " *
            "(latitude, longitude). Got: $(spec.term_vars)"))

    lat_var, lon_var = spec.term_vars
    lat = Float64.(Tables.getcolumn(data, lat_var))
    lon = Float64.(Tables.getcolumn(data, lon_var))
    n = length(lat)

    m_order = spec.m === nothing ? 2 : spec.m
    k = spec.k

    # Knot subsampling for large datasets
    max_knots = get(spec.xt, :max_knots, 2000)
    if n > max_knots
        # Subsample knots evenly
        step = n / max_knots
        knot_idx = round.(Int, range(1, n; length=max_knots))
        knot_idx = unique(knot_idx)
        lat_k = lat[knot_idx]
        lon_k = lon[knot_idx]
    else
        knot_idx = collect(1:n)
        lat_k = lat
        lon_k = lon
    end
    nk = length(knot_idx)
    k = min(k, nk - 1)  # need at least 1 column for null space

    # Compute geodesic distance matrix among knots
    D_kk = _geodesic_distance_matrix(lat_k, lon_k)

    # Build radial basis matrix
    R = _sos_kernel_matrix(D_kk, m_order)
    R = (R + R') / 2  # ensure symmetry

    # Eigendecompose
    eig = eigen(Symmetric(R))
    # Sort by absolute eigenvalue descending (like TPRS)
    idx = sortperm(abs.(eig.values); rev=true)

    # Select top k eigenpairs
    U_k = eig.vectors[:, idx[1:k]]
    λ_k = eig.values[idx[1:k]]

    # Build basis matrix
    if nk < n
        # Nystrom extension: project data onto knot eigenbasis
        D_nk = _geodesic_distance_matrix(lat, lon, lat_k, lon_k)
        R_nk = _sos_kernel_matrix(D_nk, m_order)
        # Nystrom: U_data ≈ R_nk * U_k * diag(1/λ_k)
        # Basis X_eig = U_data (scaled so penalty is simple)
        inv_λ = [abs(λ) > eps() ? 1.0 / λ : 0.0 for λ in λ_k]
        X_eig = R_nk * U_k * Diagonal(inv_λ)
    else
        X_eig = U_k
    end

    # Build penalty: D = diag(1/|eigenvalues|) for selected eigenvectors
    # Larger eigenvalues → less penalized (smoother components)
    pen_diag = [abs(λ) > eps() ? 1.0 / abs(λ) : 0.0 for λ in λ_k]
    S_pen = Diagonal(pen_diag) |> Matrix{Float64}

    # Null space: constant function on sphere (null_dim = 1)
    null_dim = 1
    pen_rank = k - null_dim

    # Add constant column for null space
    X_full = hcat(X_eig, ones(n, 1))  # n × (k+1)
    k_full = k + 1

    # Expand penalty to include null space column (unpenalized)
    S_full = zeros(k_full, k_full)
    S_full[1:k, 1:k] .= S_pen

    penalties = Matrix{Float64}[S_full]

    # Apply sum-to-zero constraint
    X_cons, S_cons, C, _ = absorb_constraints!(X_full, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],  # knots stored in metadata
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        predict_cache = SOSPredictCache(
            Float64.(lat_k), Float64.(lon_k), Matrix{Float64}(U_k),
            Float64.(λ_k), m_order, k,
        ),
    )

    return sm
end

function _predict_matrix(::SphericalSpline, smooth::ConstructedSmooth, newdata)
    spec = smooth.spec
    lat_var, lon_var = spec.term_vars

    lat_new = Float64.(Tables.getcolumn(newdata, lat_var))
    lon_new = Float64.(Tables.getcolumn(newdata, lon_var))
    n_new = length(lat_new)

    info = smooth.predict_cache
    info isa SOSPredictCache ||
        throw(ArgumentError("Cannot find spherical spline metadata for prediction"))

    lat_k = info.lat_k
    lon_k = info.lon_k
    U_k = info.U_k
    λ_k = info.lambda_k
    m_order = info.m_order
    k = info.k

    # Compute distances from new points to knots
    D_new = _geodesic_distance_matrix(lat_new, lon_new, lat_k, lon_k)
    R_new = _sos_kernel_matrix(D_new, m_order)

    # Project onto eigenbasis (Nystrom)
    inv_λ = [abs(λ) > eps() ? 1.0 / λ : 0.0 for λ in λ_k]
    X_eig = R_new * U_k * Diagonal(inv_λ)

    # Add constant column
    X_full = hcat(X_eig, ones(n_new, 1))

    # Apply constraint
    if smooth.constraint !== nothing
        C = smooth.constraint
        k_pred = size(X_full, 2)
        qr_C = qr(C')
        Z_cons = (qr_C.Q * Matrix(I, k_pred, k_pred))[:, 2:k_pred]
        return X_full * Z_cons
    end
    return X_full
end
