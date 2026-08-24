# Splines on the Sphere (SOS) — bs="sos"
#
# A direct port of mgcv's spline-on-the-sphere construction: Wahba's (1981)
# spherical-spline reproducing kernels, evaluated on great-circle angles.
# This replaces an earlier approximation that applied the PLANAR thin-plate
# kernel η(d) = d^(2m-2)log(d) to geodesic distances; that kernel is not
# positive definite on the sphere and produced materially different fits.
#
# Reference implementation (mgcv 1.9-4):
#   R/smooth.r:2882-2988   `makeR`  — the reproducing kernel + null space
#   R/smooth.r:2990-3112   `smooth.construct.sos.smooth.spec`
#   R/smooth.r:3116-3147   `Predict.matrix.sos.smooth`
#   src/misc.c:39-73       `rksos`  — Wendelberger m=0 dilogarithm series
#
# Penalty orders (mgcv's `m`, i.e. p.order), all supported here:
#   m = -2  Duchon 1st-derivative semi-kernel, -z          (null space dim 1)
#   m = -1  Duchon semi-kernel z²log(z)/(8π)               (null space dim 4)
#   m =  0  Wendelberger order 2, via `rksos`              (null space dim 1)
#   m =  1..4  Wahba pseudospline closed forms             (null space dim 1)
# mgcv's default is m = 0, and that is the default here too.
#
# UNITS: latitude/longitude are interpreted as DEGREES by default, matching
# mgcv, whose `makeR` (R/smooth.r:2892-2894) begins
#     pi180 <- pi/180; la <- la*pi180; lo <- lo*pi180
# and which calls `makeR` from both the constructor and `Predict.matrix`, so a
# single conversion point serves both. Pass `xt = Dict(:units => :radians)` to
# supply radians instead. The resolved unit is stored in `SOSPredictCache` and
# read back at prediction, so the fit and predict paths cannot disagree.
#
# Algorithm (mirroring `smooth.construct.sos.smooth.spec`):
#   1. Input: latitude and longitude (degrees by default; see UNITS above)
#   2. Knots = unique locations, subsampled to `max_knots` (default 2000)
#   3. R = makeR(knots, knots, m): reproducing kernel on great-circle angles,
#      with null-space basis T (at data) and constraint basis Tc (at knots)
#   4. Truncated eigendecomposition of R keeping the k LARGEST-MAGNITUDE
#      eigenpairs — signed, so negative eigenvalues are retained as mgcv's
#      `slanczos(R, k, -1)` does
#   5. Constraint 1'Uγ = 0 absorbed via QR of U1 = U'Tc, giving Z; the
#      penalty is Z'DZ (zero-padded on the null-space block) and the basis
#      map is UZ = U·Z
#   6. X = [R(data, knots)·UZ  T(data)], then column-rescaled by 1/sd with
#      the smallest-sd column (the constant) pinned to scale 1

"""
Spline on the sphere basis (registered as `bs=:sos`).

A direct port of mgcv's `bs="sos"`: Wahba (1981) spherical-spline reproducing
kernels on great-circle angles, supporting penalty orders `m = -2, -1, 0, …, 4`
with `m = 0` (Wendelberger) the default, as in mgcv.
"""
struct SphericalSpline <: AbstractBasisType end

# Only used to subsample knots when there are more unique locations than
# `max_knots`; mgcv does the same with `sample()` under a fixed seed.
using Random: MersenneTwister, randperm

BASIS_TYPES[:sos] = SphericalSpline()

"""
    _sos_resolve_units(spec) -> Symbol

Resolve the angle units for an `sos` term: `:degrees` (the default, matching
mgcv) or `:radians`. Throws on any other value so a typo fails at fit time
rather than silently changing the scale of the fit.
"""
function _sos_resolve_units(spec::SmoothSpec)
    u = get(spec.xt, :units, :degrees)
    u = u isa AbstractString ? Symbol(u) : u
    u === :degrees || u === :radians || throw(ArgumentError(
        "sos smooth: xt[:units] must be :degrees (default, as in mgcv) or " *
        ":radians, got $(repr(u))."))
    return u::Symbol
end

"""
    _sos_to_radians(lat, lon, units) -> (Vector{Float64}, Vector{Float64})

Convert user-supplied angles to radians. This is the single conversion point
for both construction and prediction (mirroring mgcv's `makeR`).
"""
function _sos_to_radians(lat::AbstractVector, lon::AbstractVector, units::Symbol)
    units === :radians && return (Float64.(lat), Float64.(lon))
    d2r = π / 180
    return (Float64.(lat) .* d2r, Float64.(lon) .* d2r)
end

"""
    _sos_check_units(lat, lon, units)

Advisory range check. Because the default flipped from radians to degrees,
radian-valued data would otherwise be silently reinterpreted as a sliver of
the tropics — the worst possible failure mode. Warns, never throws.
"""
function _sos_check_units(lat::AbstractVector, lon::AbstractVector, units::Symbol)
    isempty(lat) && return nothing
    lat_rng = maximum(lat) - minimum(lat)
    lon_rng = maximum(lon) - minimum(lon)
    # A degenerate (single-point) extent tells us nothing either way.
    (lat_rng > 0 || lon_rng > 0) || return nothing

    if units === :degrees &&
       all(l -> abs(l) <= π / 2, lat) && all(l -> abs(l) <= π, lon)
        @warn "sos smooth: latitude/longitude are being read as DEGREES " *
              "(the default, matching mgcv), but every value lies within " *
              "|lat| ≤ π/2 and |lon| ≤ π — the range you would expect from " *
              "RADIANS. If these are radians, pass " *
              "`xt = Dict(:units => :radians)`. If they really are degrees " *
              "(a region within ~1.6° of the equator), you can ignore this." maxlog = 1
    elseif units === :radians && (any(l -> abs(l) > π / 2, lat) ||
                                  any(l -> abs(l) > π, lon))
        @warn "sos smooth: xt[:units] = :radians was requested, but some " *
              "values exceed |lat| = π/2 or |lon| = π, which are out of " *
              "range for radians. Did you mean degrees (the default)?" maxlog = 1
    end
    return nothing
end

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
    _rksos(z, tol) -> Float64

Reproducing kernel for the m=0 spline on the sphere (Wendelberger 1981), a
direct port of mgcv's C routine `rksos` (`src/misc.c:39-73`). `z` is the
cosine of the great-circle angle; the result is the unnormalised kernel (mgcv
divides by 4π afterwards).

Both branches sum `Σ xᵏ/k²`, i.e. the dilogarithm Li₂, but they terminate on
*different* quantities — the `z ≤ 0` branch breaks on the term `xx = xᵏ/k²`
while the `z > 0` branch breaks on the running power `xk` alone. That
asymmetry is reproduced deliberately: it is what mgcv computes, and matching
it is the difference between agreeing to machine precision and agreeing to
roughly `tol`.
"""
function _rksos(z::Real, tol::Float64)
    dl1 = (2 * acos(0.0))^2 / 6      # dilog(1) = π²/6, as mgcv computes it
    xi = Float64(z)
    local rk::Float64
    if xi <= 0
        xi < -1 && (xi = -1.0)
        rk = 1.0 - dl1
        xi = xi / 2 + 0.5
        xk = xi
        for k in 1:999
            xx = xk / (k * k)
            rk += xx
            xk *= xi
            xx < tol && break
        end
    else
        xi > 1 && (xi = 1.0)
        rk = xi / 2 >= 0.5 ? 1.0 : 1.0 - log(0.5 + xi / 2) * log(0.5 - xi / 2)
        xi = 0.5 - xi / 2
        xk = xi
        for k in 1:999
            xx = xk / (k * k)
            rk -= xx
            xk *= xi
            xk < tol && break
        end
    end
    return rk
end

"""
    _sos_makeR(la, lo, lak, lok, m) -> (R, T, Tc)

Port of mgcv's `makeR` (`R/smooth.r:2882-2988`). Returns the `length(la) ×
length(lak)` reproducing-kernel matrix between evaluation points and knots,
the null-space basis `T` evaluated at the evaluation points, and the
constraint basis `Tc` evaluated at the knots.

All four angle vectors are in **radians**. mgcv's `makeR` performs the
degrees→radians conversion itself at its lines 2892-2894; GAM.jl converts once
upstream (`_sos_to_radians`) and stores radians in `SOSPredictCache`, which is
equivalent and keeps the fit and predict paths on a single resolved unit.
"""
function _sos_makeR(la::AbstractVector, lo::AbstractVector,
                    lak::AbstractVector, lok::AbstractVector, m::Int)
    n, nk = length(la), length(lak)

    # Great-circle angle between each point and each knot.
    gamma = Matrix{Float64}(undef, n, nk)
    @inbounds for j in 1:nk, i in 1:n
        v = sin(la[i]) * sin(lak[j]) + cos(la[i]) * cos(lak[j]) * cos(lo[i] - lok[j])
        gamma[i, j] = acos(clamp(v, -1.0, 1.0))
    end

    ones_T() = (ones(n, 1), ones(nk, 1))

    if m == -2
        # Duchon first-derivative proposal: Euclidean 3-distance semi-kernel.
        zeps = floatmin(Float64) * 10
        R = Matrix{Float64}(undef, n, nk)
        @inbounds for j in 1:nk, i in 1:n
            R[i, j] = -max(2 * sin(gamma[i, j] / 2), zeps)
        end
        T, Tc = ones_T()
        return R, T, Tc
    elseif m == -1
        zeps = floatmin(Float64) * 10
        R = Matrix{Float64}(undef, n, nk)
        @inbounds for j in 1:nk, i in 1:n
            zz = max(2 * sin(gamma[i, j] / 2), zeps)
            R[i, j] = zz * zz * log(zz) / (8π)
        end
        # Null space is spanned by {1, x, y, z} in ambient 3-space.
        tmat(a, o) = hcat(ones(length(a)), cos.(a) .* sin.(o),
                          cos.(a) .* cos.(o), sin.(a))
        return R, tmat(la, lo), tmat(lak, lok)
    elseif m == 0
        tol = eps(Float64)
        R = Matrix{Float64}(undef, n, nk)
        @inbounds for j in 1:nk, i in 1:n
            R[i, j] = _rksos(cos(gamma[i, j]), tol) / (4π)
        end
        T, Tc = ones_T()
        return R, T, Tc
    end

    # m = 1..4: Wahba pseudospline closed forms.
    zeps = eps(Float64) * 1e-4
    R = Matrix{Float64}(undef, n, nk)
    @inbounds for j in 1:nk, i in 1:n
        z = max(1 - cos(gamma[i, j]), zeps)
        W = z / 2
        C = sqrt(W)
        A = log(1 + 1 / C)
        C *= 2
        if m == 1
            q1 = 2 * A * W - C + 1
            R[i, j] = (q1 - 1 / 2) / (2π)
        elseif m == 2
            W2 = W * W
            q2 = A * (6 * W2 - 2 * W) - 3 * C * W + 3 * W + 1 / 2
            R[i, j] = (q2 / 2 - 1 / 6) / (2π)
        elseif m == 3
            W2 = W * W
            W3 = W2 * W
            q3 = (A * (60 * W3 - 36 * W2) + 30 * W2 + C * (8 * W - 30 * W2) -
                  3 * W + 1) / 3
            R[i, j] = (q3 / 6 - 1 / 24) / (2π)
        else # m == 4
            W2 = W * W
            W3 = W2 * W
            W4 = W3 * W
            q4 = A * (70 * W4 - 60 * W3 + 6 * W2) + 35 * W3 * (1 - C) +
                 C * 55 * W2 / 3 - 12.5 * W2 - W / 3 + 1 / 4
            R[i, j] = (q4 / 24 - 1 / 120) / (2π)
        end
    end
    T, Tc = ones_T()
    return R, T, Tc
end

"""
    _sos_clamp_m(m) -> Int

mgcv's penalty-order normalisation (`R/smooth.r:3057-3060`): a missing order
becomes 0, the value is rounded, anything below -2 becomes **-1** (not -2 —
this asymmetry is mgcv's), and anything above 4 becomes 4.
"""
function _sos_clamp_m(m)
    p = m === nothing ? 0 : round(Int, m)
    p < -2 && (p = -1)
    p > 4 && (p = 4)
    return p
end

"""
    _sos_unique_locations(lat, lon) -> Vector{Int}

Indices of the unique (lat, lon) pairs in first-appearance order, matching
mgcv's `uniquecombs` (`smooth.construct.sos.smooth.spec` line 3033).
"""
function _sos_unique_locations(lat::AbstractVector, lon::AbstractVector)
    seen = Dict{Tuple{Float64,Float64},Bool}()
    idx = Int[]
    @inbounds for i in eachindex(lat)
        key = (Float64(lat[i]), Float64(lon[i]))
        if !haskey(seen, key)
            seen[key] = true
            push!(idx, i)
        end
    end
    return idx
end

function _smooth_construct(::SphericalSpline, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 2 ||
        throw(ArgumentError("Spherical spline (sos) requires exactly 2 variables " *
            "(latitude, longitude). Got: $(spec.term_vars)"))

    lat_var, lon_var = spec.term_vars
    lat_raw = Float64.(Tables.getcolumn(data, lat_var))
    lon_raw = Float64.(Tables.getcolumn(data, lon_var))

    # Degrees by default, as in mgcv; converted once here and stored in the
    # cache so prediction reuses the same resolved unit.
    units = _sos_resolve_units(spec)
    _sos_check_units(lat_raw, lon_raw, units)
    lat, lon = _sos_to_radians(lat_raw, lon_raw, units)
    n = length(lat)

    m_order = _sos_clamp_m(spec.m)
    k = spec.k

    # Knots: unique locations, subsampled to `max_knots` if there are still
    # too many (mgcv lines 3030-3053). mgcv draws its subsample with
    # `sample()` under a fixed seed; an exact RNG match with R is not
    # attainable, so a seeded MersenneTwister is used instead. For
    # n ≤ max_knots — the overwhelmingly common case — the knot set is the
    # unique locations and therefore identical to mgcv's.
    max_knots = Int(get(spec.xt, :max_knots, 2000))::Int
    seed = Int(get(spec.xt, :seed, 1))::Int
    uidx = _sos_unique_locations(lat, lon)
    if n > max_knots && length(uidx) > max_knots
        rng = MersenneTwister(seed)
        knot_idx = uidx[sort(randperm(rng, length(uidx))[1:max_knots])]
    else
        knot_idx = uidx
    end
    lat_k = lat[knot_idx]
    lon_k = lon[knot_idx]
    nk = length(knot_idx)

    # Kernel among knots, with the constraint basis at the knots.
    R_kk, _, Tc = _sos_makeR(lat_k, lon_k, lat_k, lon_k, m_order)
    R_kk = (R_kk + R_kk') / 2   # symmetrise against round-off
    p = size(Tc, 2)             # null-space dimension: 1, or 4 when m == -1

    k = min(k, nk)
    k > p || throw(ArgumentError("sos smooth: k must exceed the null-space " *
        "dimension ($p); got k=$k with $nk knots"))

    # Truncated eigendecomposition keeping the k LARGEST-MAGNITUDE eigenpairs.
    # mgcv uses `slanczos(R, k, -1)`, and `kl < 0` selects by magnitude with
    # the sign retained (R/mgcv.r, slanczos docs) — negative eigenvalues are
    # kept, not discarded. A dense symmetric eigendecomposition is used here:
    # nk is bounded by max_knots, and it avoids a second iterative solver's
    # tolerance entering the comparison.
    if k < nk
        eig = eigen(Symmetric(R_kk))
        ord = sortperm(abs.(eig.values); rev = true)[1:k]
        U = eig.vectors[:, ord]
        Dvals = eig.values[ord]
        D = Matrix{Float64}(Diagonal(Dvals))
        U1 = U' * Tc                      # k × p
    else
        U = Matrix{Float64}(I, nk, nk)
        D = R_kk
        U1 = Tc
        k = nk
    end

    # Absorb the constraint 1'Uγ = 0: with QR of U1 = [Y Z], the penalty is
    # Z'DZ and the basis map is U·Z (mgcv lines 3078-3090).
    Fqr = qr(U1)
    Qfull = Fqr.Q * Matrix{Float64}(I, k, k)
    Z = Qfull[:, (p + 1):k]               # k × (k-p)

    S_zz = Symmetric(Z' * D * Z)
    S_full = zeros(k, k)
    S_full[1:(k - p), 1:(k - p)] .= S_zz

    UZ = U * Z                            # nk × (k-p)

    # Design matrix: kernel against knots mapped through UZ, then the
    # null-space columns evaluated at the data.
    R_nk, T_data, _ = _sos_makeR(lat, lon, lat_k, lon_k, m_order)
    X_full = hcat(R_nk * UZ, T_data)      # n × k

    # Column rescaling for conditioning (mgcv lines 3104-3109). The smallest
    # standard deviation — the constant null-space column, whose sd is 0 — is
    # pinned to 1 before inversion, which is what stops a division by zero.
    xs = vec(std(X_full; dims = 1))
    xs[xs .== minimum(xs)] .= 1.0
    xs = 1.0 ./ xs
    X_full = X_full * Diagonal(xs)
    S_full = Diagonal(xs) * S_full * Diagonal(xs)
    S_full = (S_full + S_full') / 2

    null_dim = p
    pen_rank = k - p

    penalties = Matrix{Float64}[S_full]

    # Sum-to-zero identifiability constraint, as smoothCon applies to any
    # smooth without its own `C`.
    X_cons, S_cons, C, _ = absorb_constraints!(X_full, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],  # knots stored in the predict cache
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
        # NOTE on field reuse: `SOSPredictCache` predates this port, and
        # `types.jl` is owned elsewhere, so its two payload fields carry the
        # port's quantities under their old names — `U_k` holds **UZ** (the
        # nk × (k-p) basis map) and `lambda_k` holds **xs** (the per-column
        # scale factors). Renaming them to `UZ`/`xs` would be clearer.
        predict_cache = SOSPredictCache(
            Float64.(lat_k), Float64.(lon_k), Matrix{Float64}(UZ),
            Float64.(xs), m_order, k, units,
        ),
    )

    return sm
end

function _predict_matrix(::SphericalSpline, smooth::ConstructedSmooth, newdata)
    spec = smooth.spec
    lat_var, lon_var = spec.term_vars

    lat_raw = Float64.(Tables.getcolumn(newdata, lat_var))
    lon_raw = Float64.(Tables.getcolumn(newdata, lon_var))

    info = smooth.predict_cache
    info isa SOSPredictCache ||
        throw(ArgumentError("Cannot find spherical spline metadata for prediction"))

    # Use the unit resolved at CONSTRUCTION, never `spec.xt`: that is what
    # makes it impossible for the fit and predict paths to disagree.
    lat_new, lon_new = _sos_to_radians(lat_raw, lon_raw, info.units)

    lat_k = info.lat_k
    lon_k = info.lon_k
    UZ = info.U_k            # see the field-reuse note in _smooth_construct
    xs = info.lambda_k       # per-column scale factors
    m_order = info.m_order

    # Same construction as at fit time (mgcv's Predict.matrix.sos.smooth):
    # kernel against the stored knots through UZ, then the null-space columns
    # at the new points, then the stored column scaling.
    R_new, T_new, _ = _sos_makeR(lat_new, lon_new, lat_k, lon_k, m_order)
    X_full = hcat(R_new * UZ, T_new) * Diagonal(xs)

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
