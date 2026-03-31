# Duchon spline smooth — bs="ds"
#
# Duchon splines generalize thin plate splines to allow non-integer
# smoothness orders. They include TPRS as a special case.
# For simplicity, this implementation uses the same TPRS algorithm
# with configurable smoothness order m.

"""Duchon spline basis (mgcv `bs="ds"`)."""
struct DuchonSpline <: AbstractBasisType end

BASIS_TYPES[:ds] = DuchonSpline()

function _smooth_construct(::DuchonSpline, spec::SmoothSpec, data, user_knots)
    # Duchon splines with integer m reduce to TPRS
    # Delegate to TPRS with the specified m
    return _smooth_construct(ThinPlateSpline(), spec, data, user_knots)
end

function _predict_matrix(::DuchonSpline, smooth::ConstructedSmooth, newdata)
    return _predict_matrix(ThinPlateSpline(), smooth, newdata)
end

# ─── Markov Random Field smooth — bs="mrf" ───────────────────────────────

"""Markov random field smooth (mgcv `bs="mrf"`)."""
struct MarkovRandomField <: AbstractBasisType end

BASIS_TYPES[:mrf] = MarkovRandomField()

# Module-level storage for MRF region labels (keyed by objectid of ConstructedSmooth)
const _MRF_REGION_LABELS = Dict{UInt, Vector}()

"""
    _nb_to_adjacency(nb, n_regions) -> Matrix{Float64}

Convert a neighbourhood specification to an adjacency matrix.
Accepts either a symmetric matrix or a `Vector{Vector{Int}}` of neighbour lists.
"""
function _nb_to_adjacency(nb::AbstractMatrix, n_regions::Int)
    size(nb) == (n_regions, n_regions) ||
        throw(ArgumentError("Neighbourhood matrix must be $n_regions × $n_regions, " *
            "got $(size(nb))"))
    A = Float64.(nb)
    # Ensure symmetric
    if !issymmetric(A)
        A = (A .+ A') ./ 2.0
    end
    # Zero diagonal
    for i in 1:n_regions
        A[i, i] = 0.0
    end
    return A
end

function _nb_to_adjacency(nb::AbstractVector{<:AbstractVector{<:Integer}}, n_regions::Int)
    length(nb) == n_regions ||
        throw(ArgumentError("Neighbour list must have $n_regions entries, got $(length(nb))"))
    A = zeros(n_regions, n_regions)
    for (i, neighbours) in enumerate(nb)
        for j in neighbours
            1 <= j <= n_regions ||
                throw(ArgumentError("Neighbour index $j out of range [1, $n_regions]"))
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end
    return A
end

function _smooth_construct(::MarkovRandomField, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 1 ||
        throw(ArgumentError("MRF smooth requires at least one variable"))

    haskey(spec.xt, :nb) ||
        throw(ArgumentError("MRF smooth requires a neighbourhood matrix. " *
            "Pass xt=Dict(:nb => adjacency_matrix) to s()."))

    var = spec.term_vars[1]
    col = Tables.getcolumn(data, var)
    levels = sort(unique(col))
    n_regions = length(levels)
    n = length(col)

    # Build level → index mapping
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Convert neighbourhood to adjacency matrix
    nb = spec.xt[:nb]
    A = _nb_to_adjacency(nb, n_regions)

    # k = n_regions unless user specified smaller
    k = spec.k > 0 ? min(spec.k, n_regions) : n_regions

    # Build indicator/dummy matrix (n × n_regions)
    X = zeros(n, n_regions)
    for i in 1:n
        j = level_map[col[i]]
        X[i, j] = 1.0
    end

    # Build penalty: graph Laplacian S = D - A
    D = Diagonal(vec(sum(A; dims = 2)))
    S_pen = Matrix{Float64}(D - A)
    penalties = Matrix{Float64}[S_pen]

    null_dim = 1  # constant vector is in the null space
    pen_rank = n_regions - 1

    # Apply sum-to-zero constraint (absorb like other smooths)
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64.(1:n_regions),  # dummy knots (region indices)
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )

    # Store region labels for prediction
    _MRF_REGION_LABELS[objectid(sm)] = levels

    return sm
end

function _predict_matrix(::MarkovRandomField, smooth::ConstructedSmooth, newdata)
    var = smooth.spec.term_vars[1]
    col = Tables.getcolumn(newdata, var)
    n_new = length(col)

    # Retrieve stored region labels
    levels = get(_MRF_REGION_LABELS, objectid(smooth), nothing)
    n_regions = length(smooth.knots)

    if levels === nothing
        # Fallback: use integer indices 1:n_regions
        levels = collect(1:n_regions)
    end

    level_map = Dict(lev => i for (i, lev) in enumerate(levels))

    # Build indicator matrix for new data
    X = zeros(n_new, n_regions)
    for i in 1:n_new
        j = get(level_map, col[i], nothing)
        if j !== nothing
            X[i, j] = 1.0
        end
        # Unknown regions get zero rows (no contribution)
    end

    # Apply same constraint as training
    if smooth.constraint !== nothing
        C = smooth.constraint
        Z = _constraint_basis(C, size(X, 2))
        return X * Z
    end
    return X
end

# ─── Factor-smooth interaction — bs="fs" ─────────────────────────────────
#
# A factor-smooth interaction creates a separate copy of a smooth basis for
# each level of a factor variable, with the smoothing parameter(s) shared
# across all levels.  This allows group-specific smooth curves regularized
# to a common degree of smoothness.
#
# Convention: s(x, group, bs=:fs, k=10)
#   - term_vars = [:x, :group]  — last variable is the factor
#   - The marginal smooth is built (with constraint) for the continuous variable(s)
#   - Model matrix is block-diagonal: one block per factor level
#   - Penalty matrices are block-diagonal replications of the marginal penalty

"""Factor-smooth interaction basis (mgcv `bs="fs"`)."""
struct FactorSmooth <: AbstractBasisType end

BASIS_TYPES[:fs] = FactorSmooth()

"""
    FactorSmoothInfo

Metadata for a factor-smooth interaction, stored for prediction.
"""
struct FactorSmoothInfo
    levels::Vector{Any}
    marginal_smooth::ConstructedSmooth
    factor_var::Symbol
end

# Module-level storage for factor smooth metadata (keyed by objectid)
const _FS_INFO = Dict{UInt, FactorSmoothInfo}()

function _smooth_construct(::FactorSmooth, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) >= 2 ||
        throw(ArgumentError("Factor-smooth interactions require at least 2 variables: " *
            "continuous variable(s) and a grouping factor. Got: $(spec.term_vars)"))

    # Last variable is the factor, rest are continuous
    factor_var = spec.term_vars[end]
    cont_vars = spec.term_vars[1:end-1]

    factor_col = Tables.getcolumn(data, factor_var)
    levels = sort(unique(factor_col))
    L = length(levels)
    n = length(factor_col)

    # Build marginal spec for the continuous variables using TPRS (R's default)
    marginal_spec = SmoothSpec(
        cont_vars, ThinPlateSpline(), spec.k,
        nothing, spec.id, spec.sp, spec.fx, spec.m,
        "s($(join(cont_vars, ",")),bs=tp)",
    )

    # Construct the marginal smooth WITH constraint absorption.
    # The sum-to-zero constraint removes one column (the global constant direction),
    # preventing linear dependence with the model intercept.
    marginal_sm = _smooth_construct(ThinPlateSpline(), marginal_spec, data, user_knots)
    X_marginal = marginal_sm.X    # n × k_eff (after constraint)
    k_eff = size(X_marginal, 2)

    # Block-diagonal model matrix: one block per factor level
    total_cols = L * k_eff
    X = zeros(n, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(levels))
    for i in 1:n
        l = level_map[factor_col[i]]
        col_offset = (l - 1) * k_eff
        @inbounds for j in 1:k_eff
            X[i, col_offset + j] = X_marginal[i, j]
        end
    end

    # Block-diagonal penalties: replicate each marginal penalty L times
    penalties = Matrix{Float64}[]
    for S_j in marginal_sm.S
        S_fs = zeros(total_cols, total_cols)
        for l in 1:L
            rng = ((l - 1) * k_eff + 1):(l * k_eff)
            S_fs[rng, rng] .= S_j
        end
        push!(penalties, S_fs)
    end

    # The constrained marginal has penalty rank = marginal_sm.rank.
    # Each level contributes k_eff columns with marginal_sm.rank penalized.
    pen_rank = L * marginal_sm.rank
    null_dim = total_cols - pen_rank

    sm = ConstructedSmooth(
        spec, X, penalties,
        marginal_sm.knots,
        null_dim, pen_rank,
        nothing, nothing, 0, 0,   # no additional constraint on the full fs smooth
        nothing, nothing, nothing,
        Int[],
    )

    _FS_INFO[objectid(sm)] = FactorSmoothInfo(levels, marginal_sm, factor_var)
    return sm
end

function _predict_matrix(::FactorSmooth, smooth::ConstructedSmooth, newdata)
    info = get(_FS_INFO, objectid(smooth), nothing)
    info !== nothing ||
        throw(ArgumentError("Cannot find factor smooth metadata for prediction"))

    factor_col = Tables.getcolumn(newdata, info.factor_var)
    n_new = length(factor_col)

    # Predict marginal at new data (handles constraint absorption automatically)
    marginal_sm = info.marginal_smooth
    X_marginal = _predict_matrix(marginal_sm.spec.basis, marginal_sm, newdata)
    k_eff = size(X_marginal, 2)
    L = length(info.levels)
    total_cols = L * k_eff

    X = zeros(n_new, total_cols)
    level_map = Dict(lev => i for (i, lev) in enumerate(info.levels))
    for i in 1:n_new
        l = get(level_map, factor_col[i], 0)
        if l > 0
            col_offset = (l - 1) * k_eff
            @inbounds for j in 1:k_eff
                X[i, col_offset + j] = X_marginal[i, j]
            end
        end
        # Unknown levels get zero rows → no contribution to prediction
    end

    return X
end

# ─── Soap Film smooth — bs="so" ──────────────────────────────────────────
#
# A 2D smooth that respects an irregular boundary, following Wood et al.
# (2008). The basis decomposes into:
#   1. Boundary film: cyclic spline along the boundary, extended into the
#      interior by solving the Laplace equation (∇²f = 0).
#   2. Interior wiggly: Green's-function-like basis from point sources at
#      interior knots, also via the discrete Laplacian.
#
# Reference: Wood, Bravington & Hedley (2008) "Soap film smoothing",
#            Journal of the Royal Statistical Society B, 70(5), 931-955.

"""Soap film smooth basis (mgcv `bs="so"`)."""
struct SoapFilm <: AbstractBasisType end

BASIS_TYPES[:so] = SoapFilm()

# Module-level storage for soap film prediction data (keyed by smooth label)
const _SOAP_PREDICT_DATA = Dict{String, Dict{Symbol, Any}}()

# ── Point-in-polygon (ray casting) ───────────────────────────────────────

"""
    _point_in_polygon(px, py, poly_x, poly_y) -> Bool

Ray-casting point-in-polygon test.
"""
function _point_in_polygon(px::Real, py::Real,
                           poly_x::AbstractVector, poly_y::AbstractVector)
    n = length(poly_x)
    inside = false
    j = n
    @inbounds for i in 1:n
        if ((poly_y[i] > py) != (poly_y[j] > py)) &&
           (px < (poly_x[j] - poly_x[i]) * (py - poly_y[i]) /
                 (poly_y[j] - poly_y[i]) + poly_x[i])
            inside = !inside
        end
        j = i
    end
    return inside
end

"""
    _in_soap_domain(px, py, bnd) -> Bool

Check whether `(px, py)` is inside the domain defined by boundary loops.
First loop is the outer boundary; subsequent loops are holes.
"""
function _in_soap_domain(px::Real, py::Real, bnd::Vector{Matrix{Float64}})
    _point_in_polygon(px, py, bnd[1][:, 1], bnd[1][:, 2]) || return false
    for i in 2:length(bnd)
        _point_in_polygon(px, py, bnd[i][:, 1], bnd[i][:, 2]) && return false
    end
    return true
end

# ── Closest point on polygon ─────────────────────────────────────────────

"""
    _closest_on_polygon(px, py, poly_x, poly_y) -> (dist, arc_length, total_length)

Find the closest point on a closed polygon to `(px, py)`.
Returns the distance, the arc-length parameter at the closest point,
and the total arc-length of the polygon.
"""
function _closest_on_polygon(px::Real, py::Real,
                             poly_x::AbstractVector, poly_y::AbstractVector)
    nv = length(poly_x)
    cum_len = zeros(nv)
    for i in 2:nv
        cum_len[i] = cum_len[i - 1] +
            sqrt((poly_x[i] - poly_x[i - 1])^2 + (poly_y[i] - poly_y[i - 1])^2)
    end
    close_edge = sqrt((poly_x[1] - poly_x[nv])^2 + (poly_y[1] - poly_y[nv])^2)
    total_len = cum_len[end] + close_edge

    best_dist = Inf
    best_arc  = 0.0

    @inbounds for i in 1:nv
        next_i = i < nv ? i + 1 : 1
        edx = poly_x[next_i] - poly_x[i]
        edy = poly_y[next_i] - poly_y[i]
        elen = sqrt(edx^2 + edy^2)
        elen < 1e-15 && continue

        t = clamp(((px - poly_x[i]) * edx + (py - poly_y[i]) * edy) / (elen^2),
                  0.0, 1.0)
        cx = poly_x[i] + t * edx
        cy = poly_y[i] + t * edy
        dist = sqrt((px - cx)^2 + (py - cy)^2)

        if dist < best_dist
            best_dist = dist
            best_arc  = cum_len[i] + t * elen
        end
    end
    return best_dist, best_arc, total_len
end

# ── Bilinear interpolation on a grid ─────────────────────────────────────

"""
    _soap_bilinear(grid, inside, px, py, x0, y0, dx, dy, nx, ny) -> Float64

Bilinear interpolation of `grid` (ny × nx) at `(px, py)`.
Returns `NaN` when the point is outside the domain.
"""
function _soap_bilinear(grid::AbstractMatrix{Float64}, inside::BitMatrix,
                        px::Real, py::Real,
                        x0::Float64, y0::Float64,
                        dx::Float64, dy::Float64,
                        nx::Int, ny::Int)
    fi = (px - x0) / dx + 1.0   # 1-based column
    fj = (py - y0) / dy + 1.0   # 1-based row

    i_lo = floor(Int, fi);  i_hi = i_lo + 1
    j_lo = floor(Int, fj);  j_hi = j_lo + 1

    i_lo = clamp(i_lo, 1, nx);  i_hi = clamp(i_hi, 1, nx)
    j_lo = clamp(j_lo, 1, ny);  j_hi = clamp(j_hi, 1, ny)

    s = clamp(fi - floor(fi), 0.0, 1.0)
    t = clamp(fj - floor(fj), 0.0, 1.0)

    # Fast path: all four corners inside
    if inside[j_lo, i_lo] && inside[j_hi, i_lo] &&
       inside[j_lo, i_hi] && inside[j_hi, i_hi]
        return (1 - s) * (1 - t) * grid[j_lo, i_lo] +
                    s  * (1 - t) * grid[j_lo, i_hi] +
               (1 - s) *      t  * grid[j_hi, i_lo] +
                    s  *      t  * grid[j_hi, i_hi]
    end

    # Slow path: weighted average over inside corners only
    val = 0.0;  w = 0.0
    for (jj, wj) in ((j_lo, 1.0 - t), (j_hi, t))
        for (ii, wi) in ((i_lo, 1.0 - s), (i_hi, s))
            if inside[jj, ii]
                wt = wi * wj
                val += wt * grid[jj, ii]
                w   += wt
            end
        end
    end
    return w > 0 ? val / w : NaN
end

# ── Main soap film constructor ───────────────────────────────────────────

function _smooth_construct(::SoapFilm, spec::SmoothSpec, data, user_knots)
    length(spec.term_vars) == 2 ||
        throw(ArgumentError("Soap film smooth requires exactly 2 variables"))
    haskey(spec.xt, :bnd) ||
        throw(ArgumentError("Soap film smooth requires boundary via " *
            "xt=Dict(:bnd => [boundary_matrix])"))

    # ── Unpack inputs ────────────────────────────────────────────────────
    bnd = Matrix{Float64}[Float64.(b) for b in spec.xt[:bnd]]
    nmax = Int(get(spec.xt, :nmax, 200))

    x = Float64.(Tables.getcolumn(data, spec.term_vars[1]))
    y = Float64.(Tables.getcolumn(data, spec.term_vars[2]))
    n = length(x)

    # ── Grid setup ───────────────────────────────────────────────────────
    all_bx = vcat([b[:, 1] for b in bnd]...)
    all_by = vcat([b[:, 2] for b in bnd]...)
    x_lo, x_hi = extrema(all_bx)
    y_lo, y_hi = extrema(all_by)
    x_range = x_hi - x_lo
    y_range = y_hi - y_lo

    if x_range >= y_range
        dx = x_range / max(nmax - 1, 1)
        nx = nmax
        ny = max(2, ceil(Int, y_range / dx) + 1)
    else
        dx = y_range / max(nmax - 1, 1)
        ny = nmax
        nx = max(2, ceil(Int, x_range / dx) + 1)
    end
    dy = dx                   # square cells
    x0 = x_lo - dx           # one-cell padding
    y0 = y_lo - dy
    nx += 2;  ny += 2

    # ── Classify grid points ─────────────────────────────────────────────
    inside = falses(ny, nx)
    for j in 1:ny, i in 1:nx
        gx = x0 + (i - 1) * dx
        gy = y0 + (j - 1) * dy
        inside[j, i] = _in_soap_domain(gx, gy, bnd)
    end

    is_boundary = falses(ny, nx)
    is_interior = falses(ny, nx)
    for j in 1:ny, i in 1:nx
        inside[j, i] || continue
        has_ext = false
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ni, nj = i + di, j + dj
            if ni < 1 || ni > nx || nj < 1 || nj > ny || !inside[nj, ni]
                has_ext = true; break
            end
        end
        if has_ext
            is_boundary[j, i] = true
        else
            is_interior[j, i] = true
        end
    end

    # Index interior grid points 1..ng
    G = zeros(Int, ny, nx)
    ng = 0
    for j in 1:ny, i in 1:nx
        if is_interior[j, i]
            ng += 1
            G[j, i] = ng
        end
    end
    ng > 0 || throw(ArgumentError(
        "No interior grid points. Boundary may be too small or nmax too low."))

    # ── Sparse 5-point Laplacian on interior ─────────────────────────────
    II = Int[];  JJ = Int[];  VV = Float64[]
    sizehint!(II, 5 * ng);  sizehint!(JJ, 5 * ng);  sizehint!(VV, 5 * ng)

    for j in 1:ny, i in 1:nx
        idx = G[j, i]
        idx == 0 && continue
        push!(II, idx); push!(JJ, idx); push!(VV, -4.0)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ni, nj = i + di, j + dj
            (1 <= ni <= nx && 1 <= nj <= ny) || continue
            nidx = G[nj, ni]
            if nidx > 0
                push!(II, idx); push!(JJ, nidx); push!(VV, 1.0)
            end
        end
    end
    L = sparse(II, JJ, VV, ng, ng)
    L_lu = lu(L)

    # ── Boundary arc-length parameterisation ─────────────────────────────
    outer_x = bnd[1][:, 1]
    outer_y = bnd[1][:, 2]
    nv_bnd  = size(bnd[1], 1)

    cum_arc = zeros(nv_bnd)
    for i in 2:nv_bnd
        cum_arc[i] = cum_arc[i - 1] +
            sqrt((outer_x[i] - outer_x[i - 1])^2 +
                 (outer_y[i] - outer_y[i - 1])^2)
    end
    total_arc = cum_arc[end] +
        sqrt((outer_x[1] - outer_x[end])^2 + (outer_y[1] - outer_y[end])^2)

    # Assign arc-lengths to boundary grid points
    bnd_grid_ij  = Tuple{Int,Int}[]
    bnd_arc_vals = Float64[]
    for j in 1:ny, i in 1:nx
        is_boundary[j, i] || continue
        gx = x0 + (i - 1) * dx
        gy = y0 + (j - 1) * dy
        _, arc, _ = _closest_on_polygon(gx, gy, outer_x, outer_y)
        push!(bnd_grid_ij, (i, j))
        push!(bnd_arc_vals, arc)
    end
    n_bnd_grid = length(bnd_grid_ij)

    # Boundary basis dimension
    k_bnd = min(max(spec.k ÷ 3, 6), n_bnd_grid - 1, nv_bnd)
    k_int = max(spec.k - k_bnd, 1)

    # Cyclic spline on [0, total_arc]
    bnd_knots = collect(range(0.0, total_arc; length = k_bnd + 1))
    X_bnd_grid, S_bnd = _cc_basis(bnd_arc_vals, bnd_knots)
    k_bnd_cc = size(X_bnd_grid, 2)          # == k_bnd

    # Fast lookup: (i,j) → index in bnd_grid_ij
    bnd_ij_map = Dict{Tuple{Int,Int}, Int}()
    for (bi, ij) in enumerate(bnd_grid_ij)
        bnd_ij_map[ij] = bi
    end

    # ── Solve PDE for each boundary basis function ───────────────────────
    grid_bnd = zeros(ny, nx, k_bnd_cc)

    for col in 1:k_bnd_cc
        # Prescribe boundary values
        bval = zeros(ny, nx)
        for (bi, (gi, gj)) in enumerate(bnd_grid_ij)
            bval[gj, gi] = X_bnd_grid[bi, col]
        end
        # RHS from boundary contributions
        rhs = zeros(ng)
        for j in 1:ny, i in 1:nx
            idx = G[j, i]; idx == 0 && continue
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ni, nj = i + di, j + dj
                (1 <= ni <= nx && 1 <= nj <= ny) || continue
                if is_boundary[nj, ni]
                    rhs[idx] -= bval[nj, ni]
                end
            end
        end
        u = L_lu \ rhs
        # Fill grid with PDE solution
        for j in 1:ny, i in 1:nx
            if is_interior[j, i]
                grid_bnd[j, i, col] = u[G[j, i]]
            elseif is_boundary[j, i]
                bi = get(bnd_ij_map, (i, j), 0)
                if bi > 0
                    grid_bnd[j, i, col] = X_bnd_grid[bi, col]
                end
            end
        end
    end

    # ── Interior knot placement ──────────────────────────────────────────
    interior_ij = [(i, j) for j in 1:ny for i in 1:nx if is_interior[j, i]]

    if length(interior_ij) <= k_int
        knot_ij = interior_ij
    else
        step = length(interior_ij) / k_int
        idxs = [clamp(round(Int, (i - 0.5) * step) + 1, 1, length(interior_ij))
                for i in 1:k_int]
        knot_ij = interior_ij[unique(idxs)]
    end
    k_int_actual = length(knot_ij)

    # ── Solve PDE for interior (wiggly) basis ────────────────────────────
    grid_int = zeros(ny, nx, k_int_actual)
    g_mat    = zeros(ng, k_int_actual)        # for penalty construction

    for ki in 1:k_int_actual
        gi, gj = knot_ij[ki]
        knot_idx = G[gj, gi]

        # Delta forcing → first solve
        rhs1 = zeros(ng);  rhs1[knot_idx] = 1.0
        u1 = L_lu \ rhs1
        mx1 = maximum(abs, u1)
        mx1 > 0 && (u1 ./= mx1)

        # Second solve for smoother basis
        u2 = L_lu \ u1
        mx2 = maximum(abs, u2)
        mx2 > 0 && (u2 ./= mx2)

        g_mat[:, ki] .= u2

        for j in 1:ny, i in 1:nx
            is_interior[j, i] || continue
            grid_int[j, i, ki] = u2[G[j, i]]
        end
    end

    # ── Evaluate basis at data points (bilinear interpolation) ───────────
    p = k_bnd_cc + k_int_actual
    X = zeros(n, p)

    for col in 1:k_bnd_cc
        g = @view grid_bnd[:, :, col]
        for i in 1:n
            X[i, col] = _soap_bilinear(g, inside, x[i], y[i],
                                       x0, y0, dx, dy, nx, ny)
        end
    end
    for col in 1:k_int_actual
        g = @view grid_int[:, :, col]
        for i in 1:n
            X[i, k_bnd_cc + col] = _soap_bilinear(g, inside, x[i], y[i],
                                                   x0, y0, dx, dy, nx, ny)
        end
    end
    replace!(X, NaN => 0.0)

    # ── Column scaling for conditioning ──────────────────────────────────
    irng = zeros(p)
    for j in 1:p
        lo, hi = extrema(@view X[:, j])
        rng = hi - lo
        irng[j] = rng > 0 ? 1.0 / rng : 1.0
    end
    X .= X .* irng'

    # ── Penalty matrices ─────────────────────────────────────────────────
    # 1. Boundary: cyclic-spline wiggliness penalty
    S_bnd_full = zeros(p, p)
    for a in 1:k_bnd_cc, b in 1:k_bnd_cc
        S_bnd_full[a, b] = S_bnd[a, b] * irng[a] * irng[b]
    end

    # 2. Interior: Gram matrix of PDE solutions (approximates ∫∫|∇²f|²)
    S_int = g_mat' * g_mat * (dx * dy)
    S_int_full = zeros(p, p)
    for a in 1:k_int_actual, b in 1:k_int_actual
        S_int_full[k_bnd_cc + a, k_bnd_cc + b] =
            S_int[a, b] * irng[k_bnd_cc + a] * irng[k_bnd_cc + b]
    end

    # Symmetrize to fix floating-point round-off
    S_bnd_full .= (S_bnd_full .+ S_bnd_full') ./ 2
    S_int_full .= (S_int_full .+ S_int_full') ./ 2
    penalties = Matrix{Float64}[S_bnd_full, S_int_full]

    # ── Cache for prediction ─────────────────────────────────────────────
    grid_basis = cat(grid_bnd, grid_int; dims = 3)
    _SOAP_PREDICT_DATA[spec.label] = Dict{Symbol, Any}(
        :bnd    => bnd,
        :x0     => x0,  :y0  => y0,
        :dx     => dx,  :dy  => dy,
        :nx     => nx,  :ny  => ny,
        :inside => inside,
        :grid_basis => grid_basis,
        :irng   => irng,
    )

    # ── Absorb identifiability constraints ───────────────────────────────
    null_dim = 1
    pen_rank = p - null_dim
    X_cons, S_cons, C, _ = absorb_constraints!(X, penalties)

    return ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],          # no 1-D knot vector
        null_dim, pen_rank,
        C, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )
end

# ── Prediction ───────────────────────────────────────────────────────────

function _predict_matrix(::SoapFilm, smooth::ConstructedSmooth, newdata)
    sd = get(_SOAP_PREDICT_DATA, smooth.spec.label, nothing)
    sd === nothing && throw(ArgumentError(
        "No cached soap-film data for '$(smooth.spec.label)'. " *
        "Was the smooth constructed in this session?"))

    xv = Float64.(Tables.getcolumn(newdata, smooth.spec.term_vars[1]))
    yv = Float64.(Tables.getcolumn(newdata, smooth.spec.term_vars[2]))
    nn = length(xv)

    bnd    = sd[:bnd]::Vector{Matrix{Float64}}
    x0     = sd[:x0]::Float64;   y0  = sd[:y0]::Float64
    dxg    = sd[:dx]::Float64;   dyg = sd[:dy]::Float64
    nxg    = sd[:nx]::Int;       nyg = sd[:ny]::Int
    ins    = sd[:inside]::BitMatrix
    gbasis = sd[:grid_basis]::Array{Float64, 3}   # ny × nx × p
    irng   = sd[:irng]::Vector{Float64}
    p      = size(gbasis, 3)

    X = zeros(nn, p)
    for col in 1:p
        g = @view gbasis[:, :, col]
        for i in 1:nn
            if _in_soap_domain(xv[i], yv[i], bnd)
                X[i, col] = _soap_bilinear(g, ins, xv[i], yv[i],
                                           x0, y0, dxg, dyg, nxg, nyg)
            else
                X[i, col] = NaN
            end
        end
    end
    replace!(X, NaN => 0.0)

    # Apply column scaling
    X .= X .* irng'

    # Apply constraint
    if smooth.constraint !== nothing
        Z = _constraint_basis(smooth.constraint, size(X, 2))
        return X * Z
    end
    return X
end
