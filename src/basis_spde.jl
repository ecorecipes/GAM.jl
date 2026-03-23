# SPDE (Stochastic Partial Differential Equation) Matérn smooth — bs="spde"
#
# The SPDE approach (Lindgren et al., 2011; Miller, Glennie & Seaton, 2020)
# represents a Matérn covariance function as a basis-penalty smoother using
# finite element method (FEM) matrices.
#
# For a 1D mesh with piecewise linear basis functions:
#   X = A (interpolation/projection matrix from mesh nodes to data locations)
#   Penalty matrices: S₁ = C (mass), S₂ = 2G₁ (scaled stiffness), S₃ = G₂
#   where G₂ = G₁ C⁻¹ G₁.
#
# The exact Matérn parameterization couples these via (τ, κ):
#   Q = τ²(κ⁴·C + 2κ²·G₁ + G₂)
#
# Our framework fits with 3 independent smoothing parameters (one per penalty
# matrix), which is a MORE flexible relaxation—each λⱼ is estimated by REML,
# allowing the data to determine the effective Matérn correlation structure.
# The L matrix for the exact coupling is stored in xt[:L] for reference.
#
# For 2D, users can supply pre-computed FEM matrices via xt (e.g. from INLA),
# or use the built-in regular grid triangulation.
#
# Reference: https://github.com/dill/SPDE-smoothing
# Reference: Miller, Glennie & Seaton (2020), JABES 25, 1-21.

"""SPDE Matérn smooth basis (`bs=:spde`)."""
struct SPDESmooth <: AbstractBasisType end

BASIS_TYPES[:spde] = SPDESmooth()

# Module-level storage for prediction metadata (mesh nodes, dimension info)
const _SPDE_INFO = Dict{UInt, Dict{Symbol, Any}}()

# ---------------------------------------------------------------------------
# 1D FEM helpers
# ---------------------------------------------------------------------------

"""
    _fem_mass_matrix_1d(t) -> Matrix{Float64}

Assemble the FEM mass matrix C for piecewise linear basis functions on a 1D
mesh with node positions `t`.  C[i,j] = ∫ φᵢ(x) φⱼ(x) dx.
"""
function _fem_mass_matrix_1d(t::AbstractVector{Float64})
    k = length(t)
    h = diff(t)
    C = zeros(k, k)
    @inbounds for i in 1:(k - 1)
        C[i, i] += h[i] / 3
        C[i + 1, i + 1] += h[i] / 3
        C[i, i + 1] += h[i] / 6
        C[i + 1, i] += h[i] / 6
    end
    return C
end

"""
    _fem_stiffness_matrix_1d(t) -> Matrix{Float64}

Assemble the FEM stiffness matrix G₁ for piecewise linear basis functions on a
1D mesh with node positions `t`.  G₁[i,j] = ∫ φᵢ'(x) φⱼ'(x) dx.
"""
function _fem_stiffness_matrix_1d(t::AbstractVector{Float64})
    k = length(t)
    h = diff(t)
    G = zeros(k, k)
    @inbounds for i in 1:(k - 1)
        inv_h = 1.0 / h[i]
        G[i, i] += inv_h
        G[i + 1, i + 1] += inv_h
        G[i, i + 1] -= inv_h
        G[i + 1, i] -= inv_h
    end
    return G
end

"""
    _fem_interpolation_matrix_1d(x, t) -> Matrix{Float64}

Build the FEM interpolation (projection) matrix A that maps mesh node values
to observation locations via piecewise linear interpolation.

For each data point xⱼ in mesh interval [tᵢ, tᵢ₊₁]:
  A[j, i]   = (tᵢ₊₁ - xⱼ) / (tᵢ₊₁ - tᵢ)
  A[j, i+1] = (xⱼ - tᵢ)   / (tᵢ₊₁ - tᵢ)
"""
function _fem_interpolation_matrix_1d(x::AbstractVector{Float64},
                                       t::AbstractVector{Float64})
    n = length(x)
    k = length(t)
    h = diff(t)
    A = zeros(n, k)
    @inbounds for j in 1:n
        idx = searchsortedlast(t, x[j])
        idx = clamp(idx, 1, k - 1)
        w = (x[j] - t[idx]) / h[idx]
        A[j, idx] = 1.0 - w
        A[j, idx + 1] = w
    end
    return A
end

# ---------------------------------------------------------------------------
# 2D FEM helpers (regular grid triangulation)
# ---------------------------------------------------------------------------

"""
    _fem_matrices_2d_grid(x, y, nx, ny) -> (A, C, G1, G2, nodes_x, nodes_y)

Build 2D FEM matrices on a regular grid triangulation.  The bounding box of
(x, y) is extended by 5 % on each side, divided into nx × ny cells, and each
cell is split into two triangles (lower-left and upper-right).

Returns the interpolation matrix A, mass matrix C, stiffness matrix G1,
second-order stiffness G2 = G1 * inv(C) * G1, and the node coordinate
vectors.
"""
function _fem_matrices_2d_grid(x::AbstractVector{Float64},
                                y::AbstractVector{Float64},
                                nx::Int, ny::Int)
    n = length(x)
    xlo, xhi = extrema(x)
    ylo, yhi = extrema(y)
    dx = xhi - xlo
    dy = yhi - ylo
    # Extend domain by 5 % on each side
    xlo -= 0.05 * dx; xhi += 0.05 * dx
    ylo -= 0.05 * dy; yhi += 0.05 * dy

    xs = range(xlo, xhi; length = nx)
    ys = range(ylo, yhi; length = ny)
    hx = step(xs)
    hy = step(ys)
    nk = nx * ny  # total nodes

    # Node indexing: node(ix, iy) = (iy - 1) * nx + ix
    node_idx(ix, iy) = (iy - 1) * nx + ix
    nodes_x = Float64[xs[ix] for iy in 1:ny for ix in 1:nx]
    nodes_y = Float64[ys[iy] for iy in 1:ny for ix in 1:nx]
    # fix ordering: node_idx iterates ix fast, iy slow
    nodes_x2 = zeros(nk)
    nodes_y2 = zeros(nk)
    for iy in 1:ny, ix in 1:nx
        ni = node_idx(ix, iy)
        nodes_x2[ni] = xs[ix]
        nodes_y2[ni] = ys[iy]
    end

    # Build triangles: each cell (ix, iy) → 2 triangles
    triangles = Vector{NTuple{3,Int}}()
    sizehint!(triangles, 2 * (nx - 1) * (ny - 1))
    for iy in 1:(ny - 1), ix in 1:(nx - 1)
        n1 = node_idx(ix, iy)
        n2 = node_idx(ix + 1, iy)
        n3 = node_idx(ix, iy + 1)
        n4 = node_idx(ix + 1, iy + 1)
        push!(triangles, (n1, n2, n3))  # lower-left triangle
        push!(triangles, (n2, n4, n3))  # upper-right triangle
    end

    # Assemble FEM matrices (piecewise linear on triangles)
    C = zeros(nk, nk)
    G1 = zeros(nk, nk)

    for (i1, i2, i3) in triangles
        # Triangle area via cross product
        x1, y1 = nodes_x2[i1], nodes_y2[i1]
        x2, y2 = nodes_x2[i2], nodes_y2[i2]
        x3, y3 = nodes_x2[i3], nodes_y2[i3]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        area < 1e-15 && continue

        # Mass matrix contributions:  ∫ φᵢ φⱼ = area/12 (i≠j), area/6 (i=j)
        idx = (i1, i2, i3)
        for a in 1:3, b in 1:3
            if a == b
                C[idx[a], idx[b]] += area / 6.0
            else
                C[idx[a], idx[b]] += area / 12.0
            end
        end

        # Stiffness matrix: ∫ ∇φᵢ · ∇φⱼ
        # Gradients of barycentric coordinates on the triangle
        denom = 2.0 * area
        g1 = [(y2 - y3) / denom, (x3 - x2) / denom]
        g2 = [(y3 - y1) / denom, (x1 - x3) / denom]
        g3 = [(y1 - y2) / denom, (x2 - x1) / denom]
        grads = (g1, g2, g3)

        for a in 1:3, b in 1:3
            G1[idx[a], idx[b]] += area * (grads[a][1] * grads[b][1] +
                                           grads[a][2] * grads[b][2])
        end
    end

    # Ensure exact symmetry
    C = (C + C') / 2
    G1 = (G1 + G1') / 2

    # Second-order stiffness: G2 = G1 * C⁻¹ * G1
    # Use a regularized inverse (C can be singular at boundary nodes)
    C_reg = C + 1e-10 * I
    G2 = G1 * (C_reg \ G1)
    G2 = (G2 + G2') / 2  # symmetrize

    # Interpolation matrix A (barycentric coordinates)
    A = zeros(n, nk)
    for j in 1:n
        xj, yj = x[j], y[j]
        best_tri = 0
        best_lam = (0.0, 0.0, 0.0)
        best_dist = Inf

        for (ti, (i1, i2, i3)) in enumerate(triangles)
            x1, y1 = nodes_x2[i1], nodes_y2[i1]
            x2, y2 = nodes_x2[i2], nodes_y2[i2]
            x3, y3 = nodes_x2[i3], nodes_y2[i3]

            det_T = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            abs(det_T) < 1e-15 && continue

            l1 = ((x2 - x3) * (yj - y3) + (y3 - y2) * (xj - x3)) / det_T
            l2 = ((x3 - x1) * (yj - y1) + (y1 - y3) * (xj - x1)) / det_T
            l3 = 1.0 - l1 - l2

            if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10
                # Point is inside (or on boundary of) this triangle
                A[j, i1] = max(l1, 0.0)
                A[j, i2] = max(l2, 0.0)
                A[j, i3] = max(l3, 0.0)
                best_tri = ti
                break
            else
                # Track nearest triangle for points just outside domain
                dist = max(-l1, 0.0) + max(-l2, 0.0) + max(-l3, 0.0)
                if dist < best_dist
                    best_dist = dist
                    best_tri = ti
                    best_lam = (l1, l2, l3)
                end
            end
        end

        if best_tri > 0 && A[j, :] == zeros(nk)
            # Clamp to nearest triangle
            i1, i2, i3 = triangles[best_tri]
            l1 = max(best_lam[1], 0.0)
            l2 = max(best_lam[2], 0.0)
            l3 = max(best_lam[3], 0.0)
            s = l1 + l2 + l3
            if s > 0
                A[j, i1] = l1 / s
                A[j, i2] = l2 / s
                A[j, i3] = l3 / s
            else
                A[j, i1] = 1.0 / 3
                A[j, i2] = 1.0 / 3
                A[j, i3] = 1.0 / 3
            end
        end
    end

    return A, C, G1, G2, nodes_x2, nodes_y2
end

# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

function _smooth_construct(::SPDESmooth, spec::SmoothSpec, data, user_knots)
    dim = length(spec.term_vars)
    dim ∈ (1, 2) || throw(ArgumentError(
        "SPDE smooth supports 1D or 2D. Got $(dim)D: $(spec.term_vars)"))

    k = spec.k

    # Check for user-supplied FEM matrices via xt
    has_precomputed = all(haskey(spec.xt, s) for s in (:C, :G1, :G2, :A))

    if has_precomputed
        # Use pre-computed FEM matrices (e.g. from INLA for 2D meshes)
        A = Float64.(spec.xt[:A])
        C_fem = Float64.(spec.xt[:C])
        G1_fem = Float64.(spec.xt[:G1])
        G2_fem = Float64.(spec.xt[:G2])
        nk = size(C_fem, 1)
        mesh_nodes = get(spec.xt, :mesh_nodes, Float64[])
        info = Dict{Symbol, Any}(:dim => dim, :precomputed => true,
                                  :nk => nk, :mesh_nodes => mesh_nodes)
    elseif dim == 1
        var = spec.term_vars[1]
        x = Float64.(Tables.getcolumn(data, var))
        n = length(x)
        k = max(k, 5)  # minimum mesh nodes

        # Build 1D mesh: evenly spaced nodes covering the data range
        t = collect(range(minimum(x), maximum(x); length = k))

        # Interpolation matrix
        A = _fem_interpolation_matrix_1d(x, t)
        nk = k

        # FEM matrices
        C_fem = _fem_mass_matrix_1d(t)
        G1_fem = _fem_stiffness_matrix_1d(t)
        # G2 = G1 * C⁻¹ * G1  (regularized for numerical stability)
        C_reg = C_fem + 1e-10 * I
        G2_fem = G1_fem * (C_reg \ G1_fem)
        G2_fem = (G2_fem + G2_fem') / 2

        info = Dict{Symbol, Any}(:dim => 1, :precomputed => false,
                                  :mesh_nodes => t, :nk => nk)
    else  # dim == 2
        xvar, yvar = spec.term_vars
        x = Float64.(Tables.getcolumn(data, xvar))
        y = Float64.(Tables.getcolumn(data, yvar))

        # Grid dimensions: aim for ~k nodes total
        side = max(round(Int, sqrt(k)), 3)
        k_actual = side * side
        A, C_fem, G1_fem, G2_fem, nodes_x, nodes_y =
            _fem_matrices_2d_grid(x, y, side, side)
        nk = k_actual

        info = Dict{Symbol, Any}(:dim => 2, :precomputed => false,
                                  :nk => nk,
                                  :mesh_nodes_x => nodes_x,
                                  :mesh_nodes_y => nodes_y,
                                  :grid_nx => side, :grid_ny => side)
    end

    X = A  # model matrix is the interpolation matrix

    # Three penalty matrices: mass, scaled stiffness, second-order stiffness
    # Combined Matérn penalty would be: Q = τ²(κ⁴·C + 2κ²·G₁ + G₂)
    # We store each separately; the fitting engine assigns independent λⱼ.
    S1 = Matrix{Float64}(C_fem)
    S2 = 2.0 * Matrix{Float64}(G1_fem)
    S3 = Matrix{Float64}(G2_fem)
    penalties = Matrix{Float64}[S1, S2, S3]

    # Matérn SPDE has no null space when κ > 0
    null_dim = 0
    pen_rank = nk

    # Apply sum-to-zero constraint
    X_cons, S_cons, C_con, _ = absorb_constraints!(X, penalties)

    sm = ConstructedSmooth(
        spec, X_cons, S_cons,
        Float64[],  # knots stored in metadata
        null_dim, pen_rank,
        C_con, nothing, 0, 0,
        nothing, nothing, nothing,
        Int[],
    )

    # Store the L matrix for exact Matérn coupling reference:
    # L = [2 4; 2 2; 2 0]  means log(λⱼ) = Lⱼ₁·log(τ) + Lⱼ₂·log(κ)
    # i.e., λ₁ = τ²κ⁴, λ₂ = 2τ²κ², λ₃ = τ²
    info[:L] = [2 4; 2 2; 2 0]
    _SPDE_INFO[objectid(sm)] = info

    return sm
end

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

function _predict_matrix(::SPDESmooth, smooth::ConstructedSmooth, newdata)
    spec = smooth.spec
    dim = length(spec.term_vars)

    info = get(_SPDE_INFO, objectid(smooth), nothing)
    info !== nothing || throw(ArgumentError(
        "Cannot find SPDE smooth metadata for prediction. " *
        "This can happen if the smooth object was serialized/deserialized."))

    if info[:precomputed]
        throw(ArgumentError(
            "Prediction with pre-computed FEM matrices requires supplying " *
            "the interpolation matrix A for new data via predict_matrix " *
            "directly, or re-constructing the smooth with new data."))
    end

    if dim == 1
        var = spec.term_vars[1]
        x_new = Float64.(Tables.getcolumn(newdata, var))
        t = info[:mesh_nodes]
        A_new = _fem_interpolation_matrix_1d(x_new, t)
    else  # dim == 2
        xvar, yvar = spec.term_vars
        x_new = Float64.(Tables.getcolumn(newdata, xvar))
        y_new = Float64.(Tables.getcolumn(newdata, yvar))
        nx = info[:grid_nx]
        ny = info[:grid_ny]
        nk = info[:nk]
        nodes_x = info[:mesh_nodes_x]
        nodes_y = info[:mesh_nodes_y]

        # Rebuild grid information for interpolation
        xlo = minimum(nodes_x)
        xhi = maximum(nodes_x)
        ylo = minimum(nodes_y)
        yhi = maximum(nodes_y)
        xs = range(xlo, xhi; length = nx)
        ys = range(ylo, yhi; length = ny)
        node_idx(ix, iy) = (iy - 1) * nx + ix

        # Build triangles
        triangles = Vector{NTuple{3,Int}}()
        for iy in 1:(ny - 1), ix in 1:(nx - 1)
            n1 = node_idx(ix, iy)
            n2 = node_idx(ix + 1, iy)
            n3 = node_idx(ix, iy + 1)
            n4 = node_idx(ix + 1, iy + 1)
            push!(triangles, (n1, n2, n3))
            push!(triangles, (n2, n4, n3))
        end

        n_new = length(x_new)
        A_new = zeros(n_new, nk)
        for j in 1:n_new
            xj, yj = x_new[j], y_new[j]
            best_tri = 0
            best_lam = (0.0, 0.0, 0.0)
            best_dist = Inf

            for (ti, (i1, i2, i3)) in enumerate(triangles)
                x1, y1 = nodes_x[i1], nodes_y[i1]
                x2, y2 = nodes_x[i2], nodes_y[i2]
                x3, y3 = nodes_x[i3], nodes_y[i3]

                det_T = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                abs(det_T) < 1e-15 && continue

                l1 = ((x2 - x3) * (yj - y3) + (y3 - y2) * (xj - x3)) / det_T
                l2 = ((x3 - x1) * (yj - y1) + (y1 - y3) * (xj - x1)) / det_T
                l3 = 1.0 - l1 - l2

                if l1 >= -1e-10 && l2 >= -1e-10 && l3 >= -1e-10
                    A_new[j, i1] = max(l1, 0.0)
                    A_new[j, i2] = max(l2, 0.0)
                    A_new[j, i3] = max(l3, 0.0)
                    best_tri = ti
                    break
                else
                    dist = max(-l1, 0.0) + max(-l2, 0.0) + max(-l3, 0.0)
                    if dist < best_dist
                        best_dist = dist
                        best_tri = ti
                        best_lam = (l1, l2, l3)
                    end
                end
            end

            if best_tri > 0 && all(A_new[j, :] .== 0.0)
                i1, i2, i3 = triangles[best_tri]
                l1 = max(best_lam[1], 0.0)
                l2 = max(best_lam[2], 0.0)
                l3 = max(best_lam[3], 0.0)
                s = l1 + l2 + l3
                if s > 0
                    A_new[j, i1] = l1 / s
                    A_new[j, i2] = l2 / s
                    A_new[j, i3] = l3 / s
                else
                    A_new[j, i1] = 1.0 / 3
                    A_new[j, i2] = 1.0 / 3
                    A_new[j, i3] = 1.0 / 3
                end
            end
        end
    end

    # Apply constraint absorption
    if smooth.constraint !== nothing
        C_con = smooth.constraint
        k_pred = size(A_new, 2)
        qr_C = qr(C_con')
        Z_cons = (qr_C.Q * Matrix(I, k_pred, k_pred))[:, 2:k_pred]
        return A_new * Z_cons
    end
    return A_new
end
