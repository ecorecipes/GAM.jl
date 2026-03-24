# Visualization types for GAM smooth effects
#
# This file defines wrapper types for plot dispatch. The actual @recipe
# implementations live in ext/GAMPlotsExt.jl and are loaded automatically
# when the user does `using Plots`.

"""
    GamPlot

Wrapper for dispatching plot recipes on GAM models.
Create via [`gamplot`](@ref).
"""
struct GamPlot
    model::GamModel
    select::Union{Int, Nothing}   # which smooth to plot (nothing = all)
    residuals::Bool               # show partial residuals
    se::Bool                      # show confidence bands
    n_grid::Int                   # number of grid points
end

"""
    gamplot(model; select=nothing, residuals=false, se=true, n_grid=200)

Create a plot specification for a fitted GAM. Requires `using Plots`.

# Arguments
- `select`: which smooth to plot (nothing = all 1D smooths)
- `residuals`: overlay partial residuals
- `se`: show ±2 SE confidence bands
- `n_grid`: number of grid points for the smooth curve

# Example
```julia
using Plots
m = gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df)
plot(gamplot(m; se=true, residuals=true))
```
"""
function gamplot(m::GamModel; select=nothing, residuals::Bool=false,
    se::Bool=true, n_grid::Int=200)
    return GamPlot(m, select, residuals, se, n_grid)
end

"""
    GamContourPlot

Wrapper for 2D smooth effect contour plots.
Create via [`gamcontour`](@ref).
"""
struct GamContourPlot
    model::GamModel
    select::Int
    n_grid::Int
end

"""
    gamcontour(model; select=1, n_grid=50)

Create a contour plot specification for a 2D smooth in a fitted GAM.
Requires `using Plots`.
"""
function gamcontour(m::GamModel; select::Int=1, n_grid::Int=50)
    return GamContourPlot(m, select, n_grid)
end

# ─── vis_gam: 2D smooth visualization data ──────────────────────────────────

"""
    VisGamData

Data container for 2D smooth surface visualization, returned by [`vis_gam`](@ref).

# Fields
- `x1`, `x2`: 1-D grid vectors for the two covariates
- `z`: `n_grid × n_grid` matrix of predicted smooth effect
- `se`: optional `n_grid × n_grid` matrix of standard errors (or `nothing`)
- `x1_label`, `x2_label`, `z_label`, `smooth_label`: axis / title labels
"""
struct VisGamData
    x1::Vector{Float64}
    x2::Vector{Float64}
    z::Matrix{Float64}
    se::Union{Matrix{Float64}, Nothing}
    x1_label::String
    x2_label::String
    z_label::String
    smooth_label::String
end

"""
    _exclude_too_far(x1_grid, x2_grid, x1_data, x2_data, dist) -> Matrix{Bool}

For each grid point `(x1_grid[i], x2_grid[j])`, compute the minimum scaled
Euclidean distance to the nearest observed data point. Returns a Bool matrix
where `true` means the grid point is farther than `dist` from any data point
(and should be masked to `NaN`).
"""
function _exclude_too_far(x1_grid::AbstractVector, x2_grid::AbstractVector,
                          x1_data::AbstractVector, x2_data::AbstractVector,
                          dist::Real)
    n1 = length(x1_grid)
    n2 = length(x2_grid)
    nd = length(x1_data)
    mask = Matrix{Bool}(undef, n1, n2)

    r1 = maximum(x1_data) - minimum(x1_data)
    r2 = maximum(x2_data) - minimum(x2_data)
    r1 = r1 > 0 ? r1 : one(r1)
    r2 = r2 > 0 ? r2 : one(r2)

    for j in 1:n2, i in 1:n1
        min_d = Inf
        @inbounds for k in 1:nd
            d = sqrt(((x1_grid[i] - x1_data[k]) / r1)^2 +
                     ((x2_grid[j] - x2_data[k]) / r2)^2)
            d < min_d && (min_d = d)
        end
        mask[i, j] = min_d > dist
    end
    return mask
end

"""
    vis_gam(m::GamModel; select=1, n_grid=30, type=:link, se=false, too_far=0.0)

Compute predicted surface data for a 2D smooth in a fitted GAM.

Returns a [`VisGamData`](@ref) containing grid vectors, a prediction matrix,
optional standard errors, and labels suitable for surface / contour plotting.

# Arguments
- `select`: index of the smooth term in `m.smooths` (must be 2-D)
- `n_grid`: number of grid points along each axis
- `type`: `:link` (default) for the linear predictor scale, `:response` to
  apply the inverse link
- `se`: if `true`, compute pointwise standard errors
- `too_far`: if > 0, mask grid points whose scaled distance to the nearest
  observed data point exceeds this threshold (set to `NaN`)
"""
function vis_gam(m::GamModel; select::Int=1, n_grid::Int=30,
                 type::Symbol=:link, se::Bool=false, too_far::Real=0.0)
    # ── Argument validation ──────────────────────────────────────────────
    n_smooth = length(m.smooths)
    if select < 1 || select > n_smooth
        throw(ArgumentError(
            "select=$select is out of range; model has $n_smooth smooth(s)"))
    end
    if type ∉ (:link, :response)
        throw(ArgumentError(
            "type must be :link or :response, got :$type"))
    end

    sm   = m.smooths[select]
    spec = sm.spec
    vars = spec.term_vars

    if length(vars) != 2
        throw(ArgumentError(
            "vis_gam requires a 2D smooth (got $(length(vars))D); " *
            "use `gamplot` for 1D smooths"))
    end

    var1, var2 = vars
    sm_idx = sm.first_para:sm.last_para
    beta_s = m.coefficients[sm_idx]

    # ── Build evaluation grid ────────────────────────────────────────────
    x1_data = _get_covariate_from_model(m, sm, var1)
    x2_data = _get_covariate_from_model(m, sm, var2)

    x1_grid = collect(range(extrema(x1_data)...; length=n_grid))
    x2_grid = collect(range(extrema(x2_data)...; length=n_grid))

    # Flat vectors for all n_grid² combinations (column-major order)
    x1_flat = repeat(x1_grid, n_grid)
    x2_flat = repeat(x2_grid, inner=n_grid)
    newdata = NamedTuple{(var1, var2)}((x1_flat, x2_flat))

    # ── Predict ──────────────────────────────────────────────────────────
    X_pred = predict_matrix(sm, newdata)
    z_flat = X_pred * beta_s

    # Standard errors
    se_mat = nothing
    if se
        Vp_s = m.Vp[sm_idx, sm_idx]
        se_flat = sqrt.(max.(diag(X_pred * Vp_s * X_pred'), 0.0))
    end

    # Inverse link
    if type == :response
        z_flat = GLM.linkinv.(Ref(m.link), z_flat)
    end

    # Reshape to grid matrices
    z_mat = reshape(z_flat, n_grid, n_grid)
    if se
        se_mat = reshape(se_flat, n_grid, n_grid)
    end

    # ── too_far masking ──────────────────────────────────────────────────
    if too_far > 0
        far_mask = _exclude_too_far(x1_grid, x2_grid, x1_data, x2_data, too_far)
        z_mat[far_mask] .= NaN
        if se_mat !== nothing
            se_mat[far_mask] .= NaN
        end
    end

    # ── Labels ───────────────────────────────────────────────────────────
    z_label = type == :response ? "Response" : "Effect"
    smooth_label = "$(spec.label), edf=$(round(m.edf[select]; digits=1))"

    return VisGamData(x1_grid, x2_grid, z_mat, se_mat,
                      string(var1), string(var2), z_label, smooth_label)
end
