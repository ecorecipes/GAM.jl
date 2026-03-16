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
