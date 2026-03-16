# Knot placement utilities

"""
    place_knots(x::AbstractVector, k::Int) -> Vector{Float64}

Place `k` knots at evenly-spaced quantiles of `x`.
Equivalent to mgcv's `place.knots`.
"""
function place_knots(x::AbstractVector{<:Real}, k::Int)
    k >= 1 || throw(ArgumentError("k must be ≥ 1, got $k"))
    xu = sort(unique(x))
    n = length(xu)
    if n <= k
        return Float64.(xu)
    end
    # Evenly-spaced quantile positions
    probs = range(0, 1; length = k)
    return Float64.(quantile(xu, probs))
end

"""
    place_knots(x::AbstractVector, k::Int, lo::Real, hi::Real) -> Vector{Float64}

Place `k` interior knots between boundary knots `lo` and `hi`.
"""
function place_knots(x::AbstractVector{<:Real}, k::Int, lo::Real, hi::Real)
    k >= 1 || throw(ArgumentError("k must be ≥ 1"))
    # Interior knots at evenly-spaced quantiles within [lo, hi]
    xf = filter(xi -> lo <= xi <= hi, x)
    if isempty(xf)
        return range(lo, hi; length = k) |> collect |> Vector{Float64}
    end
    probs = range(0, 1; length = k + 2)[2:(end - 1)]
    return Float64.(quantile(sort(xf), probs))
end

"""
    knot_quantiles(x::AbstractVector, n_interior::Int) -> Vector{Float64}

Compute `n_interior` interior knot positions as evenly spaced quantiles of unique values of `x`.
"""
function knot_quantiles(x::AbstractVector{<:Real}, n_interior::Int)
    n_interior >= 0 || throw(ArgumentError("n_interior must be ≥ 0"))
    xu = sort(unique(x))
    if n_interior == 0
        return Float64[]
    end
    probs = range(0, 1; length = n_interior + 2)[2:(end - 1)]
    return Float64.(quantile(xu, probs))
end
