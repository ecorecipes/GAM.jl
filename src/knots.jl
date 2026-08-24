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
    _bspline_knot_vector(x, k, m2; user_knots=nothing, extend_frac=0.001)

Shared B-spline knot-vector builder used by the ps/bs/ad/sc/scad bases.

Places `nk = k - m2 + 1` evenly spaced knots spanning the (slightly
extended) data range, then extends by `m2` knots on each side so that a
B-spline basis of degree `m2` has `k` columns. Following mgcv, the data
range is first extended by `extend_frac * range` on each side so boundary
observations sit strictly inside the outer interior knots.

If `user_knots` is supplied it is used as the interior knot sequence
(no range extension is applied to user knots).
"""
function _bspline_knot_vector(x::AbstractVector{<:Real}, k::Int, m2::Int;
    user_knots = nothing, extend_frac::Float64 = 0.001)
    nk = k - m2 + 1
    nk >= 2 || throw(ArgumentError(
        "k=$k too small for B-spline of degree $m2 (need k ≥ $(m2 + 1))"))

    if user_knots !== nothing
        interior = Float64.(user_knots)
        dk = length(interior) > 1 ? interior[2] - interior[1] :
             (maximum(x) - minimum(x))
        return vcat(
            [interior[1] - dk * i for i in m2:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:m2],
        )
    end

    lo, hi = minimum(x), maximum(x)
    ext = extend_frac * (hi - lo)
    lo -= ext
    hi += ext
    k_new = collect(range(lo, hi; length = nk))
    dk = k_new[2] - k_new[1]
    return vcat(
        [k_new[1] - dk * i for i in m2:-1:1],
        k_new,
        [k_new[end] + dk * i for i in 1:m2],
    )
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
