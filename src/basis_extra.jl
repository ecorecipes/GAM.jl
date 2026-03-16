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

function _smooth_construct(::MarkovRandomField, spec::SmoothSpec, data, user_knots)
    throw(ArgumentError("MRF smooths require a neighbourhood matrix. " *
        "Not yet fully implemented — use bs=:re for random effects."))
end

# ─── Factor-smooth interaction — bs="fs" ─────────────────────────────────

"""Factor-smooth interaction basis (mgcv `bs="fs"`)."""
struct FactorSmooth <: AbstractBasisType end

BASIS_TYPES[:fs] = FactorSmooth()

function _smooth_construct(::FactorSmooth, spec::SmoothSpec, data, user_knots)
    throw(ArgumentError("Factor-smooth interactions (bs=:fs) are not yet implemented. " *
        "Use s(x, by=:group) for varying coefficient models."))
end
