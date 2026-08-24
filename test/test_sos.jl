using Test
using GAM
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, coef, predict

const rng_sos = StableRNG(123)

# Angles are supplied in DEGREES throughout, matching mgcv's `bs="sos"`
# (mgcv's `makeR` multiplies by pi/180 internally). The `_geodesic_distance`
# helper is a lower-level routine that still takes radians, and is tested as
# such at the bottom of this file.

@testset "Spherical Spline (sos)" begin
    @testset "Construction with simulated lat/lon data" begin
        n = 200
        lat = 90.0 .* (2 .* rand(rng_sos, n) .- 1)    # [-90, 90] degrees
        lon = 180.0 .* (2 .* rand(rng_sos, n) .- 1)   # [-180, 180] degrees

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=20)

        @test spec.basis isa SphericalSpline
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{SphericalSpline}
        @test size(sm.X, 1) == n
    end

    @testset "Basis matrix dimensions correct" begin
        n = 100
        lat = 45.0 .* rand(rng_sos, n)
        lon = 90.0 .* rand(rng_sos, n)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=15)
        sm = smooth_construct(spec, data)

        # Matches mgcv: makeR gives a k-column basis (k-p kernel columns plus
        # p null-space columns), and the centering constraint removes one, so
        # `smoothCon(s(...,k=15), absorb.cons=TRUE)` also yields k-1 = 14.
        @test size(sm.X, 2) == 14
        @test size(sm.X, 1) == n
    end

    @testset "Penalty matrix is PSD and correct dimension" begin
        n = 100
        lat = 45.0 .* rand(rng_sos, n)
        lon = 90.0 .* rand(rng_sos, n)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=15)
        sm = smooth_construct(spec, data)

        @test length(sm.S) == 1
        S = sm.S[1]
        ncols = size(sm.X, 2)
        @test size(S) == (ncols, ncols)

        # Should be symmetric
        @test norm(S - S') < 1e-10

        # Should be positive semi-definite
        evals = eigvals(Symmetric(S))
        @test all(evals .>= -1e-8)
    end

    @testset "Fitting a GAM with spherical smooth" begin
        n = 300
        lat = 90.0 .* (2 .* rand(rng_sos, n) .- 1)
        lon = 180.0 .* (2 .* rand(rng_sos, n) .- 1)
        # Smooth function on sphere: f = sin(lat) * cos(lon), lat/lon in degrees
        f_true = sin.(deg2rad.(lat)) .* cos.(deg2rad.(lon))
        y = f_true .+ 0.2 .* randn(rng_sos, n)

        df = DataFrame(lat=lat, lon=lon, y=y)
        m = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df)

        @test m isa GamModel
        @test m.converged
        @test length(coef(m)) > 0
        # Should explain a reasonable amount of variance
        @test cor(fitted(m), y)^2 > 0.3
    end

    @testset "Prediction works on new lat/lon points" begin
        n = 200
        lat = 90.0 .* (2 .* rand(rng_sos, n) .- 1)
        lon = 180.0 .* (2 .* rand(rng_sos, n) .- 1)
        f_true = sin.(deg2rad.(lat)) .* cos.(deg2rad.(lon))
        y = f_true .+ 0.2 .* randn(rng_sos, n)

        df = DataFrame(lat=lat, lon=lon, y=y)
        m = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df)

        # Predict on new data
        n_new = 50
        lat_new = 90.0 .* (2 .* rand(rng_sos, n_new) .- 1)
        lon_new = 180.0 .* (2 .* rand(rng_sos, n_new) .- 1)
        df_new = DataFrame(lat=lat_new, lon=lon_new)

        pred = predict(m, df_new)
        @test length(pred) == n_new
        @test all(isfinite.(pred))
    end

    @testset "Handles knot subsampling for large n" begin
        n = 3000
        lat = 90.0 .* (2 .* rand(rng_sos, n) .- 1)
        lon = 180.0 .* (2 .* rand(rng_sos, n) .- 1)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=20,
                 xt=Dict{Symbol,Any}(:max_knots => 500))
        sm = smooth_construct(spec, data)

        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 19  # mgcv's k columns, less the constraint
    end

    @testset "Null space dimension" begin
        n = 100
        lat = 45.0 .* rand(rng_sos, n)
        lon = 90.0 .* rand(rng_sos, n)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=15)
        sm = smooth_construct(spec, data)

        # Null space should be 1 (constant on sphere)
        @test sm.null_dim == 1
    end

    @testset "Requires exactly 2 variables" begin
        data = DataFrame(x=rand(10))
        spec = s(:x, bs=:sos, k=5)
        @test_throws ArgumentError smooth_construct(spec, data)
    end

    # ── Angle units ────────────────────────────────────────────────────────
    # Degrees is the default (mgcv's convention); radians are opt-in via
    # xt[:units]. The two must agree exactly on equivalent inputs, for BOTH
    # the fit and prediction on new data — `bs=:gp` previously had a bug where
    # construction and prediction resolved an xt option independently and
    # could silently disagree, so prediction is pinned explicitly here.
    @testset "Angle units: degrees default ≡ radians opt-in" begin
        rng_u = StableRNG(4242)
        n = 250
        lat_d = 70.0 .* (2 .* rand(rng_u, n) .- 1)
        lon_d = 170.0 .* (2 .* rand(rng_u, n) .- 1)
        y = sin.(deg2rad.(lat_d)) .+ cos.(deg2rad.(lon_d)) .+ 0.2 .* randn(rng_u, n)

        df_deg = DataFrame(lat=lat_d, lon=lon_d, y=y)
        df_rad = DataFrame(lat=deg2rad.(lat_d), lon=deg2rad.(lon_d), y=y)

        m_deg = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df_deg)
        m_rad = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20,
                                    xt = Dict(:units => :radians))), df_rad)

        @test fitted(m_deg) ≈ fitted(m_rad) atol = 1e-12
        @test coef(m_deg) ≈ coef(m_rad) atol = 1e-12
        @test m_deg.edf_total ≈ m_rad.edf_total atol = 1e-12

        # Prediction on new data must use the unit resolved at construction.
        nd_deg = DataFrame(lat=[10.0, -45.0, 60.0], lon=[0.0, 120.0, -30.0])
        nd_rad = DataFrame(lat=deg2rad.(nd_deg.lat), lon=deg2rad.(nd_deg.lon))
        @test predict(m_deg, nd_deg) ≈ predict(m_rad, nd_rad) atol = 1e-12
    end

    @testset "Angle units: cache records the resolved unit" begin
        n = 80
        lat = 45.0 .* rand(rng_sos, n)
        lon = 90.0 .* rand(rng_sos, n)
        data = DataFrame(lat=lat, lon=lon)

        sm_deg = smooth_construct(s(:lat, :lon, bs=:sos, k=10), data)
        @test sm_deg.predict_cache.units === :degrees
        # Knots are always stored in radians regardless of the input unit.
        @test maximum(abs, sm_deg.predict_cache.lat_k) <= π / 2 + 1e-12

        data_r = DataFrame(lat=deg2rad.(lat), lon=deg2rad.(lon))
        sm_rad = smooth_construct(
            s(:lat, :lon, bs=:sos, k=10, xt=Dict(:units => :radians)), data_r)
        @test sm_rad.predict_cache.units === :radians
        @test sm_deg.predict_cache.lat_k ≈ sm_rad.predict_cache.lat_k atol = 1e-12
    end

    @testset "Angle units: invalid value rejected at construction" begin
        @test_throws ArgumentError s(:lat, :lon, bs=:sos, xt=Dict(:units => :radian))
        @test_throws ArgumentError s(:lat, :lon, bs=:sos, xt=Dict(:units => :deg))
        # Valid values must still construct, including the string spelling.
        @test s(:lat, :lon, bs=:sos, xt=Dict(:units => :radians)) isa SmoothSpec
        @test s(:lat, :lon, bs=:sos, xt=Dict(:units => "degrees")) isa SmoothSpec
        # The guard is sos-specific: other bases ignore :units.
        @test s(:x, bs=:tp, xt=Dict(:units => :nonsense)) isa SmoothSpec
    end

    @testset "Angle units: radian-valued data warns under the degrees default" begin
        n = 120
        rng_w = StableRNG(99)
        # Radian-valued data read as degrees: the misuse this warning exists for.
        lat_r = (π / 2) .* (2 .* rand(rng_w, n) .- 1)
        lon_r = π .* (2 .* rand(rng_w, n) .- 1)
        data_r = DataFrame(lat=lat_r, lon=lon_r)
        @test_logs (:warn, r"read as DEGREES"i) match_mode = :any begin
            smooth_construct(s(:lat, :lon, bs=:sos, k=10), data_r)
        end

        # Genuine degree data spanning a real region must NOT warn.
        # Bare `@test_logs` asserts that NO log records are produced at all.
        lat_d = 60.0 .* (2 .* rand(rng_w, n) .- 1)
        lon_d = 150.0 .* (2 .* rand(rng_w, n) .- 1)
        data_d = DataFrame(lat=lat_d, lon=lon_d)
        @test_logs smooth_construct(s(:lat, :lon, bs=:sos, k=10), data_d)

        # Out-of-range values under an explicit :radians request also warn.
        data_big = DataFrame(lat=lat_d, lon=lon_d)
        @test_logs (:warn, r"exceed"i) match_mode = :any begin
            smooth_construct(
                s(:lat, :lon, bs=:sos, k=10, xt=Dict(:units => :radians)), data_big)
        end
    end

    @testset "Geodesic distance properties" begin
        # This helper works in RADIANS (it is below the units layer).
        # Same point → distance 0
        @test GAM._geodesic_distance(0.0, 0.0, 0.0, 0.0) ≈ 0.0

        # Antipodal points → distance π
        @test GAM._geodesic_distance(0.0, 0.0, 0.0, π) ≈ π

        # North pole to equator → π/2
        @test GAM._geodesic_distance(π/2, 0.0, 0.0, 0.0) ≈ π/2
    end

    @testset "rksos matches mgcv's C routine" begin
        # Reference values produced by calling mgcv 1.9-4's compiled `rksos`
        # (src/misc.c:39-73) directly:
        #   .C(mgcv:::C_rksos, z=..., n=..., eps=.Machine$double.eps)
        # Chosen to span both branches and every edge case the C guards:
        # antipodal (-1), the branch discontinuity at 0, and coincident (1).
        z = [-1.0, -0.999, -0.9, -0.5, -0.1, -1e-8, 0.0, 1e-8,
             0.1, 0.5, 0.9, 0.999, 1 - 1e-12, 1.0]
        ref = [-0.64493406684822641, -0.64443400433433373, -0.59429477438373024,
               -0.37728142776549389, -0.13053507769401454, -0.062693547314685863,
               -0.062693540383214036, -0.062693533451742153, 0.0082235646586754051,
               0.33353532608207981, 0.79569973012181283, 0.99569853582670609,
               0.99999999998533662, 1.0]
        got = [GAM._rksos(zi, eps(Float64)) for zi in z]
        @test got ≈ ref rtol = 1e-15
        # Out-of-domain arguments are clamped, as the C does.
        @test GAM._rksos(-2.0, eps(Float64)) == GAM._rksos(-1.0, eps(Float64))
        @test GAM._rksos(2.0, eps(Float64)) == GAM._rksos(1.0, eps(Float64))
    end

    @testset "mgcv penalty-order normalisation" begin
        # R/smooth.r:3057-3060 — note that below -2 maps to -1, not -2.
        @test GAM._sos_clamp_m(nothing) == 0    # mgcv's default is m = 0
        @test GAM._sos_clamp_m(2) == 2
        @test GAM._sos_clamp_m(9) == 4
        @test GAM._sos_clamp_m(-5) == -1
        @test GAM._sos_clamp_m(-2) == -2
    end

    @testset "makeR null-space dimensions match mgcv" begin
        la = [10.0, -20.0, 33.0]; lo = [5.0, 40.0, -70.0]
        r = π / 180
        for (m, p) in ((-2, 1), (-1, 4), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1))
            _, T, Tc = GAM._sos_makeR(la .* r, lo .* r, la .* r, lo .* r, m)
            @test size(T, 2) == p
            @test size(Tc, 2) == p
        end
    end

    @testset "sp transfers to mgcv (basis is exactly mgcv's)" begin
        # The generalized eigenvalues of (S, X'X) determine edf(sp), so a
        # basis that reproduces mgcv's edf at mgcv's own sp is mgcv's basis up
        # to reparameterization. Unlike bs=:tp, sos sp values are portable.
        rng = StableRNG(4321)
        n = 300
        lat = 60.0 .* (2 .* rand(rng, n) .- 1)
        lon = 60.0 .* (2 .* rand(rng, n) .- 1)
        f = 2 .* sin.(deg2rad.(lat)) .+ 1.5 .* cos.(deg2rad.(lon))
        y = f .+ 0.3 .* randn(rng, n)
        df = DataFrame(lat = lat, lon = lon, y = y)

        m = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df)
        @test m.converged
        @test size(m.smooths[1].X, 2) == 19

        # Refitting at the selected sp must reproduce the same fit: this is
        # the round-trip that would break if the reported sp were on a
        # different penalty scale from the one the basis is built on. It is
        # what makes mgcv's sp values portable into GAM.jl for this basis.
        sp_sel = exp(m.sp[1])
        spec = s(:lat, :lon, bs = :sos, k = 20, sp = sp_sel)
        gf = GAM.GamFormula(:y, Symbol[], true, [spec])
        m2 = gam(gf, df)
        @test m2.converged
        @test m2.edf_total ≈ m.edf_total rtol = 1e-6
        @test fitted(m2) ≈ fitted(m) rtol = 1e-6
    end
end
