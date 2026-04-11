using Test
using GAM
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, coef, predict

const rng_sos = StableRNG(123)

@testset "Spherical Spline (sos)" begin
    @testset "Construction with simulated lat/lon data" begin
        n = 200
        lat = π/2 .* (2 .* rand(rng_sos, n) .- 1)  # [-π/2, π/2]
        lon = π .* (2 .* rand(rng_sos, n) .- 1)      # [-π, π]

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=20)

        @test spec.basis isa SphericalSpline
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{SphericalSpline}
        @test size(sm.X, 1) == n
    end

    @testset "Basis matrix dimensions correct" begin
        n = 100
        lat = π/4 .* rand(rng_sos, n)
        lon = π/2 .* rand(rng_sos, n)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=15)
        sm = smooth_construct(spec, data)

        # k=15 eigenvectors + 1 null space column - 1 constraint = 15 columns
        @test size(sm.X, 2) == 15
        @test size(sm.X, 1) == n
    end

    @testset "Penalty matrix is PSD and correct dimension" begin
        n = 100
        lat = π/4 .* rand(rng_sos, n)
        lon = π/2 .* rand(rng_sos, n)

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
        lat = π/2 .* (2 .* rand(rng_sos, n) .- 1)
        lon = π .* (2 .* rand(rng_sos, n) .- 1)
        # Smooth function on sphere: f(lat, lon) = sin(lat) * cos(lon)
        f_true = sin.(lat) .* cos.(lon)
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
        lat = π/2 .* (2 .* rand(rng_sos, n) .- 1)
        lon = π .* (2 .* rand(rng_sos, n) .- 1)
        f_true = sin.(lat) .* cos.(lon)
        y = f_true .+ 0.2 .* randn(rng_sos, n)

        df = DataFrame(lat=lat, lon=lon, y=y)
        m = gam(@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df)

        # Predict on new data
        n_new = 50
        lat_new = π/2 .* (2 .* rand(rng_sos, n_new) .- 1)
        lon_new = π .* (2 .* rand(rng_sos, n_new) .- 1)
        df_new = DataFrame(lat=lat_new, lon=lon_new)

        pred = predict(m, df_new)
        @test length(pred) == n_new
        @test all(isfinite.(pred))
    end

    @testset "Handles knot subsampling for large n" begin
        n = 3000
        lat = π/2 .* (2 .* rand(rng_sos, n) .- 1)
        lon = π .* (2 .* rand(rng_sos, n) .- 1)

        data = DataFrame(lat=lat, lon=lon)
        spec = s(:lat, :lon, bs=:sos, k=20,
                 xt=Dict{Symbol,Any}(:max_knots => 500))
        sm = smooth_construct(spec, data)

        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 20  # k + 1 null - 1 constraint = k
    end

    @testset "Null space dimension" begin
        n = 100
        lat = π/4 .* rand(rng_sos, n)
        lon = π/2 .* rand(rng_sos, n)

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

    @testset "Geodesic distance properties" begin
        # Same point → distance 0
        @test GAM._geodesic_distance(0.0, 0.0, 0.0, 0.0) ≈ 0.0

        # Antipodal points → distance π
        @test GAM._geodesic_distance(0.0, 0.0, 0.0, π) ≈ π

        # North pole to equator → π/2
        @test GAM._geodesic_distance(π/2, 0.0, 0.0, 0.0) ≈ π/2
    end
end
