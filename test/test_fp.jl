using Test
using GAM
using LinearAlgebra
using Statistics
using Random

const fp_rng = MersenneTwister(456)

@testset "Fractional polynomial smooth (bs=:fp)" begin

    @testset "Construction with positive x data" begin
        n = 100
        x = collect(range(0.1, 5.0; length = n))
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        # FP2 (default degree=2) produces 2 columns
        @test size(sm.X, 2) == 2
        # No penalty for FP
        @test isempty(sm.S)
    end

    @testset "FP1 degree option" begin
        n = 100
        x = collect(range(0.1, 5.0; length = n))
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true, xt = Dict{Symbol,Any}(:degree => 1))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 2) == 1  # FP1 gives 1 column
    end

    @testset "Correct powers selected for sqrt relationship" begin
        n = 200
        x = collect(range(0.1, 10.0; length = n))
        y = sqrt.(x) .+ 0.01 .* randn(fp_rng, n)
        data = (x = x,)

        # Provide response hint for power selection
        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 1, :y => y))
        sm = smooth_construct(spec, data)

        powers = sm.spec.xt[:_selected_powers]
        @test length(powers) == 1
        @test powers[1] ≈ 0.5  # should select p=0.5 for sqrt
    end

    @testset "FP2 with known quadratic relationship" begin
        n = 200
        x = collect(range(0.1, 5.0; length = n))
        y = 2.0 .* x .+ 0.5 .* x .^ 2 .+ 0.1 .* randn(fp_rng, n)
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 2, :y => y))
        sm = smooth_construct(spec, data)

        powers = sm.spec.xt[:_selected_powers]
        @test length(powers) == 2
        # Should select powers (1, 2) for linear + quadratic
        @test 1.0 in powers
        @test 2.0 in powers
    end

    @testset "Handles log transform (p=0)" begin
        n = 200
        x = collect(range(0.1, 10.0; length = n))
        y = 3.0 .* log.(x) .+ 0.01 .* randn(fp_rng, n)
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 1, :y => y))
        sm = smooth_construct(spec, data)

        powers = sm.spec.xt[:_selected_powers]
        @test length(powers) == 1
        @test powers[1] ≈ 0.0  # p=0 means log(x)
    end

    @testset "Prediction applies correct power transform" begin
        n = 100
        x = collect(range(0.5, 5.0; length = n))
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 2))
        sm = smooth_construct(spec, data)

        # Predict at new points
        x_new = collect(range(1.0, 4.0; length = 50))
        newdata = (x = x_new,)
        Xp = predict_matrix(sm, newdata)

        @test size(Xp, 1) == 50
        @test size(Xp, 2) == size(sm.X, 2)
        @test all(isfinite, Xp)
    end

    @testset "Handles non-positive x (auto-shift)" begin
        n = 100
        x = collect(range(-2.0, 3.0; length = n))
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        x_shift = sm.spec.xt[:_x_shift]
        @test x_shift > 0.0  # shift applied since x has non-positive values
        @test all(isfinite, sm.X)

        # Prediction with non-positive x should also work
        x_new = collect(range(-1.0, 2.0; length = 50))
        newdata = (x = x_new,)
        Xp = predict_matrix(sm, newdata)
        @test all(isfinite, Xp)
    end

    @testset "Custom candidate powers" begin
        n = 100
        x = collect(range(0.1, 5.0; length = n))
        y = x .^ 3 .+ 0.1 .* randn(fp_rng, n)
        data = (x = x,)

        custom_powers = [1.0, 2.0, 3.0]
        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 1, :powers => custom_powers, :y => y))
        sm = smooth_construct(spec, data)

        powers = sm.spec.xt[:_selected_powers]
        @test powers[1] ≈ 3.0  # should pick cubic
    end

    @testset "GAM fitting with FP basis" begin
        n = 200
        x = sort(rand(fp_rng, n) .* 4.0 .+ 0.5)
        y = 2.0 .* sqrt.(x) .+ 0.3 .* randn(fp_rng, n)
        data = (x = x, y = y)

        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 1, :y => y))
        sm = smooth_construct(spec, data)

        # Simple regression with FP basis
        X = hcat(ones(n), sm.X)
        β = X \ y
        y_hat = X * β
        y_true = 2.0 .* sqrt.(x)

        # Should capture the sqrt relationship well
        cor_val = cor(y_true, y_hat)
        @test cor_val > 0.95
    end

    @testset "FP2 repeated power (p1 == p2)" begin
        n = 200
        x = collect(range(0.5, 5.0; length = n))
        # y = x * log(x) which is the FP2 basis for repeated power p=1
        y = x .* log.(x) .+ 0.01 .* randn(fp_rng, n)
        data = (x = x,)

        spec = s(:x, bs = :fp, k = 10, fx = true,
            xt = Dict{Symbol,Any}(:degree => 2, :y => y))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 2) == 2
    end
end
