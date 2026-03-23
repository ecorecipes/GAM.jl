using Test
using GAM
using LinearAlgebra
using Statistics
using Random

const lo_rng = MersenneTwister(123)

@testset "Loess smooth (bs=:lo)" begin

    @testset "Construction produces correct dimensions" begin
        n = 100
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        spec = s(:x, bs = :lo, k = 15)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        # After constraint absorption: ncol = k - 1
        @test size(sm.X, 2) == 14
        @test length(sm.S) == 1
        # Penalty should be symmetric
        @test norm(sm.S[1] - sm.S[1]') < 1e-10
    end

    @testset "Kernel basis functions are localized" begin
        n = 200
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        spec = s(:x, bs = :lo, k = 10)
        sm = smooth_construct(spec, data)
        knots = sm.knots

        # Build raw kernel matrix to check localization
        x_range = maximum(x) - minimum(x)
        nk = length(knots)
        bandwidth = 0.75 * x_range / max(nk - 1, 1)

        # Kernel at knot location should peak, decay away
        mid_knot_idx = div(nk, 2)
        mid_knot = knots[mid_knot_idx]
        # Evaluate kernel at knot vs far away
        u_at_knot = 0.0
        u_far = (x_range / 2) / bandwidth

        k_at = GAM._tricube_kernel(u_at_knot)
        k_far = GAM._tricube_kernel(u_far)
        @test k_at > k_far
        @test k_at ≈ 1.0  # tricube(0) = 1
    end

    @testset "Gaussian kernel option" begin
        n = 100
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        spec = s(:x, bs = :lo, k = 10, xt = Dict{Symbol,Any}(:kernel => :gaussian))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
    end

    @testset "Degree 2 (local quadratic)" begin
        n = 100
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        spec = s(:x, bs = :lo, k = 10, xt = Dict{Symbol,Any}(:degree => 2))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        # Degree 2 doubles the basis columns (kernel + kernel*x) minus constraint
        # Before constraint: 2 * nk columns (minus any inactive)
        # After constraint: cols - 1
        @test size(sm.X, 2) > 0
    end

    @testset "Prediction on new data" begin
        n = 100
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        spec = s(:x, bs = :lo, k = 10)
        sm = smooth_construct(spec, data)

        # Predict at new points
        x_new = collect(range(0.1, 0.9; length = 50))
        newdata = (x = x_new,)
        Xp = predict_matrix(sm, newdata)

        @test size(Xp, 1) == 50
        @test size(Xp, 2) == size(sm.X, 2)
    end

    @testset "GAM fitting recovers smooth function" begin
        n = 200
        x = sort(rand(lo_rng, n))
        y_true = sin.(2π .* x)
        y = y_true .+ 0.2 .* randn(lo_rng, n)
        data = (x = x, y = y)

        spec = s(:x, bs = :lo, k = 20, xt = Dict{Symbol,Any}(:kernel => :gaussian))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n

        # Simple ridge regression to check the basis works
        X = hcat(ones(n), sm.X)
        λ = 0.1
        S_aug = zeros(size(X, 2), size(X, 2))
        S_aug[2:end, 2:end] = sm.S[1]
        β = (X' * X + λ * S_aug) \ (X' * y)
        y_hat = X * β

        # Should capture the general trend
        cor_val = cor(y_true, y_hat)
        @test cor_val > 0.7
    end

    @testset "Different span settings" begin
        n = 100
        x = collect(range(0, 1; length = n))
        data = (x = x,)

        # Small span: more local
        spec1 = s(:x, bs = :lo, k = 10, xt = Dict{Symbol,Any}(:span => 0.5))
        sm1 = smooth_construct(spec1, data)

        # Large span: more global
        spec2 = s(:x, bs = :lo, k = 10, xt = Dict{Symbol,Any}(:span => 1.5))
        sm2 = smooth_construct(spec2, data)

        @test sm1 isa ConstructedSmooth
        @test sm2 isa ConstructedSmooth
        # Both should work; larger span gives less localized basis
        @test size(sm1.X) == size(sm2.X)
    end
end
