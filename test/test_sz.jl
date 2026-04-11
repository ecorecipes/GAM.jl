using Test
using GAM
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, coef, predict

const rng_sz = StableRNG(456)

@testset "Constrained Factor Smooth (sz)" begin
    @testset "Construction with factor + continuous variable" begin
        n = 200
        x = randn(rng_sz, n)
        group = repeat(["A", "B", "C"], outer=ceil(Int, n/3))[1:n]

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=8)

        @test spec.basis isa ConstrainedFactorSmooth
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{ConstrainedFactorSmooth}
        @test size(sm.X, 1) == n
    end

    @testset "Correct number of columns (k_base * n_levels minus constraints)" begin
        n = 200
        x = randn(rng_sz, n)
        group = repeat(["A", "B", "C"], outer=ceil(Int, n/3))[1:n]

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=8)
        sm = smooth_construct(spec, data)

        # k=8 TPRS → 7 cols after TPRS constraint
        # Per-level constraint removes 1 more → 6 cols per level
        # 3 levels × 6 = 18 total columns
        k_eff = 7  # k - 1 for TPRS constraint
        k_constrained = k_eff - 1  # -1 for per-level sz constraint
        n_levels = 3
        expected_cols = n_levels * k_constrained
        @test size(sm.X, 2) == expected_cols
    end

    @testset "Penalty structure correct (block diagonal)" begin
        n = 200
        x = randn(rng_sz, n)
        group = repeat(["A", "B"], outer=ceil(Int, n/2))[1:n]

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=8)
        sm = smooth_construct(spec, data)

        @test length(sm.S) >= 1

        for S in sm.S
            ncols = size(sm.X, 2)
            @test size(S) == (ncols, ncols)
            # Symmetric
            @test norm(S - S') < 1e-10
            # PSD
            evals = eigvals(Symmetric(S))
            @test all(evals .>= -1e-8)
        end

        # Check block-diagonal structure: off-diagonal blocks should be zero
        S = sm.S[1]
        ncols = size(sm.X, 2)
        k_per_level = ncols ÷ 2  # 2 levels
        # Off-diagonal block should be zero
        off_block = S[1:k_per_level, (k_per_level+1):end]
        @test norm(off_block) < 1e-10
    end

    @testset "GAM fitting works" begin
        n = 300
        x = range(-3, 3; length=n) |> collect
        group = repeat(["A", "B", "C"], outer=ceil(Int, n/3))[1:n]
        # Different smooth functions per group
        f = map(1:n) do i
            if group[i] == "A"
                sin(x[i])
            elseif group[i] == "B"
                cos(x[i])
            else
                0.5 * x[i]^2 - 1.0
            end
        end
        y = f .+ 0.3 .* randn(rng_sz, n)

        df = DataFrame(x=x, group=group, y=y)
        m = gam(@formulak(y ~ s(x, group, bs = :sz, k = 8)), df)

        @test m isa GamModel
        @test m.converged
        @test length(coef(m)) > 0
        # Should explain variance
        @test cor(fitted(m), y)^2 > 0.3
    end

    @testset "Predictions correct per factor level" begin
        n = 300
        x = range(-2, 2; length=n) |> collect
        group = repeat(["A", "B"], outer=ceil(Int, n/2))[1:n]
        f = [group[i] == "A" ? sin(x[i]) : cos(x[i]) for i in 1:n]
        y = f .+ 0.2 .* randn(rng_sz, n)

        df = DataFrame(x=x, group=group, y=y)
        m = gam(@formulak(y ~ s(x, group, bs = :sz, k = 8)), df)

        # Predict for each group separately
        x_new = range(-1.5, 1.5; length=50) |> collect
        for g in ["A", "B"]
            df_new = DataFrame(x=x_new, group=fill(g, 50))
            pred = predict(m, df_new)
            @test length(pred) == 50
            @test all(isfinite.(pred))
        end
    end

    @testset "Factor variable specified via xt" begin
        n = 100
        x = randn(rng_sz, n)
        group = repeat(["X", "Y"], outer=50)

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=6,
                 xt=Dict{Symbol,Any}(:factor => :group))
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{ConstrainedFactorSmooth}
        @test size(sm.X, 1) == n
    end

    @testset "Requires at least 2 variables" begin
        data = DataFrame(x=rand(10))
        spec = s(:x, bs=:sz, k=5)
        @test_throws ArgumentError smooth_construct(spec, data)
    end

    @testset "Observations zero for other levels" begin
        # Each observation should only contribute to its own level's columns
        n = 100
        x = randn(rng_sz, n)
        group = repeat(["A", "B"], outer=50)

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=6)
        sm = smooth_construct(spec, data)

        ncols = size(sm.X, 2)
        k_per_level = ncols ÷ 2

        # Observations in group A should have zeros in group B columns
        mask_A = group .== "A"
        mask_B = group .== "B"
        @test norm(sm.X[mask_A, (k_per_level+1):end]) < 1e-10
        @test norm(sm.X[mask_B, 1:k_per_level]) < 1e-10
    end
end
