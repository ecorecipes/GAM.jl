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

        # Raw (uncentered) TPRS marginal: k = 8 columns per level; the
        # sum-to-zero-across-levels constraint is absorbed via the
        # (L-1)-column level-contrast basis Q_L: (L-1) * k columns total.
        k_eff = 8
        n_levels = 3
        expected_cols = (n_levels - 1) * k_eff
        @test size(sm.X, 2) == expected_cols
    end

    @testset "Penalty structure correct (block diagonal)" begin
        n = 300
        x = randn(rng_sz, n)
        group = repeat(["A", "B", "C"], outer=ceil(Int, n/3))[1:n]

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

        # There is now one penalty PER LEVEL, matching mgcv
        # (`R/smooth.r:2281-2286`); see test_sz_penalties.jl. Each individual
        # penalty is (q_i q_i') ⊗ S_marginal and therefore has NON-zero
        # off-diagonal blocks — it is the SUM over levels that recovers the
        # block-diagonal I_{L-1} ⊗ S_marginal this test used to assert of
        # `sm.S[1]` back when the smooth carried a single summed penalty.
        S = sum(sm.S)
        ncols = size(sm.X, 2)
        n_blocks = 2            # L - 1 contrast blocks for 3 levels
        k_per_block = ncols ÷ n_blocks
        off_block = S[1:k_per_block, (k_per_block+1):end]
        @test norm(off_block) < 1e-10
        @test S[1:k_per_block, 1:k_per_block] ≈
              S[(k_per_block+1):end, (k_per_block+1):end]
        # And the decomposition is genuine: a single penalty is not itself
        # block-diagonal, which is what lets each level be smoothed on its own.
        @test norm(sm.S[1][1:k_per_block, (k_per_block+1):end]) > 1e-6
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

    @testset "Level smooths sum to zero across levels at each x" begin
        # The defining sz constraint (mgcv): Σ_l f_l(x) = 0 for every x.
        # In the contrast parameterization the design rows for the same x
        # across all levels therefore sum to exactly zero.
        n = 100
        x = randn(rng_sz, n)
        group = repeat(["A", "B", "C"], outer=ceil(Int, n/3))[1:n]

        data = DataFrame(x=x, group=group)
        spec = s(:x, :group, bs=:sz, k=6)
        sm = smooth_construct(spec, data)

        x0 = collect(range(-1.5, 1.5; length=9))
        row_sum = zeros(length(x0), size(sm.X, 2))
        for g in ["A", "B", "C"]
            nd = DataFrame(x=x0, group=fill(g, length(x0)))
            row_sum .+= predict_matrix(sm, nd)
        end
        @test norm(row_sum) < 1e-10
    end
end
