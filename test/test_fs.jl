@testset "Factor-smooth interactions (bs=:fs)" begin
    using Random, DataFrames, Statistics, Distributions, LinearAlgebra, StatsAPI

    import StatsAPI: predict

    # ─── SmoothSpec defaults ──────────────────────────────────────────────

    @testset "SmoothSpec construction" begin
        sp = s(:x, :group, bs = :fs)
        @test sp isa SmoothSpec{FactorSmooth}
        @test sp.term_vars == [:x, :group]
        @test sp.k == 10   # default 1D marginal

        sp2 = s(:x, :group, bs = :fs, k = 8)
        @test sp2.k == 8

        # Need at least 2 variables
        @test_throws ArgumentError smooth_construct(
            s(:x, bs = :fs), (x = randn(10), g = repeat(["a", "b"], 5)))
    end

    # ─── Basis construction ───────────────────────────────────────────────

    @testset "Construction dimensions" begin
        Random.seed!(1)
        n = 60
        groups = repeat(["a", "b", "c"], inner = n ÷ 3)
        x = randn(n)
        data = (x = x, group = groups)

        k = 8
        spec = s(:x, :group, bs = :fs, k = k)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{FactorSmooth}
        n_levels = 3
        k_eff = k - 1   # after sum-to-zero constraint on marginal
        @test size(sm.X) == (n, n_levels * k_eff)

        # No additional constraint on the full fs smooth
        @test sm.constraint === nothing

        # Penalty dimensions match
        @test all(S -> size(S) == (n_levels * k_eff, n_levels * k_eff), sm.S)

        # Penalty rank and null_dim
        # TPRS 1D: original null_dim=2, after constraint: null_dim_constrained=1
        @test sm.null_dim == n_levels * 1
        @test sm.rank == n_levels * (k - 2)   # marginal rank = k - original_null_dim
    end

    @testset "Block-diagonal structure" begin
        Random.seed!(2)
        n = 90
        groups = repeat(1:3, inner = 30)
        x = randn(n)
        data = (x = x, group = groups)

        spec = s(:x, :group, bs = :fs, k = 6)
        sm = smooth_construct(spec, data)
        n_levels = 3
        k_eff = size(sm.X, 2) ÷ n_levels

        # Each observation should only have non-zero entries in its group's block
        for i in 1:n
            l = groups[i]
            for bl in 1:n_levels
                cols = ((bl - 1) * k_eff + 1):(bl * k_eff)
                if bl == l
                    @test any(sm.X[i, cols] .!= 0.0)
                else
                    @test all(sm.X[i, cols] .== 0.0)
                end
            end
        end

        # Penalties should be block-diagonal
        for S_fs in sm.S
            for l1 in 1:n_levels
                r1 = ((l1 - 1) * k_eff + 1):(l1 * k_eff)
                for l2 in 1:n_levels
                    r2 = ((l2 - 1) * k_eff + 1):(l2 * k_eff)
                    if l1 == l2
                        # Diagonal blocks should be identical
                        block = S_fs[r1, r2]
                        ref_block = S_fs[1:k_eff, 1:k_eff]
                        @test block ≈ ref_block
                    else
                        # Off-diagonal blocks should be zero
                        @test all(S_fs[r1, r2] .== 0.0)
                    end
                end
            end
        end
    end

    # ─── Prediction matrix ────────────────────────────────────────────────

    @testset "Prediction matrix" begin
        Random.seed!(3)
        n = 60
        groups = repeat(["a", "b", "c"], inner = 20)
        x = randn(n)
        data = (x = x, group = groups)

        spec = s(:x, :group, bs = :fs, k = 6)
        sm = smooth_construct(spec, data)
        k_eff = size(sm.X, 2) ÷ 3

        # Predict at new data with known levels — correct dimensions
        n_new = 15
        new_groups = repeat(["a", "b", "c"], inner = 5)
        new_x = randn(n_new)
        newdata = (x = new_x, group = new_groups)

        X_new = predict_matrix(sm, newdata)
        @test size(X_new) == (n_new, 3 * k_eff)

        # Block-diagonal structure preserved in prediction
        level_map = Dict("a" => 1, "b" => 2, "c" => 3)
        for i in 1:n_new
            l = level_map[new_groups[i]]
            for bl in 1:3
                cols = ((bl - 1) * k_eff + 1):(bl * k_eff)
                if bl == l
                    @test any(X_new[i, cols] .!= 0.0)
                else
                    @test all(X_new[i, cols] .== 0.0)
                end
            end
        end

        # Unknown level should give zero row
        unknown_data = (x = [0.5], group = ["z"])
        X_unk = predict_matrix(sm, unknown_data)
        @test all(X_unk .== 0.0)
    end

    # ─── Model fitting ────────────────────────────────────────────────────

    @testset "Fitting with group-specific curves" begin
        Random.seed!(42)
        n_per_group = 100
        n_groups = 3
        n = n_per_group * n_groups

        group_labels = repeat(["A", "B", "C"], inner = n_per_group)
        x = repeat(range(0, 2π; length = n_per_group), n_groups)

        # Group-specific sine curves with different amplitudes
        y = zeros(n)
        amplitudes = [1.0, 2.0, 0.5]
        for (g, amp) in enumerate(amplitudes)
            idx = ((g - 1) * n_per_group + 1):(g * n_per_group)
            y[idx] = amp .* sin.(x[idx]) .+ 0.2 .* randn(n_per_group)
        end

        df = DataFrame(x = x, y = y, group = group_labels)

        m = gam(@formulak(y ~ s(x, group, bs = :fs, k = 10)), df)

        @test m.converged
        @test length(m.fitted_values) == n
        @test m.deviance_val < sum((y .- mean(y)) .^ 2)  # better than null

        # Predictions should differ across groups
        pred_A = m.fitted_values[group_labels .== "A"]
        pred_B = m.fitted_values[group_labels .== "B"]
        pred_C = m.fitted_values[group_labels .== "C"]

        # Group B has largest amplitude → largest range in predictions
        @test (maximum(pred_B) - minimum(pred_B)) >
              (maximum(pred_C) - minimum(pred_C))
    end

    @testset "Prediction at new data" begin
        Random.seed!(99)
        n_per_group = 80
        n_groups = 2
        n = n_per_group * n_groups

        group_labels = repeat(["X", "Y"], inner = n_per_group)
        x = repeat(range(0, 2π; length = n_per_group), n_groups)

        y = zeros(n)
        y[1:n_per_group] = sin.(x[1:n_per_group]) .+ 0.1 .* randn(n_per_group)
        y[(n_per_group + 1):end] = cos.(x[(n_per_group + 1):end]) .+ 0.1 .* randn(n_per_group)

        df = DataFrame(x = x, y = y, group = group_labels)

        m = gam(@formulak(y ~ s(x, group, bs = :fs, k = 10)), df)
        @test m.converged

        # Predict at new x values for each group
        new_x = collect(range(0.5, 5.5; length = 20))
        newdf = DataFrame(
            x = repeat(new_x, 2),
            group = repeat(["X", "Y"], inner = 20),
        )

        preds = predict(m, newdf; type = :response)
        @test length(preds) == 40

        # Predictions for same x but different groups should differ
        pred_X = preds[1:20]
        pred_Y = preds[21:40]
        @test !all(pred_X .≈ pred_Y)
    end

    # ─── Edge cases ───────────────────────────────────────────────────────

    @testset "Integer factor levels" begin
        Random.seed!(7)
        n = 80
        groups = repeat(1:4, inner = 20)
        x = randn(n)
        y = sin.(x) .* groups .+ 0.2 .* randn(n)
        data = (x = x, group = groups, y = y)

        spec = s(:x, :group, bs = :fs, k = 6)
        sm = smooth_construct(spec, data)
        n_levels = 4
        k_eff = size(sm.X, 2) ÷ n_levels
        @test size(sm.X, 2) == n_levels * k_eff

        # Prediction has correct dimensions and block structure
        X_pred = predict_matrix(sm, data)
        @test size(X_pred) == size(sm.X)
    end

    @testset "Two groups" begin
        Random.seed!(8)
        n = 100
        groups = repeat(["lo", "hi"], inner = 50)
        x = randn(n)
        y = sin.(x) .+ (groups .== "hi") .* 2.0 .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y, group = groups)

        m = gam(@formulak(y ~ s(x, group, bs = :fs, k = 8)), df)
        @test m.converged
        @test m.deviance_val < sum((y .- mean(y)) .^ 2)
    end
end
