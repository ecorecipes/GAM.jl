using Test
using GAM
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, predict

const ad_rng = StableRNG(123)

@testset "Adaptive Smooth" begin

    # ── Construction basics ──────────────────────────────────────────────
    @testset "SmoothSpec with :ad basis" begin
        sp = s(:x, bs = :ad, k = 15)
        @test sp.basis isa AdaptiveSmooth
        @test sp.k == 15
    end

    @testset "Construct adaptive smooth" begin
        n = 200
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        sp = s(:x, bs = :ad, k = 15)
        sm = smooth_construct(sp, df)

        @test size(sm.X, 1) == n
        # After constraint absorption: k_eff = k - 1 (sum-to-zero)
        @test size(sm.X, 2) == 14

        # Should have 5 penalty matrices (default n_penalties)
        @test length(sm.S) == 5
        for S_j in sm.S
            @test size(S_j) == (14, 14)
        end
    end

    @testset "Custom n_penalties via xt" begin
        n = 200
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        sp3 = s(:x, bs = :ad, k = 20, xt = Dict{Symbol,Any}(:n_penalties => 3))
        sm3 = smooth_construct(sp3, df)
        @test length(sm3.S) == 3

        sp8 = s(:x, bs = :ad, k = 20, xt = Dict{Symbol,Any}(:n_penalties => 8))
        sm8 = smooth_construct(sp8, df)
        @test length(sm8.S) == 8
    end

    @testset "Single penalty (n_penalties=1) matches P-spline" begin
        n = 100
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        sp_ad = s(:x, bs = :ad, k = 10, xt = Dict{Symbol,Any}(:n_penalties => 1))
        sm_ad = smooth_construct(sp_ad, df)
        sp_ps = s(:x, bs = :ps, k = 10)
        sm_ps = smooth_construct(sp_ps, df)

        # Basis matrices should be identical
        @test sm_ad.X ≈ sm_ps.X atol = 1e-10

        # Single adaptive penalty should equal the P-spline penalty
        @test length(sm_ad.S) == 1
        @test length(sm_ps.S) == 1
        @test sm_ad.S[1] ≈ sm_ps.S[1] atol = 1e-10
    end

    # ── Penalty properties ───────────────────────────────────────────────
    @testset "Penalties are PSD" begin
        n = 200
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        sp = s(:x, bs = :ad, k = 20)
        sm = smooth_construct(sp, df)

        for (j, S_j) in enumerate(sm.S)
            eigs = eigvals(Symmetric(S_j))
            @test all(eigs .>= -1e-10) || "Penalty $j has negative eigenvalue: $(minimum(eigs))"
        end
    end

    @testset "Penalties sum to approximately full penalty (pre-constraint)" begin
        n = 200
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        # Compare raw D'WD penalties before absorb_constraints! rescaling.
        # Build the base P-spline penalty and the adaptive local penalties manually.
        k = 15
        m_order = 2
        spline_order = m_order + 2
        m2 = spline_order - 1
        nk = k - m2 + 1
        lo, hi = 0.0, 1.0
        k_new = range(lo, hi, length = nk) |> collect
        dk = k_new[2] - k_new[1]
        knot_vec = vcat(
            [k_new[1] - dk * i for i in m2:-1:1],
            k_new,
            [k_new[end] + dk * i for i in 1:m2],
        )
        X = GAM._bspline_basis(collect(x), knot_vec, spline_order)
        actual_k = size(X, 2)

        S_full = GAM._diff_penalty(actual_k, m_order)
        D = GAM._ad_diff_matrix(actual_k, m_order)
        n_rows = size(D, 1)
        n_pen = 5
        pou_weights = GAM._partition_of_unity_weights(n_rows, n_pen)

        S_sum = zeros(actual_k, actual_k)
        for j in 1:n_pen
            W_j = Diagonal(pou_weights[j])
            S_sum .+= D' * W_j * D
        end
        # Sum of locally weighted penalties == full penalty (since weights are partition of unity)
        @test S_sum ≈ S_full atol = 1e-12
    end

    @testset "Partition of unity weights sum to 1" begin
        for (n_rows, n_pen) in [(10, 3), (20, 5), (50, 8), (5, 1)]
            weights = GAM._partition_of_unity_weights(n_rows, n_pen)
            @test length(weights) == n_pen
            for i in 1:n_rows
                total = sum(weights[j][i] for j in 1:n_pen)
                @test total ≈ 1.0 atol = 1e-12
            end
        end
    end

    # ── Prediction matrix ────────────────────────────────────────────────
    @testset "predict_matrix works" begin
        n = 200
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        sp = s(:x, bs = :ad, k = 15)
        sm = smooth_construct(sp, df)

        # Predict at same data → should match X
        Xp = predict_matrix(sm, df)
        @test Xp ≈ sm.X atol = 1e-10

        # Predict at new data
        df_new = DataFrame(x = [0.25, 0.5, 0.75])
        Xp_new = predict_matrix(sm, df_new)
        @test size(Xp_new) == (3, size(sm.X, 2))
    end

    # ── GAM fitting ──────────────────────────────────────────────────────
    @testset "Fit GAM with adaptive smooth" begin
        n = 300
        x = sort(randn(ad_rng, n))
        y = sin.(2.0 .* x) .+ 0.3 .* randn(ad_rng, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, bs = :ad, k = 15)), df)
        @test m.converged
        @test length(m.sp) == 5  # 5 smoothing parameters (one per local penalty)
        @test m.edf_total > 1.0
    end

    @testset "Adaptive smooth on varying-smoothness data" begin
        # Left half: smooth (low frequency); right half: wiggly (high frequency)
        n = 400
        x = sort(rand(ad_rng, n) .* 2π)
        y = similar(x)
        for i in eachindex(x)
            if x[i] < π
                y[i] = sin(x[i]) + 0.2 * randn(ad_rng)
            else
                y[i] = sin(3.0 * x[i]) + 0.2 * randn(ad_rng)
            end
        end
        df = DataFrame(x = x, y = y)

        m_ad = gam(@gam_formula(y ~ s(x, bs = :ad, k = 25)), df)
        m_ps = gam(@gam_formula(y ~ s(x, bs = :ps, k = 25)), df)

        @test m_ad.converged
        @test m_ps.converged

        # Both should fit; adaptive should have at least comparable deviance explained
        dev_ad = 1.0 - m_ad.deviance_val / m_ad.null_deviance
        dev_ps = 1.0 - m_ps.deviance_val / m_ps.null_deviance
        @test dev_ad > 0.3  # should explain substantial variance
        @test dev_ps > 0.3
    end

    @testset "Adaptive smooth with different n_penalties" begin
        n = 200
        x = sort(randn(ad_rng, n))
        y = cos.(x) .+ 0.2 .* randn(ad_rng, n)
        df = DataFrame(x = x, y = y)

        for np in [2, 3, 5, 7]
            sp = s(:x, bs = :ad, k = 15, xt = Dict{Symbol,Any}(:n_penalties => np))
            gf = GamFormula(:y, Symbol[], true, SmoothSpec[sp])
            m = gam(gf, df)
            @test m.converged
            @test length(m.sp) == np
        end
    end

    @testset "Predict from fitted adaptive GAM" begin
        n = 200
        x = sort(randn(ad_rng, n))
        y = sin.(x) .+ 0.2 .* randn(ad_rng, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, bs = :ad, k = 15)), df)
        @test m.converged

        df_new = DataFrame(x = range(-2.0, 2.0, length = 50))
        yhat = predict(m, df_new)
        @test length(yhat) == 50
        @test all(isfinite, yhat)
    end
end
