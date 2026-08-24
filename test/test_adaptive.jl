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
        pou_weights = GAM._adaptive_weight_basis(n_rows, n_pen)

        S_sum = zeros(actual_k, actual_k)
        for j in 1:n_pen
            W_j = Diagonal(pou_weights[j])
            S_sum .+= D' * W_j * D
        end
        # Sum of locally weighted penalties == full penalty (since weights are partition of unity)
        @test S_sum ≈ S_full atol = 1e-12
    end

    @testset "Partition of unity weights sum to 1" begin
        # Holds for the B-spline branches (n_pen >= 3) and the degenerate
        # n_pen == 1 case. It deliberately does NOT hold for n_pen == 2, where
        # mgcv uses [1  x] — see the mgcv-convention testset below.
        for (n_rows, n_pen) in [(10, 3), (20, 5), (50, 8), (5, 1)]
            weights = GAM._adaptive_weight_basis(n_rows, n_pen)
            @test length(weights) == n_pen
            for i in 1:n_rows
                total = sum(weights[j][i] for j in 1:n_pen)
                @test total ≈ 1.0 atol = 1e-12
            end
        end
    end

    # ── mgcv convention for the adaptive weight basis ────────────────────
    # mgcv's smooth.construct.ad.smooth.spec (R/smooth.r:2467-2477) evaluates a
    # fixed P-spline basis of dimension `n_penalties` at x = 1:(nk-2)/nk:
    #   n_pen == 2 -> [1  x];  n_pen == 3 -> order 3 (m=1);  n_pen >= 4 -> order 4.
    # GAM.jl previously used order 3 for every n_pen, which matched mgcv only at
    # n_pen == 3 and left every other basis size wrong.
    @testset "Weight basis follows mgcv (order 4, with k=2/k=3 special cases)" begin
        # n_pen == 2 is mgcv's [1  x], NOT a partition of unity
        n_rows = 18
        w2 = GAM._adaptive_weight_basis(n_rows, 2)
        @test length(w2) == 2
        @test all(w2[1] .== 1.0)
        @test w2[2] ≈ collect(1.0:n_rows) ./ (n_rows + 2) atol = 1e-14
        rowsums2 = [w2[1][i] + w2[2][i] for i in 1:n_rows]
        @test !all(isapprox.(rowsums2, 1.0; atol = 1e-8))

        # B-spline order shows up directly as the number of non-zero columns per
        # row: an order-p B-spline basis has at most p non-zeros in any row.
        maxnz(W) = maximum(count(>(1e-12), [W[j][i] for j in eachindex(W)]) for i in 1:length(W[1]))
        @test maxnz(GAM._adaptive_weight_basis(23, 3)) <= 3   # order 3 at n_pen == 3
        @test maxnz(GAM._adaptive_weight_basis(38, 5)) == 4   # order 4 otherwise
        @test maxnz(GAM._adaptive_weight_basis(38, 8)) == 4

        # Reference column sums taken from mgcv 1.9-4 for nk=40, n_pen=5
        # (V <- ps2$X with x = 1:38/40, k=5, bs="ps", m=2, fx=TRUE).
        W5 = GAM._adaptive_weight_basis(38, 5)
        @test [sum(c) for c in W5] ≈ [0.8513, 9.5772, 17.1430, 9.5772, 0.8513] atol = 1e-3
    end

    @testset "`m` sets the number of sub-penalties (mgcv convention)" begin
        n = 300
        x = range(0.0, 1.0, length = n)
        df = DataFrame(x = collect(x))

        # mgcv: m is p.order = penalty basis size, default 5
        sm_default = smooth_construct(s(:x, bs = :ad, k = 20), df)
        @test length(sm_default.S) == 5

        for np in (3, 4, 6)
            sm = smooth_construct(s(:x, bs = :ad, k = 20, m = np), df)
            @test length(sm.S) == np
            # the smoothing basis stays a cubic P-spline regardless of m
            @test sm.null_dim == 2
        end

        # `m` and xt[:n_penalties] must agree when both are given
        @test length(smooth_construct(
            s(:x, bs = :ad, k = 20, m = 4,
              xt = Dict{Symbol,Any}(:n_penalties => 4)), df).S) == 4
        @test_throws ArgumentError smooth_construct(
            s(:x, bs = :ad, k = 20, m = 4,
              xt = Dict{Symbol,Any}(:n_penalties => 6)), df)

        # mgcv errors rather than silently shrinking an oversized penalty basis
        @test_throws ArgumentError smooth_construct(s(:x, bs = :ad, k = 8, m = 12), df)
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

        m = gam(@formulak(y ~ s(x, bs = :ad, k = 15)), df)
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

        # Adaptive smooths carry one smoothing parameter per sub-penalty, and
        # matching mgcv's order-4 weight basis flattens the REML surface enough
        # that EFS needs more than the default 200 outer iterations to declare
        # convergence here. The estimate is already stable to ~1e-3 in edf at
        # the default; the extra iterations only tighten the stopping rule.
        ad_ctrl = gam_control(outer_maxit = 1000)
        m_ad = gam(@formulak(y ~ s(x, bs = :ad, k = 25)), df; control = ad_ctrl)
        m_ps = gam(@formulak(y ~ s(x, bs = :ps, k = 25)), df)

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

        # See the note above on outer_maxit for adaptive smooths.
        np_ctrl = gam_control(outer_maxit = 1000)
        for np in [2, 3, 5, 7]
            sp = s(:x, bs = :ad, k = 15, xt = Dict{Symbol,Any}(:n_penalties => np))
            gf = GamFormula(:y, Symbol[], true, SmoothSpec[sp])
            m = gam(gf, df; control = np_ctrl)
            @test m.converged
            @test length(m.sp) == np
        end
    end

    @testset "Predict from fitted adaptive GAM" begin
        n = 200
        x = sort(randn(ad_rng, n))
        y = sin.(x) .+ 0.2 .* randn(ad_rng, n)
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, bs = :ad, k = 15)), df)
        @test m.converged

        df_new = DataFrame(x = range(-2.0, 2.0, length = 50))
        yhat = predict(m, df_new)
        @test length(yhat) == 50
        @test all(isfinite, yhat)
    end
end
