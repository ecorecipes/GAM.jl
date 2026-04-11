using Test, GAM, DataFrames, Random, Statistics, LinearAlgebra
using StatsAPI: predict

@testset "Cyclic P-spline (bs=:cps)" begin

    @testset "SmoothSpec construction" begin
        sp = s(:x, bs = :cps, k = 15)
        @test sp isa SmoothSpec
        @test sp.basis isa CyclicPSpline
        @test sp.k == 15

        # cps() convenience function
        sp2 = cps(:x; k = 12)
        @test sp2.basis isa CyclicPSpline
        @test sp2.k == 12
    end

    @testset "Basis matrix dimensions" begin
        Random.seed!(123)
        n = 200
        x = range(0, 2π; length = n) |> collect
        y = sin.(x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        for k in [8, 12, 20]
            spec = s(:x, bs = :cps, k = k)
            sm = GAM.smooth_construct(spec, df)
            # After constraint absorption: k - 1 columns
            @test size(sm.X, 1) == n
            @test size(sm.X, 2) == k - 1
            # Penalty matrix matches
            @test length(sm.S) == 1
            @test size(sm.S[1]) == (k - 1, k - 1)
        end
    end

    @testset "Penalty is symmetric PSD" begin
        Random.seed!(42)
        n = 100
        x = range(0, 2π; length = n) |> collect
        y = sin.(x)
        df = DataFrame(x = x, y = y)

        spec = s(:x, bs = :cps, k = 15)
        sm = GAM.smooth_construct(spec, df)
        S = sm.S[1]

        # Symmetric
        @test S ≈ S' atol = 1e-12

        # Positive semi-definite (all eigenvalues ≥ 0)
        evals = eigvals(Symmetric(S))
        @test all(evals .>= -1e-10)
    end

    @testset "Null space dimension" begin
        Random.seed!(42)
        n = 100
        x = range(0, 2π; length = n) |> collect
        y = sin.(x)
        df = DataFrame(x = x, y = y)

        spec = s(:x, bs = :cps, k = 15)
        sm = GAM.smooth_construct(spec, df)

        # For cyclic differences the null space is constants only (dim=1)
        @test sm.null_dim == 1

        # Verify via eigenvalues of the raw (pre-absorption) penalty
        S_raw = GAM._cyclic_diff_penalty(15, 2)
        evals = eigvals(Symmetric(S_raw))
        n_zero = count(e -> abs(e) < 1e-10, evals)
        @test n_zero == 1
    end

    @testset "Cyclic diff penalty construction" begin
        # Order 1
        S1 = GAM._cyclic_diff_penalty(5, 1)
        @test size(S1) == (5, 5)
        @test S1 ≈ S1' atol = 1e-14
        evals1 = eigvals(Symmetric(S1))
        @test count(e -> abs(e) < 1e-10, evals1) == 1

        # Order 2
        S2 = GAM._cyclic_diff_penalty(8, 2)
        @test size(S2) == (8, 8)
        @test S2 ≈ S2' atol = 1e-14
        evals2 = eigvals(Symmetric(S2))
        @test count(e -> abs(e) < 1e-10, evals2) == 1

        # Order 3
        S3 = GAM._cyclic_diff_penalty(10, 3)
        @test S3 ≈ S3' atol = 1e-14
        evals3 = eigvals(Symmetric(S3))
        @test count(e -> abs(e) < 1e-10, evals3) == 1
    end

    @testset "GAM fit recovers periodic signal" begin
        Random.seed!(42)
        n = 300
        x = range(0, 2π; length = n + 1)[1:n] |> collect
        y_true = sin.(x) .+ 0.5 .* cos.(2 .* x)
        y = y_true .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, bs = :cps, k = 15)), df)
        @test m.converged

        # Correlation with true signal should be high
        cor_val = cor(m.fitted_values, y_true)
        @test cor_val > 0.95

        # RMSE should be low
        rmse = sqrt(mean((m.fitted_values .- y_true) .^ 2))
        @test rmse < 0.3
    end

    @testset "Compare :cps and :cc on periodic data" begin
        Random.seed!(42)
        n = 200
        x = range(0, 2π; length = n + 1)[1:n] |> collect
        y_true = sin.(x)
        y = y_true .+ 0.15 .* randn(n)
        df = DataFrame(x = x, y = y)

        m_cps = gam(@formulak(y ~ s(x, bs = :cps, k = 15)), df)
        m_cc = gam(@formulak(y ~ s(x, bs = :cc, k = 15)), df)

        @test m_cps.converged
        @test m_cc.converged

        # Both should recover the signal well
        @test cor(m_cps.fitted_values, y_true) > 0.95
        @test cor(m_cc.fitted_values, y_true) > 0.95

        # Fitted values should be reasonably close to each other
        @test cor(m_cps.fitted_values, m_cc.fitted_values) > 0.95
    end

    @testset "Prediction wraps at boundaries" begin
        Random.seed!(42)
        n = 200
        # Include both endpoints so period = exactly 2π
        x = collect(range(0.0, 2π; length = n))
        y = sin.(x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, bs = :cps, k = 15)), df)

        # Predict at points just inside boundaries
        eps_val = 1e-6
        x_lo = [0.0 + eps_val]
        x_hi = [2π - eps_val]
        df_lo = DataFrame(x = x_lo)
        df_hi = DataFrame(x = x_hi)

        pred_lo = predict(m, df_lo)
        pred_hi = predict(m, df_hi)

        # For a periodic function on [0, 2π], values near 0 and 2π should match
        @test abs(pred_lo[1] - pred_hi[1]) < 0.1

        # Predict at points outside the domain — should wrap
        x_outside = [2π + 0.5, -0.5, 4π + 0.3]
        x_wrapped = [0.5, 2π - 0.5, 0.3]
        df_outside = DataFrame(x = x_outside)
        df_wrapped = DataFrame(x = x_wrapped)

        pred_outside = predict(m, df_outside)
        pred_wrapped = predict(m, df_wrapped)

        @test pred_outside ≈ pred_wrapped atol = 1e-10
    end

    @testset "Different penalty orders" begin
        Random.seed!(42)
        n = 200
        x = range(0, 2π; length = n + 1)[1:n] |> collect
        y = sin.(x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        # Order 1 penalty
        m1 = gam(@formulak(y ~ s(x, bs = :cps, k = 15, m = 1)), df)
        @test m1.converged
        @test cor(m1.fitted_values, sin.(x)) > 0.9

        # Order 3 penalty
        m3 = gam(@formulak(y ~ s(x, bs = :cps, k = 15, m = 3)), df)
        @test m3.converged
        @test cor(m3.fitted_values, sin.(x)) > 0.9
    end

    @testset "Basis rows sum to 1 (partition of unity)" begin
        # B-spline basis should form a partition of unity before constraint absorption
        Random.seed!(42)
        n = 50
        k = 10
        x = range(0, 2π; length = n + 1)[1:n] |> collect
        y = sin.(x)
        df = DataFrame(x = x, y = y)

        m_order = 2
        spline_order = m_order + 2
        degree = spline_order - 1
        ndx = k

        lo, hi = extrema(x)
        interior = collect(range(lo, hi; length = ndx + 1))
        dk = interior[2] - interior[1]
        knot_vec = vcat(
            [interior[1] - dk * i for i in degree:-1:1],
            interior,
            [interior[end] + dk * i for i in 1:degree],
        )

        X_full = GAM._bspline_basis(x, knot_vec, spline_order)
        X_wrapped = X_full[:, 1:ndx]
        for j in 1:degree
            X_wrapped[:, j] .+= X_full[:, ndx + j]
        end

        # Each row should sum to approximately 1
        row_sums = vec(sum(X_wrapped; dims = 2))
        @test all(isapprox.(row_sums, 1.0; atol = 1e-10))
    end
end
