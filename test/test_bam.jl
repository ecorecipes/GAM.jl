@testset "BAM (Big Additive Models)" begin
    @testset "Discretization" begin
        # Basic discretization
        x = collect(range(0, 10; length=5000))
        data = (x=x,)
        disc = discretize_covariates(data, [:x]; max_unique=100)

        @test disc isa DiscretizedData
        @test disc.n == 5000
        @test length(disc.unique_values[:x]) <= 100
        @test length(disc.indices[:x]) == 5000
        @test minimum(disc.indices[:x]) >= 1
        @test maximum(disc.indices[:x]) <= length(disc.unique_values[:x])

        # With few unique values (no binning needed)
        x_small = repeat([1.0, 2.0, 3.0, 4.0, 5.0], 1000)
        data_small = (x=x_small,)
        disc_small = discretize_covariates(data_small, [:x]; max_unique=100)
        @test length(disc_small.unique_values[:x]) == 5
    end

    @testset "BamControl construction" begin
        ctrl = bam_control()
        @test ctrl.chunk_size == 10000
        @test ctrl.discrete == true
        @test ctrl.max_unique == 1000
        @test ctrl.nthreads >= 1

        ctrl2 = bam_control(chunk_size=5000, discrete=false)
        @test ctrl2.chunk_size == 5000
        @test ctrl2.discrete == false
    end

    @testset "BAM Gaussian — small data matches gam" begin
        rng_bam = StableRNG(42)
        n = 200
        x = range(0, 2π; length=n) |> collect
        y_true = sin.(x)
        y = y_true .+ 0.3 .* randn(rng_bam, n)
        df = DataFrame(x=x, y=y)

        # Fit with gam
        m_gam = gam(@gam_formula(y ~ s(x, k=10, bs=:cr)), df)

        # Fit with bam (same data, should give similar results)
        m_bam = bam(@gam_formula(y ~ s(x, k=10, bs=:cr)), df;
            bam_ctrl=bam_control(chunk_size=50))

        @test m_bam isa GamModel
        @test m_bam.converged
        @test m_bam.n_smooth == 1

        # Coefficients should be very similar (same algorithm, just chunked)
        @test cor(m_gam.fitted_values, m_bam.fitted_values) > 0.999

        # EDF should be close
        @test abs(sum(edf(m_gam)) - sum(edf(m_bam))) < 1.0

        # Fit quality
        rmse_bam = sqrt(mean((m_bam.fitted_values .- y_true) .^ 2))
        @test rmse_bam < 0.5
    end

    @testset "BAM Gaussian — large data" begin
        rng_bam = StableRNG(123)
        n = 10000
        x = randn(rng_bam, n)
        y = sin.(x) .+ 0.3 .* randn(rng_bam, n)
        df = DataFrame(x=x, y=y)

        m = bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df;
            bam_ctrl=bam_control(chunk_size=2000))

        @test m isa GamModel
        @test m.converged
        @test nobs(m) == n
        @test length(coef(m)) > 1

        # Should capture the sine curve
        x_grid = range(-3, 3; length=100) |> collect
        pred_df = DataFrame(x=x_grid)
        pred = predict(m, pred_df)
        @test length(pred) == 100
        # Check correlation with true function
        @test cor(pred, sin.(x_grid)) > 0.95
    end

    @testset "BAM Poisson" begin
        rng_bam = StableRNG(456)
        n = 5000
        x = randn(rng_bam, n)
        mu_true = exp.(1.0 .+ 0.5 .* sin.(x))
        y = Float64[rand(rng_bam, Poisson(m)) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = bam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df;
            family=Poisson(), link=LogLink(),
            bam_ctrl=bam_control(chunk_size=1000))

        @test m isa GamModel
        @test m.converged
        @test all(m.fitted_values .> 0)
        @test m.deviance_val >= 0
    end

    @testset "BAM two smooths" begin
        rng_bam = StableRNG(789)
        n = 5000
        x1 = randn(rng_bam, n)
        x2 = randn(rng_bam, n)
        y = sin.(x1) .+ 0.5 .* x2 .^ 2 .+ 0.3 .* randn(rng_bam, n)
        df = DataFrame(x1=x1, x2=x2, y=y)

        m = bam(@gam_formula(y ~ s(x1, k=10, bs=:cr) + s(x2, k=10, bs=:cr)), df;
            bam_ctrl=bam_control(chunk_size=1000))

        @test m isa GamModel
        @test m.converged
        @test m.n_smooth == 2
        @test length(edf(m)) == 2
        @test all(edf(m) .> 1.0)
    end

    @testset "BAM predict and StatsBase interface" begin
        rng_bam = StableRNG(101)
        n = 2000
        x = range(0, 2π; length=n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng_bam, n)
        df = DataFrame(x=x, y=y)

        m = bam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df;
            bam_ctrl=bam_control(chunk_size=500))

        @test nobs(m) == n
        @test length(coef(m)) == size(m.X, 2)
        @test deviance(m) >= 0
        @test isfinite(loglikelihood(m))
        @test dof_residual(m) > 0
        @test m.scale > 0

        # Predict on new data
        new_x = range(0, 2π; length=50) |> collect
        new_df = DataFrame(x=new_x)
        pred = predict(m, new_df)
        @test length(pred) == 50
        @test cor(pred, sin.(new_x)) > 0.95

        # Residuals
        r = residuals(m; type=:response)
        @test length(r) == n
    end

    @testset "BAM with FormulaTerm" begin
        rng_bam = StableRNG(202)
        n = 500
        x = randn(rng_bam, n)
        y = 2.0 .* x .+ 0.5 .* randn(rng_bam, n)
        df = DataFrame(x=x, y=y)

        # Test that bam works with @formula too
        m = bam(@formula(y ~ x), df;
            bam_ctrl=bam_control(chunk_size=100))

        @test m isa GamModel
        @test m.converged
    end

    @testset "Chunked accumulation correctness" begin
        # Verify chunked X'WX matches non-chunked
        rng_bam = StableRNG(303)
        n = 500
        p = 10
        X = randn(rng_bam, n, p)
        w = abs.(randn(rng_bam, n)) .+ 0.1
        z = randn(rng_bam, n)

        # Direct computation
        WX = Diagonal(w) * X
        XtWX_direct = X' * WX
        XtWz_direct = X' * (w .* z)

        # Chunked computation
        XtWX_chunked = zeros(p, p)
        XtWz_chunked = zeros(p)
        GAM._accumulate_XtWX_XtWz_chunked!(XtWX_chunked, XtWz_chunked,
            X, w, z, 100)

        @test XtWX_chunked ≈ XtWX_direct atol=1e-10
        @test XtWz_chunked ≈ XtWz_direct atol=1e-10

        # Different chunk sizes should give same result
        XtWX_tiny = zeros(p, p)
        XtWz_tiny = zeros(p)
        GAM._accumulate_XtWX_XtWz_chunked!(XtWX_tiny, XtWz_tiny,
            X, w, z, 7)  # prime-sized chunks

        @test XtWX_tiny ≈ XtWX_direct atol=1e-10
        @test XtWz_tiny ≈ XtWz_direct atol=1e-10
    end

    @testset "Expand discretized X" begin
        X_unique = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 unique × 2 cols
        indices = [1, 2, 3, 1, 2, 3, 1]  # 7 obs
        X_full = GAM.expand_discretized_X(X_unique, indices, 7)

        @test size(X_full) == (7, 2)
        @test X_full[1, :] == [1.0, 2.0]
        @test X_full[4, :] == [1.0, 2.0]
        @test X_full[3, :] == [5.0, 6.0]
    end
end
