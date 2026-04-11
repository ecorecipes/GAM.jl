using Test
using GAM
using DataFrames
using Random
using StatsAPI: deviance
using LinearAlgebra: diag
using GLM: LogLink
using Statistics: var, mean, cor, cov
using Distributions

@testset "Gratia-like diagnostics & visualization" begin

    # Shared test data
    rng = MersenneTwister(42)
    n = 200
    x = sort(rand(rng, n))
    y = sin.(2π .* x) .+ 0.3 .* randn(rng, n)
    df = DataFrame(x = x, y = y)
    m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df)

    # Multi-smooth model
    rng2 = MersenneTwister(123)
    n2 = 300
    x1 = rand(rng2, n2)
    x2 = rand(rng2, n2)
    y2 = sin.(2π .* x1) .+ cos.(2π .* x2) .+ 0.5 .* randn(rng2, n2)
    df2 = DataFrame(x1 = x1, x2 = x2, y = y2)
    m2 = gam(@formulak(y ~ s(x1, k = 10, bs = :cr) + s(x2, k = 10, bs = :cr)), df2)

    # ─── smooth_estimates ────────────────────────────────────────────────

    @testset "smooth_estimates" begin
        se = smooth_estimates(m)
        @test se isa SmoothEstimates
        @test length(se.estimate) == 100  # default n=100
        @test length(se.se) == 100
        @test all(se.se .>= 0)
        @test haskey(se.covariates, :x)
        @test length(se.covariates[:x]) == 100

        # With custom n
        se50 = smooth_estimates(m; n = 50)
        @test length(se50.estimate) == 50

        # Select by index
        se_sel = smooth_estimates(m; select = 1)
        @test length(se_sel.estimate) == 100
        @test all(s -> s == "s(x,bs=cr)", se_sel.smooth)

        # Multi-smooth
        se2 = smooth_estimates(m2)
        @test length(unique(se2.smooth)) == 2

        # Select by label
        se_lab = smooth_estimates(m2; select = "s(x1,bs=cr)")
        @test all(s -> s == "s(x1,bs=cr)", se_lab.smooth)
        @test length(se_lab.estimate) == 100

        # Custom data
        custom_data = (x = collect(range(0.1, 0.9; length = 20)),)
        se_custom = smooth_estimates(m; data = custom_data)
        @test length(se_custom.estimate) == 20

        # SE should be finite
        @test all(isfinite, se.estimate)
        @test all(isfinite, se.se)

        # show method
        buf = IOBuffer()
        show(buf, se)
        @test occursin("SmoothEstimates", String(take!(buf)))
    end

    # ─── partial_residuals ───────────────────────────────────────────────

    @testset "partial_residuals" begin
        pr = partial_residuals(m)
        @test pr isa Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}
        @test haskey(pr, "s(x,bs=cr)")
        x_vals, p_resid = pr["s(x,bs=cr)"]
        @test length(x_vals) == n
        @test length(p_resid) == n
        @test all(isfinite, p_resid)

        # Multi-smooth
        pr2 = partial_residuals(m2)
        @test length(pr2) == 2
        @test haskey(pr2, "s(x1,bs=cr)")
        @test haskey(pr2, "s(x2,bs=cr)")
    end

    # ─── data_slice ──────────────────────────────────────────────────────

    @testset "data_slice" begin
        ds = data_slice(m; var = :x, n = 50)
        @test length(ds.x) == 50
        @test ds.x[1] <= ds.x[end]  # sorted grid

        # Non-existent variable
        @test_throws ArgumentError data_slice(m; var = :z)
    end

    # ─── derivatives ─────────────────────────────────────────────────────

    @testset "derivatives" begin
        de = derivatives(m; n = 50)
        @test de isa DerivativeEstimates
        @test length(de.derivative) == 50
        @test length(de.se) == 50
        @test length(de.lower) == 50
        @test length(de.upper) == 50
        @test de.order == 1
        @test de.type == :central
        @test all(de.lower .<= de.derivative)
        @test all(de.derivative .<= de.upper)

        # Forward differences
        de_fwd = derivatives(m; n = 30, type = :forward)
        @test de_fwd.type == :forward
        @test length(de_fwd.derivative) == 30

        # Backward differences
        de_bwd = derivatives(m; n = 30, type = :backward)
        @test de_bwd.type == :backward

        # Second order
        de2 = derivatives(m; n = 30, order = 2)
        @test de2.order == 2
        @test length(de2.derivative) == 30

        # Forward and central should give similar results
        @test cor(de_fwd.derivative, derivatives(m; n = 30, type = :central).derivative) > 0.97

        # Derivative of sin(2πx) should be approximately 2π·cos(2πx)
        de_fine = derivatives(m; n = 100, type = :central, eps = 1e-5)
        x_grid = de_fine.x
        expected_deriv = 2π .* cos.(2π .* x_grid)
        # Correlation should be high even if values aren't exact
        @test cor(de_fine.derivative, expected_deriv) > 0.95

        # Multi-smooth derivatives
        de2_multi = derivatives(m2; n = 50)
        @test length(unique(de2_multi.smooth)) == 2

        # Error handling
        @test_throws ArgumentError derivatives(m; type = :invalid)
        @test_throws ArgumentError derivatives(m; order = 3)

        # show method
        buf = IOBuffer()
        show(buf, de)
        @test occursin("DerivativeEstimates", String(take!(buf)))
    end

    # ─── posterior_samples ───────────────────────────────────────────────

    @testset "posterior_samples" begin
        ps = posterior_samples(m; n = 100, seed = 42)
        @test size(ps) == (100, length(m.coefficients))
        @test all(isfinite, ps)

        # Reproducibility with seed
        ps2 = posterior_samples(m; n = 100, seed = 42)
        @test ps ≈ ps2

        # Different seed gives different results
        ps3 = posterior_samples(m; n = 100, seed = 99)
        @test !(ps ≈ ps3)

        # Mean should be close to estimated coefficients
        mean_coef = vec(mean(ps; dims = 1))
        @test cor(mean_coef, m.coefficients) > 0.9
    end

    # ─── fitted_samples ──────────────────────────────────────────────────

    @testset "fitted_samples" begin
        fs = fitted_samples(m; n = 50, seed = 42, scale = :response)
        @test size(fs) == (n, 50)
        @test all(isfinite, fs)

        # Link scale
        fs_link = fitted_samples(m; n = 50, seed = 42, scale = :link)
        @test size(fs_link) == (n, 50)
        # For Gaussian with identity link, link and response should be equal
        @test fs ≈ fs_link

        # Mean of samples should be close to fitted values
        mean_fitted = vec(mean(fs; dims = 2))
        @test cor(mean_fitted, m.fitted_values) > 0.99
    end

    # ─── smooth_samples ──────────────────────────────────────────────────

    @testset "smooth_samples" begin
        ss = smooth_samples(m; n = 50, seed = 42)
        @test ss isa Dict
        @test haskey(ss, "s(x,bs=cr)")
        x_grid, draws = ss["s(x,bs=cr)"]
        @test length(x_grid) == 100  # default n_grid
        @test size(draws) == (100, 50)
        @test all(isfinite, draws)

        # Multi-smooth
        ss2 = smooth_samples(m2; n = 20, seed = 42)
        @test length(ss2) == 2
    end

    # ─── predicted_samples ───────────────────────────────────────────────

    @testset "predicted_samples" begin
        pps = predicted_samples(m; n = 20, seed = 42)
        @test size(pps) == (n, 20)
        @test all(isfinite, pps)

        # Predicted samples should have more variance than fitted samples
        fs = fitted_samples(m; n = 20, seed = 42)
        @test var(vec(pps)) > var(vec(fs))
    end

    # ─── appraise ────────────────────────────────────────────────────────

    @testset "appraise" begin
        ad = appraise(m)
        @test ad isa AppraiseData
        @test length(ad.residuals_deviance) == n
        @test length(ad.residuals_pearson) == n
        @test length(ad.linear_predictor) == n
        @test length(ad.observed) == n
        @test length(ad.fitted) == n
        @test length(ad.qq_theoretical) == n
        @test length(ad.qq_sample) == n

        # QQ data should be sorted
        @test issorted(ad.qq_sample)
        @test issorted(ad.qq_theoretical)

        # Observed should be the original y
        @test ad.observed ≈ m.y

        # show method
        buf = IOBuffer()
        show(buf, ad)
        @test occursin("AppraiseData", String(take!(buf)))
    end

    # ─── rootogram ───────────────────────────────────────────────────────

    @testset "rootogram" begin
        # Need a Poisson model for rootogram
        rng_p = MersenneTwister(77)
        n_p = 300
        x_p = sort(rand(rng_p, n_p))
        mu_p = exp.(1.0 .+ 2.0 .* sin.(2π .* x_p))
        y_p = Float64.([rand(rng_p, Distributions.Poisson(max(m, 0.1))) for m in mu_p])
        df_p = DataFrame(x = x_p, y = y_p)
        m_p = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df_p;
            family = Poisson(), link = LogLink())

        rd = rootogram(m_p)
        @test rd isa RootogramData
        @test rd.count[1] == 0
        @test all(rd.observed .>= 0)
        @test all(rd.expected .>= 0)
        @test rd.sqrt_observed ≈ sqrt.(rd.observed)
        @test rd.sqrt_expected ≈ sqrt.(rd.expected)
        @test sum(rd.observed) ≈ n_p  # total frequency = n

        # show method
        buf = IOBuffer()
        show(buf, rd)
        @test occursin("RootogramData", String(take!(buf)))
    end

    # ─── model_edf ───────────────────────────────────────────────────────

    @testset "model_edf" begin
        @test model_edf(m) > 1.0  # at least intercept
        @test model_edf(m) < 15.0  # less than max k
        @test model_edf(m) ≈ m.edf_total
    end

    # ─── overview ────────────────────────────────────────────────────────

    @testset "overview" begin
        ov = overview(m)
        @test ov isa OverviewTable
        @test length(ov.label) == 1
        @test ov.label[1] == "s(x,bs=cr)"
        @test ov.dimension[1] == 1
        @test ov.basis_size[1] == 14  # k=15, minus 1 for identifiability
        @test 0 < ov.edf[1] <= 14
        @test 0 < ov.edf_ratio[1] <= 1

        # Multi-smooth
        ov2 = overview(m2)
        @test length(ov2.label) == 2

        # show method
        buf = IOBuffer()
        show(buf, ov)
        s = String(take!(buf))
        @test occursin("GAM Overview", s)
        @test occursin("s(x,bs=cr)", s)
    end

    # ─── Model stores data ───────────────────────────────────────────────

    @testset "data storage" begin
        @test m.data !== nothing
        @test :x in Tables.columnnames(m.data)
        @test :y in Tables.columnnames(m.data)
        @test length(Tables.getcolumn(m.data, :x)) == n
    end

    # ─── Edge cases ──────────────────────────────────────────────────────

    @testset "edge cases" begin
        # Invalid smooth selection
        @test_throws ArgumentError smooth_estimates(m; select = "nonexistent")
        @test_throws BoundsError smooth_estimates(m; select = 5)

        # Derivatives with simultaneous intervals
        de_sim = derivatives(m; n = 30, interval = :simultaneous,
            n_sim = 100, seed = 42)
        @test length(de_sim.derivative) == 30
        @test all(de_sim.lower .<= de_sim.derivative)

        # Zero draws
        ps0 = posterior_samples(m; n = 1, seed = 42)
        @test size(ps0) == (1, length(m.coefficients))
    end
end
