using Test
using GAM
using DataFrames
using Random
using StatsAPI: deviance
using LinearAlgebra: diag, norm
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

    # smooth_estimates must equal the documented linear-algebra identity
    # f̂ = X_pred β_s, se = sqrt(diag(X_pred Vp_s X_predᵀ)), recomputed here
    # from the model's own blocks. The checks above only pin lengths and
    # finiteness; this catches a misaligned coefficient range, a wrong Vp
    # block, or scrambled row order in the concatenated output.
    @testset "smooth_estimates linear-algebra identity" begin
        se_id = smooth_estimates(m2; n = 25, overall_uncertainty = false)
        row0 = 0
        for (si, vsym) in ((1, :x1), (2, :x2))
            sm = m2.smooths[si]
            idx = sm.first_para:sm.last_para
            # evaluate on the SAME grid the function used for this smooth
            gvals = se_id.covariates[vsym][(row0 + 1):(row0 + 25)]
            Xp = predict_matrix(sm, NamedTuple{(vsym,)}((gvals,)))
            f_true = Xp * m2.coefficients[idx]
            se_true = sqrt.(max.(diag(Xp * m2.Vp[idx, idx] * Xp'), 0.0))
            @test maximum(abs.(se_id.estimate[(row0 + 1):(row0 + 25)] .- f_true)) < 1e-10
            @test maximum(abs.(se_id.se[(row0 + 1):(row0 + 25)] .- se_true)) < 1e-10
            row0 += 25
        end

        # overall_uncertainty=true must reproduce the intercept-inclusive
        # formula: X_full has the smooth block plus a ones column at the
        # intercept, se = sqrt(diag(X_full Vp X_fullᵀ)) over the FULL Vp.
        gridc = (x = collect(range(0.1, 0.9; length = 20)),)
        se_ou = smooth_estimates(m; data = gridc, overall_uncertainty = true)
        sm = m.smooths[1]
        idx = sm.first_para:sm.last_para
        Xp = predict_matrix(sm, gridc)
        p = length(m.coefficients)
        Xf = zeros(20, p)
        Xf[:, idx] .= Xp
        Xf[:, 1] .= 1.0
        se_true = sqrt.(max.(diag(Xf * m.Vp * Xf'), 0.0))
        @test maximum(abs.(se_ou.se .- se_true)) < 1e-10
        @test maximum(abs.(se_ou.estimate .- Xp * m.coefficients[idx])) < 1e-10
        # and it must genuinely differ from the conditional se
        se_cond = smooth_estimates(m; data = gridc, overall_uncertainty = false)
        @test maximum(abs.(se_ou.se .- se_cond.se)) > 1e-6
    end

    # ─── partial_residuals ───────────────────────────────────────────────

    @testset "partial_residuals" begin
        pr = partial_residuals(m)
        @test pr isa PartialResiduals
        @test "s(x,bs=cr)" in pr.smooth
        mask = pr.smooth .== "s(x,bs=cr)"
        @test count(mask) == n
        @test all(isfinite, pr.residual[mask])
        @test pr.xname[findfirst(mask)] == "x"
        @test length(pr.x) == length(pr.residual) == length(pr.smooth)

        # Multi-smooth: long format with one block per smooth
        pr2 = partial_residuals(m2)
        @test sort(unique(pr2.smooth)) == sort(["s(x1,bs=cr)", "s(x2,bs=cr)"])
        @test length(pr2.residual) == 2 * n2

        # show method
        buf = IOBuffer()
        show(buf, pr)
        @test occursin("PartialResiduals", String(take!(buf)))
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

        # Moment check against the claimed sampling distribution N(β̂, Vp):
        # with 5000 draws the mean's SE is sqrt(diag(Vp)/5000) per coordinate
        # (4-SE band, measured max standardised deviation 2.41), and the
        # sample covariance's Frobenius error scales like sqrt(2p/n) ≈ 0.077
        # at p=15 (measured 0.050; bound 0.12). A dropped Cholesky factor or
        # a Vp/Vc mix-up fails here while passing the correlation test above.
        psm = posterior_samples(m; n = 5000, seed = 42)
        mc = vec(mean(psm; dims = 1))
        mc_se = sqrt.(diag(m.Vp) ./ 5000)
        @test maximum(abs.(mc .- m.coefficients) ./ mc_se) < 4.0
        @test norm(cov(psm) .- m.Vp) / norm(m.Vp) < 0.12
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
        # default is the gratia-style simulated reference envelope
        ad = appraise(m; seed = 42)
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

        # simulated reference quantiles track the sample closely for a
        # well-specified Gaussian model, and are reproducible under a seed
        @test cor(ad.qq_theoretical, ad.qq_sample) > 0.95
        ad_rep = appraise(m; seed = 42)
        @test ad_rep.qq_theoretical ≈ ad.qq_theoretical

        # normal-theory fallback still available
        ad_n = appraise(m; method = :normal)
        @test issorted(ad_n.qq_theoretical)
        @test_throws ArgumentError appraise(m; method = :bogus)

        # Randomness that is an implementation detail of a reported quantity is
        # seeded by default, so a checked-in diagnostic figure does not change
        # on every re-render; randomness the caller explicitly asked for is not.
        # (Two unseeded vignette QQ panels were being redrawn every render.)
        @test appraise(m).qq_theoretical == appraise(m).qq_theoretical
        @test appraise(m; seed = nothing).qq_theoretical !=
              appraise(m; seed = nothing).qq_theoretical
        @test derivatives(m; n = 20, interval = :simultaneous, n_sim = 200).lower ==
              derivatives(m; n = 20, interval = :simultaneous, n_sim = 200).lower
        # ... while the sampling functions must keep drawing fresh values
        @test posterior_samples(m; n = 20) != posterior_samples(m; n = 20)
        @test fitted_samples(m; n = 10) != fitted_samples(m; n = 10)
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

    # ─── Tables.jl interface ─────────────────────────────────────────────

    @testset "Tables.jl interface" begin
        using Tables
        pr = partial_residuals(m)
        @test Tables.istable(typeof(pr))
        df_pr = DataFrame(pr)
        @test names(df_pr) == ["smooth", "xname", "x", "residual"]
        @test nrow(df_pr) == length(pr.x)
        @test df_pr.residual == pr.residual

        se_t = smooth_estimates(m; n = 25)
        df_se = DataFrame(se_t)
        @test "estimate" in names(df_se) && "se" in names(df_se)
        @test nrow(df_se) == length(se_t.estimate)

        d_t = derivatives(m; select = 1, n = 25)
        df_d = DataFrame(d_t)
        @test names(df_d) == ["smooth", "x", "derivative", "se", "lower", "upper"]
        @test df_d.derivative == d_t.derivative
    end

end
