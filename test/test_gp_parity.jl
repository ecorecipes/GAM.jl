# mgcv parity for `bs=:gp` (Kammann & Wand Matérn / Gaussian-process smooths).
#
# Reference values are from mgcv 1.9-4 and are pinned here so the suite needs
# no R. They were produced with:
#
#     n <- 200; x <- (1:n)/(n+1); y <- sin(2*pi*x) + 0.3*sin(37*(1:n))
#     gam(y ~ s(x, bs="gp", k=10, m=<m>), data=..., method="REML")
#
# The edf comparisons are made at mgcv's OWN selected sp, which removes the
# smoothing-parameter optimizer from the comparison and tests the basis and
# penalty alone. Free-fit agreement is checked separately and is looser purely
# because GAM.jl selects by EFS where mgcv uses outer Newton.

@testset "GP smooth (bs=:gp) — mgcv parity" begin
    n = 200
    x = collect((1:n) ./ (n + 1))
    y = sin.(2π .* x) .+ 0.3 .* sin.(37 .* (1:n))
    df = DataFrame(x = x, y = y)

    # mgcv 1.9-4 reference: (m, selected sp, sum(edf))
    mgcv_ref = [
        (1, 0.0220623425646975, 8.6022467961),
        (2, 0.0166036341173075, 8.4663577391),
        (3, 0.000986257393143823, 7.6115158776),
        (4, 4.27855169159946e-5, 6.6106271533),
        (5, 1.85397728208035e-6, 5.8526601437),
        (-3, 0.00121562363280307, 8.1334217050),
    ]

    fit_fixed_sp(spec) = gam(GAM.GamFormula(:y, Symbol[], true, [spec]), df)

    @testset "gp.defn resolves as mgcv does" begin
        sm = GAM.smooth_construct(GAM.s(:x; bs = :gp, k = 10), (x = x,))
        c = sm.predict_cache
        @test c isa GAM.GPPredictCache
        # mgcv: gp.defn = c(3, 0.990049751243781, 1) — type 3 is the default,
        # and rho is the largest pairwise distance among the KNOTS.
        @test c.gptype == 3
        @test c.rho ≈ 0.990049751243781 atol = 1e-12
        @test c.kappa == 1.0
        @test c.stationary == false
        @test c.shift ≈ 0.5 atol = 1e-12
        # Knots are all unique covariate values (n < max.knots = 2000).
        @test length(c.knots) == n
        # Unpenalized null space [1, x]; rank = k - M.
        @test sm.null_dim == 2
        @test sm.rank == 8
    end

    @testset "null space is [1, x], or [1] when stationary" begin
        smn = GAM.smooth_construct(GAM.s(:x; bs = :gp, k = 10, m = 3), (x = x,))
        sms = GAM.smooth_construct(GAM.s(:x; bs = :gp, k = 10, m = -3), (x = x,))
        @test smn.null_dim == 2
        @test sms.null_dim == 1
        @test sms.predict_cache.stationary
        # Stationary keeps one more penalized column for the same k.
        @test sms.rank == 9
    end

    @testset "edf matches mgcv at mgcv's own sp" begin
        for (m, sp_r, edf_r) in mgcv_ref
            mf = fit_fixed_sp(GAM.s(:x; bs = :gp, k = 10, m = m, sp = sp_r))
            edf_j = sum(GAM.edf(mf)) + 1
            @test edf_j ≈ edf_r atol = 1e-4
        end
    end

    @testset "free fit selects essentially mgcv's sp" begin
        # Default m = 3, as in mgcv.
        mfree = gam(@formula(y ~ s(x, bs = :gp, k = 10)), df)
        @test sum(GAM.edf(mfree)) + 1 ≈ 7.6115158776 atol = 1e-3
        @test exp(mfree.sp[1]) / 0.000986257393143823 ≈ 1.0 atol = 1e-3
    end

    @testset "xt[:rho] override matches mgcv m = c(3, rho)" begin
        spec = GAM.s(:x; bs = :gp, k = 10, m = 3,
                     xt = Dict{Symbol, Any}(:rho => 0.5), sp = 0.00567495830558465)
        mf = fit_fixed_sp(spec)
        @test sum(GAM.edf(mf)) + 1 ≈ 7.6989025228 atol = 1e-4
        sm = GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 10, m = 3, xt = Dict{Symbol, Any}(:rho => 0.5)),
            (x = x,))
        @test sm.predict_cache.rho == 0.5
    end

    @testset "prediction reuses the fit-time correlation definition" begin
        # A basis built with one correlation and predicted with another
        # disagrees silently, so the definition must be resolved once and
        # stored — not re-derived from spec.xt at predict time.
        for m in (1, 2, 3, 4, 5, -3)
            sm = GAM.smooth_construct(GAM.s(:x; bs = :gp, k = 10, m = m), (x = x,))
            Xp = GAM.predict_matrix(sm, (x = x,))
            @test Xp ≈ sm.X atol = 1e-10
        end
        m3 = gam(@formula(y ~ s(x, bs = :gp, k = 10)), df)
        @test predict(m3, df) ≈ fitted(m3) atol = 1e-8
        # Genuinely new data stays finite and is centred by the fit-time shift.
        newd = DataFrame(x = collect(range(0.02, 0.98; length = 37)))
        @test all(isfinite, predict(m3, newd))
    end

    @testset "legacy named correlation functions still work" begin
        for cf in (:matern32, :matern52, :exponential, :gaussian, :sqexp,
                   :power_exp, :mgcv_m32)
            spec = GAM.s(:x; bs = :gp, k = 10,
                         xt = Dict{Symbol, Any}(:corfun => cf))
            m = gam(GAM.GamFormula(:y, Symbol[], true, [spec]), df)
            @test m.converged
            @test predict(m, df) ≈ fitted(m) atol = 1e-8
            @test m.smooths[1].predict_cache.corfun === cf
        end
    end

    @testset "max_knots caps the knot set" begin
        nbig = 300
        xb = collect((1:nbig) ./ (nbig + 1))
        yb = sin.(2π .* xb)
        smc = GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 10, xt = Dict{Symbol, Any}(:max_knots => 50)),
            (x = xb,))
        @test length(smc.predict_cache.knots) <= 50
        # Uncapped, every unique value is a knot.
        smu = GAM.smooth_construct(GAM.s(:x; bs = :gp, k = 10), (x = xb,))
        @test length(smu.predict_cache.knots) == nbig
    end

    @testset "argument validation" begin
        # |m| outside mgcv's 1..5
        @test_throws ArgumentError GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 10, m = 6), (x = x,))
        # kappa outside (0, 2]
        @test_throws ArgumentError GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 10, m = 2,
                  xt = Dict{Symbol, Any}(:kappa => 3.0)), (x = x,))
        # k below the null-space dimension + 1
        @test_throws ArgumentError GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 2, m = 3), (x = x,))
        # fewer unique covariate values than k, as mgcv errors
        @test_throws ArgumentError GAM.smooth_construct(
            GAM.s(:x; bs = :gp, k = 10), (x = repeat([0.1, 0.2, 0.3], 20),))
    end
end
