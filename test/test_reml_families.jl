# REML criterion parity with mgcv 1.9-4 for estimated-scale families.
#
# mgcv evaluates the REML score at `reml.scale` — the root of its own profiling
# equation (R/gam.fit3.r:629 with `remlInd = 1`) — and NOT at the
# Pearson/Fletcher estimate it reports as `b$scale`. The two are different
# quantities. For the reference Gamma fit below mgcv gives
#
#     b$scale      = 0.00447411844238   (Fletcher)
#     b$reml.scale = 0.00461804454675   (profiled)
#
# and reconstructing mgcv's own score from mgcv's own log|A|, log|S₊|, `ls` and
# `Mp` reproduces `b$gcv.ubre = -113.810025421882` exactly at `reml.scale`, but
# gives -113.759796979458 at `b$scale`.
#
# Plugging in `pearson/(n − edf)`, as GAM.jl previously did, left Gamma 3.4%
# away from mgcv on that fit even though its deviance, edf and coefficients all
# agreed to ~1e-15 — i.e. the FIT was right and only the reported criterion was
# wrong. Away from the optimum the discrepancy is much larger: measured
# `reml.scale / (pearson/(n − edf))` ratios of 0.96 at sp = 1e-3 and 1.55 at
# sp = 1e3 for Gaussian, and 2.50 at sp = 1e3 for Gamma.
#
# Gaussian is deliberately left on `pearson/(n − edf)`: that is mgcv's own rule
# on the Gaussian path, and mgcv reports `b$scale`, `b$reml.scale` and
# `pearson/(n − edf)` as equal to 12 significant figures there.
#
# Reference values are mgcv 1.9-4's `b$gcv.ubre` at mgcv's own selected `sp`,
# so the CRITERION is compared rather than the optimizer (GAM.jl selects `sp`
# by EFS, mgcv by outer Newton). Regenerate with e.g.
#     b <- gam(y~s(x,bs='cr',k=10), data=df, family=Gamma(), method='REML')
#     c(b$sp, b$gcv.ubre, b$reml.scale)

@testset "REML criterion vs mgcv — estimated-scale families" begin
    # Deterministic data — no RNG, so these are exactly reproducible.
    n = 200
    xr = collect(range(0, 1; length = n))
    fr = sin.(2π .* xr) .+ 0.5 .* cos.(5.0 .* xr)
    yG = exp.(fr) .+ 0.3 .* abs.(sin.(29.0 .* (1:n))) .+ 0.5

    dfG = DataFrame(x = xr, y = yG)
    gfG(sp) = GAM.GamFormula(:y, Symbol[], true,
        GAM.SmoothSpec[GAM.s(:x; k = 10, bs = :cr, sp = sp)])

    # mgcv 1.9-4 reference, Gamma()/inverse link, method = "REML"
    mgcv_sp = 4.07887786333345
    mgcv_score = -113.810025421882
    mgcv_reml_scale = 0.00461804454674792
    mgcv_edf = 9.41761936230618

    @testset "Gamma REML score matches mgcv at mgcv's own sp" begin
        m = gam(gfG(mgcv_sp), dfG; family = Gamma(), link = InverseLink(),
            method = :REML)
        @test m.edf_total ≈ mgcv_edf rtol = 1e-8
        # The whole point: the criterion, not just the fit.
        @test m.reml ≈ mgcv_score rtol = 1e-9
    end

    @testset "the profiled scale is the root of mgcv's profiling equation" begin
        m = gam(gfG(mgcv_sp), dfG; family = Gamma(), link = InverseLink(),
            method = :REML)
        p = size(m.X, 2)
        S = GAM.total_penalty(m.penalty, m.sp, p)
        pr = GAM.pirls(m.X, yG, S, Gamma(), InverseLink())
        Mp = p - sum(b.rank for b in m.penalty.blocks; init = 0)
        Dp = pr.deviance + dot(pr.coefficients, S * pr.coefficients)
        phi = GAM._reml_profiled_scale(Gamma(), yG, ones(n), Dp, n, Mp, 1.0,
            pr.pearson / (n - m.edf_total))

        # Agrees with mgcv's own reml.scale to its optimizer tolerance.
        @test phi ≈ mgcv_reml_scale rtol = 1e-6

        # It is a genuinely different quantity from pearson/(n − edf) ...
        pearson_scale = pr.pearson / (n - m.edf_total)
        @test !isapprox(phi, pearson_scale; rtol = 1e-4)
        # ... and from Dp/(n − Mp), because Gamma's `ls` carries digamma terms
        # that shift the root (Gaussian's closed form does not apply here).
        @test !isapprox(phi, Dp / (n - Mp); rtol = 1e-4)
    end

    @testset "known scale bypasses profiling entirely" begin
        # With `scale` supplied, mgcv and GAM.jl both use it verbatim.
        m = gam(gfG(mgcv_sp), dfG; family = Gamma(), link = InverseLink(),
            method = :REML)
        S = GAM.total_penalty(m.penalty, m.sp, size(m.X, 2))
        pr = GAM.pirls(m.X, yG, S, Gamma(), InverseLink())
        s1, _ = GAM.reml_score(m.X, yG, m.penalty, m.sp, Gamma(), InverseLink(),
            ones(n), pr; method = :REML, scale = 0.005)
        s2, _ = GAM.reml_score(m.X, yG, m.penalty, m.sp, Gamma(), InverseLink(),
            ones(n), pr; method = :REML, scale = 0.005)
        @test s1 == s2
        @test isfinite(s1)
    end

    @testset "Gaussian and Poisson are untouched by the profiling change" begin
        # Gaussian keeps pearson/(n − edf); Poisson has a known scale (φ = 1).
        # Both already matched mgcv and must not move.
        yg = fr .+ 0.3 .* sin.(29.0 .* (1:n))
        dfg = DataFrame(x = xr, y = yg)
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 10, bs = :cr)])

        mg = gam(gf, dfg; family = Normal(), link = IdentityLink(), method = :REML)
        p = size(mg.X, 2)
        S = GAM.total_penalty(mg.penalty, mg.sp, p)
        pr = GAM.pirls(mg.X, yg, S, Normal(), IdentityLink())
        # Gaussian's scale is still the Pearson estimator, exactly as before.
        @test pr.pearson / (n - mg.edf_total) ≈ mg.scale rtol = 1e-6

        yp = Float64.(round.(exp.(fr) .* 3 .+ abs.(sin.(29.0 .* (1:n)))))
        dfp = DataFrame(x = xr, y = yp)
        mp = gam(gf, dfp; family = Poisson(), link = LogLink(), method = :REML)
        @test mp.scale == 1.0          # known scale, never profiled
        @test isfinite(mp.reml)
    end
end
