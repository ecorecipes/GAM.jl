"""
Smoothing-parameter uncertainty: edf1, edf2 and the corrected covariance Vc
(Wood, Pya & Säfken 2016), following mgcv's `gam.fit3.post.proc`.

The cross-implementation checks against live mgcv live in `test_rcall.jl`;
this file pins the algebra, the orderings, the laziness contract and the
fallback behaviour, none of which need R.
"""

using Test
using GAM
using DataFrames
using Random
using Statistics
using LinearAlgebra
using StatsAPI: predict

@testset "edf1 / edf2 / Vc" begin

    Random.seed!(4242)
    n = 300
    x0 = rand(n); x1 = rand(n)
    ytrue = 2 .* sin.(π .* x0) .+ exp.(2 .* x1)
    df = DataFrame(y = ytrue .+ 1.2 .* randn(n), x0 = x0, x1 = x1)
    f2 = GAM.@formula(y ~ s(x0) + s(x1))

    @testset "edf1_from_F matches its definition" begin
        # edf1 = 2·diag(F) − diag(F²); check against a dense F² for a random F
        Random.seed!(1)
        F = randn(7, 7)
        want = [2 * F[i, i] - sum(F[i, j] * F[j, i] for j in 1:7) for i in 1:7]
        @test GAM.edf1_from_F(F) ≈ want
        # and against the trace identity 2tr(F) − tr(F²)
        @test sum(GAM.edf1_from_F(F)) ≈ 2 * tr(F) - tr(F * F)
    end

    @testset "_dchol is the derivative of the Cholesky factor" begin
        Random.seed!(2)
        p = 6
        B = randn(p, p); A = B'B + p * I
        dA = Symmetric(randn(p, p)) |> Matrix
        U = Matrix(cholesky(Symmetric(A)).U)
        analytic = GAM._dchol(dA, U)
        h = 1e-6
        fd = (Matrix(cholesky(Symmetric(A + h * dA)).U) -
              Matrix(cholesky(Symmetric(A - h * dA)).U)) / (2h)
        @test maximum(abs.(analytic - fd)) < 1e-6 * maximum(abs.(fd))
    end

    @testset "lazy: Vc is deferred until first access" begin
        m = gam(f2, df)
        # edf1 is eager (free from F); edf2/Vc are not yet materialised
        @test !isempty(m.edf1)
        @test length(m.edf1) == length(m.coefficients)
        @test isempty(m.edf2)
        @test size(m.Vc) == (0, 0)
        @test m.vc_thunk !== nothing
        # forcing populates the cache and clears the thunk
        @test has_vc(m)
        @test m.vc_thunk === nothing
        @test length(m.edf2) == length(m.coefficients)
        @test size(m.Vc) == (length(m.coefficients), length(m.coefficients))
        # idempotent
        Vc1 = copy(m.Vc)
        @test has_vc(m)
        @test m.Vc == Vc1
    end

    @testset "orderings and structure" begin
        m = gam(f2, df)
        @test has_vc(m)
        e1 = m.edf1
        e2 = edf2(m)
        # edf ≤ edf1 termwise (per smooth), and edf ≤ edf2 ≤ edf1 in total
        @test all(ref_df(m) .>= GAM.edf(m) .- 1e-8)
        @test sum(e2) >= m.edf_total - 1e-8
        @test sum(e2) <= sum(e1) + 1e-8
        # Ref.df is exactly the per-smooth sum of edf1 (mgcv's convention)
        @test ref_df(m) ≈ GAM.smooth_edf(e1, m.smooths)
        # Vc is symmetric and at least as wide as Vp in every direction:
        # Vc − Vp = Vc1 + Vc2 is a sum of two PSD matrices
        Vc = vcov_corrected(m)
        @test Vc ≈ Vc'
        D = Symmetric(Vc - m.Vp)
        @test minimum(eigvals(D)) > -1e-8 * maximum(abs.(Vc))
        @test all(diag(Vc) .>= diag(m.Vp) .- 1e-10)
    end

    @testset "AIC uses edf2; conditional_aic uses edf" begin
        m = gam(f2, df)
        @test GAM.aic(m) - conditional_aic(m) ≈ 2 * (sum(edf2(m)) - m.edf_total)
        @test GAM.dof(m) - conditional_dof(m) ≈ sum(edf2(m)) - m.edf_total
        # edf2 ≥ edf, so the corrected AIC is the larger of the two
        @test GAM.aic(m) >= conditional_aic(m) - 1e-8
        @test conditional_aic(m) ≈ -2 * GAM.loglikelihood(m) + 2 * conditional_dof(m)
    end

    @testset "unconditional widens intervals" begin
        m = gam(f2, df)
        _, se_c = predict(m, df; type = :link, se = true)
        _, se_u = predict(m, df; type = :link, se = true, unconditional = true)
        @test all(se_u .>= se_c .- 1e-10)
        @test mean(se_u) > mean(se_c)          # strictly wider on average

        sc = smooth_estimates(m; select = 1, n = 20)
        su = smooth_estimates(m; select = 1, n = 20, unconditional = true)
        @test all(su.se .>= sc.se .- 1e-10)
        @test mean(su.se) > mean(sc.se)

        # posterior_samples and derivatives accept the flag too
        d1 = derivatives(m; select = 1, n = 10)
        d2 = derivatives(m; select = 1, n = 10, unconditional = true)
        @test all(d2.se .>= d1.se .- 1e-10)
    end

    @testset "unavailable: GCV, fixed sp, and the fallbacks" begin
        # mgcv likewise leaves Vc unset for GCV/UBRE
        mg = gam(f2, df; method = :GCV)
        @test !has_vc(mg)
        @test GAM.aic(mg) ≈ conditional_aic(mg)
        @test GAM.dof(mg) ≈ conditional_dof(mg)
        @test vcov_corrected(mg) === mg.Vp
        # edf1 is still available for a GCV fit, so Ref.df is still mgcv-like
        @test !isempty(mg.edf1)
        @test all(ref_df(mg) .>= GAM.edf(mg) .- 1e-8)
        # asking for unconditional intervals warns rather than silently
        # returning the narrower Vp
        @test_logs (:warn, r"unconditional=true requested") match_mode = :any begin
            predict(mg, df; type = :link, se = true, unconditional = true)
        end

        # every sp fixed -> no smoothing-parameter uncertainty to propagate
        mf = gam(GAM.@formulak(y ~ s(x0, sp = 1.0) + s(x1, sp = 1.0)), df)
        @test !has_vc(mf)
        @test GAM.aic(mf) ≈ conditional_aic(mf)
    end

    @testset "non-Gaussian families" begin
        Random.seed!(99)
        lp = (ytrue .- mean(ytrue)) ./ 3
        dp = DataFrame(y = Float64.(rand.(GAM.Distributions.Poisson.(exp.(lp)))),
            x0 = x0, x1 = x1)
        mp = gam(f2, dp; family = GAM.Distributions.Poisson(),
            link = GAM.GLM.LogLink())
        @test has_vc(mp)
        @test sum(edf2(mp)) >= mp.edf_total - 1e-8
        @test sum(edf2(mp)) <= sum(mp.edf1) + 1e-8
        @test issymmetric(round.(vcov_corrected(mp); digits = 10))

        dg = DataFrame(y = [rand(GAM.Distributions.Gamma(3.0, exp(l + 1) / 3.0))
                            for l in lp], x0 = x0, x1 = x1)
        mgam = gam(f2, dg; family = GAM.Distributions.Gamma(),
            link = GAM.GLM.LogLink())
        @test has_vc(mgam)
        @test sum(edf2(mgam)) >= mgam.edf_total - 1e-8
        @test all(isfinite, vcov_corrected(mgam))
    end

    @testset "anova table reports mgcv's Ref.df and the test rank separately" begin
        m = gam(f2, df)
        at = anova_gam(m)
        @test at.smooth_table.ref_df ≈ ref_df(m)
        @test haskey(at.smooth_table, :test_rank)
        # the rank the statistic was built with is an integer
        @test all(x -> isapprox(x, round(x)), at.smooth_table.test_rank)
    end

    @testset "fitters without Vc degrade gracefully" begin
        # bam has its own fitting path and supplies neither edf1 nor Vc; the
        # accessors must still work and fall back rather than erroring
        mb = bam(f2, df)
        @test !has_vc(mb)
        @test vcov_corrected(mb) === mb.Vp
        @test ref_df(mb) == GAM.edf(mb)
        @test GAM.aic(mb) ≈ conditional_aic(mb)
        # The old guard here (`isempty || right length`) was vacuous — it
        # passed if edf2 regressed to empty. The fallback contract is now
        # pinned for real below.
        @test length(edf2(mb)) == length(mb.coefficients)
    end

    @testset "edf2 fallback is per-coefficient edf, not edf1" begin
        # mgcv builds Vc even at fully fixed sp, where it collapses to Vp and
        # therefore edf2 collapses to per-coefficient edf = diag(F). Our
        # fallback used to return edf1 instead (measured live: sum 9.088 vs
        # mgcv's 8.032 on a fixed-sp Gamma control), contradicting both mgcv
        # and edf2's own docstring. diag(F) = 1 − diag(Vp·S_λ)/φ, so the two
        # decisive properties are: the fallback sums to edf_total exactly,
        # and it differs from edf1 whenever the fit is genuinely penalized.
        dfe = DataFrame(x = collect(range(0, 1; length = 250)))
        dfe.y = sin.(2π .* dfe.x) .+ 0.3 .* randn(StableRNG(77), 250)
        gfe = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 10, sp = 0.5)])
        me = gam(gfe, dfe)                       # fixed sp ⇒ no Vc
        @test !has_vc(me)
        e2 = edf2(me)
        @test length(e2) == length(me.coefficients)
        @test sum(e2) ≈ me.edf_total atol = 1e-8
        # edf1 ≥ edf strictly for a penalized smooth: the old fallback fails
        # this assertion, the corrected one passes it.
        @test sum(e2) < sum(me.edf1) - 1e-3
    end
end
