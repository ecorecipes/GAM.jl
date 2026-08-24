using Test
using GAM
using RCall
using DataFrames
using Random
using Statistics
using ForwardDiff
using StatsAPI: fitted
using GAM: ScatFamily, scat_Dd, _scat_loglik, _deviance, _variance

# Direct comparison against mgcv's scat() (mgcv >= 1.9). The Dd derivatives are
# transcribed from mgcv's source, so they are held to 1e-12 — the two agree to
# the last ulp, and only the order of floating-point operations differs. Fitted
# models are compared much more loosely, because GAM.jl estimates (ν, σ) by
# alternating profile likelihood while mgcv folds them into the REML criterion.

R"suppressMessages(library(mgcv))"

@testset "Scaled t (scat) — R comparison" begin

    # ========================================================================
    # Family primitives: exact agreement with mgcv's own closures
    # ========================================================================
    @testset "Dd, deviance, aic and variance match mgcv exactly" begin
        rng = MersenneTwister(11)
        n = 60
        y = randn(rng, n) .* 2.0 .+ 1.0
        mu = randn(rng, n) .* 1.5
        min_df = 3.0

        for (ν, σ) in ((5.7, 1.3), (3.5, 0.4), (25.0, 2.0))
            f = ScatFamily(; nu = ν, sigma = σ, min_df = min_df)
            wt = ones(n)               # mgcv drops the prior weight from EDmu2
            jl = scat_Dd(f, y, mu, wt; level = 1)

            R"""
            fam <- scat(theta = c($ν, $σ), min.df = $min_df)
            th  <- fam$getTheta(FALSE)
            dd  <- fam$Dd($y, $mu, th, $wt, level = 1)
            dev <- sum(fam$dev.resids($y, $mu, $wt))
            aic <- fam$aic($y, $mu, th, $wt, 0)
            vv  <- fam$variance($mu)
            """
            # The internal parameterisation itself must line up:
            # ν = exp(θ₁) + min.df, σ = exp(θ₂).
            @test rcopy(R"th") ≈ [log(ν - min_df), log(σ)] rtol = 1e-12

            ddR = rcopy(R"dd")
            # Agreement is to the last ulp — the formulae are the same, only the
            # order of floating-point operations differs from R's.
            for k in (:Dmu, :Dmu2, :Dmu3, :Dth, :Dmuth, :Dmu2th, :EDmu2, :EDmu2th, :EDmu3)
                @test jl[k] ≈ ddR[k] rtol = 1e-12
            end

            @test _deviance(f, y, mu, wt) ≈ rcopy(R"dev") rtol = 1e-12
            @test _scat_loglik(ν, σ, y, mu, wt) ≈ -rcopy(R"aic") / 2 rtol = 1e-12
            # mgcv returns a single value here: V(μ) does not depend on μ.
            @test all(_variance(f, mu) .≈ rcopy(R"vv"))
        end
    end

    @testset "EDmu2 differs from mgcv only by the prior weight" begin
        # mgcv's scat() omits `wt` from EDmu2 and from the first column of
        # EDmu2th, while including it everywhere else (and while mgcv's own
        # nb() includes it). GAM.jl includes it; this pins the difference down
        # to exactly that factor so the divergence stays deliberate.
        rng = MersenneTwister(11)
        n = 40
        y = randn(rng, n)
        mu = randn(rng, n) .* 0.5
        wt = rand(rng, n) .+ 0.5
        f = ScatFamily(; nu = 6.0, sigma = 1.0)
        jl = scat_Dd(f, y, mu, wt; level = 1)
        R"""
        fam <- scat(theta = c(6.0, 1.0))
        dd  <- fam$Dd($y, $mu, fam$getTheta(FALSE), $wt, level = 1)
        """
        ddR = rcopy(R"dd")
        @test jl[:EDmu2] ≈ wt .* ddR[:EDmu2] rtol = 1e-12
        @test jl[:EDmu2th][:, 1] ≈ wt .* ddR[:EDmu2th][:, 1] rtol = 1e-12
        # ... and the unweighted values genuinely disagree, so the test above is
        # pinning down a real divergence rather than a tautology.
        @test !isapprox(jl[:EDmu2], ddR[:EDmu2]; rtol = 1e-3)
        # Everything else is unaffected by the discrepancy.
        @test jl[:Dmu] ≈ ddR[:Dmu] rtol = 1e-12
        @test jl[:Dth] ≈ ddR[:Dth] rtol = 1e-12
    end

    # ========================================================================
    # Fitted models
    # ========================================================================
    f_true(x) = sin(2π * x) + 0.4 * x

    function _t_draws(rng, n, ν)
        [randn(rng) / sqrt(sum(abs2, randn(rng, Int(ν))) / ν) for _ in 1:n]
    end

    function _data(kind; n = 300, seed = 7)
        rng = MersenneTwister(seed)
        x = sort(rand(rng, n))
        ft = f_true.(x)
        if kind === :gauss
            y = ft .+ 0.3 .* randn(rng, n)
        elseif kind === :t3
            y = ft .+ 0.3 .* _t_draws(rng, n, 3.0)
        else
            y = ft .+ 0.3 .* randn(rng, n)
            y[[10, 60, 120, 180, 240, 290]] .+= [8.0, -9.0, 7.0, -8.5, 9.5, -7.5]
        end
        DataFrame(x = x, y = y), ft
    end

    @testset "fits agree with mgcv::gam(family = scat())" begin
        for kind in (:gauss, :t3, :outlier)
            df, ft = _data(kind)
            m = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
            @rput df
            R"""
            mr <- gam(y ~ s(x, k = 10), data = df, family = scat(), method = "REML")
            th <- mr$family$getTheta(TRUE)
            out <- list(fit = as.numeric(fitted(mr)), edf = sum(mr$edf),
                        nu = th[1], sig = th[2])
            """
            r = rcopy(R"out")

            @test m.converged
            # Fitted values track mgcv closely. They are not identical because
            # the two packages estimate (ν, σ) by different criteria; the
            # residual disagreement is ~1% of the fitted range.
            @test cor(fitted(m), r[:fit]) > 0.999
            @test maximum(abs.(fitted(m) .- r[:fit])) <
                  0.03 * (maximum(r[:fit]) - minimum(r[:fit]))
            @test m.edf_total ≈ r[:edf] atol = 0.3
            @test m.family.sigma ≈ r[:sig] rtol = 0.10

            if kind === :gauss
                # Both must run ν off to the Gaussian limit.
                @test m.family.nu > 100.0
                @test r[:nu] > 100.0
            else
                @test m.family.nu ≈ r[:nu] rtol = 0.25
                @test m.family.nu < 15.0
            end
        end
    end

    @testset "robustness: scat recovers the truth where Gaussian fails" begin
        # The substantive claim for the family, measured against R so that both
        # the robust and the non-robust reference come from mgcv.
        df, ft = _data(:outlier)
        m = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
        @rput df
        R"""
        ms <- gam(y ~ s(x, k = 10), data = df, family = scat(), method = "REML")
        mg <- gam(y ~ s(x, k = 10), data = df, method = "REML")
        out <- list(scat = as.numeric(fitted(ms)), gauss = as.numeric(fitted(mg)))
        """
        r = rcopy(R"out")
        rmse(p) = sqrt(mean(abs2, p .- ft))

        # GAM.jl's robust fit must be in the same league as mgcv's, and both
        # must beat the Gaussian fit by a wide margin.
        @test rmse(fitted(m)) < 1.3 * rmse(r[:scat])
        @test rmse(fitted(m)) < 0.4 * rmse(r[:gauss])
        @test rmse(r[:scat]) < 0.4 * rmse(r[:gauss])

        # Heavy-tailed (not merely contaminated) noise: same story, smaller gap.
        df3, ft3 = _data(:t3)
        m3 = gam(@formula(y ~ s(x, k = 10)), df3; family = ScatFamily())
        @rput df3
        R"""
        mg3 <- gam(y ~ s(x, k = 10), data = df3, method = "REML")
        gauss3 <- as.numeric(fitted(mg3))
        """
        rmse3(p) = sqrt(mean(abs2, p .- ft3))
        @test rmse3(fitted(m3)) < rmse3(rcopy(R"gauss3"))
    end

    @testset "fixed theta reproduces mgcv's fixed-theta fit" begin
        # With (ν, σ) held fixed both packages optimise the same penalised
        # deviance, so agreement should be much tighter than in the estimated
        # case — this isolates the fitting path from the θ-estimation path.
        df, _ = _data(:t3)
        ν, σ = 5.0, 0.32
        m = gam(@formula(y ~ s(x, k = 10)), df;
            family = ScatFamily(; nu = ν, sigma = σ, estimate_theta = false))
        @rput df
        R"""
        mr <- gam(y ~ s(x, k = 10), data = df, family = scat(theta = c($ν, $σ)),
                  method = "REML")
        out <- list(fit = as.numeric(fitted(mr)), edf = sum(mr$edf),
                    dev = deviance(mr))
        """
        r = rcopy(R"out")
        @test m.family.nu == ν
        @test m.family.sigma == σ
        @test maximum(abs.(fitted(m) .- r[:fit])) <
              0.02 * (maximum(r[:fit]) - minimum(r[:fit]))
        @test m.edf_total ≈ r[:edf] atol = 0.3
        @test GAM.deviance(m) ≈ r[:dev] rtol = 0.02
    end
end
