# Regression tests for core-inference and smoothing-selection fixes.
using Test
using GAM
using DataFrames
using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsAPI: fitted, predict, coef, deviance, loglikelihood, aic, residuals, vcov

@testset "Core inference / smoothing-selection fixes" begin

    @testset "leverage sums to EDF (weighted, non-Gaussian)" begin
        rng = Xoshiro(1)
        n = 400
        x = rand(rng, n) .* 2pi
        # Poisson
        yp = Float64.(rand.(rng, Poisson.(exp.(0.4 .* sin.(x) .+ 1.0))))
        mp = gam(GAM.@formula(y ~ s(x, k=10)), DataFrame(y=yp, x=x), Poisson(),
                 GAM.GLM.LogLink())
        @test isapprox(sum(mp.hat_matrix_diag), mp.edf_total; rtol=1e-6)
        # Weighted Gaussian
        w = rand(rng, n) .+ 0.5
        mw = gam(GAM.@formula(y ~ s(x, k=8)),
                 DataFrame(y=sin.(x) .+ 0.2 .* randn(rng, n), x=x); weights=w)
        @test isapprox(sum(mw.hat_matrix_diag), mw.edf_total; rtol=1e-6)
    end

    @testset "Ve = F*Vp (no extra hat factor) — Ve ⪯ Vp diag for Gaussian" begin
        rng = Xoshiro(2)
        n = 300
        x = rand(rng, n)
        m = gam(GAM.@formula(y ~ s(x, k=10)),
                DataFrame(y=sin.(2pi .* x) .+ 0.2 .* randn(rng, n), x=x))
        # frequentist variance ≤ Bayesian variance elementwise on the diagonal
        @test all(diag(m.Ve) .<= diag(m.Vp) .+ 1e-8)
        @test issymmetric(Symmetric(m.Ve))
    end

    @testset "GCV actually optimizes (differs from REML, recovers signal)" begin
        rng = Xoshiro(3)
        n = 400
        x = rand(rng, n) .* 2pi
        f = sin.(x)
        y = f .+ 0.3 .* randn(rng, n)
        df = DataFrame(y=y, x=x)
        m_reml = gam(GAM.@formula(y ~ s(x, k=15)), df; method=:REML)
        m_gcv  = gam(GAM.@formula(y ~ s(x, k=15)), df; method=:GCV)
        # GCV should not be a no-op pinned to the REML answer
        @test !isapprox(m_gcv.sp, m_reml.sp; rtol=1e-3)
        # but should still recover the signal well
        @test cor(fitted(m_gcv), f) > 0.95
    end

    @testset "UBRE works for Poisson, errors for Gaussian" begin
        rng = Xoshiro(4)
        n = 400
        x = rand(rng, n) .* 2pi
        yp = Float64.(rand.(rng, Poisson.(exp.(0.5 .* sin.(x) .+ 1.0))))
        mp = gam(GAM.@formula(y ~ s(x, k=12)), DataFrame(y=yp, x=x), Poisson(),
                 GAM.GLM.LogLink(); method=:UBRE)
        @test mp.converged
        @test cor(fitted(mp), exp.(0.5 .* sin.(x) .+ 1.0)) > 0.9
        # UBRE assumes known scale → not valid for Gaussian
        @test_throws ArgumentError gam(GAM.@formula(y ~ s(x, k=12)),
            DataFrame(y=sin.(x) .+ randn(rng, n), x=x); method=:UBRE)
    end

    @testset "Gamma log-likelihood is family-specific (not Gaussian)" begin
        rng = Xoshiro(5)
        n = 400
        x = rand(rng, n) .* 2pi
        mu = exp.(0.5 .+ 0.3 .* sin.(x))
        yg = rand.(rng, Gamma.(2.0, mu ./ 2.0))
        mg = gam(GAM.@formula(y ~ s(x, k=10)), DataFrame(y=yg, x=x), Gamma(),
                 GAM.GLM.LogLink())
        # Compare with the explicit family likelihood at the fitted values
        phi = mg.scale
        ll_manual = sum(logpdf(Gamma(mg.weights[i]/phi, fitted(mg)[i]*phi/mg.weights[i]),
                               yg[i]) for i in 1:n)
        @test isapprox(loglikelihood(mg), ll_manual; rtol=1e-8)
        # deviance residuals use the Gamma form, not the Gaussian fallback
        rd = residuals(mg; type=:deviance)
        @test !isapprox(rd, sign.(yg .- fitted(mg)) .* sqrt.(abs.(yg .- fitted(mg)).^2))
        @test all(isfinite, rd)
    end

    @testset "null deviance uses weighted mean" begin
        rng = Xoshiro(6)
        n = 300
        x = rand(rng, n)
        y = 2.0 .+ sin.(2pi .* x) .+ 0.2 .* randn(rng, n)
        w = rand(rng, n) .* 3 .+ 0.2
        m = gam(GAM.@formula(y ~ s(x, k=8)), DataFrame(y=y, x=x); weights=w)
        # null deviance from the weighted-mean null model
        mu0 = sum(w .* y) / sum(w)
        @test isapprox(m.null_deviance, sum(w .* (y .- mu0).^2); rtol=1e-8)
    end

    @testset "Wood (2013) smooth p-values are sane" begin
        rng = Xoshiro(7)
        n = 500
        x = rand(rng, n) .* 2pi
        z = rand(rng, n)
        # x has real signal, z is noise
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(y=y, x=x, z=z)
        m = gam(GAM.@formula(y ~ s(x, k=15) + s(z, k=10)), df)
        at = GAM.anova_gam(m)
        ix = findfirst(==("s(x,bs=tp)"), at.smooth_table.label)
        iz = findfirst(==("s(z,bs=tp)"), at.smooth_table.label)
        @test at.smooth_table.p_value[ix] < 1e-3      # real effect: significant
        @test at.smooth_table.p_value[iz] > 0.05      # noise: not significant
        @test all(0 .<= at.smooth_table.p_value .<= 1)
    end
end
