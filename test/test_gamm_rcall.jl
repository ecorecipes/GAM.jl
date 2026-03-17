# Integration tests: compare GAM.jl GAMM results against R mgcv::gamm via RCall
#
# These tests verify that GAM.jl's gamm() produces statistically equivalent
# results to R mgcv's gamm(). We compare random effect estimates, variance
# components, and fitted values on the same data.
#
# Requirements: R with mgcv and nlme installed, RCall.jl

using Test
using GAM
using RCall
using DataFrames
using Statistics
using StatsAPI: fitted, coef, deviance, nobs
using StableRNGs
import Distributions: Poisson

@testset "GAMM R Integration Tests (mgcv::gamm)" begin

    # ========================================================================
    # Gaussian GAMM: random intercept
    # ========================================================================
    @testset "Gaussian GAMM: random intercept vs R gamm()" begin
        rng = StableRNG(42)
        n_groups = 10
        n_per = 50
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re = randn(rng, n_groups) * 0.5
        y = sin.(x) .+ true_re[group] .+ 0.3 .* randn(rng, n)
        group_str = string.(group)
        df = DataFrame(x = x, y = y, group = group_str)

        # Julia fit
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
        re_jl = ranef(m_jl)
        est_jl = vec(re_jl.group.effects)
        vc_jl = VarCorr(m_jl)
        σ_re_jl = vc_jl[1].std

        # R fit
        @rput x y group_str
        reval("library(mgcv); library(nlme)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~1), data=df_r)")
        reval("re_r <- unlist(ranef(m_r\$lme)\$group)")
        reval("vc <- VarCorr(m_r\$lme)")
        reval("sigma_re_r <- as.numeric(vc[rownames(vc)=='(Intercept)', 'StdDev'])")
        # Use fitted from lme (includes RE) for comparison
        reval("fitted_r <- fitted(m_r\$lme)")

        re_r = rcopy(reval("re_r"))
        σ_re_r = rcopy(reval("sigma_re_r"))
        fitted_r = rcopy(reval("fitted_r"))

        # Julia and R random effects should be nearly identical
        @test cor(est_jl, re_r) > 0.99

        # Variance components should be in same ballpark
        @test σ_re_jl > 0.1
        @test σ_re_r > 0.1

        # Fitted values should be highly correlated
        # (fitted includes both smooth + RE contributions)
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.95
    end

    # ========================================================================
    # Poisson GAMM: random intercept
    # ========================================================================
    @testset "Poisson GAMM: random intercept vs R gamm()" begin
        rng = StableRNG(123)
        n_groups = 8
        n_per = 80
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re = randn(rng, n_groups) * 0.3
        η = 0.5 .* x .+ true_re[group]
        y = [rand(rng, Poisson(exp(η[i]))) for i in 1:n]
        yf = Float64.(y)
        group_str = string.(group)
        df = DataFrame(x = x, y = yf, group = group_str)

        # Julia fit
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df, Poisson())
        re_jl = ranef(m_jl)
        est_jl = vec(re_jl.group.effects)

        # R fit (gammPQL for non-Gaussian)
        @rput x yf group_str
        reval("library(mgcv); library(MASS)")
        reval("df_r <- data.frame(x=x, y=yf, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~1), family=poisson(), data=df_r)")
        reval("re_r <- unlist(ranef(m_r\$lme)\$group)")
        reval("fitted_r <- fitted(m_r\$gam)")

        re_r = rcopy(reval("re_r"))
        fitted_r = rcopy(reval("fitted_r"))

        # Random effects should correlate with truth
        @test cor(est_jl, true_re) > 0.7
        @test cor(re_r, true_re) > 0.7

        # Julia and R random effects should be correlated
        @test cor(est_jl, re_r) > 0.7

        # Fitted values should be correlated (on response scale)
        # Note: Julia uses PIRLS+REML, R uses PQL — different algorithms
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.85
    end

    # ========================================================================
    # Compare GAMM vs s(group, bs="re") in both Julia and R
    # ========================================================================
    @testset "GAMM vs s(group, bs='re') equivalence (Julia and R)" begin
        rng = StableRNG(999)
        n_groups = 6
        n_per = 60
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re = randn(rng, n_groups) * 0.4
        y = cos.(x) .+ true_re[group] .+ 0.25 .* randn(rng, n)
        df = DataFrame(x = x, y = y, group = group)

        # Julia: GAMM
        m_gamm = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
        # Julia: GAM with s(group, bs=:re)
        m_gam_re = gam(@gam_formula(y ~ s(x) + s(group, bs = :re)), df)

        # R: GAM with s(group, bs="re")
        @rput x y group
        reval("library(mgcv)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group))")
        reval("m_r <- gam(y ~ s(x) + s(group, bs='re'), data=df_r, method='REML')")
        reval("fitted_r <- fitted(m_r)")
        reval("scale_r <- m_r\$scale")

        fitted_r_vec = rcopy(reval("fitted_r"))
        r_scale = rcopy(reval("scale_r"))

        # All three fits should produce similar fitted values
        fit_gamm = fitted(m_gamm)
        fit_gam_re = fitted(m_gam_re)

        @test cor(fit_gamm, fit_gam_re) > 0.95
        @test cor(fit_gamm, fitted_r_vec) > 0.95
        @test cor(fit_gam_re, fitted_r_vec) > 0.99

        # Scale estimates should be similar
        @test abs(m_gamm.gam_model.scale - r_scale) / r_scale < 0.5
        @test abs(m_gam_re.scale - r_scale) / r_scale < 0.2
    end
end
