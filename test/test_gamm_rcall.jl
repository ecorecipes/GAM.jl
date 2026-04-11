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
import Distributions: Poisson, Binomial, Gamma

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
        @test cor(fit_jl, fitted_r) > 0.99
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
        # Use full fitted values from lme (includes RE), on response scale
        reval("fitted_r <- exp(as.numeric(fitted(m_r\$lme)))")

        re_r = rcopy(reval("re_r"))
        fitted_r = rcopy(reval("fitted_r"))

        # Random effects should correlate with truth
        @test cor(est_jl, true_re) > 0.7
        @test cor(re_r, true_re) > 0.7

        # Julia and R random effects should be correlated
        @test cor(est_jl, re_r) > 0.99

        # Fitted values should be correlated (on response scale, including RE)
        # Both Julia and R use PQL, so full fitted values should match closely.
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.99
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
        m_gam_re = gam(@formulak(y ~ s(x) + s(group, bs = :re)), df)

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

        @test cor(fit_gamm, fit_gam_re) > 0.99
        @test cor(fit_gamm, fitted_r_vec) > 0.99
        @test cor(fit_gam_re, fitted_r_vec) > 0.99

        # Scale estimates should be similar
        @test abs(m_gamm.gam_model.scale - r_scale) / r_scale < 0.5
        @test abs(m_gam_re.scale - r_scale) / r_scale < 0.2
    end

    # ========================================================================
    # Random intercept + slope: gamm(y ~ s(x) + (x|group)) vs R
    # ========================================================================
    @testset "Random intercept + slope vs R gamm()" begin
        rng = StableRNG(77)
        n_groups = 8
        n_per = 60
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re_int = randn(rng, n_groups) * 0.4
        true_re_slope = randn(rng, n_groups) * 0.2
        y = sin.(x) .+ true_re_int[group] .+ true_re_slope[group] .* x .+ 0.3 .* randn(rng, n)
        group_str = string.(group)
        df = DataFrame(x = x, y = y, group = group_str)

        # Julia: random intercept + slope
        m_jl = gamm(@gamm_formula(y ~ s(x) + (x | group)), df)
        re_jl = ranef(m_jl)
        vc_jl = VarCorr(m_jl)

        @test m_jl isa GammModel
        @test m_jl.gam_model.converged

        # R: gamm(y ~ s(x), random=list(group=~x))
        @rput x y group_str
        reval("library(mgcv); library(nlme)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~x), data=df_r)")
        reval("fitted_r <- fitted(m_r\$lme)")
        reval("vc <- VarCorr(m_r\$lme)")
        reval("sigma_int_r <- as.numeric(vc[rownames(vc)=='(Intercept)', 'StdDev'])")
        reval("sigma_slope_r <- as.numeric(vc[rownames(vc)=='x', 'StdDev'])")

        fitted_r = rcopy(reval("fitted_r"))
        σ_int_r = rcopy(reval("sigma_int_r"))
        σ_slope_r = rcopy(reval("sigma_slope_r"))

        # Fitted values should be well correlated
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.99

        # Both Julia and R should recover reasonable variance components
        @test vc_jl[1].std > 0.0
        @test σ_int_r > 0.0
        @test σ_slope_r > 0.0
    end

    # ========================================================================
    # Multiple RE groups: (1|site) + (1|subject) vs R
    # ========================================================================
    @testset "Multiple RE groups vs R gamm()" begin
        rng = StableRNG(55)
        n_sites = 5
        n_subjects = 10
        n_per = 30
        n = n_sites * n_subjects * n_per ÷ n_sites  # 300
        site = repeat(1:n_sites, inner = n ÷ n_sites)
        subject = repeat(1:n_subjects, n ÷ n_subjects)
        x = randn(rng, n)
        re_site = randn(rng, n_sites) * 0.3
        re_subject = randn(rng, n_subjects) * 0.4
        y = sin.(x) .+ re_site[site] .+ re_subject[subject] .+ 0.25 .* randn(rng, n)
        site_str = string.(site)
        subject_str = string.(subject)
        df = DataFrame(x = x, y = y, site = site_str, subject = subject_str)

        # Julia: two crossed random intercepts
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | site) + (1 | subject)), df)
        vc_jl = VarCorr(m_jl)

        @test m_jl isa GammModel
        @test m_jl.gam_model.converged
        @test length(m_jl.random_effects) == 2
        # VarCorr: 2 RE + 1 Residual
        @test length(vc_jl) == 3
        @test vc_jl[1].group == :site
        @test vc_jl[2].group == :subject
        @test vc_jl[3].group == :Residual

        # R: gamm with two random effects
        @rput x y site_str subject_str
        reval("library(mgcv); library(nlme)")
        reval("df_r <- data.frame(x=x, y=y, site=factor(site_str), subject=factor(subject_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(site=~1, subject=~1), data=df_r)")
        reval("vc <- VarCorr(m_r\$lme)")
        reval("fitted_r <- fitted(m_r\$lme)")

        fitted_r = rcopy(reval("fitted_r"))

        # Fitted values should be well correlated
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.99

        # Both should have positive variance components
        @test vc_jl[1].variance > 0.0  # site
        @test vc_jl[2].variance > 0.0  # subject
    end

    # ========================================================================
    # Variance component comparison (σ²_re)
    # ========================================================================
    @testset "Variance component magnitude comparison vs R" begin
        rng = StableRNG(321)
        n_groups = 12
        n_per = 50
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        σ_re_true = 0.6
        true_re = randn(rng, n_groups) * σ_re_true
        σ_eps_true = 0.25
        y = sin.(x) .+ true_re[group] .+ σ_eps_true .* randn(rng, n)
        group_str = string.(group)
        df = DataFrame(x = x, y = y, group = group_str)

        # Julia
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
        vc_jl = VarCorr(m_jl)
        σ_re_jl = vc_jl[1].std
        σ_res_jl = vc_jl[end].std

        # R
        @rput x y group_str
        reval("library(mgcv); library(nlme)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~1), data=df_r)")
        reval("vc <- VarCorr(m_r\$lme)")
        reval("sigma_re_r <- as.numeric(vc[rownames(vc)=='(Intercept)', 'StdDev'])")
        reval("sigma_res_r <- as.numeric(vc[rownames(vc)=='Residual', 'StdDev'])")

        σ_re_r = rcopy(reval("sigma_re_r"))
        σ_res_r = rcopy(reval("sigma_res_r"))

        # Both should be in the right ballpark of truth
        @test σ_re_jl > σ_re_true * 0.3  # not too small
        @test σ_re_jl < σ_re_true * 2.5  # not too large
        @test σ_re_r > σ_re_true * 0.3
        @test σ_re_r < σ_re_true * 2.5

        # Julia and R RE std dev should be in the same order of magnitude
        ratio = σ_re_jl / σ_re_r
        @test ratio > 0.3 && ratio < 3.0

        # Residual std should also be comparable
        ratio_res = σ_res_jl / σ_res_r
        @test ratio_res > 0.3 && ratio_res < 3.0
    end

    # ========================================================================
    # Binomial GAMM: random intercept
    # ========================================================================
    @testset "Binomial GAMM: random intercept vs R gamm()" begin
        rng = StableRNG(456)
        n_groups = 8
        n_per = 100
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re = randn(rng, n_groups) * 0.4
        η = 0.8 .* x .+ true_re[group]
        p = 1.0 ./ (1.0 .+ exp.(-η))
        y = Float64.([rand(rng) < p[i] ? 1.0 : 0.0 for i in 1:n])
        group_str = string.(group)
        df = DataFrame(x = x, y = y, group = group_str)

        # Julia fit (default link for Binomial is LogitLink)
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df, Binomial())
        re_jl = ranef(m_jl)
        est_jl = vec(re_jl.group.effects)

        # R fit
        @rput x y group_str
        reval("library(mgcv); library(MASS)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~1), family=binomial(), data=df_r)")
        reval("re_r <- unlist(ranef(m_r\$lme)\$group)")
        reval("fitted_r <- 1/(1+exp(-as.numeric(fitted(m_r\$lme))))")

        re_r = rcopy(reval("re_r"))
        fitted_r = rcopy(reval("fitted_r"))

        # RE should correlate with truth
        @test cor(est_jl, true_re) > 0.7
        @test cor(re_r, true_re) > 0.7

        # Julia and R RE should match closely
        @test cor(est_jl, re_r) > 0.99

        # Full fitted values (response scale, including RE)
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.99
    end

    # ========================================================================
    # Gamma GAMM: random intercept
    # ========================================================================
    @testset "Gamma GAMM: random intercept vs R gamm()" begin
        rng = StableRNG(789)
        n_groups = 8
        n_per = 80
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(rng, n)
        true_re = randn(rng, n_groups) * 0.2
        η = 1.0 .+ 0.5 .* x .+ true_re[group]  # positive mean on log scale
        μ = exp.(η)
        shape = 5.0
        y = [rand(rng, Gamma(shape, μ[i] / shape)) for i in 1:n]
        group_str = string.(group)
        df = DataFrame(x = x, y = y, group = group_str)

        # Julia fit (use LogLink for Gamma to match R)
        m_jl = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df, Gamma(); link = LogLink())
        re_jl = ranef(m_jl)
        est_jl = vec(re_jl.group.effects)

        # R fit
        @rput x y group_str
        reval("library(mgcv); library(MASS)")
        reval("df_r <- data.frame(x=x, y=y, group=factor(group_str))")
        reval("m_r <- gamm(y ~ s(x), random=list(group=~1), family=Gamma(link='log'), data=df_r)")
        reval("re_r <- unlist(ranef(m_r\$lme)\$group)")
        reval("fitted_r <- exp(as.numeric(fitted(m_r\$lme)))")

        re_r = rcopy(reval("re_r"))
        fitted_r = rcopy(reval("fitted_r"))

        # RE should correlate with truth
        @test cor(est_jl, true_re) > 0.7
        @test cor(re_r, true_re) > 0.7

        # Julia and R RE should match closely
        @test cor(est_jl, re_r) > 0.99

        # Full fitted values (response scale, including RE)
        fit_jl = fitted(m_jl)
        @test cor(fit_jl, fitted_r) > 0.99
    end
end
