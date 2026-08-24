# R comparison tests for multi-parameter models (evgam)
# Compares Julia evgam() output against R evgam package

using Test
using GAM
using RCall
using DataFrames
using Statistics
using StableRNGs
using Distributions
using StatsAPI: fitted

@testset "evgam R comparison" begin

    # ================================================================
    # Test 20: GEV constant model vs R optim (evgam has bug with y~1)
    # ================================================================
    @testset "20. GEV constant model vs R" begin
        # Generate GEV data in R and fit via direct MLE using optim
        R"""
        set.seed(42)
        n <- 300L
        y <- evd::rgev(n, loc=2.0, scale=0.5, shape=0.1)
        # Direct MLE via optim (parameterized as mu, log_sigma, xi)
        gev_nll <- function(par, y) {
            mu <- par[1]; lpsi <- par[2]; xi <- par[3]
            -sum(evd::dgev(y, loc=mu, scale=exp(lpsi), shape=xi, log=TRUE))
        }
        fit_r <- optim(c(mean(y), log(sd(y)), 0.05), gev_nll, y=y, method="Nelder-Mead",
                       control=list(maxit=5000))
        coef_r <- fit_r$par
        nll_r <- fit_r$value
        """
        coef_r = rcopy(R"coef_r")
        nll_r = rcopy(R"nll_r")
        y_r = rcopy(R"y")

        # Fit in Julia with same data
        df = DataFrame(y=Float64.(y_r))
        m_j = evgam(
            [@formulak(y ~ 1), @formulak(y ~ 1), @formulak(y ~ 1)],
            df, GEVFamily()
        )

        # Compare coefficients (3 params: μ, log σ, ξ)
        @test length(m_j.coefficients) == 3

        # Location
        @test m_j.coefficients[1] ≈ coef_r[1] atol=0.1
        # Log-scale
        @test param_coef(m_j, 2)[1] ≈ coef_r[2] atol=0.1
        # Shape
        @test param_coef(m_j, 3)[1] ≈ coef_r[3] atol=0.1

        # NLL should be very close (both are MLE)
        @test m_j.nll ≈ nll_r atol=0.5
    end

    # ================================================================
    # Test 21: GEV with smooth location vs R evgam
    # ================================================================
    @testset "21. GEV smooth location vs R" begin
        R"""
        set.seed(123)
        library(evgam)
        n <- 400L
        x <- seq(0, 3, length.out=n)
        mu_true <- 2 + 0.5 * sin(2*pi*x)
        y <- evd::rgev(n, loc=mu_true, scale=0.5, shape=0.1)
        dat <- data.frame(y=y, x=x)
        m_r <- evgam(list(y ~ s(x, bs="cr", k=10), ~ 1, ~ 1), dat, family="gev")
        coef_r <- m_r$coefficients
        nll_r <- -m_r$logLik
        sp_r <- m_r$sp
        """

        coef_r = rcopy(R"coef_r")
        nll_r = rcopy(R"nll_r")
        y_r = rcopy(R"dat$y")
        x_r = rcopy(R"dat$x")

        df = DataFrame(y=Float64.(y_r), x=Float64.(x_r))
        m_j = evgam(
            [@formulak(y ~ s(x, bs=:cr, k=10)),
             @formulak(y ~ 1),
             @formulak(y ~ 1)],
            df, GEVFamily()
        )

        @test m_j.converged

        # NLL should be within 5% of R
        @test abs(m_j.nll - nll_r) / abs(nll_r) < 0.05

        # Log-scale and shape should agree
        @test param_coef(m_j, 2)[1] ≈ coef_r[end-1] atol=0.15
        @test param_coef(m_j, 3)[1] ≈ coef_r[end] atol=0.15
    end

    # ================================================================
    # Test 22: GPD constant model vs R optim (evgam has bug with y~1)
    # ================================================================
    @testset "22. GPD constant model vs R" begin
        R"""
        set.seed(77)
        n <- 500L
        y <- evd::rgpd(n, loc=0, scale=1.0, shape=0.15)
        # Direct MLE via optim
        gpd_nll <- function(par, y) {
            lpsi <- par[1]; xi <- par[2]
            -sum(evd::dgpd(y, loc=0, scale=exp(lpsi), shape=xi, log=TRUE))
        }
        fit_r <- optim(c(log(sd(y)), 0.05), gpd_nll, y=y, method="Nelder-Mead",
                       control=list(maxit=5000))
        coef_r <- fit_r$par
        nll_r <- fit_r$value
        """

        coef_r = rcopy(R"coef_r")
        nll_r = rcopy(R"nll_r")
        y_r = rcopy(R"y")

        df = DataFrame(y=Float64.(y_r))
        m_j = evgam(
            [@formulak(y ~ 1), @formulak(y ~ 1)],
            df, GPDFamily()
        )

        @test length(m_j.coefficients) == 2

        # Log-scale
        @test param_coef(m_j, 1)[1] ≈ coef_r[1] atol=0.1
        # Shape
        @test param_coef(m_j, 2)[1] ≈ coef_r[2] atol=0.1

        @test m_j.nll ≈ nll_r atol=0.5
    end

    # ================================================================
    # Test 23: GPD with smooth log-scale vs R evgam
    # ================================================================
    @testset "23. GPD smooth log-scale vs R" begin
        R"""
        set.seed(55)
        library(evgam)
        n <- 400L
        x <- seq(0, 3, length.out=n)
        sigma_true <- exp(0.3 * x)
        y <- evd::rgpd(n, loc=0, scale=sigma_true, shape=0.1)
        dat <- data.frame(y=y, x=x)
        m_r <- evgam(list(y ~ s(x, bs="cr", k=8), ~ 1), dat, family="gpd")
        coef_r <- m_r$coefficients
        nll_r <- -m_r$logLik
        """

        coef_r = rcopy(R"coef_r")
        nll_r = rcopy(R"nll_r")
        y_r = rcopy(R"dat$y")
        x_r = rcopy(R"dat$x")

        df = DataFrame(y=Float64.(y_r), x=Float64.(x_r))
        m_j = evgam(
            [@formulak(y ~ s(x, bs=:cr, k=8)),
             @formulak(y ~ 1)],
            df, GPDFamily()
        )

        @test m_j.converged

        # NLL within 5%
        @test abs(m_j.nll - nll_r) / abs(nll_r) < 0.05

        # Shape should agree well
        @test param_coef(m_j, 2)[1] ≈ coef_r[end] atol=0.15
    end

    # ================================================================
    # Test 24: GEV NLL value matches R
    # ================================================================
    @testset "24. GEV NLL matches R exactly" begin
        R"""
        library(evgam)
        y <- c(2.5, 3.1, 1.8, 4.0, 2.2)
        # Compute NLL at known parameter values
        mu <- 2.0; lpsi <- 0.3; xi <- 0.15
        sigma <- exp(lpsi)
        nll_r <- -sum(evd::dgev(y, loc=mu, scale=sigma, shape=xi, log=TRUE))
        """
        nll_r = rcopy(R"nll_r")

        fam = GEVFamily()
        y = [2.5, 3.1, 1.8, 4.0, 2.2]
        η = [fill(2.0, 5), fill(0.3, 5), fill(0.15, 5)]
        nll_j = nll_total(fam, y, η)

        @test nll_j ≈ nll_r atol=1e-8
    end

    # ================================================================
    # Test 25: GPD NLL value matches R
    # ================================================================
    @testset "25. GPD NLL matches R exactly" begin
        R"""
        library(evgam)
        y <- c(0.5, 1.2, 0.8, 2.5, 0.3)
        lpsi <- 0.2; xi <- 0.2
        sigma <- exp(lpsi)
        nll_r <- -sum(evd::dgpd(y, loc=0, scale=sigma, shape=xi, log=TRUE))
        """
        nll_r = rcopy(R"nll_r")

        fam = GPDFamily()
        y = [0.5, 1.2, 0.8, 2.5, 0.3]
        η = [fill(0.2, 5), fill(0.2, 5)]
        nll_j = nll_total(fam, y, η)

        @test nll_j ≈ nll_r atol=1e-8
    end
    # ── Round-4 parity: fitted GEV location curve vs evgam ─────────────────
    # Measured: elementwise max-abs 1.85e-4, cor 1.000000 on the fitted
    # location linear predictor (smooth coefficients themselves are not
    # compared — parameterizations differ; the fitted curve is the
    # basis-independent quantity).
    @testset "GEV smooth location curve vs evgam" begin
        rng = StableRNG(504)
        n = 500
        xe = rand(rng, n)
        muloc = 2.0 .+ sin.(2π .* xe)
        ye = [rand(rng, GeneralizedExtremeValue(m, 0.8, 0.1)) for m in muloc]
        dfe = (y=ye, x=xe)
        me = evgam([GAM.@formula(y ~ s(x, k=8, bs=:cr)), GAM.@formula(y ~ 1),
                    GAM.@formula(y ~ 1)], dfe, GEVFamily())
        eta_loc_j = param_eta(me, 1)
        @rput ye xe
        RCall.reval("""
        dre <- data.frame(y=ye, x=xe)
        mre <- evgam(list(y ~ s(x, k=8, bs="cr"), ~1, ~1), data=dre, family="gev")
        eta_loc_r <- as.vector(predict(mre)[,1])
        """)
        eta_loc_r = rcopy(Vector{Float64}, R"eta_loc_r")
        @test maximum(abs.(eta_loc_j .- eta_loc_r)) < 5e-3
        @test cor(eta_loc_j, eta_loc_r) > 0.9999
    end
end

# ── Multi-parameter offset parity vs mgcv gaulss ────────────────────────────
# gaulss lives in mgcv (not evgam), but this file already carries the RCall
# gating for multi-parameter live comparisons. Measured agreement at the time
# of writing: fitted-location max-abs 2.1e-5, cor 1.0 (mgcv's fitted() for
# gaulss includes the offset, matching Julia's stored-offset convention).
@testset "gamlss GaussianLS offset vs mgcv gaulss" begin
    rng = StableRNG(2026)
    n = 400
    x = sort(rand(rng, n)) .* 2
    off = 1.5 .* x .+ 0.3 .* sin.(3 .* x)
    y = off .+ sin.(2π .* x) .+ 0.25 .* randn(rng, n)
    df = (y = y, x = x)

    m_jl = gamlss([GAM.@formulak(y ~ s(x, k = 12, bs = :cr)),
                   GAM.@formulak(y ~ 1)],
                  df, GaussianLS(); offset = off)
    eta_jl = m_jl.fitted_eta[1]

    @rput y x off
    RCall.reval("""
    suppressMessages(library(mgcv))
    d_off <- data.frame(y = y, x = x, off = off)
    m_r_off <- gam(list(y ~ s(x, k = 12, bs = "cr") + offset(off), ~ 1),
                   family = gaulss(), data = d_off, method = "REML")
    fit_r_off <- fitted(m_r_off)[, 1]
    """)
    fit_r = rcopy(Vector{Float64}, R"fit_r_off")

    @test maximum(abs.(eta_jl .- fit_r)) < 5e-4      # ~25× observed 2.1e-5
    @test cor(eta_jl, fit_r) > 0.99999
end
