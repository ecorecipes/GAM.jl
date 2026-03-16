# R comparison tests for multi-parameter models (evgam)
# Compares Julia evgam() output against R evgam package

using Test
using GAM
using RCall
using DataFrames
using Statistics

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
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
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
            [@gam_formula(y ~ s(x, bs=:cr, k=10)),
             @gam_formula(y ~ 1),
             @gam_formula(y ~ 1)],
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
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1)],
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
            [@gam_formula(y ~ s(x, bs=:cr, k=8)),
             @gam_formula(y ~ 1)],
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
end
