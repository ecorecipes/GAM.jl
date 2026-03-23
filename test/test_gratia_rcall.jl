using Test
using GAM
using DataFrames
using Random
using RCall
using LinearAlgebra
using Statistics: cor, mean
using GLM: LogLink

@testset "Gratia R comparison" begin

    # Load gratia in R
    R"""
    suppressPackageStartupMessages({
        library(mgcv)
        library(gratia)
    })
    """

    # ─── smooth_estimates comparison ─────────────────────────────────────

    @testset "smooth_estimates vs gratia" begin
        # Generate shared data
        R"""
        set.seed(42)
        n <- 200
        x <- sort(runif(n))
        y <- sin(2 * pi * x) + 0.3 * rnorm(n)
        dat <- data.frame(x = x, y = y)
        m_r <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat, method = "REML")

        se_r <- smooth_estimates(m_r, n = 100)
        """

        x_r = rcopy(R"dat$x")
        y_r = rcopy(R"dat$y")
        df = DataFrame(x = x_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df)

        # Get R smooth estimates
        est_r = rcopy(R"se_r$.estimate")
        se_r = rcopy(R"se_r$.se")
        x_grid_r = rcopy(R"se_r$x")

        # Julia smooth estimates on same grid
        se_jl = smooth_estimates(m_jl; data = (x = x_grid_r,))
        est_jl = se_jl.estimate
        se_jl_vals = se_jl.se

        # Estimates should be highly correlated
        c = cor(est_jl, est_r)
        @test c > 0.99
        println("smooth_estimates correlation: $c")

        # SE should be highly correlated
        # (Small differences due to smoothing parameter optimization converging
        # to slightly different optima between Julia and R)
        @test cor(se_jl_vals, se_r) > 0.98
    end

    # ─── derivatives comparison ──────────────────────────────────────────

    @testset "derivatives vs gratia" begin
        R"""
        set.seed(42)
        n <- 200
        x <- sort(runif(n))
        y <- sin(2 * pi * x) + 0.3 * rnorm(n)
        dat <- data.frame(x = x, y = y)
        m_r <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat, method = "REML")

        deriv_r <- derivatives(m_r, n = 50, type = "central", eps = 1e-7)
        """

        x_r = rcopy(R"dat$x")
        y_r = rcopy(R"dat$y")
        df = DataFrame(x = x_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df)

        d_est_r = rcopy(R"deriv_r$.derivative")
        d_se_r = rcopy(R"deriv_r$.se")

        de_jl = derivatives(m_jl; n = 50, type = :central, eps = 1e-7)

        # Derivative estimates should correlate well
        c = cor(de_jl.derivative, d_est_r)
        @test c > 0.95
        println("derivatives correlation: $c")
    end

    # ─── posterior_samples distribution check ────────────────────────────

    @testset "posterior samples distribution" begin
        R"""
        set.seed(42)
        n <- 200
        x <- sort(runif(n))
        y <- sin(2 * pi * x) + 0.3 * rnorm(n)
        dat <- data.frame(x = x, y = y)
        m_r <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat, method = "REML")
        coef_r <- coef(m_r)
        vcov_r <- vcov(m_r)
        """

        x_r = rcopy(R"dat$x")
        y_r = rcopy(R"dat$y")
        df = DataFrame(x = x_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df)

        coef_r = rcopy(R"coef_r")
        vcov_r = rcopy(R"vcov_r")

        # Sample from Julia posterior
        ps = posterior_samples(m_jl; n = 5000, seed = 42)
        mean_coef = vec(mean(ps; dims = 1))
        cov_coef = cov(ps)

        # Coefficient means should be close to point estimates
        @test cor(mean_coef, m_jl.coefficients) > 0.99

        # Julia Vp should be close to R vcov
        # (Differences reflect slightly different smoothing parameter optima)
        @test cor(vec(m_jl.Vp), vec(vcov_r)) > 0.96
    end

    # ─── fitted_samples on shared data ───────────────────────────────────

    @testset "fitted_samples vs R predict" begin
        R"""
        set.seed(42)
        n <- 200
        x <- sort(runif(n))
        y <- sin(2 * pi * x) + 0.3 * rnorm(n)
        dat <- data.frame(x = x, y = y)
        m_r <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat, method = "REML")
        fitted_r <- predict(m_r, type = "response")
        """

        x_r = rcopy(R"dat$x")
        y_r = rcopy(R"dat$y")
        fitted_r = rcopy(R"fitted_r")
        df = DataFrame(x = x_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df)

        # Mean of fitted_samples should match fitted values
        fs = fitted_samples(m_jl; n = 1000, seed = 42)
        mean_fs = vec(mean(fs; dims = 2))

        @test cor(mean_fs, m_jl.fitted_values) > 0.999
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    # ─── Multi-smooth comparison ─────────────────────────────────────────

    @testset "multi-smooth smooth_estimates" begin
        R"""
        set.seed(123)
        n <- 300
        x1 <- runif(n)
        x2 <- runif(n)
        y <- sin(2 * pi * x1) + cos(2 * pi * x2) + 0.5 * rnorm(n)
        dat <- data.frame(x1 = x1, x2 = x2, y = y)
        m_r <- gam(y ~ s(x1, k = 10, bs = "cr") + s(x2, k = 10, bs = "cr"),
                    data = dat, method = "REML")

        se_r1 <- smooth_estimates(m_r, select = "s(x1)", n = 100)
        se_r2 <- smooth_estimates(m_r, select = "s(x2)", n = 100)
        """

        x1_r = rcopy(R"dat$x1")
        x2_r = rcopy(R"dat$x2")
        y_r = rcopy(R"dat$y")
        df = DataFrame(x1 = x1_r, x2 = x2_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x1, k = 10, bs = :cr) + s(x2, k = 10, bs = :cr)), df)

        est_r1 = rcopy(R"se_r1$.estimate")
        est_r2 = rcopy(R"se_r2$.estimate")
        x_grid_r1 = rcopy(R"se_r1$x1")
        x_grid_r2 = rcopy(R"se_r2$x2")

        # Julia evaluation on same grids
        se_jl1 = smooth_estimates(m_jl;
            select = "s(x1,bs=cr)",
            data = (x1 = x_grid_r1,))
        se_jl2 = smooth_estimates(m_jl;
            select = "s(x2,bs=cr)",
            data = (x2 = x_grid_r2,))

        c1 = cor(se_jl1.estimate, est_r1)
        c2 = cor(se_jl2.estimate, est_r2)
        @test c1 > 0.99
        @test c2 > 0.99
        println("Multi-smooth: s(x1) cor=$c1, s(x2) cor=$c2")
    end

    # ─── Poisson model comparison ────────────────────────────────────────

    @testset "Poisson smooth_estimates" begin
        R"""
        set.seed(77)
        n <- 300
        x <- sort(runif(n))
        mu <- exp(1 + 2 * sin(2 * pi * x))
        y <- rpois(n, mu)
        dat <- data.frame(x = x, y = y)
        m_r <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat,
                    family = poisson(), method = "REML")
        se_r <- smooth_estimates(m_r, n = 100)
        """

        x_r = rcopy(R"dat$x")
        y_r = Float64.(rcopy(R"dat$y"))
        df = DataFrame(x = x_r, y = y_r)
        m_jl = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df;
            family = Poisson(), link = LogLink())

        est_r = rcopy(R"se_r$.estimate")
        x_grid_r = rcopy(R"se_r$x")
        se_jl = smooth_estimates(m_jl; data = (x = x_grid_r,))

        c = cor(se_jl.estimate, est_r)
        @test c > 0.95
        println("Poisson smooth_estimates correlation: $c")
    end
end
