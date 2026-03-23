using Test, GAM, RCall, Random, Statistics

R"""
library(egpd)
"""

@testset "EGPD R comparison" begin

    # ================================================================
    # NLL value comparison (single observation)
    # ================================================================
    @testset "EGPD1 NLL matches R" begin
        test_cases = [
            (2.0, log(1.5), 0.2, log(1.5)),
            (0.5, log(0.8), 0.1, log(2.0)),
            (5.0, log(3.0), 0.3, log(0.5)),
            (1.0, log(1.0), -0.1, log(1.0)),
        ]
        for (y, lpsi, xi, lkappa) in test_cases
            R"""
            X <- matrix(1, nrow=1, ncol=1)
            off <- list(numeric(0), numeric(0), numeric(0))
            nll_r <- egpd:::egpd1d0(list($(lpsi), $(xi), $(lkappa)), X, X, X, $(y), 0L, 0L, off)
            """
            @rget nll_r
            nll_jl = GAM.nll_obs(EGPD1Family(), y, [lpsi, xi, lkappa])
            @test abs(nll_jl - nll_r) < 1e-12
        end
    end

    @testset "EGPD2 NLL matches R" begin
        test_cases = [
            (2.0, log(1.5), 0.2, log(1.0), log(1.0), 0.0),
            (1.0, log(1.0), 0.1, log(0.5), log(2.0), 1.0),
            (5.0, log(3.0), 0.3, log(2.0), log(0.5), -1.0),
        ]
        for (y, lpsi, xi, lk1, ldk, logit_p) in test_cases
            R"""
            X <- matrix(1, nrow=1, ncol=1)
            off <- list(numeric(0), numeric(0), numeric(0), numeric(0), numeric(0))
            nll_r <- egpd:::egpd2d0(list($(lpsi), $(xi), $(lk1), $(ldk), $(logit_p)), X, X, X, X, X, $(y), 0L, 0L, off)
            """
            @rget nll_r
            nll_jl = GAM.nll_obs(EGPD2Family(), y, [lpsi, xi, lk1, ldk, logit_p])
            @test abs(nll_jl - nll_r) < 1e-12
        end
    end

    @testset "EGPD3 NLL matches R" begin
        test_cases = [
            (2.0, log(1.5), 0.2, log(1.0)),
            (0.5, log(0.8), 0.1, log(0.5)),
            (5.0, log(3.0), 0.3, log(2.0)),
        ]
        for (y, lpsi, xi, ldelta) in test_cases
            R"""
            X <- matrix(1, nrow=1, ncol=1)
            off <- list(numeric(0), numeric(0), numeric(0))
            nll_r <- egpd:::egpd3d0(list($(lpsi), $(xi), $(ldelta)), X, X, X, $(y), 0L, 0L, off)
            """
            @rget nll_r
            nll_jl = GAM.nll_obs(EGPD3Family(), y, [lpsi, xi, ldelta])
            @test abs(nll_jl - nll_r) < 1e-12
        end
    end

    @testset "EGPD4 NLL matches R" begin
        test_cases = [
            (2.0, log(1.5), 0.2, log(1.0), log(2.0)),
            (0.5, log(0.8), 0.05, log(0.5), log(3.0)),
            (5.0, log(3.0), 0.3, log(2.0), log(1.0)),
        ]
        for (y, lpsi, xi, ldelta, lkappa) in test_cases
            R"""
            X <- matrix(1, nrow=1, ncol=1)
            off <- list(numeric(0), numeric(0), numeric(0), numeric(0))
            nll_r <- egpd:::egpd4d0(list($(lpsi), $(xi), $(ldelta), $(lkappa)), X, X, X, X, $(y), 0L, 0L, off)
            """
            @rget nll_r
            nll_jl = GAM.nll_obs(EGPD4Family(), y, [lpsi, xi, ldelta, lkappa])
            @test abs(nll_jl - nll_r) < 1e-12
        end
    end

    # ================================================================
    # Gradient comparison against R's d12 functions
    # ================================================================
    @testset "EGPD1 gradients match R" begin
        R"""
        X <- matrix(1, nrow=1, ncol=1)
        off <- list(numeric(0), numeric(0), numeric(0))
        d12_r <- egpd:::egpd1d12(list(log(1.5), 0.2, log(1.5)), X, X, X, 2.0, 0L, 0L, off)
        """
        @rget d12_r

        import ForwardDiff
        fam = EGPD1Family()
        η = [log(1.5), 0.2, log(1.5)]
        g_jl = ForwardDiff.gradient(v -> GAM.nll_obs(fam, 2.0, v), η)
        h_jl = ForwardDiff.hessian(v -> GAM.nll_obs(fam, 2.0, v), η)

        # R d12 columns: [d_psi, d_xi, d_kappa, d_pp, d_px, d_xp_or_pk, d_xx, d_xk, d_kk]
        # = gradient (3) + upper-tri Hessian (6)
        @test abs(d12_r[1] - g_jl[1]) < 1e-8
        @test abs(d12_r[2] - g_jl[2]) < 1e-8
        @test abs(d12_r[3] - g_jl[3]) < 1e-8

        # Hessian: d_pp, d_px, d_pk, d_xx, d_xk, d_kk
        @test abs(d12_r[4] - h_jl[1,1]) < 1e-8
        @test abs(d12_r[5] - h_jl[1,2]) < 1e-8
        @test abs(d12_r[6] - h_jl[1,3]) < 1e-8
        @test abs(d12_r[7] - h_jl[2,2]) < 1e-8
        @test abs(d12_r[8] - h_jl[2,3]) < 1e-8
        @test abs(d12_r[9] - h_jl[3,3]) < 1e-8
    end

    @testset "EGPD3 gradients match R" begin
        R"""
        X <- matrix(1, nrow=1, ncol=1)
        off <- list(numeric(0), numeric(0), numeric(0))
        d12_r3 <- egpd:::egpd3d12(list(log(1.5), 0.2, log(1.0)), X, X, X, 2.0, 0L, 0L, off)
        """
        @rget d12_r3

        import ForwardDiff
        fam = EGPD3Family()
        η = [log(1.5), 0.2, log(1.0)]
        g_jl = ForwardDiff.gradient(v -> GAM.nll_obs(fam, 2.0, v), η)
        h_jl = ForwardDiff.hessian(v -> GAM.nll_obs(fam, 2.0, v), η)

        @test abs(d12_r3[1] - g_jl[1]) < 1e-8
        @test abs(d12_r3[2] - g_jl[2]) < 1e-8
        @test abs(d12_r3[3] - g_jl[3]) < 1e-8
        @test abs(d12_r3[4] - h_jl[1,1]) < 1e-8
        @test abs(d12_r3[5] - h_jl[1,2]) < 1e-8
        @test abs(d12_r3[6] - h_jl[1,3]) < 1e-8
        @test abs(d12_r3[7] - h_jl[2,2]) < 1e-8
        @test abs(d12_r3[8] - h_jl[2,3]) < 1e-8
        @test abs(d12_r3[9] - h_jl[3,3]) < 1e-8
    end

    # ================================================================
    # Full model fitting comparison: EGPD1
    # ================================================================
    @testset "EGPD1 constant model matches R" begin
        R"""
        set.seed(42)
        y_r <- regpd(500, sigma=2, xi=0.1, kappa=1.5, type=1)
        rdf <- data.frame(y=y_r)
        m_r <- egpd(list(y ~ 1, ~ 1, ~ 1), data=rdf, family="egpd", egpd.args=list(m=1), trace=0)
        coefs_r <- c(m_r$logscale$coefficients, m_r$shape$coefficients, m_r$logkappa$coefficients)
        """
        @rget y_r coefs_r

        y = Float64.(y_r)
        df = (; y=y)
        m = evgam(
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df, EGPD1Family()
        )

        @test m.converged
        @test maximum(abs.(m.coefficients .- coefs_r)) < 0.01
    end

    @testset "EGPD3 constant model matches R" begin
        R"""
        set.seed(42)
        y_r3 <- regpd(500, sigma=2, xi=0.1, delta=1.5, type=4)
        rdf3 <- data.frame(y=y_r3)
        m_r3 <- egpd(list(y ~ 1, ~ 1, ~ 1), data=rdf3, family="egpd", egpd.args=list(m=3), trace=0)
        coefs_r3 <- c(m_r3$logscale$coefficients, m_r3$shape$coefficients, m_r3$logdelta$coefficients)
        """
        @rget y_r3 coefs_r3

        y3 = Float64.(y_r3)
        df3 = (; y=y3)
        m3 = evgam(
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df3, EGPD3Family()
        )

        @test m3.converged
        @test maximum(abs.(m3.coefficients .- coefs_r3)) < 0.01
    end

    # ================================================================
    # Smooth model fitting comparison: EGPD1
    # ================================================================
    @testset "EGPD1 smooth model matches R" begin
        R"""
        set.seed(123)
        n <- 800
        x <- seq(0, 5, length.out=n)
        lpsi <- log(1 + 0.5 * x)
        sigma <- exp(lpsi)
        xi <- rep(0.1, n)
        kappa <- rep(1.5, n)
        # Generate using inverse CDF: u^(1/kappa) gives GPD uniform, then quantile
        u <- runif(n)
        u_gpd <- u^(1/kappa)
        y <- sigma * ((1 - u_gpd)^(-xi) - 1) / xi
        rdf <- data.frame(y=y, x=x)
        m_r <- egpd(list(y ~ s(x, bs="cr", k=8), ~ 1, ~ 1), data=rdf, family="egpd", egpd.args=list(m=1), trace=0)
        lpsi_fitted_r <- m_r$logscale$fitted
        xi_fitted_r <- m_r$shape$coefficients[1]
        lkappa_fitted_r <- m_r$logkappa$coefficients[1]
        """
        @rget y x lpsi_fitted_r xi_fitted_r lkappa_fitted_r

        y_jl = Float64.(y)
        x_jl = Float64.(x)
        df = (; y=y_jl, x=x_jl)
        m = evgam(
            [@gam_formula(y ~ s(x, bs=:cr, k=8)), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df, EGPD1Family()
        )

        @test m.converged

        # Fitted log-scale should be correlated with R
        lpsi_fitted_jl = GAM.param_eta(m, 1)
        corr = cor(lpsi_fitted_jl, Float64.(lpsi_fitted_r))
        @test corr > 0.95

        # Shape and kappa intercepts should be similar
        @test abs(GAM.param_coef(m, 2)[1] - xi_fitted_r) < 0.15
        @test abs(GAM.param_coef(m, 3)[1] - lkappa_fitted_r) < 0.5
    end

    # ================================================================
    # Multi-observation NLL matches R (vector check)
    # ================================================================
    @testset "EGPD1 vector NLL matches R" begin
        R"""
        set.seed(99)
        y_v <- regpd(100, sigma=1.5, xi=0.15, kappa=2.0, type=1)
        n <- length(y_v)
        X <- matrix(1, nrow=n, ncol=1)
        off <- list(numeric(0), numeric(0), numeric(0))
        nll_r_v <- egpd:::egpd1d0(list(log(1.5), 0.15, log(2.0)), X, X, X, y_v, 0L, 0L, off)
        """
        @rget y_v nll_r_v

        y_jl = Float64.(y_v)
        n = length(y_jl)
        η_list = [fill(log(1.5), n), fill(0.15, n), fill(log(2.0), n)]
        nll_jl = GAM.nll_total(EGPD1Family(), y_jl, η_list)
        @test abs(nll_jl - nll_r_v) < 1e-8
    end

end
