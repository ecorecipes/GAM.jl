using Test, GAM, DataFrames, Random, Statistics, StatsAPI, RCall
using StableRNGs

R"library(scam)"

@testset "SCAM R comparison" begin

    Random.seed!(42)
    n = 200
    x = sort(rand(n))

    @testset "MPI basis matches R" begin
        @rput x
        R"""
        library(splines)
        q <- 10; m <- 2
        nk <- q + m + 2
        xk <- rep(0, nk)
        xk[(m+2):(q+1)] <- seq(min(x), max(x), length=q-m)
        for (i in 1:(m+1)) xk[i] <- xk[m+2] - (m+2-i)*(xk[m+3]-xk[m+2])
        for (i in (q+2):(q+m+2)) xk[i] <- xk[q+1] + (i-q-1)*(xk[m+3]-xk[m+2])
        X1_r <- splineDesign(xk, x, ord=m+2)
        Sig_r <- matrix(1, q, q)
        Sig_r[upper.tri(Sig_r)] <- 0
        X_r <- X1_r %*% Sig_r
        X_r <- X_r[,-1]
        cmx_r <- colMeans(X_r)
        X_r <- sweep(X_r, 2, cmx_r)
        """
        X_r = rcopy(R"X_r")
        cmx_r = rcopy(R"cmx_r")

        spec = GAM.SmoothSpec([:x], GAM.MonoIncBasis(), 10, nothing, nothing,
            nothing, false, nothing, "s(x)")
        sm = smooth_construct(spec, DataFrame(x = x))

        @test sm.cmX ≈ cmx_r atol = 1e-10
        @test sm.X ≈ X_r atol = 1e-10
    end

    @testset "MPI fitted values match R" begin
        y = 3.0 .* x .+ 0.2 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="mpi", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :mpi, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.999
    end

    @testset "MPD fitted values match R" begin
        y = 3.0 .- 3.0 .* x .+ 0.2 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="mpd", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :mpd, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.999
    end

    @testset "CX fitted values match R" begin
        y = x .^ 2 .+ 0.1 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="cx", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :cx, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "CV fitted values match R" begin
        y = sqrt.(x) .+ 0.1 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="cv", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :cv, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "MICX fitted values match R" begin
        y = x .^ 2 .+ 0.1 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="micx", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :micx, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "MDCV fitted values match R" begin
        y = -x .^ 2 .+ 0.1 .* randn(n)
        @rput x y
        R"""
        m_r <- scam(y ~ s(x, bs="mdcv", k=10), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y ~ s(x, bs = :mdcv, k = 10)), DataFrame(x = x, y = y))
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "Poisson SCAM matches R" begin
        y_pois = [max(1, round(Int, 10 * xi^2 + rand())) for xi in x]
        y_f = Float64.(y_pois)
        @rput x y_f
        R"""
        m_r <- scam(y_f ~ s(x, bs="mpi", k=10), family=poisson(), data=data.frame(x=x, y_f=y_f))
        fitted_r <- fitted(m_r)
        """
        fitted_r = rcopy(R"fitted_r")

        m_jl = scam(@formulak(y_f ~ s(x, bs = :mpi, k = 10)),
            DataFrame(x = x, y_f = y_f); family = Poisson())
        @test cor(m_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "GCV smoothing-parameter parity" begin
        # Round-3 finding, now FIXED: the cyclic golden-section optimizer
        # could return a criterion-worse point than R scam's optimum (warm-
        # started evaluations at distant sp reported inconsistent
        # (deviance, edf) pairs). The optimizer now cold-starts its coarse
        # global scan and refits unconverged warm evaluations, so the chosen
        # point is criterion-optimal under Julia's own GCV. Raw sp/edf
        # equality with R is NOT asserted: the GCV landscape on problems
        # like this is flat over several log-sp units (criterion spread
        # ~1e-3), and BFGS (R) vs global-scan (Julia) legitimately stop at
        # different points on the ridge — the assertable facts are
        # criterion quality and fit agreement.
        rng_g = StableRNG(2024)
        n_g = 300
        x_g = sort(rand(rng_g, n_g)) .* 4
        y_g = 3.0 ./ (1.0 .+ exp.(-2.0 .* (x_g .- 2.0))) .+ 0.3 .* randn(rng_g, n_g)
        df_g = DataFrame(x = x_g, y = y_g)
        m_jl = scam(GAM.@formulak(y ~ s(x, bs = :mpi, k = 12)), df_g)
        @rput y_g x_g
        RCall.reval("""
        dat_g <- data.frame(x = x_g, y = y_g)
        m_rg <- scam(y ~ s(x, bs = "mpi", k = 12), data = dat_g)
        sp_rg <- m_rg[["sp"]][1]; edf_rg <- sum(m_rg[["edf"]])
        fit_rg <- as.vector(fitted(m_rg))
        """)
        sp_r = rcopy(R"sp_rg")
        edf_r = rcopy(R"edf_rg")
        fit_r = rcopy(Vector{Float64}, R"fit_rg")

        # The optimized criterion is stored on the model (NaN for REML fits)
        @test isfinite(m_jl.criterion)
        @test isnan(m_jl.reml)

        # Criterion quality: Julia's chosen point is at least as good, under
        # Julia's own GCV, as R's optimum evaluated by an independent
        # fixed-sp refit — the direct statement of the fixed defect.
        m_at_r = scam(GAM.@formulak(y ~ s(x, bs = :mpi, k = 12, sp = sp_r)), df_g)
        gcv_at_r = n_g * m_at_r.deviance_val / (n_g - m_at_r.edf_total)^2
        @test m_jl.criterion <= gcv_at_r * 1.001

        # Fit agreement with R along the (flat) ridge
        @test cor(m_jl.fitted_values, fit_r) > 0.999
        @test abs(m_jl.edf_total - edf_r) < 4.0
    end
end
