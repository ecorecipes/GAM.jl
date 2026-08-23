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

    @testset "GCV smoothing-parameter parity (known divergence)" begin
        # Round-3 finding: with method=:GCV (the default), scam()'s cyclic
        # golden-section optimizer can return a smoothing parameter that is
        # WORSE under its own GCV criterion. Measured on this problem:
        # Julia sp = 6.1e5 (edf 2.0, GCV 0.1319) vs R scam sp = 3.4e-3
        # (edf 7.28); Julia's own GCV at R's sp is 0.1007 < 0.1319, so the
        # optimizer — not the criterion — is at fault. Fitted values still
        # agree here only because the target is an easy sigmoid.
        # The @test_broken lines pin the divergence and will flip when the
        # optimizer is fixed (e.g. multi-start or BFGS as in R scam).
        rng_g = StableRNG(2024)
        n_g = 300
        x_g = sort(rand(rng_g, n_g)) .* 4
        y_g = 3.0 ./ (1.0 .+ exp.(-2.0 .* (x_g .- 2.0))) .+ 0.3 .* randn(rng_g, n_g)
        m_jl = scam(GAM.@formulak(y ~ s(x, bs = :mpi, k = 12)),
            DataFrame(x = x_g, y = y_g))
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
        @test cor(m_jl.fitted_values, fit_r) > 0.999
        @test_broken abs(m_jl.sp[1] - log(sp_r)) < 1.0
        @test_broken abs(m_jl.edf_total - edf_r) < 1.0
        # (A direct self-criterion probe — GCV at Julia's chosen sp vs at
        # R's sp under Julia's own formula — showed 0.1319 vs 0.1007 in a
        # standalone run, i.e. the optimizer returned a criterion-worse
        # point; that probe is warm-start-order sensitive, so it is
        # documented here rather than asserted.)
    end
end
