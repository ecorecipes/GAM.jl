@testset "Unified gam() API dispatch" begin
    using Random, DataFrames, Statistics, StatsAPI, Distributions

    @testset "gam() auto-detects SCAM for GamFormula with mpi" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = 3.0 .* x .+ 0.2 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gam(@gam_formula(y ~ s(x, bs=:mpi, k=10)), df)
        @test m.converged || m.deviance_val < 20.0

        # Check monotonicity
        xp = collect(0.01:0.01:0.99)
        pred = StatsAPI.predict(m, DataFrame(x=xp))
        @test all(diff(pred) .>= -1e-8)
    end

    @testset "gam() auto-SCAM matches scam() results" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = 3.0 .* x .+ 0.2 .* randn(n)
        df = DataFrame(x=x, y=y)

        m_gam = gam(@gam_formula(y ~ s(x, bs=:mpi, k=10)), df)
        m_scam = scam(@gam_formula(y ~ s(x, bs=:mpi, k=10)), df)

        @test m_gam.deviance_val ≈ m_scam.deviance_val atol=1e-6
        @test m_gam.fitted_values ≈ m_scam.fitted_values atol=1e-6
    end

    @testset "gam() auto-SCAM for convex constraint" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = x .^ 2 .+ 0.1 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gam(@gam_formula(y ~ s(x, bs=:cx, k=10)), df)
        @test m.deviance_val < 5.0
        @test cor(m.fitted_values, x .^ 2) > 0.98
    end

    @testset "gam() with unconstrained smooths does NOT use SCAM" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.1 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gam(@gam_formula(y ~ s(x, k=20)), df)
        @test m.converged
        @test cor(m.fitted_values, sin.(2π .* x)) > 0.95
    end

    @testset "gam() with MultiParameterFamily dispatches to GAMLSS" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0.1, 3.0; length=n))
        y = 1.0 .+ 0.5 .* sin.(x) .+ 0.2 .* randn(n)
        y = max.(y, 0.01)
        df = DataFrame(x=x, y=y)

        # Single formula → replicated for all params
        m = gam(@gam_formula(y ~ s(x)), df, GammaLocationScale())
        @test m isa GAM.MultiParameterModel
        @test m.converged
    end

    @testset "gam() with vector of formulas + MultiParameterFamily" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0.1, 3.0; length=n))
        y = 1.0 .+ 0.5 .* sin.(x) .+ 0.2 .* randn(n)
        y = max.(y, 0.01)
        df = DataFrame(x=x, y=y)

        m = gam(
            [@gam_formula(y ~ s(x)), @gam_formula(y ~ 1)],
            df, GammaLocationScale()
        )
        @test m isa GAM.MultiParameterModel
        @test m.converged
    end

    @testset "gam() with DistFamily (GaussianLS)" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gam(@gam_formula(y ~ s(x)), df, GaussianLS())
        @test m isa GAM.MultiParameterModel
        @test m.converged
    end

    @testset "gam() GAMLSS matches gamlss() results" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x=x, y=y)

        formulas = [@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))]
        m_gam = gam(formulas, df, GaussianLS())
        m_gamlss = gamlss(formulas, df, GaussianLS())

        @test m_gam.converged == m_gamlss.converged
        @test m_gam.nll ≈ m_gamlss.nll atol=1e-6
    end

    @testset "scam() still works as before" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = 3.0 .* x .+ 0.2 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = scam(@gam_formula(y ~ s(x, bs=:mpi, k=10)), df)
        @test m.converged || m.deviance_val < 20.0
    end

    @testset "gamlss() still works as before" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gamlss(
            [@gam_formula(y ~ s(x)), @gam_formula(y ~ s(x))],
            df, GaussianLS()
        )
        @test m.converged
    end
end
