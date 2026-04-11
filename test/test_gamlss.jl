@testset "GAMLSS (Location-Scale-Shape)" begin
    using Random, DataFrames, Statistics, Distributions

    @testset "GaussianLS: location and scale recovery" begin
        Random.seed!(42)
        n = 500
        x = collect(range(0, 2π; length=n))
        μ_true = sin.(x)
        σ_true = 0.5 .+ 0.3 .* cos.(x)
        y = μ_true .+ σ_true .* randn(n)
        df = DataFrame(x=x, y=y)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ s(x))],
            df, GaussianLS())

        @test m.converged
        μ_fit = m.fitted_eta[1]
        σ_fit = exp.(m.fitted_eta[2])
        @test cor(μ_fit, μ_true) > 0.99
        @test cor(σ_fit, σ_true) > 0.95
        @test abs(mean(σ_fit) - mean(σ_true)) < 0.1
    end

    @testset "GaussianLS: constant variance recovers gam()" begin
        Random.seed!(43)
        n = 300
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x=x, y=y)

        # Standard GAM
        m_gam = gam(@formulak(y ~ s(x)), df)

        # GAMLSS with intercept-only σ formula
        m_ls = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ 1)],
            df, GaussianLS())

        μ_gam = m_gam.fitted_values
        μ_ls = m_ls.fitted_eta[1]

        # μ estimates should be very close
        @test cor(μ_gam, μ_ls) > 0.999
        # σ should be approximately constant
        σ_fit = exp.(m_ls.fitted_eta[2])
        @test std(σ_fit) / mean(σ_fit) < 0.01  # CV < 1%
    end

    @testset "GammaLS: positive data" begin
        Random.seed!(44)
        n = 400
        x = collect(range(0.1, 3.0; length=n))
        μ_true = 1.0 .+ 0.5 .* sin.(2.0 .* x)
        σ_true = 0.3 .+ 0.1 .* cos.(x)
        y = [rand(Gamma(1 / (σ^2), μ * σ^2)) for (μ, σ) in zip(μ_true, σ_true)]
        df = DataFrame(x=x, y=y)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ s(x))],
            df, GammaLS())

        @test m.converged
        μ_fit = exp.(m.fitted_eta[1])
        @test cor(μ_fit, μ_true) > 0.9
    end

    @testset "BetaLS: bounded data" begin
        Random.seed!(45)
        n = 400
        x = collect(range(0, 2π; length=n))
        μ_true = 0.3 .+ 0.2 .* sin.(x)
        φ_true = 10.0
        y = [rand(Beta(μ * φ_true, (1 - μ) * φ_true)) for μ in μ_true]
        df = DataFrame(x=x, y=y)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ 1)],
            df, BetaLS())

        @test m.converged
        μ_fit = 1.0 ./ (1.0 .+ exp.(-m.fitted_eta[1]))
        @test cor(μ_fit, μ_true) > 0.9
    end

    @testset "gamlss() single formula replication" begin
        Random.seed!(46)
        n = 200
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.5 .* randn(n)
        df = DataFrame(x=x, y=y)

        # Single formula should be replicated for both params
        m = gamlss(@formulak(y ~ s(x)), df, GaussianLS())
        @test m.converged
    end

    @testset "Family interface completeness" begin
        for fam in [GaussianLS(), GammaLS(), BetaLS(), NegBinLS(),
                    GammaLocationScale(), BetaRegression(),
                    NegativeBinomialLocationScale(), InverseGaussianLocationScale()]
            @test nparams(fam) == 2
            @test length(param_names(fam)) == 2
            @test length(GAM.param_links(fam)) == 2
        end
    end

    @testset "gamlss with Normal() directly" begin
        Random.seed!(50)
        n = 300
        x = collect(range(0, 2π; length=n))
        μ_true = sin.(x)
        σ_true = 0.5 .+ 0.2 .* cos.(x)
        y = μ_true .+ σ_true .* randn(n)
        df = DataFrame(x=x, y=y)

        # Direct Distributions.jl type
        m1 = gamlss([@formulak(y ~ s(x)), @formulak(y ~ s(x))],
                     df, Normal())
        @test m1.converged
        @test cor(m1.fitted_eta[1], μ_true) > 0.99

        # Legacy alias gives identical result
        m2 = gamlss([@formulak(y ~ s(x)), @formulak(y ~ s(x))],
                     df, GaussianLS())
        @test maximum(abs.(m1.fitted_eta[1] - m2.fitted_eta[1])) < 1e-10
    end

    @testset "gamlss with custom links" begin
        Random.seed!(51)
        n = 200
        x = collect(range(0, 2π; length=n))
        y = sin.(x) .+ 0.5 .* randn(n)
        df = DataFrame(x=x, y=y)

        # Normal with LogLink for σ (default) vs IdentityLink
        m = gamlss([@formulak(y ~ s(x)), @formulak(y ~ 1)],
                   df, Normal(); links=[IdentityLink(), LogLink()])
        @test m.converged
    end

    @testset "GammaLocationScale with custom links" begin
        Random.seed!(52)
        n = 300
        x = collect(range(0.1, 3.0; length=n))
        μ_true = 1.0 .+ 0.5 .* sin.(2.0 .* x)
        y = [rand(Gamma(1 / 0.09, μ * 0.09)) for μ in μ_true]
        df = DataFrame(x=x, y=y)

        m = gamlss([@formulak(y ~ s(x)), @formulak(y ~ 1)],
                   df, GammaLocationScale(links=[LogLink(), LogLink()]))
        @test m.converged
        μ_fit = exp.(m.fitted_eta[1])
        @test cor(μ_fit, μ_true) > 0.9
    end
end
