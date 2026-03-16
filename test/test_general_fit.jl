@testset "General Fit (WPS Algorithm)" begin
    using Random, DataFrames, Statistics, Distributions

    Random.seed!(42)
    n = 200
    x = range(0, 2π; length=n)

    @testset "Gaussian: general matches PIRLS" begin
        Random.seed!(42)
        y = sin.(collect(x)) .+ 0.3 .* randn(n)
        df = DataFrame(x=collect(x), y=y)
        m_pirls = gam(@formula(y ~ s(x)), df)
        m_general = gam(@formula(y ~ s(x)), df; optimizer=:general)

        @test maximum(abs.(m_pirls.fitted_values .- m_general.fitted_values)) < 1e-10
        @test abs(m_pirls.deviance_val - m_general.deviance_val) < 1e-6
        @test abs(m_pirls.scale - m_general.scale) < 1e-6
    end

    @testset "Poisson: general matches PIRLS" begin
        Random.seed!(43)
        mu_true = exp.(0.5 .* sin.(collect(x)))
        y_pois = Float64.([rand(Poisson(m)) for m in mu_true])
        df = DataFrame(x=collect(x), y=y_pois)
        m_pirls = gam(@formula(y ~ s(x)), df; family=Poisson(), link=LogLink())
        m_general = gam(@formula(y ~ s(x)), df; family=Poisson(), link=LogLink(),
            optimizer=:general)

        @test maximum(abs.(m_pirls.fitted_values .- m_general.fitted_values)) < 1e-5
        @test abs(m_pirls.deviance_val - m_general.deviance_val) < 1e-3
    end

    @testset "Binomial: general matches PIRLS" begin
        Random.seed!(44)
        p_true = 1.0 ./ (1.0 .+ exp.(-2.0 .* sin.(collect(x))))
        y_bin = Float64.([rand() < p for p in p_true])
        df = DataFrame(x=collect(x), y=y_bin)
        m_pirls = gam(@formula(y ~ s(x)), df; family=Bernoulli(), link=LogitLink())
        m_general = gam(@formula(y ~ s(x)), df; family=Bernoulli(), link=LogitLink(),
            optimizer=:general)

        @test maximum(abs.(m_pirls.fitted_values .- m_general.fitted_values)) < 1e-8
        @test abs(m_pirls.deviance_val - m_general.deviance_val) < 1e-4
    end

    @testset "Gamma: general matches PIRLS" begin
        Random.seed!(45)
        mu_gam = 1.0 ./ (0.5 .+ abs.(sin.(collect(x))))
        y_gamma = [rand(Gamma(5.0, m / 5.0)) for m in mu_gam]
        df = DataFrame(x=collect(x), y=y_gamma)
        m_pirls = gam(@formula(y ~ s(x)), df; family=Gamma(), link=InverseLink())
        m_general = gam(@formula(y ~ s(x)), df; family=Gamma(), link=InverseLink(),
            optimizer=:general)

        @test maximum(abs.(m_pirls.fitted_values .- m_general.fitted_values)) < 1e-5
        @test abs(m_pirls.deviance_val - m_general.deviance_val) < 1e-3
    end

    @testset "ForwardDiff fallback for non-canonical link" begin
        # Gamma with LogLink (non-canonical) — exercises ForwardDiff path
        # PIRLS uses expected info, Newton uses observed info → slightly different sp
        Random.seed!(123)
        mu_gam = exp.(0.5 .* sin.(collect(x)))
        y_gamma = [rand(Gamma(5.0, m / 5.0)) for m in mu_gam]
        df = DataFrame(x=collect(x), y=y_gamma)

        m_pirls = gam(@formula(y ~ s(x)), df; family=Gamma(), link=LogLink())
        m_general = gam(@formula(y ~ s(x)), df; family=Gamma(), link=LogLink(),
            optimizer=:general)

        @test maximum(abs.(m_pirls.fitted_values .- m_general.fitted_values)) < 0.5
        @test cor(m_pirls.fitted_values, m_general.fitted_values) > 0.99
    end

    @testset "Invalid optimizer argument" begin
        df = DataFrame(x=collect(x), y=sin.(collect(x)))
        @test_throws ArgumentError gam(@formula(y ~ s(x)), df; optimizer=:invalid)
    end

    @testset "Distributions.logpdf derivatives consistency" begin
        # Verify ForwardDiff through Distributions.logpdf gives correct derivatives
        # This validates the core mechanism of the general fit method
        using ForwardDiff, GLM

        Random.seed!(99)
        test_cases = [
            (Normal(), IdentityLink(), 1.5, 0.8),
            (Poisson(), LogLink(), 3.0, 1.2),
            (Bernoulli(), LogitLink(), 1.0, 0.5),
            (Gamma(), InverseLink(), 2.0, 0.4),
        ]

        for (fam, lnk, y_val, eta_val) in test_cases
            # Helper that builds ℓ(η) = logpdf(D(linkinv(η)), y)
            ll_fn = let f = fam, l = lnk, yv = y_val
                η -> logpdf(GAM._make_distribution(f, GLM.linkinv(l, η)), yv)
            end

            dl_fd = ForwardDiff.derivative(ll_fn, eta_val)
            d2l_fd = ForwardDiff.derivative(
                η -> ForwardDiff.derivative(ll_fn, η), eta_val)

            h = 1e-5
            dl_num = (ll_fn(eta_val + h) - ll_fn(eta_val - h)) / (2h)
            d2l_num = (ll_fn(eta_val + h) - 2 * ll_fn(eta_val) + ll_fn(eta_val - h)) / h^2

            @test abs(dl_fd - dl_num) < 1e-5
            @test abs(d2l_fd - d2l_num) < 1e-3
        end
    end
end
