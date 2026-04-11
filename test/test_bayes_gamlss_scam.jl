using Test
using GAM
using DataFrames
using Distributions
using LinearAlgebra: diag
using StatsAPI: nobs, coef, coeftable, confint, vcov

# Bayesian GAMLSS and SCAM tests via Turing.jl
#
# Run standalone: julia --project -e 'include("test/test_bayes_gamlss_scam.jl")'
# Requires Turing.jl to be installed.

@testset "Bayesian GAMLSS and SCAM" begin
    using StableRNGs
    using Turing

    rng = StableRNG(777)
    n = 100

    @testset "Bayesian GAMLSS Normal" begin
        x = range(0, 2π; length = n)
        μ_true = sin.(x)
        σ_true = 0.3 .+ 0.1 .* cos.(x)
        y = μ_true .+ σ_true .* randn(rng, n)
        df = DataFrame(x = collect(x), y = y)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ s(x))],
            df, Normal();
            priors = PriorSpec(), nsamples = 300, nchains = 1
        )
        @test m isa BayesGamModel
        @test nobs(m) == n

        # Should have coefficients for both μ and σ parameters
        c = coef(m)
        @test length(c) == 4  # intercept + null-space for each param

        ct = coeftable(m)
        @test length(ct.rownms) == 4
        @test "mu_(Intercept)" in ct.rownms

        # Credible intervals
        ci = confint(m)
        @test size(ci, 1) == length(c)
        @test size(ci, 2) == 2
        @test all(ci[:, 1] .< ci[:, 2])  # lower < upper

        # Vcov matrix
        V = vcov(m)
        @test size(V) == (length(c), length(c))
        @test all(diag(V) .>= 0)

        ll = pointwise_loglikelihood(m)
        l = loo(m)
        l_is = loo(m; method = :is)
        d = pareto_k_diagnostic(l)
        w = waic(m)
        @test size(ll, 2) == n
        @test length(l.pointwise_elpd) == n
        @test length(l.pareto_k) == n
        @test length(l.n_eff) == n
        @test isfinite(l.elpd_loo)
        @test isfinite(l.looic)
        @test l.method == :psis
        @test all(isfinite, l.pareto_k)
        @test l_is.method == :is
        @test all(isnan, l_is.pareto_k)
        @test d isa PSISKDiagnostic
        @test d.pareto_k == l.pareto_k
        @test length(w.pointwise_elpd) == n
        @test isfinite(w.elpd_waic)
        @test isfinite(w.waic)
    end

    @testset "Bayesian GAMLSS single formula replicated" begin
        x = range(0, 2π; length = n)
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = collect(x), y = y)

        # Single formula gets replicated for both μ and σ
        m = gamlss(@formulak(y ~ s(x)), df, Normal();
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m isa BayesGamModel
        @test nobs(m) == n
        @test length(coef(m)) > 0
    end

    @testset "Bayesian GAMLSS GammaLocationScale" begin
        x = range(0.1, 3; length = n)
        mu_g = exp.(0.5 .+ 0.3 .* x)
        sigma_g = fill(0.3, n)
        y_g = [rand(rng, Gamma(1 / sigma_g[i]^2, mu_g[i] * sigma_g[i]^2))
               for i in 1:n]
        df = DataFrame(x = collect(x), y = y_g)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ 1)],
            df, GammaLocationScale();
            priors = PriorSpec(), nsamples = 300, nchains = 1
        )
        @test m isa BayesGamModel
        @test nobs(m) == n

        c = coef(m)
        @test length(c) >= 2  # at least intercepts for both params

        ct = coeftable(m)
        @test length(ct.rownms) >= 2
    end

    @testset "Bayesian GAMLSS BetaRegression" begin
        x = range(0, 1; length = n)
        mu_b = 0.3 .+ 0.4 .* x
        phi_b = fill(20.0, n)
        y_b = [rand(rng, Beta(mu_b[i] * phi_b[i], (1 - mu_b[i]) * phi_b[i]))
               for i in 1:n]
        df = DataFrame(x = collect(x), y = y_b)

        m = gamlss(
            [@formulak(y ~ s(x)), @formulak(y ~ 1)],
            df, BetaRegression();
            priors = PriorSpec(), nsamples = 300, nchains = 1
        )
        @test m isa BayesGamModel
        @test nobs(m) == n
        @test length(coef(m)) >= 2
    end

    @testset "Bayesian SCAM monotone increasing" begin
        x = sort(rand(rng, n))
        y_true = 2 .* x .+ 0.5 .* x .^ 2
        y = y_true .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :mpi, k = 8)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m isa BayesGamModel
        @test nobs(m) == n

        c = coef(m)
        @test length(c) >= 1

        # Compare with frequentist — intercepts should be reasonably close
        m_freq = scam(@formulak(y ~ s(x, bs = :mpi, k = 8)), df)
        @test abs(c[1] - coef(m_freq)[1]) < 0.5

        l = loo(m)
        w = waic(m)
        @test length(l.pointwise_elpd) == n
        @test length(l.pareto_k) == n
        @test isfinite(l.elpd_loo)
        @test isfinite(l.looic)
        @test length(w.pointwise_elpd) == n
        @test isfinite(w.elpd_waic)
        @test isfinite(w.waic)
    end

    @testset "Bayesian SCAM monotone decreasing" begin
        x = sort(rand(rng, n))
        y = 3.0 .- 2 .* x .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :mpd, k = 8)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m isa BayesGamModel
        @test nobs(m) == n
    end

    @testset "Bayesian SCAM convex" begin
        x = sort(rand(rng, n))
        y = x .^ 2 .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :cx, k = 8)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m isa BayesGamModel
        @test nobs(m) == n
    end

    @testset "Bayesian GAM via @formula" begin
        x = range(0, 2π; length = n)
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = collect(x), y = y)

        # @formula with Bayesian dispatch
        m = gam(@formula(y ~ s(x)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m isa BayesGamModel
        @test nobs(m) == n
        @test length(coef(m)) >= 1

        # @formula with basis alias
        m2 = gam(@formula(y ~ cr(x)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)
        @test m2 isa BayesGamModel
    end

    @testset "Posterior samples extraction" begin
        x = range(0, 2π; length = n)
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = collect(x), y = y)

        m = gam(@formula(y ~ s(x)), df;
            priors = PriorSpec(), nsamples = 300, nchains = 1)

        ps = posterior_samples(m)
        @test ps isa Matrix{Float64}
        @test size(ps, 2) == length(coef(m))
        @test size(ps, 1) > 0

        # Subsampled
        ps_sub = posterior_samples(m; n = 50)
        @test size(ps_sub, 1) == 50
    end
end
