# End-to-end Bayesian GAM tests via Turing.jl
#
# Run standalone: julia --project -e 'include("test/test_bayes_e2e.jl")'
# Requires Turing.jl to be installed.

using Test
using GAM
using Turing
using MCMCChains
using DataFrames
using Random
using Distributions
using StatsAPI
using LinearAlgebra
using Statistics: mean, std

@testset "Bayesian GAM end-to-end" begin

    @testset "Gaussian GAM — y = sin(2πx) + ε" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y_true = sin.(2π .* x)
        y = y_true .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x = x)

        # Frequentist reference
        m_freq = gam(@gam_formula(y ~ s(x, k = 10)), df)
        freq_int = StatsAPI.coef(m_freq)[1]
        freq_σ = sqrt(m_freq.scale)

        # Bayesian fit
        m_bayes = gam(@gam_formula(y ~ s(x, k = 10)), df;
            priors = PriorSpec(sds = Exponential(1.0)),
            nsamples = 1000, nchains = 1)

        @test m_bayes isa BayesGamModel
        @test StatsAPI.nobs(m_bayes) == n

        # Intercept should be close to frequentist
        bayes_int = StatsAPI.coef(m_bayes)[1]
        @test abs(bayes_int - freq_int) < 0.15

        # σ should be close to frequentist
        chains = m_bayes.chains
        σ_bayes = mean(vec(chains[Symbol("σ_obs")].data))
        @test abs(σ_bayes - freq_σ) < 0.1

        # Chain should be mixing (non-zero std)
        β1_std = std(vec(chains[Symbol("β[1]")].data))
        @test β1_std > 0.001

        # StatsBase interface
        ct = StatsAPI.coeftable(m_bayes)
        @test size(ct.cols[1], 1) == length(StatsAPI.coef(m_bayes))

        ci = StatsAPI.confint(m_bayes)
        @test size(ci, 2) == 2
        @test all(ci[:, 1] .< ci[:, 2])  # lower < upper

        V = StatsAPI.vcov(m_bayes)
        @test size(V, 1) == size(V, 2)
        @test all(eigvals(Symmetric(V)) .> -1e-10)  # positive semi-definite

        # show() should not error
        io = IOBuffer()
        show(io, MIME("text/plain"), m_bayes)
        s = String(take!(io))
        @test occursin("Bayesian", s)
        @test occursin("Normal", s)
        @test occursin("IdentityLink", s)
    end

    @testset "Poisson GAM — count data" begin
        Random.seed!(123)
        n = 200
        x = sort(rand(n))
        λ_true = exp.(1.0 .+ 1.5 .* sin.(2π .* x))
        y = [rand(Poisson(λ)) for λ in λ_true]
        df = DataFrame(y = y, x = x)

        m_freq = gam(@gam_formula(y ~ s(x, k = 10)), df;
            family = Poisson(), link = LogLink())
        m_bayes = gam(@gam_formula(y ~ s(x, k = 10)), df;
            family = Poisson(), link = LogLink(),
            priors = PriorSpec(sds = Exponential(1.0)),
            nsamples = 1000, nchains = 1)

        @test m_bayes isa BayesGamModel

        # Intercept on log scale should be reasonable
        freq_int = StatsAPI.coef(m_freq)[1]
        bayes_int = StatsAPI.coef(m_bayes)[1]
        @test abs(bayes_int - freq_int) < 0.5

        # No σ_obs for Poisson
        @test !(Symbol("σ_obs") in names(m_bayes.chains))
    end

    @testset "Bernoulli GAM — binary data" begin
        Random.seed!(456)
        n = 300
        x = sort(rand(n))
        p_true = 1 ./ (1 .+ exp.(-(2 .* sin.(2π .* x))))
        y = Float64.([rand(Bernoulli(p)) for p in p_true])
        df = DataFrame(y = y, x = x)

        m_freq = gam(@gam_formula(y ~ s(x, k = 8)), df;
            family = Bernoulli(), link = LogitLink())
        m_bayes = gam(@gam_formula(y ~ s(x, k = 8)), df;
            family = Bernoulli(), link = LogitLink(),
            priors = PriorSpec(sds = Exponential(1.0)),
            nsamples = 1000, nchains = 1)

        @test m_bayes isa BayesGamModel

        freq_int = StatsAPI.coef(m_freq)[1]
        bayes_int = StatsAPI.coef(m_bayes)[1]
        @test abs(bayes_int - freq_int) < 0.5

        # No σ_obs for Bernoulli
        @test !(Symbol("σ_obs") in names(m_bayes.chains))
    end

    @testset "Multiple smooths" begin
        Random.seed!(789)
        n = 200
        x1 = sort(rand(n))
        x2 = randn(n)
        y = sin.(2π .* x1) .+ 0.5 .* x2 .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x1 = x1, x2 = x2)

        m_bayes = gam(@gam_formula(y ~ s(x1, k = 8) + s(x2, k = 8)), df;
            priors = PriorSpec(sds = Exponential(1.0)),
            nsamples = 1000, nchains = 1)

        @test m_bayes isa BayesGamModel
        @test m_bayes.n_smooth == 2
        @test length(m_bayes.smooth_labels) == 2

        # Should have σ_s for both smooths
        chains = m_bayes.chains
        @test Symbol("σ_s[1]") in names(chains)
        @test Symbol("σ_s[2]") in names(chains)

        # Both smooth SDs should be positive
        σ_s1 = mean(vec(chains[Symbol("σ_s[1]")].data))
        σ_s2 = mean(vec(chains[Symbol("σ_s[2]")].data))
        @test σ_s1 > 0
        @test σ_s2 > 0
    end

    @testset "Custom priors" begin
        Random.seed!(42)
        n = 100
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x = x)

        # Tight prior on sds → more smoothing (smaller σ_s)
        m_tight = gam(@gam_formula(y ~ s(x, k = 10)), df;
            priors = PriorSpec(sds = Exponential(0.1)),
            nsamples = 500, nchains = 1)

        # Wide prior on sds → less smoothing (larger σ_s)
        m_wide = gam(@gam_formula(y ~ s(x, k = 10)), df;
            priors = PriorSpec(sds = Exponential(5.0)),
            nsamples = 500, nchains = 1)

        σ_tight = mean(vec(m_tight.chains[Symbol("σ_s[1]")].data))
        σ_wide = mean(vec(m_wide.chains[Symbol("σ_s[1]")].data))
        @test σ_tight < σ_wide
    end

    @testset "Building blocks: gam_smooth + gam_matrices" begin
        Random.seed!(42)
        n = 100
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x = x)

        # gam_smooth
        smm = gam_smooth(:x, df; bs = :cr, k = 10)
        @test smm isa SmoothMixedModel
        @test size(smm.Xf, 1) == n
        @test !smm.fixed

        # gam_matrices
        gf = @gam_formula(y ~ s(x, k = 10))
        X, sms, labels = gam_matrices(gf, df)
        @test size(X, 1) == n
        @test size(X, 2) == 1  # intercept only
        @test length(sms) == 1
        @test length(labels) == 1

        # Custom @model using building blocks
        Xf = sms[1].Xf
        Zs = sms[1].Zs[1]
        X_fixed = hcat(X, Xf)

        @model function custom_gam(y, X_f, Z)
            n_f = size(X_f, 2)
            n_z = size(Z, 2)
            β ~ MvNormal(zeros(n_f), 10.0 * I)
            σ ~ truncated(Normal(0, 2.5); lower = 0.0)
            σ_s ~ Exponential(1.0)
            z ~ MvNormal(zeros(n_z), I)
            η = X_f * β .+ σ_s .* (Z * z)
            y ~ MvNormal(η, σ^2 * I)
        end

        chains = sample(custom_gam(y, X_fixed, Zs), NUTS(), 500; progress = false)
        σ_post = mean(vec(chains[:σ].data))
        @test 0.1 < σ_post < 1.0  # reasonable noise estimate
    end

    @testset "Bayesian GAMM — smooth + random intercept" begin
        Random.seed!(101)
        n = 100
        n_groups = 5
        x = randn(n)
        group = repeat(1:n_groups, inner = n ÷ n_groups)
        group_effects = randn(n_groups) * 0.5
        y = sin.(x) .+ group_effects[group] .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, group = string.(group))

        # Fit via @formula with re()
        m = gamm(@formula(y ~ cr(x, 10) + re(group)), df;
            priors = PriorSpec(sds = Exponential(1.0), sigma = Exponential(1.0)),
            nsamples = 500, nchains = 1)

        @test m isa GAM.BayesGamModel
        @test length(m.coef_names) >= 2  # intercept + smooth fixed part

        # Posterior means should be reasonable
        β_mean = GAM._bayes_coef_means(m)
        @test length(β_mean) > 0
        @test all(isfinite, β_mean)

        # Fit via @gamm_formula
        m2 = gamm(@gamm_formula(y ~ s(x, k = 10) + (1|group)), df;
            priors = PriorSpec(sds = Exponential(1.0), sigma = Exponential(1.0)),
            nsamples = 500, nchains = 1)

        @test m2 isa GAM.BayesGamModel
    end

    @testset "Convergence diagnostics — R-hat and ESS" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y_true = sin.(2π .* x)
        y = y_true .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x = x)

        m = gam(@gam_formula(y ~ s(x, k = 10)), df;
            priors = PriorSpec(sds = Exponential(1.0)),
            nsamples = 1000, nchains = 2)

        @test m isa BayesGamModel

        chains = m.chains
        diag = MCMCChains.ess_rhat(chains)
        ess_vals = diag[:, :ess]
        rhat_vals = diag[:, :rhat]

        # R-hat < 1.05 for all parameters (chains have converged)
        @test all(rhat_vals .< 1.05)

        # ESS > 100 for all parameters (sufficient effective samples)
        @test all(ess_vals .> 100)
    end

    @testset "Bayesian performance" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.3 .* randn(n)
        df = DataFrame(y = y, x = x)

        t = @elapsed begin
            m = gam(@gam_formula(y ~ s(x, k = 10)), df;
                priors = PriorSpec(sds = Exponential(1.0)),
                nsamples = 500, nchains = 1)
        end

        @test m isa BayesGamModel
        @test t < 120  # generous timeout for CI
    end
end
