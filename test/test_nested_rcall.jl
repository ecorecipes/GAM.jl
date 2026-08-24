# Nested effects: live comparison against R's gamFactory (gam_nl / s_nest)
#
# Requires R with gamFactory (github.com/mfasiolo/gamFactory) installed;
# gated by the availability check in runtests.jl.

using Test
using GAM
using StatsAPI: fitted, predict
using RCall
using Random, Statistics, LinearAlgebra
using StableRNGs

@testset "Nested effects vs gamFactory" begin

    @testset "Single-index (trans_linear): agreement with gamFactory" begin
        rng = StableRNG(407)
        n = 400
        X = randn(rng, n, 3)
        a_true = normalize([0.7, 0.5, 0.2])
        u = X * a_true
        f_true = sin.(1.5 .* u)
        y = f_true .+ 0.2 .* randn(rng, n)

        # Julia fit
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        m_jl = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 10)), df)
        @test m_jl.converged
        a_jl = inner_coef(m_jl)
        fit_jl = fitted(m_jl)

        # R / gamFactory fit on the same data (fam_gaussian is a two-LP
        # Gaussian; the second predictor is intercept-only, like ours)
        @rput y X
        RCall.reval("""
        library(gamFactory)
        dat <- list(y = y, X = X)
        fit_r <- gam_nl(list(y ~ s_nest(X, trans = trans_linear(), k = 10), ~ 1),
                        family = fam_gaussian(), data = dat)
        cf <- coef(fit_r)
        a_r <- cf[2:4]
        fit_vals_r <- fitted(fit_r)[, 1]
        """)
        a_r = rcopy(Vector{Float64}, R"a_r")
        a_r ./= norm(a_r)
        fit_r = rcopy(Vector{Float64}, R"fit_vals_r")

        # Index directions agree (up to sign) and both recover the truth
        @test abs(cor(X * a_jl, X * a_r)) > 0.999
        @test abs(cor(X * a_jl, u)) > 0.99
        @test abs(cor(X * a_r, u)) > 0.99
        @test abs(dot(a_jl, a_r)) > 0.999

        # Fitted values agree closely between implementations
        @test cor(fit_jl, fit_r) > 0.999
        @test maximum(abs.(fit_jl .- fit_r)) < 0.15
        # and both recover the true function
        @test cor(fit_jl, f_true) > 0.99
        @test cor(fit_r, f_true) > 0.99
    end

    @testset "Poisson single-index: agreement with gamFactory" begin
        rng = StableRNG(408)
        n = 500
        X = randn(rng, n, 3)
        a_true = normalize([0.6, 0.35, 0.15])
        u = X * a_true
        η = 0.3 .+ 0.8 .* tanh.(2 .* u)
        y = Float64.(rand.(rng, Poisson.(exp.(η))))

        df = (y = y, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        m_jl = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 10)), df; family = Poisson())
        a_jl = inner_coef(m_jl)

        @rput y X
        RCall.reval("""
        library(gamFactory)
        dat <- list(y = y, X = X)
        fit_r <- gam_nl(y ~ s_nest(X, trans = trans_linear(), k = 10),
                        family = fam_poisson(), data = dat)
        cf <- coef(fit_r)
        a_r <- cf[2:4]
        mu_r <- as.vector(fitted(fit_r))
        """)
        a_r = rcopy(Vector{Float64}, R"a_r")
        a_r ./= norm(a_r)
        mu_r = rcopy(Vector{Float64}, R"mu_r")

        @test abs(dot(a_jl, a_r)) > 0.99
        @test cor(fitted(m_jl), mu_r) > 0.99
    end
end
