# Regression tests for by= variable support (previously silently ignored).
using Test
using GAM
using DataFrames
using Random
using Statistics
using StatsAPI: fitted, predict, coef

@testset "by= variable support" begin

    @testset "numeric by (varying coefficient)" begin
        rng = Xoshiro(99)
        n = 500
        x = rand(rng, n)
        v = randn(rng, n)
        f1 = sin.(2pi .* x)
        y = v .* f1 .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, v = v, y = y)

        m = gam(GAM.@formula(y ~ v + s(x, k = 10, by = v)), df)
        # The basis must actually be modulated by v: recovers v*sin(2πx)
        @test sqrt(mean((fitted(m) .- v .* f1) .^ 2)) < 0.05
        # predict reuses the same by transform
        @test isapprox(predict(m, df), fitted(m); atol = 1e-7)
        # the smooth spec carries the by variable
        @test m.smooths[1].spec.by == :v
    end

    @testset "by is honored — differs from no-by fit" begin
        rng = Xoshiro(7)
        n = 400
        x = rand(rng, n)
        v = randn(rng, n)
        y = v .* sin.(2pi .* x) .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, v = v, y = y)
        m_by = gam(GAM.@formula(y ~ v + s(x, k = 10, by = v)), df)
        m_no = gam(GAM.@formula(y ~ v + s(x, k = 10)), df)
        # with by, deviance is far smaller (the no-by model can't fit v*f(x))
        @test m_by.deviance_val < 0.25 * m_no.deviance_val
    end

    @testset "factor by (smooth per level)" begin
        rng = Xoshiro(99)
        n = 500
        x = rand(rng, n)
        g = rand(rng, ["a", "b", "c"], n)
        fa(gi, xi) = gi == "a" ? 2xi : gi == "b" ? cos(2pi * xi) : -xi^2
        y = [fa(g[i], x[i]) for i in 1:n] .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, g = g, y = y)

        m = gam(GAM.@formula(y ~ g + s(x, k = 10, by = g)), df)
        truth = [fa(g[i], x[i]) for i in 1:n]
        @test cor(fitted(m), truth) > 0.99
        # one smoothing parameter per factor level
        @test length(m.sp) == 3
        @test isapprox(predict(m, df), fitted(m); atol = 1e-7)
    end

    @testset "factor by — unseen level at predict warns, contributes zero" begin
        rng = Xoshiro(3)
        n = 300
        x = rand(rng, n)
        g = rand(rng, ["a", "b"], n)
        y = [g[i] == "a" ? 2x[i] : -x[i]^2 for i in 1:n] .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, g = g, y = y)
        m = gam(GAM.@formula(y ~ g + s(x, k = 8, by = g)), df)

        newdf = DataFrame(x = [0.5, 0.5], g = ["a", "c"])  # "c" unseen
        local p
        @test_logs (:warn,) match_mode = :any begin
            p = predict(m, newdf)
        end
        @test all(isfinite, p)
    end
end
