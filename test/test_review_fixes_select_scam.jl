# Regression tests for the two added features:
#   1. select=TRUE double-penalty term selection
#   2. offset / by for the shape-constrained (SCAM/SCASM) solvers
using Test
using GAM
using DataFrames
using Random
using Distributions
using Statistics
using StatsAPI: fitted, predict, deviance

@testset "select=TRUE and SCAM/SCASM offset & by" begin

    @testset "select=TRUE shrinks noise terms to ~zero" begin
        rng = Xoshiro(20)
        n = 600
        x1 = rand(rng, n); x2 = rand(rng, n); x3 = rand(rng, n)
        y = sin.(2pi .* x1) .+ 0.3 .* randn(rng, n)
        df = DataFrame(y = y, x1 = x1, x2 = x2, x3 = x3)

        m_sel = gam(GAM.@formula(y ~ s(x1, k=12) + s(x2, k=12) + s(x3, k=12)),
                    df; select = true)
        m_no = gam(GAM.@formula(y ~ s(x1, k=12) + s(x2, k=12) + s(x3, k=12)),
                   df; select = false)

        # each smooth gets a second (null-space) penalty under select
        @test length(m_sel.sp) == 2 * length(m_no.sp)
        # signal term keeps substantial EDF
        @test m_sel.edf[1] > 3.0
        # noise terms shrink well below the EDF=1 floor that no-select leaves
        @test m_sel.edf[2] < 0.5
        @test m_sel.edf[3] < 0.5
        @test m_no.edf[2] > 0.8   # without select, the linear part survives
    end

    @testset "select=TRUE works for non-Gaussian families" begin
        rng = Xoshiro(21)
        n = 500
        x1 = rand(rng, n) .* 2pi; x2 = rand(rng, n)
        yp = Float64.(rand.(rng, Poisson.(exp.(0.5 .* sin.(x1) .+ 1.0))))
        df = DataFrame(y = yp, x1 = x1, x2 = x2)
        m = gam(GAM.@formula(y ~ s(x1, k=12) + s(x2, k=12)), df, Poisson(),
                GAM.GLM.LogLink(); select = true)
        @test m.converged
        @test m.edf[1] > 2.0     # real signal
        @test m.edf[2] < 0.6     # noise selected out
    end

    @testset "SCAM offset" begin
        rng = Xoshiro(22)
        n = 500
        x = sort(rand(rng, n))
        off = randn(rng, n) .* 0.3
        y = 2 .* x .+ off .+ 0.1 .* randn(rng, n)   # monotone + offset
        df = DataFrame(y = y, x = x)
        m = gam(GAM.@formula(y ~ s(x, bs=:mpi, k=12)), df; offset = off)
        @test m.converged
        @test isapprox(predict(m, df; offset = off), fitted(m); atol = 1e-6)
        # fit without the offset is materially different
        m0 = gam(GAM.@formula(y ~ s(x, bs=:mpi, k=12)), df)
        @test deviance(m) < deviance(m0)
    end

    @testset "SCASM offset" begin
        rng = Xoshiro(23)
        n = 400
        x = sort(rand(rng, n))
        off = randn(rng, n) .* 0.2
        y = sin.(pi .* x) .+ off .+ 0.1 .* randn(rng, n)
        df = DataFrame(y = y, x = x)
        m = gam(GAM.@formulak(y ~ s(x, bs=:sc, xt=["m+"], k=12)), df; offset = off)
        @test m.converged
        @test isapprox(predict(m, df; offset = off), fitted(m); atol = 1e-6)
    end

    @testset "SCAM factor-by (monotone smooth per level)" begin
        rng = Xoshiro(31)
        n = 600
        x = rand(rng, n)
        g = rand(rng, ["a", "b"], n)
        f(gi, xi) = gi == "a" ? 3 * xi : 1.5 * sqrt(xi)
        y = [f(g[i], x[i]) for i in 1:n] .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, g = g, y = y)
        m = gam(GAM.@formula(y ~ g + s(x, bs=:mpi, k=10, by=g)), df)
        tr = [f(g[i], x[i]) for i in 1:n]
        @test cor(fitted(m), tr) > 0.99
        @test isapprox(predict(m, df), fitted(m); atol = 1e-6)
        # each level's fitted smooth is monotone increasing
        xg = collect(0.05:0.05:0.95)
        for lev in ["a", "b"]
            pg = predict(m, DataFrame(x = xg, g = fill(lev, length(xg))))
            @test all(diff(pg) .>= -1e-6)
        end
    end

    @testset "parametric factor levels reused at prediction" begin
        # predicting on a subset of factor levels must still produce the right
        # dummy columns (uses training levels, not newdata levels)
        rng = Xoshiro(32)
        n = 300
        x = rand(rng, n)
        g = rand(rng, ["a", "b", "c"], n)
        y = (g .== "a") .* 1.0 .+ (g .== "b") .* 2.0 .+ x .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, g = g, y = y)
        m = gam(GAM.@formula(y ~ g + s(x, k=8)), df)
        # predict with only one level present
        p = predict(m, DataFrame(x = [0.5, 0.6], g = ["a", "a"]))
        @test length(p) == 2
        @test all(isfinite, p)
    end
end
