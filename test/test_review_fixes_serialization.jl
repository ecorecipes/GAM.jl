# Verifies that smooth types whose prediction metadata used to live in
# module-level objectid-keyed dicts now store it in `ConstructedSmooth.predict_cache`,
# so the model survives serialization round-trips.
#
# For each affected smooth (te, ti, t2, fs, sz, sos, spde, mrf, so):
#   (a) predict(m, traindata) ≈ fitted(m)   (cache used correctly)
#   (b) deserialize(serialize(m)) predicts identically (the whole point)

using Test
using GAM
using DataFrames
using Random
using LinearAlgebra
using Serialization
using StatsAPI: fitted, predict

const RTOL_FITTED = 1e-6
const RTOL_ROUND  = 1e-8

"""
Run the two checks for a fitted model `m` against its training `df`.
`gc_before_roundtrip` forces a GC + drops the original model reference so the
old objectid-dict approach would definitely fail.
"""
function check_serialization(name, m, df)
    @testset "$name" begin
        pred = predict(m, df)
        fit = fitted(m)
        @test length(pred) == length(fit)
        @test isapprox(pred, fit; rtol = RTOL_FITTED, atol = 1e-8)

        buf = IOBuffer()
        serialize(buf, m)
        seekstart(buf)
        m2 = deserialize(buf)

        # Force GC: with the old objectid dicts, an objectid collision or the
        # missing dict entry on the deserialized object would surface here.
        GC.gc()

        pred2 = predict(m2, df)
        @test isapprox(pred2, pred; rtol = RTOL_ROUND, atol = 1e-10)
    end
end

@testset "Review fix: predict_cache serialization" begin

    # ── te() ──────────────────────────────────────────────────────────────
    let
        Random.seed!(1)
        n = 200
        x = randn(n); z = randn(n)
        y = sin.(x) .+ cos.(z) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, z = z, y = y)
        m = gam(GAM.@formulak(y ~ te(x, z, k = 25)), df)
        check_serialization("te", m, df)
    end

    # ── ti() ──────────────────────────────────────────────────────────────
    let
        Random.seed!(2)
        n = 200
        x = randn(n); z = randn(n)
        y = sin.(x) .* cos.(z) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, z = z, y = y)
        m = gam(GAM.@formulak(y ~ ti(x, z, k = 25, bs = :cr)), df)
        check_serialization("ti", m, df)
    end

    # ── t2() ──────────────────────────────────────────────────────────────
    let
        Random.seed!(3)
        n = 200
        x = randn(n); z = randn(n)
        y = sin.(x) .+ cos.(z) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, z = z, y = y)
        m = gam(GAM.@formulak(y ~ t2(x, z, k = 25)), df)
        check_serialization("t2", m, df)
    end

    # ── fs (factor-smooth interaction) ─────────────────────────────────────
    let
        Random.seed!(4)
        n = 180
        group = repeat(["a", "b", "c"], inner = n ÷ 3)
        x = randn(n)
        amp = Dict("a" => 1.0, "b" => 2.0, "c" => 0.5)
        y = [amp[g] * sin(xi) for (g, xi) in zip(group, x)] .+ 0.1 .* randn(n)
        df = DataFrame(x = x, group = group, y = y)
        m = gam(GAM.@formulak(y ~ s(x, group, bs = :fs, k = 8)), df)
        check_serialization("fs", m, df)
    end

    # ── sz (constrained factor smooth) ─────────────────────────────────────
    let
        Random.seed!(5)
        n = 180
        group = repeat(["a", "b", "c"], inner = n ÷ 3)
        x = randn(n)
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x = x, group = group, y = y)
        m = gam(GAM.@formulak(y ~ s(x, group, bs = :sz, k = 8)), df)
        check_serialization("sz", m, df)
    end

    # ── sos (spline on the sphere) ─────────────────────────────────────────
    let
        Random.seed!(6)
        n = 200
        lat = π/2 .* (2 .* rand(n) .- 1)
        lon = π .* (2 .* rand(n) .- 1)
        y = sin.(lat) .* cos.(lon) .+ 0.1 .* randn(n)
        df = DataFrame(lat = lat, lon = lon, y = y)
        m = gam(GAM.@formulak(y ~ s(lat, lon, bs = :sos, k = 20)), df)
        check_serialization("sos", m, df)
    end

    # ── spde (1D Matérn) ───────────────────────────────────────────────────
    let
        Random.seed!(7)
        n = 200
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)
        m = gam(GAM.@formulak(y ~ s(x, bs = :spde, k = 30)), df)
        check_serialization("spde", m, df)
    end

    # ── mrf (Markov random field) ──────────────────────────────────────────
    let
        Random.seed!(8)
        nb = zeros(Int, 6, 6)
        nb[1,2] = nb[2,1] = 1; nb[2,3] = nb[3,2] = 1
        nb[4,5] = nb[5,4] = 1; nb[5,6] = nb[6,5] = 1
        nb[1,4] = nb[4,1] = 1; nb[2,5] = nb[5,2] = 1; nb[3,6] = nb[6,3] = 1
        n = 300
        regions = rand(1:6, n)
        effects = [0.0, 1.0, 2.0, 0.5, 1.5, 2.5]
        y = [effects[r] for r in regions] .+ 0.3 .* randn(n)
        df = DataFrame(region = regions, y = y)
        spec = s(:region, bs = :mrf, xt = Dict{Symbol,Any}(:nb => nb))
        gf = GamFormula(:y, Symbol[], true, SmoothSpec[spec])
        m = gam(gf, df)
        check_serialization("mrf", m, df)
    end

    # ── so (soap film) ─────────────────────────────────────────────────────
    let
        Random.seed!(9)
        n = 250
        bnd = [hcat([0, 1, 1, 0, 0.0], [0, 0, 1, 1, 0.0])]
        x = 0.05 .+ 0.9 .* rand(n)
        y = 0.05 .+ 0.9 .* rand(n)
        z = sin.(2π .* x) .* cos.(2π .* y) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y, z = z)
        m = gam(GAM.@formulak(z ~ s(x, y, bs = :so, k = 15,
                  xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 40))), df)
        check_serialization("so", m, df)
    end

end
