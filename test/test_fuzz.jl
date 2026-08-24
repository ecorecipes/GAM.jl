# Seeded property-based fuzz over random model configurations
#
# Promoted from the round-4 review's fuzz harness, which found four defects
# in configuration space that the reference-problem suite missed (Gamma
# canonical-link domain violations, the penalized :fp path, TPRS with
# k > n_unique, and the absolute scale floor). Each curated seed draws a
# random combination of bases/family/size/data-quirk and asserts the fit
# either succeeds with sane outputs or is rejected with an informative
# ArgumentError.

using Test
using GAM
using DataFrames
using Random
using Statistics
using LinearAlgebra
using StableRNGs
using Distributions: Bernoulli
using StatsAPI: fitted, predict

const _FUZZ_BASES = [:tp, :cr, :cs, :ts, :cc, :ps, :cps, :bs, :ad, :gp, :re, :fp, :lo]
const _FUZZ_FAMS = [Normal(), Poisson(), Bernoulli(), Gamma()]

function _fuzz_draw(seed::Int)
    rng = StableRNG(seed)
    n = rand(rng, [30, 60, 120, 300, 800])
    nsm = rand(rng, 1:3)
    fam = rand(rng, _FUZZ_FAMS)
    quirk = rand(rng, [:none, :ties, :bigscale, :smallscale, :nearconst_y,
        :zeroweights, :offset, :by_num, :by_factor])
    cols = Dict{Symbol, Any}()
    specs = GAM.SmoothSpec[]
    η = fill(0.2, n)
    share = nsm >= 2 && rand(rng) < 0.3
    for j in 1:nsm
        bs = rand(rng, _FUZZ_BASES)
        bs === :lo && n > 300 && (bs = :cr)
        k = rand(rng, 4:25)
        # `bs=:ad` defaults to mgcv's 5 adaptive sub-penalties, and mgcv errors
        # when the smoothing basis cannot hold them. Clamp AFTER the draw so the
        # seeded RNG stream — and every other seed's configuration — is unchanged.
        bs === :ad && (k = max(k, 8))
        if bs === :re
            g = rand(rng, 1:rand(rng, 3:8), n)
            v = Symbol("g$j")
            cols[v] = g
            push!(specs, s(v; bs = :re))
        else
            v = (share && j == 2) ? :x1 : Symbol("x$j")
            if !haskey(cols, v)
                x = rand(rng, n)
                quirk === :ties && (x = round.(x; digits = 1))
                quirk === :bigscale && (x .*= 1e6)
                quirk === :smallscale && (x .*= 1e-6)
                cols[v] = x
            end
            xs = (cols[v] .- minimum(cols[v])) ./
                 max(maximum(cols[v]) - minimum(cols[v]), eps())
            η .+= 0.8 .* sin.(2π .* xs)
            byv = nothing
            if j == 1 && quirk === :by_num
                cols[:z] = randn(rng, n)
                byv = :z
            elseif j == 1 && quirk === :by_factor
                cols[:gby] = rand(rng, ["a", "b"], n)
                byv = :gby
            end
            push!(specs, byv === nothing ? s(v; bs = bs, k = k) :
                         s(v; bs = bs, k = k, by = byv))
        end
    end
    quirk === :nearconst_y && (η .= 0.2 .+ 1e-9 .* randn(rng, n))
    ηc = clamp.(η, -4.0, 4.0)
    y = fam isa Normal ? η .+ 0.3 .* randn(rng, n) :
        fam isa Poisson ? Float64.([rand(rng, Poisson(exp(e))) for e in ηc]) :
        fam isa Bernoulli ? Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-ηc))) :
        [rand(rng, Gamma(2.0, exp(e) / 2.0)) + 1e-8 for e in ηc]
    cols[:y] = y
    w = quirk === :zeroweights ? (wv = ones(n); wv[1:max(1, n ÷ 10)] .= 0.0; wv) : nothing
    off = quirk === :offset ? fill(0.7, n) : nothing
    return DataFrame(cols), GAM.GamFormula(:y, Symbol[], true, specs), fam, w, off, specs
end

# ok-seeds fit and satisfy the invariants; rejected-seeds raise an
# informative ArgumentError (currently: TPRS with k > n_unique under ties)
const _FUZZ_OK = [1, 3, 5, 9, 13, 21, 34, 55, 61, 77, 89, 103, 109, 121]
const _FUZZ_REJECTED = [2, 4, 6, 50]

@testset "Configuration fuzz (seeded property test)" begin
    @testset "valid configurations fit sanely" begin
        for seed in _FUZZ_OK
            df, gf, fam, w, off, specs = _fuzz_draw(seed)
            m = gam(gf, df; family = fam, weights = w, offset = off)
            f = fitted(m)
            @test all(isfinite, f)
            p = predict(m, df; type = :response, offset = off)
            @test maximum(abs.(p .- f)) < 1e-5 * (1 + maximum(abs.(f)))
            @test 0.0 < m.edf_total <= length(m.coefficients) + 1e-6
            ev = eigvals(Symmetric(m.Vp))
            @test minimum(ev) > -1e-6 * max(maximum(ev), 1.0)
        end
    end

    @testset "invalid configurations rejected informatively" begin
        for seed in _FUZZ_REJECTED
            df, gf, fam, w, off, _ = _fuzz_draw(seed)
            @test_throws ArgumentError gam(gf, df; family = fam,
                weights = w, offset = off)
        end
    end

    # ── Targeted regressions for the four fuzz-found defects ──

    @testset "F1: Gamma canonical-link domain consistency (seed 103)" begin
        df, gf, fam, w, off, _ = _fuzz_draw(103)
        @test fam isa Gamma
        m = gam(gf, df; family = fam)
        p = predict(m, df; type = :response)
        @test all(>(0.0), p)                       # no negative Gamma means
        @test p ≈ fitted(m) atol = 1e-8            # predict consistent with fit
    end

    @testset "F2: penalized :fp fits (auto-unpenalized)" begin
        rng = StableRNG(7)
        n = 120
        x = rand(rng, n) .* 2 .+ 0.5
        y = sin.(x) .+ 0.1 .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        m = gam(GAM.@formula(y ~ s(x, bs = :fp, k = 4)), df)   # default fx handling
        @test m.converged
        @test cor(fitted(m), y) > 0.5
    end

    @testset "F3: TPRS k > n_unique errors informatively (both modes)" begin
        for n in (800, 200)   # crash mode and silent-edf-0 mode pre-fix
            rng = StableRNG(42)
            x = round.(rand(rng, n); digits = 1)   # 11 unique values
            y = sin.(2π .* x) .+ 0.1 .* randn(rng, n)
            df = DataFrame(x = x, y = y)
            err = try
                gam(GAM.@formula(y ~ s(x, k = 13, bs = :tp)), df)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("unique covariate", sprint(showerror, err))
            mcr = gam(GAM.@formula(y ~ s(x, k = 8, bs = :cr)), df)
            @test mcr.edf_total > 1.0
        end
    end

    @testset "F4: scale invariance of the fit under response rescaling" begin
        rng = StableRNG(9)
        n = 250
        x = rand(rng, n) .* 2π
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        ma = gam(GAM.@formula(y ~ s(x, k = 12, bs = :cr)), DataFrame(x = x, y = y))
        mb = gam(GAM.@formula(y ~ s(x, k = 12, bs = :cr)),
            DataFrame(x = x, y = y .* 1e-8))
        mc = gam(GAM.@formula(y ~ s(x, k = 12, bs = :cr)),
            DataFrame(x = x, y = y .* 1e8))
        # Rescaled fits agree with the unit fit to convergence tolerance and
        # with each other essentially exactly (the pre-fix behavior collapsed
        # tiny-y edf from ~9 to ~3 via the absolute scale floor).
        @test mb.edf_total ≈ ma.edf_total atol = 1e-2
        @test mc.edf_total ≈ ma.edf_total atol = 1e-2
        @test mb.edf_total ≈ mc.edf_total atol = 1e-8
        @test cor(fitted(ma), fitted(mb) ./ 1e-8) > 1 - 1e-8
        @test cor(fitted(ma), fitted(mc) ./ 1e8) > 1 - 1e-8
        @test mb.scale / ma.scale ≈ 1e-16 rtol = 1e-4
    end

    @testset "F5: na_action = :omit equals fitting the hand-filtered table" begin
        # Property: for randomly placed missing/NaN/Inf entries across the
        # response, the covariates, the weights and the offset, an :omit fit
        # must be indistinguishable from fitting the table the caller would
        # have built by dropping exactly those rows — and na_omit_rows must
        # name that row set.
        for seed in 1:6
            rng = StableRNG(600 + seed)
            n = 150
            x = rand(rng, n)
            z = rand(rng, n)
            df = DataFrame(x = x, z = z,
                y = sin.(2π .* x) .+ 0.4 .* z .+ 0.3 .* randn(rng, n))

            allowmissing!(df, :y)
            allowmissing!(df, :x)
            bad_y = rand(rng, 1:n, 2)
            bad_x = rand(rng, 1:n, 2)
            bad_z = rand(rng, 1:n, 1)
            df.y[bad_y] .= missing
            df.x[bad_x] .= missing
            df.z[bad_z] .= (seed % 2 == 0 ? NaN : Inf)

            w = 0.5 .+ rand(rng, n)
            off = 0.1 .* randn(rng, n)
            expected = sort(unique(vcat(bad_y, bad_x, bad_z)))

            f = GAM.@formulak(y ~ s(x, k = 6, bs = :cr) + s(z, k = 6, bs = :cr))
            keep = GAM.na_omit_rows(df, :y, [:x, :z])
            @test keep == sort(setdiff(1:n, expected))

            m = bam(f, df; weights = w, offset = off, na_action = :omit)
            m_ref = bam(f, df[keep, :]; weights = w[keep], offset = off[keep])
            @test length(fitted(m)) == length(keep)
            @test coef(m) ≈ coef(m_ref) atol = 1e-12
            @test deviance(m) ≈ deviance(m_ref) atol = 1e-12

            # and :fail still refuses the same table
            @test_throws ArgumentError bam(f, df; weights = w, offset = off)
        end
    end
end
