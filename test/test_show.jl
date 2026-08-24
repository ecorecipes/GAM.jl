using Test
using GAM
using DataFrames
using StableRNGs

# `summary(m)` is the first thing an mgcv user types. Before 0.3 it fell
# through to `Base.summary`'s type-name fallback and printed "GamModel".
@testset "show / summary" begin
    rng = StableRNG(11)
    n = 200
    x1 = rand(rng, n) .* 2π
    x2 = rand(rng, n)
    y = sin.(x1) .+ 3 .* x2 .^ 2 .+ randn(rng, n) .* 0.3
    df = DataFrame(; y, x1, x2)
    m = gam(@formula(y ~ s(x1, k = 10, bs = :cr) + s(x2, k = 8)), df)

    shown = sprint(show, MIME"text/plain"(), m)
    summarised = sprint(show, MIME"text/plain"(), summary(m))

    @testset "summary matches the show output" begin
        @test summarised == shown
        @test !occursin("GamModel", summarised)
        # `summary` must stay printable through the plain-text path too, since
        # that is what the REPL falls back on inside containers.
        @test sprint(show, summary(m)) == shown
    end

    @testset "mgcv-style sections are present" begin
        @test occursin("Parametric coefficients:", shown)
        @test occursin("Approximate significance of smooth terms:", shown)
        @test occursin("edf", shown)
        @test occursin("Ref.df", shown)
        @test occursin("Deviance explained", shown)
    end

    @testset "significance codes" begin
        # Both smooths are strongly significant on this data, so the legend and
        # at least one starred row must appear.
        @test occursin("Signif. codes:", shown)
        @test occursin("***", shown)
    end

    @testset "footer reports the selection criterion" begin
        # Default is REML, reported as mgcv reports it.
        @test occursin("-REML", shown)
        @test occursin("Scale est.", shown)
        @test occursin("n = $n", shown)

        m_gcv = gam(@formula(y ~ s(x1, k = 10, bs = :cr)), df; method = :GCV)
        @test occursin("GCV", sprint(show, MIME"text/plain"(), m_gcv))
    end
end
