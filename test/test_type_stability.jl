# Allocation regression guards for the type-stability bug class.
#
# Round 5 found that thin-plate construction made ~580,000 allocations because
# a generic `hcat` dispatch lost concrete type inference and boxed every
# element access. Round 6 found two more instances of the same class:
# `basis_gp.jl`'s `scale` read (Dict{Symbol,Any} → Float64(::Any) → Any) and
# `_bspline_basis` allocating its Cox-de Boor buffers inside the per-row loop.
#
# These are invisible to correctness tests — output stays bit-identical — so
# they are pinned here by allocation count instead. Bounds are ~3-5x the
# measured post-fix values: loose enough not to be flaky under load (allocation
# counts are contention-immune, but GC and inlining decisions can shift them
# slightly), tight enough to catch a regression of the 1000x kind these were.

@testset "Type stability / allocation guards" begin

    @testset "_bspline_basis allocation is O(1), not O(rows)" begin
        knots = collect(range(-0.15, 1.15; length = 20))
        x_small = collect(range(0.01, 0.99; length = 100))
        x_large = collect(range(0.01, 0.99; length = 10_000))

        GAM._bspline_basis(x_small, knots, 4)          # compile
        a_small = @allocations GAM._bspline_basis(x_small, knots, 4)
        a_large = @allocations GAM._bspline_basis(x_large, knots, 4)

        # The buffers are allocated once per call, so a 100x longer input must
        # not cost 100x more allocations. Before the fix this was ~10 per row.
        @test a_small < 40
        @test a_large < 40
        @test a_large <= a_small + 5
    end

    @testset "gp basis construction does not box the inner loop" begin
        rng = StableRNG(4242)
        x = sort(rand(rng, 1000))
        tbl = (x = x,)
        spec = s(:x; bs = :gp, k = 20)

        smooth_construct(spec, tbl)                    # compile
        a = @allocations smooth_construct(spec, tbl)

        # Before annotating `scale`, this was ~40,000 allocations at n=1000
        # (one per element of the n x k correlation loop).
        @test a < 500
    end

    @testset "P-spline construction allocation is bounded" begin
        rng = StableRNG(4243)
        x = sort(rand(rng, 5000))
        tbl = (x = x,)
        spec = s(:x; bs = :ps, k = 20)

        smooth_construct(spec, tbl)                    # compile
        a = @allocations smooth_construct(spec, tbl)

        # ~50,000 before hoisting the B-spline buffers.
        @test a < 1200
    end

    @testset "SCAM fit allocation is bounded" begin
        rng = StableRNG(4244)
        n = 2000
        x = sort(rand(rng, n))
        y = 2.0 .* x .+ 0.1 .* randn(rng, n)
        df = (x = x, y = y)
        f = @formulak(y ~ s(x, bs = :mpi, k = 12))

        gam(f, df)                                     # compile
        a = @allocations gam(f, df)

        # SCAM rebuilds its design each P-IRLS iteration through the exp
        # reparameterization, so it amplifies any per-row basis allocation.
        @test a < 20_000
    end

    @testset "gp rejects k > number of unique covariate values" begin
        # mgcv: "A term has fewer unique covariate combinations than specified
        # maximum degrees of freedom". Previously gp fitted silently.
        x_tied = repeat(collect(range(0.0, 1.0; length = 5)), inner = 20)
        tbl = (x = x_tied,)
        @test_throws ArgumentError smooth_construct(s(:x; bs = :gp, k = 10), tbl)
        # Within the unique-value budget it still constructs.
        @test smooth_construct(s(:x; bs = :gp, k = 5), tbl) isa ConstructedSmooth
    end
end
