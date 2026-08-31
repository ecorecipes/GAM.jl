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
        # Tests the boxing class DIRECTLY: the correlation evaluation must
        # cost O(1) allocations however many elements it fills. Before `scale`
        # was annotated this was ~40,000 at n=1000 — one per element of the
        # n x k loop — so flatness in the element count is the sharp test.
        # `_gp_E` takes n x d covariate MATRICES since multi-dimensional GP
        # smooths landed; a 1-D smooth is the d = 1 column case.
        knots_small = reshape(collect(range(-0.5, 0.5; length = 100)), :, 1)
        knots_large = reshape(collect(range(-0.5, 0.5; length = 1000)), :, 1)
        GAM._gp_E(knots_small, knots_small, 1.0, 3, 1.0, :mgcv, Float64[])
        GAM._gp_E(knots_large, knots_large, 1.0, 3, 1.0, :mgcv, Float64[])
        aE_small = @allocations GAM._gp_E(
            knots_small, knots_small, 1.0, 3, 1.0, :mgcv, Float64[])
        aE_large = @allocations GAM._gp_E(
            knots_large, knots_large, 1.0, 3, 1.0, :mgcv, Float64[])
        @test aE_small < 20
        @test aE_large < 20                # 100x the elements, same cost
        @test aE_large <= aE_small + 5

        # The multi-dimensional path sums over d inside the same inner loop,
        # which is exactly where a boxed accumulator would reappear, so guard
        # it too rather than only the d = 1 case the original bug was found in.
        k2_small = hcat(knots_small, reverse(knots_small; dims = 1))
        k2_large = hcat(knots_large, reverse(knots_large; dims = 1))
        GAM._gp_E(k2_small, k2_small, 1.0, 3, 1.0, :mgcv, Float64[])
        GAM._gp_E(k2_large, k2_large, 1.0, 3, 1.0, :mgcv, Float64[])
        a2_small = @allocations GAM._gp_E(
            k2_small, k2_small, 1.0, 3, 1.0, :mgcv, Float64[])
        a2_large = @allocations GAM._gp_E(
            k2_large, k2_large, 1.0, 3, 1.0, :mgcv, Float64[])
        @test a2_small < 20
        @test a2_large < 20
        @test a2_large <= a2_small + 5

        # [E(x,knt)*UZ | T(x)] is built in chunks, so it is O(1) too.
        # `shift` is a per-covariate vector and `knots` an nk x d matrix since
        # multi-dimensional GP smooths landed; d = 1 here.
        UZ = Matrix{Float64}(I, size(knots_large, 1), 18)
        cache = GAM.GPPredictCache([0.0], knots_large, UZ, 3, 1.0, 1.0,
                                   false, :mgcv, Float64[])
        xr = reshape(collect(range(-0.5, 0.5; length = size(knots_large, 1))), :, 1)
        GAM._gp_model_matrix(xr, cache)                # compile
        @test (@allocations GAM._gp_model_matrix(xr, cache)) < 40

        # Whole-construction bound, measured at n=400: the largest size still
        # taking `_tprs_top_eigen`'s DENSE branch (crossover max(400, 4k)).
        # Above it, gp legitimately spends ~1900 allocations in the shared
        # Lanczos eigensolver, because mgcv's construction eigen-reduces the
        # FULL knot correlation matrix (knots = the unique covariate values,
        # capped at max.knots) where the pre-port basis used a ~k x k Nystrom
        # approximation and so stayed on the dense branch. Asserting a total
        # above the crossover pins the eigensolver's iteration count rather
        # than gp's type stability, which is what this testset is for.
        rng = StableRNG(4242)
        x = sort(rand(rng, 400))
        tbl = (x = x,)
        spec = s(:x; bs = :gp, k = 20)

        smooth_construct(spec, tbl)                    # compile
        @test (@allocations smooth_construct(spec, tbl)) < 500
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
