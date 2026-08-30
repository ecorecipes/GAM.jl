# Discretised tensor smooths — `bam(..., discrete = true)` with `te`
#
# A `te` block is stored as its per-marginal bases at the unique covariate
# values plus one index vector per marginal; the row tensor is never formed.
# That is sound only because GAM.jl's `te` applies a SINGLE overall
# sum-to-zero constraint to the raw row tensor rather than reparameterising
# each marginal, so the constraint can be applied AFTER accumulation
# (mgcv does the same at `discrete.c:2229-2266`).
#
# The first testset pins that structural precondition directly. If a future
# change to `_construct_tensor` absorbed a per-marginal transform instead, the
# kernels here would silently compute the wrong block, and this is what would
# fail first.

@testset "Discrete tensor smooths" begin

    function _mk(n; nu = 40, seed = 11, three = false)
        rng = MersenneTwister(seed)
        g1 = collect(range(0, 1; length = nu))
        g2 = collect(range(0, 1; length = nu + 7))
        x1 = g1[rand(rng, 1:nu, n)]
        x2 = g2[rand(rng, 1:(nu + 7), n)]
        y = sin.(2π .* x1) .+ cos.(3 .* x2) .+ 0.2 .* randn(rng, n)
        if three
            g3 = collect(range(0, 1; length = nu - 5))
            x3 = g3[rand(rng, 1:(nu - 5), n)]
            return DataFrame(x1 = x1, x2 = x2, x3 = x3, y = y .+ 0.5 .* x3)
        end
        return DataFrame(x1 = x1, x2 = x2, y = y)
    end

    @testset "te has the post-hoc constraint structure the kernels assume" begin
        # X_cons == X_raw * Z, with no per-marginal reparameterization.
        df = _mk(200)
        for spec in (GAM.te(:x1, :x2; k = [5, 4]), GAM.te(:x1, :x2; k = [3, 3]))
            sm = smooth_construct(spec, df)
            cache = sm.predict_cache
            @test cache isa GAM.TensorPredictCache
            @test isempty(cache.marginal_Zs)          # no per-marginal Z
            @test sm.constraint !== nothing           # one overall constraint
            Xraw = GAM._row_kronecker([rm.X for rm in cache.raw_marginals])
            Z = GAM._constraint_basis(sm.constraint, size(Xraw, 2))
            @test sm.X ≈ Xraw * Z atol = 1e-12
        end
    end

    @testset "discrete reproduces the dense fit on discretisable data" begin
        # Covariates take few distinct values, so binning is lossless and the
        # only difference is the representation. Tolerances are tight on
        # purpose: a looser bar would hide a genuine kernel bug.
        cases = (
            ("te 2d",       GAM.@formula(y ~ te(x1, x2, k = [6, 5])),  _mk(4000)),
            ("te 3d",       GAM.@formula(y ~ te(x1, x2, x3, k = [4, 5, 3])),
                _mk(4000; three = true)),
            ("te + s()",    GAM.@formula(y ~ te(x1, x2, k = [5, 5]) + s(x2, k = 6, bs = :cr)),
                _mk(4000)),
        )
        for (lbl, f, df) in cases
            md = bam(f, df; discrete = false)
            mq = bam(f, df; discrete = true)
            scale = max(maximum(abs, coef(md)), 1e-300)
            @test maximum(abs.(coef(md) .- coef(mq))) / scale < 1e-10
            @test md.edf_total ≈ mq.edf_total rtol = 1e-9
            @test maximum(abs.(fitted(md) .- fitted(mq))) < 1e-9
            @test deviance(md) ≈ deviance(mq) rtol = 1e-9
        end
    end

    @testset "kernels agree with the dense accumulator" begin
        df = _mk(3000)
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.te(:x1, :x2; k = [5, 4])])
        _, X, _, sms, _ = GAM.setup_gam(gf, df)
        Dd = GAM.bam_design(X)
        Dq = GAM.bam_design(X, sms, df, true)
        @test Dq isa GAM.DiscreteDesign
        @test length(Dq.tblocks) == 1
        @test GAM.ncols(Dq) == GAM.ncols(Dd)

        rng = MersenneTwister(4)
        n = size(X, 1)
        w = abs.(randn(rng, n)) .+ 0.5
        z = randn(rng, n)
        p = size(X, 2)

        A = zeros(p, p); B = zeros(p, p)
        GAM.accumulate_XtWX!(A, Dd, w)
        GAM.accumulate_XtWX!(B, Dq, w)
        @test maximum(abs.(A .- B)) / maximum(abs.(A)) < 1e-12

        A2 = zeros(p, p); a2 = zeros(p); B2 = zeros(p, p); b2 = zeros(p)
        GAM.accumulate_XtWX_XtWz!(A2, a2, Dd, w, z)
        GAM.accumulate_XtWX_XtWz!(B2, b2, Dq, w, z)
        @test maximum(abs.(A2 .- B2)) / maximum(abs.(A2)) < 1e-12
        @test maximum(abs.(a2 .- b2)) / maximum(abs.(a2)) < 1e-12

        beta = randn(rng, p)
        e1 = zeros(n); e2 = zeros(n)
        GAM.mul_eta!(e1, Dd, beta)
        GAM.mul_eta!(e2, Dq, beta)
        @test maximum(abs.(e1 .- e2)) / maximum(abs.(e1)) < 1e-12

        # The leverage sweep materialises rows a chunk at a time; those rows
        # must match the dense model matrix exactly.
        H = zeros(17, p)
        GAM._gather_rows!(H, Dq, 11, 27)
        @test maximum(abs.(H .- X[11:27, :])) < 1e-12
    end

    @testset "ti and t2 are not discretised" begin
        # Both reparameterize before the tensor product, so neither has the
        # single post-hoc Z the kernels rely on. They must fall back rather
        # than be silently mis-accumulated.
        # n = 4000 so the 40×40 grid is well under n: on this data a plain
        # te IS discretised (positive control below). Without that control
        # the fallback assertion is vacuous — a discretiser broken into
        # returning DenseDesign for everything would pass it.
        df = _mk(4000)
        gf_te = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.te(:x1, :x2; k = [4, 4])])
        _, X_te, _, sms_te, _ = GAM.setup_gam(gf_te, df)
        D_te = GAM.bam_design(X_te, sms_te, df, true)
        @test D_te isa GAM.DiscreteDesign
        @test length(D_te.tblocks) == 1

        for spec in (GAM.ti(:x1, :x2; k = [4, 4]), GAM.t2(:x1, :x2; k = [4, 4]))
            gf = GAM.GamFormula(:y, Symbol[], true, GAM.SmoothSpec[spec])
            _, X, _, sms, _ = GAM.setup_gam(gf, df)
            D = GAM.bam_design(X, sms, df, true)
            # The intended fallback, stated positively: no tensor kernel may
            # claim this smooth. Either the whole design stays dense, or a
            # discrete design carries it outside tblocks (checked as empty).
            if D isa GAM.DiscreteDesign
                @test isempty(D.tblocks)
            else
                @test D isa GAM.DenseDesign
            end
        end
    end

    @testset "a te that cannot be discretised falls back" begin
        # Fully continuous covariates below the grid resolution: every value is
        # its own bin, so `m == n` and discretising buys nothing. The block must
        # stay dense rather than be represented at full width.
        rng = MersenneTwister(9)
        n = 800
        df = DataFrame(x1 = rand(rng, n), x2 = rand(rng, n))
        df.y = randn(rng, n)
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.te(:x1, :x2; k = [4, 4])])
        _, X, _, sms, _ = GAM.setup_gam(gf, df)
        D = GAM.bam_design(X, sms, df, true)   # m_grid = 1000 > n = 800
        @test D isa GAM.DenseDesign
        # And the fit is unaffected by asking for discretisation.
        m1 = bam(GAM.@formula(y ~ te(x1, x2, k = [4, 4])), df; discrete = false)
        m2 = bam(GAM.@formula(y ~ te(x1, x2, k = [4, 4])), df; discrete = true)
        @test coef(m1) == coef(m2)
    end

    @testset "pb pathology warns" begin
        # pb = p_raw / p_last drives a pb(pb+1)/2 sub-block walk, so cost grows
        # as pb^2. mgcv pays ~100 s for one XᵀWX on te(6,6,6,6) at n = 1e6.
        rng = MersenneTwister(6)
        n = 600
        nu = 30
        g = collect(range(0, 1; length = nu))
        df = DataFrame(x1 = g[rand(rng, 1:nu, n)], x2 = g[rand(rng, 1:nu, n)],
            x3 = g[rand(rng, 1:nu, n)], x4 = g[rand(rng, 1:nu, n)])
        df.y = randn(rng, n)
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.te(:x1, :x2, :x3, :x4; k = [4, 4, 4, 4])])
        _, X, _, sms, _ = GAM.setup_gam(gf, df)
        @test_logs (:warn, r"pb = ") match_mode = :any begin
            GAM.bam_design(X, sms, df, true)
        end
    end

    @testset "leverage sweep does not allocate per chunk" begin
        # The sweep used to build a fresh `nr x praw` row tensor for EVERY
        # chunk and apply the constraint with a scalar triple loop. At
        # n = 2e5 that was 349 MiB of garbage and 14.13 s, against the dense
        # path's 4.1 MiB and 0.17 s -- while the accumulation kernels this
        # milestone was benchmarked on were already faster than dense. The
        # scratch is now caller-owned and the constraint is one gemm.
        #
        # Allocation is the guard rather than time, because it reproduces
        # under load. The bound is deliberately far below the old behaviour
        # and far above the fixed cost, so it catches a reintroduced
        # per-chunk allocation without being brittle.
        df = _mk(20_000)
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.te(:x1, :x2; k = [6, 5], bs = [:cr, :cr])])
        _, X, _, sms, _ = GAM.setup_gam(gf, df)
        Dq = GAM.bam_design(X, sms, df, true)
        @test !isempty(Dq.tblocks)

        n, p = GAM.nrows(Dq), GAM.ncols(Dq)
        w = ones(n)
        XtWX = zeros(p, p)
        GAM.accumulate_XtWX!(XtWX, Dq, w)
        A = cholesky(Symmetric(XtWX + 1e-8I))

        GAM.design_finalize(Dq, w, XtWX, A; compute_hat_diag = true)   # warm
        alloc = @allocated GAM.design_finalize(Dq, w, XtWX, A;
            compute_hat_diag = true)
        # One 1024-row scratch plus the chunk buffer is ~2 MiB; the old
        # per-chunk path allocated ~35 MiB at this n.
        @test alloc < 12 * 2^20

        # And a caller-supplied scratch removes the last per-call allocation
        # from the row gather itself.
        Rbuf = GAM._tensor_row_scratch(Dq, 64)
        H = zeros(64, p)
        GAM._gather_rows!(H, Dq, 101, 164, Rbuf)                        # warm
        @test (@allocated GAM._gather_rows!(H, Dq, 101, 164, Rbuf)) == 0

        # The rows themselves are unchanged by the gemm rewrite.
        @test maximum(abs.(H .- X[101:164, :])) < 1e-12
    end
end
