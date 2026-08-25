@testset "Discrete design (bam discrete=true)" begin

    # ------------------------------------------------------------------
    # Binning: mgcv's compress.df, both branches
    # ------------------------------------------------------------------
    @testset "binning branches" begin
        # Exact branch: at most `m` distinct values are kept losslessly.
        x = repeat([1.0, 2.0, 3.0, 4.0, 5.0], 200)
        xu, k, exact = GAM._bin_covariate(x, 100)
        @test exact
        @test length(xu) == 5
        @test sort(xu) == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test all(i -> xu[k[i]] == x[i], eachindex(x))   # lossless, bit-level
        @test eltype(k) === Int32

        # Rounding branch: more distinct values than the grid allows.
        rng_b = StableRNG(4)
        xc = rand(rng_b, 5000)
        xu2, k2, exact2 = GAM._bin_covariate(xc, 50)
        @test !exact2
        @test length(xu2) <= 50
        @test maximum(i -> abs(xu2[k2[i]] - xc[i]), eachindex(xc)) <=
              (maximum(xc) - minimum(xc)) / (50 - 1)   # within half a grid step
        @test extrema(k2)[1] >= 1
        @test extrema(k2)[2] <= length(xu2)

        # mgcv permutes the unique values (bam.r:173-176). Here each block owns
        # its own index vector, so the permutation cannot couple two marginals
        # and is numerically inert: `xu[k]` is unchanged by it.
        xus, ks, _ = GAM._bin_covariate(xc, 50; shuffle = false)
        @test [xu2[k2[i]] for i in eachindex(xc)] == [xus[ks[i]] for i in eachindex(xc)]
        @test sort(xu2) == sort(xus)
    end

    @testset "grid resolution is validated" begin
        rng_v = StableRNG(5)
        df_v = DataFrame(x = rand(rng_v, 200), y = randn(rng_v, 200))
        gf_v = GAM.GamFormula(:y, Symbol[], true, [GAM.s(:x; k = 8, bs = :cr)])
        @test_throws ArgumentError bam(gf_v, df_v; discrete = 1)
    end

    # ------------------------------------------------------------------
    # Exactness: pre-binned covariates involve no rounding, so the discrete
    # representation must reproduce the dense fit to solver precision.
    # ------------------------------------------------------------------
    @testset "pre-binned covariates reproduce the dense fit" begin
        rng_e = StableRNG(11)
        n = 20_000
        b(v) = round.(v .* 199) ./ 199          # 200 distinct values
        x1 = b(rand(rng_e, n))
        x2 = b(rand(rng_e, n))
        y = sin.(2π .* x1) .+ 0.5 .* x2 .+ 0.3 .* randn(rng_e, n)
        df = DataFrame(x1 = x1, x2 = x2, y = y)
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 20, bs = :cr), GAM.s(:x2; k = 20, bs = :cr)])

        md = bam(gf, df)
        mk = bam(gf, df; discrete = true)

        relc = maximum(abs.(coef(md) .- coef(mk))) / maximum(abs.(coef(md)))
        @test relc < 1e-10
        @test maximum(abs.(fitted(md) .- fitted(mk))) < 1e-9
        @test abs(md.edf_total - mk.edf_total) < 1e-8
        @test abs(md.deviance_val - mk.deviance_val) < 1e-6
    end

    # ------------------------------------------------------------------
    # The kernels, checked directly against the dense ones on exact data.
    # ------------------------------------------------------------------
    @testset "kernels agree with the dense path" begin
        rng_k = StableRNG(12)
        n = 4000
        x1 = round.(rand(rng_k, n) .* 99) ./ 99
        x2 = round.(rand(rng_k, n) .* 99) ./ 99
        y = sin.(2π .* x1) .+ x2 .+ 0.2 .* randn(rng_k, n)
        df = DataFrame(x1 = x1, x2 = x2, y = y)
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 12, bs = :cr), GAM.s(:x2; k = 12, bs = :cr)])
        _, X, _, sm, _ = GAM.setup_gam(gf, df)

        Dd = GAM.bam_design(X)
        Dk = GAM.bam_design(X, sm, df, true)
        @test Dk isa GAM.DiscreteDesign
        @test GAM.ncols(Dk) == GAM.ncols(Dd)
        @test GAM.nrows(Dk) == GAM.nrows(Dd)
        @test GAM.intercept_col(Dk) == GAM.intercept_col(Dd)
        @test all(blk -> blk.exact, Dk.blocks)

        p = GAM.ncols(Dd)
        w = 0.5 .+ rand(rng_k, n)
        z = randn(rng_k, n)
        beta = randn(rng_k, p)

        e1 = zeros(n); e2 = zeros(n)
        GAM.mul_eta!(e1, Dd, beta); GAM.mul_eta!(e2, Dk, beta)
        @test maximum(abs.(e1 .- e2)) < 1e-10 * maximum(abs.(e1))

        A1 = zeros(p, p); A2 = zeros(p, p)
        GAM.accumulate_XtWX!(A1, Dd, w); GAM.accumulate_XtWX!(A2, Dk, w)
        @test maximum(abs.(A1 .- A2)) < 1e-9 * maximum(abs.(A1))
        @test issymmetric(round.(A2; digits = 10))

        B1 = zeros(p, p); B2 = zeros(p, p)
        b1 = zeros(p); b2 = zeros(p)
        GAM.accumulate_XtWX_XtWz!(B1, b1, Dd, w, z)
        GAM.accumulate_XtWX_XtWz!(B2, b2, Dk, w, z)
        @test maximum(abs.(B1 .- B2)) < 1e-9 * maximum(abs.(B1))
        @test maximum(abs.(b1 .- b2)) < 1e-9 * maximum(abs.(b1))

        # `hat_diag` is O(n*p^2) either way, but must still be right — the
        # discrete path gathers rows a chunk at a time rather than holding X.
        S = GAM.total_penalty(GAM.setup_penalties(sm, 1),
            GAM.setup_penalties(sm, 1).sp, p)
        Ach = cholesky(Symmetric(B1 .+ S))
        f1 = GAM.design_finalize(Dd, w, B1, Ach)
        f2 = GAM.design_finalize(Dk, w, B2, Ach)
        @test maximum(abs.(f1[1] .- f2[1])) < 1e-9
        @test length(f2[2]) == n
        @test maximum(abs.(f1[2] .- f2[2])) < 1e-8
        # ...and may be declined, as on the Gaussian fast path.
        @test isempty(GAM.design_finalize(Dk, w, B2, Ach; compute_hat_diag = false)[2])
    end

    # ------------------------------------------------------------------
    # Scope: only 1-D smooths without `by=` are discretised; everything else
    # stays dense, so an unsupported term costs correctness nothing.
    # ------------------------------------------------------------------
    @testset "unsupported terms stay dense" begin
        rng_s = StableRNG(13)
        n = 2000
        x1 = round.(rand(rng_s, n) .* 49) ./ 49
        x2 = rand(rng_s, n)
        g = string.(repeat(1:10, inner = div(n, 10)))
        by = rand(rng_s, n)
        y = sin.(2π .* x1) .+ 0.2 .* randn(rng_s, n)
        df = DataFrame(x1 = x1, x2 = x2, g = g, by = by, y = y)

        # discrete=false always yields the dense design.
        gf1 = GAM.GamFormula(:y, Symbol[], true, [GAM.s(:x1; k = 10, bs = :cr)])
        _, X1, _, sm1, _ = GAM.setup_gam(gf1, df)
        @test GAM.bam_design(X1, sm1, df, false) isa GAM.DenseDesign

        # A tensor is not discretised in M1 — and with nothing else to
        # discretise, the design falls back to dense entirely.
        gft = GAM.GamFormula(:y, Symbol[], true,
            [GAM.te(:x1, :x2; k = 4, bs = [:cr, :cr])])
        _, Xt, _, smt, _ = GAM.setup_gam(gft, df)
        @test GAM.bam_design(Xt, smt, df, true) isa GAM.DenseDesign

        # A `by=` smooth is skipped, but a plain 1-D sibling is still taken.
        gfb = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 8, bs = :cr, by = :by)])
        _, Xb, _, smb, _ = GAM.setup_gam(gfb, df)
        Db = GAM.bam_design(Xb, smb, df, true)
        @test Db isa GAM.DiscreteDesign
        @test length(Db.blocks) == 1
        @test Db.blocks[1].label == smb[1].spec.label

        # A mixed model still fits, and matches its dense counterpart because
        # every discretised covariate here is exactly representable.
        mdb = bam(gfb, df)
        mkb = bam(gfb, df; discrete = true)
        @test maximum(abs.(fitted(mdb) .- fitted(mkb))) < 1e-8
    end

    # ------------------------------------------------------------------
    # Approximation: continuous covariates are rounded onto the grid, which
    # is where discrete=true stops being exact. Fitted values stay close;
    # individual smoothing parameters need not, so they are NOT pinned.
    # ------------------------------------------------------------------
    @testset "continuous covariates: approximate but close" begin
        rng_c = StableRNG(14)
        n = 30_000
        x1 = rand(rng_c, n)
        x2 = rand(rng_c, n)
        y = sin.(2π .* x1) .+ x2 .^ 2 .+ 0.3 .* randn(rng_c, n)
        df = DataFrame(x1 = x1, x2 = x2, y = y)
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 20, bs = :cr), GAM.s(:x2; k = 20, bs = :cr)])

        md = bam(gf, df)
        mk = bam(gf, df; discrete = true)
        rngf = maximum(fitted(md)) - minimum(fitted(md))

        @test !all(blk -> blk.exact, GAM.bam_design(
            GAM.setup_gam(gf, df)[2], GAM.setup_gam(gf, df)[4], df, true).blocks)
        @test maximum(abs.(fitted(md) .- fitted(mk))) / rngf < 3e-3
        @test abs(md.edf_total - mk.edf_total) / md.edf_total < 1e-3

        # A coarser grid is a worse approximation — the knob does something.
        mc = bam(gf, df; discrete = 40)
        @test maximum(abs.(fitted(md) .- fitted(mc))) >
              maximum(abs.(fitted(md) .- fitted(mk)))
    end

    # ------------------------------------------------------------------
    # The exported utility now shares the fitter's binning rule.
    # ------------------------------------------------------------------
    @testset "discretize_covariates uses the fitter's rule" begin
        rng_d = StableRNG(15)
        x = rand(rng_d, 5000)
        d = discretize_covariates((x = x,), [:x]; max_unique = 100)
        @test d isa DiscretizedData
        @test d.n == 5000
        @test length(d.unique_values[:x]) <= 100
        @test issorted(d.unique_values[:x])
        xu, k, _ = GAM._bin_covariate(x, 100; shuffle = false)
        @test d.unique_values[:x] == xu
        @test d.indices[:x] == Int.(k)
    end

    # ------------------------------------------------------------------
    # Reduced construction: build the smooth at the m unique covariate
    # values instead of all n rows. This is a MEMORY path — it exists to
    # avoid the construction transient, which for thin-plate at
    # max_knots = 2000 is an n x 2000 dense matrix and dominates peak RSS
    # (measured 4113 MB peak against 167 MB retained at n = 1e5).
    # ------------------------------------------------------------------
    @testset "reduced construction reproduces the dense basis" begin
        # Replicate an exact grid so binning is lossless: any discrepancy is
        # then the construction, not the approximation.
        m, rep = 200, 25
        grid = collect(range(0.0, 1.0; length = m))
        x = repeat(grid, inner = rep)
        rng_r = StableRNG(5)
        df = DataFrame(x = x, y = sin.(2π .* x) .+ 0.3 .* randn(rng_r, m * rep))

        for bs in (:tp, :cr, :ps, :bs, :cc)
            gf = GAM.GamFormula(:y, Symbol[], true,
                GAM.SmoothSpec[GAM.s(:x; k = 12, bs = bs)])
            _, Xd, _, smd, _ = GAM.setup_gam(gf, df)
            _, Xr, _, smr, _ = GAM.setup_gam_discrete(gf, df, 1000)

            # Knots must come from the FULL covariate. Compared numerically
            # rather than with `==`: the mean-centring shift is subtracted
            # and re-added, so a minority of entries differ by one ulp
            # (measured max|Δ| 5.6e-17, 110/120 bitwise equal).
            @test length(smd[1].knots) == length(smr[1].knots)
            @test maximum(abs, smd[1].knots .- smr[1].knots) < 1e-14

            @test size(Xr) == size(Xd)
            @test maximum(abs, Xd .- Xr) / maximum(abs, Xd) < 1e-11
            @test maximum(abs, smd[1].S[1] .- smr[1].S[1]) /
                  maximum(abs, smd[1].S[1]) < 1e-11
            # sm.X is expanded back to n rows: ConstructedSmooth.X and
            # GamModel.X are concretely-typed dense n-row matrices.
            @test size(smr[1].X, 1) == m * rep
        end
    end

    @testset "reduced construction must not shuffle the grid" begin
        # `_bin_covariate` permutes the unique values by default, as mgcv
        # does. That is inert when the basis is rebuilt as `Xd[k, :]`, but
        # TPRS takes the unique values AS its knots, and knot ORDER changes
        # the parameterization: with the shuffle on, the knot SETS still
        # match the dense path exactly while the basis differs by a relative
        # 1.97. This pins the un-shuffled call in `_reduced_smooth`.
        x = repeat(collect(range(0.0, 1.0; length = 120)), inner = 10)
        t = (x = x,)
        spec = GAM.s(:x; k = 10, bs = :tp)
        r = GAM._reduced_smooth(spec, t, 1000, length(x))
        @test r !== nothing
        sm_red, _, counts = r
        @test issorted(sm_red.knots)
        @test sum(counts) == length(x)

        sm_full = GAM.smooth_construct(spec, t)
        @test length(sm_full.knots) == length(sm_red.knots)
        @test maximum(abs, sm_full.knots .- sm_red.knots) < 1e-14
    end

    @testset "side constraints fall back to dense construction" begin
        # Smooths sharing a covariate trigger `side_constrain!`, which needs
        # the n-row blocks, so reduced construction must not be attempted.
        shared = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 8), GAM.s(:x, :z; k = 16)])
        @test GAM._smooths_share_variables(shared.smooth_specs)

        ns = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 8), GAM.s(:z; k = 8)])
        @test !GAM._smooths_share_variables(ns.smooth_specs)

        rng_s = StableRNG(21)
        n = 3000
        df = DataFrame(x = rand(rng_s, n), z = rand(rng_s, n))
        df.y = sin.(2π .* df.x) .+ df.z .^ 2 .+ 0.3 .* randn(rng_s, n)
        # Falls back for construction, still fits, and matches the dense
        # construction path exactly (the design layer may still discretise).
        _, Xf, _, _, _ = GAM.setup_gam_discrete(shared, df, 1000)
        _, Xd, _, _, _ = GAM.setup_gam(shared, df)
        @test Xf == Xd
    end
end
