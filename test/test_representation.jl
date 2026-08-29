# Representation-in-use guards for `bam(...; discrete = true)`.
#
# These features' value IS a representation: a compact design that computes the
# same answer more cheaply. A correctness test cannot tell "using the compact
# representation" apart from "silently fell back to dense and got the same
# answer" — and that gap has bitten three times:
#
#   1. Discretised tensors ran 52.09 s against 5.48 s dense, and used more
#      memory. It shipped green, and surfaced only when an unrelated memory
#      investigation measured it. The cause was `design_finalize` allocating
#      348.9 MiB against dense's 4.1 MiB — an 85x blow-up invisible end to end,
#      where the ratio was only ~1.07x.
#   2. Slimming the `ti`/`t2` prediction caches also silently demoted discrete
#      `te` fits to dense. Every prediction test passed; a structural assertion
#      in test_discrete_tensor.jl failed and saved it.
#   3. A `_tensor_block` guard added to stop a `BoundsError` turned a crash into
#      a silent dense fallback — correct, and undetectable by results alone.
#
# So: assert on STRUCTURE (which blocks exist), on RECONSTRUCTION (the compact
# block really encodes the dense columns), and on PER-OPERATION allocations
# (where a representation regression actually shows up). Nothing here asserts
# wall time, which is not reproducible under load.

@testset "Representation in use (discrete=true)" begin

    # Deterministic, exactly-binnable data: every covariate takes few enough
    # distinct values that discretisation involves no rounding, so any
    # difference from the dense fit is a bug rather than an approximation.
    _rep_rng = StableRNG(9001)
    _rep_n = 2400
    _bin(v, m) = round.(v .* (m - 1)) ./ (m - 1)
    _rep_df = DataFrame(
        x1 = _bin(rand(_rep_rng, _rep_n), 40),
        x2 = _bin(rand(_rep_rng, _rep_n), 40),
        sl = _bin(rand(_rep_rng, _rep_n), 20),
        g  = string.(repeat(1:20, inner = div(_rep_n, 20))),
        f  = string.(repeat(1:4, inner = div(_rep_n, 4))),
    )
    _rep_df.y = sin.(2π .* _rep_df.x1) .+ 0.3 .* randn(_rep_rng, _rep_n)

    # Build the design the way `bam` does, so these test the real path.
    function _design(specs; discrete = true, df = _rep_df)
        gf = GAM.GamFormula(:y, Symbol[], true, collect(specs))
        _, X, _, sm, _ = GAM.setup_gam(gf, df)
        return GAM.bam_design(X, sm, df, discrete), X, gf
    end

    # Reconstruct a DiscreteBlock's dense rows from (Xd, k, scale).
    # Row `i` is `scale[i] * Xd[k[i], :]`, or `Xd[k[i], :]` when unscaled.
    function _reconstruct(blk, n)
        out = zeros(n, size(blk.Xd, 2))
        @inbounds for i in 1:n
            row = @view blk.Xd[blk.k[i], :]
            s = blk.scale === nothing ? 1.0 : blk.scale[i]
            for j in axes(out, 2)
                out[i, j] = s * row[j]
            end
        end
        return out
    end

    # ------------------------------------------------------------------
    # 1. The representation is actually engaged.
    #
    # `isa DiscreteDesign` alone is NOT sufficient: a DiscreteDesign whose
    # smooths all fell back to the dense remainder is exactly failure mode 3.
    # So assert the block counts too.
    # ------------------------------------------------------------------
    @testset "supported paths produce compact blocks" begin
        # A plain 1-D smooth becomes one discrete block.
        D, = _design([GAM.s(:x1; k = 10, bs = :cr)])
        @test D isa GAM.DiscreteDesign
        @test length(D.blocks) == 1
        @test isempty(D.tblocks)
        @test D.blocks[1].exact                # exactly binnable, no rounding
        @test D.blocks[1].scale === nothing    # unscaled path

        # Two 1-D smooths become two blocks, not one block plus a dense
        # remainder — the shape a partial fallback would produce.
        D2, = _design([GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 10, bs = :cr)])
        @test length(D2.blocks) == 2
        @test Set(b.label for b in D2.blocks) ==
              Set(["s(x1,bs=cr)", "s(x2,bs=cr)"])

        # A `te` becomes one TensorBlock and NO 1-D blocks. This is the
        # assertion that caught failure mode 2.
        Dt, = _design([GAM.te(:x1, :x2; k = 5, bs = [:cr, :cr])])
        @test Dt isa GAM.DiscreteDesign
        @test length(Dt.tblocks) == 1
        @test isempty(Dt.blocks)
        @test length(Dt.tblocks[1].Xds) == 2   # per-marginal bases, not a row tensor
        @test length(Dt.tblocks[1].ks) == 2

        # A factor random effect becomes a compact block whose cell basis is
        # the identity — the whole point of the representation.
        Dr, = _design([GAM.s(:g; bs = :re)])
        @test Dr isa GAM.DiscreteDesign
        @test length(Dr.blocks) == 1
        blk = Dr.blocks[1]
        @test blk.Xd == Matrix{Float64}(I, size(blk.Xd)...)
        @test blk.m == length(unique(_rep_df.g))

        # A random slope (numeric + factor) uses the scaled form: the cell
        # basis is fixed, the per-row multiplier is the slope covariate.
        Ds, = _design([GAM.s(:sl, :g; bs = :re)])
        @test Ds isa GAM.DiscreteDesign
        @test length(Ds.blocks) == 1
        @test Ds.blocks[1].scale !== nothing

        # Mixed: a 1-D smooth beside a random effect gives two blocks.
        Dm, = _design([GAM.s(:x1; k = 10, bs = :cr), GAM.s(:g; bs = :re)])
        @test length(Dm.blocks) == 2

        # A FACTOR `by=` becomes one ByBlock holding a single shared basis
        # plus a grid index and a level index — not L replicated blocks, and
        # not a dense fallback. `Xd` is `m x kb`, so it must be narrower than
        # the block's `kb*L` columns; that is what distinguishes the compact
        # form from a replicated one.
        Dby, Xby = _design([GAM.s(:x1; k = 8, bs = :cr, by = :f)])
        @test Dby isa GAM.DiscreteDesign
        @test length(Dby.byblocks) == 1
        @test isempty(Dby.blocks)
        bby = Dby.byblocks[1]
        @test bby.L == length(unique(_rep_df.f))
        @test size(bby.Xd, 2) == bby.kb
        @test length(bby.cols) == bby.kb * bby.L
        @test size(bby.Xd, 1) == bby.m
        @test bby.m < size(Xby, 1)          # actually reduced

        # Reconstruction: row i is zero except in its own level's kb columns.
        # Structure alone would pass for a block encoding the wrong matrix.
        let n = size(Xby, 1), recon = zeros(n, length(bby.cols))
            for i in 1:n
                l = bby.lev[i]
                l == 0 && continue
                off = (l - 1) * bby.kb
                for c in 1:bby.kb
                    recon[i, off + c] = bby.Xd[bby.k[i], c]
                end
            end
            @test maximum(abs.(recon .- Xby[:, bby.cols])) < 1e-12
        end

        # Mixed with a plain smooth: both engage.
        Dbp, = _design([GAM.s(:x1; k = 8, bs = :cr, by = :f),
                        GAM.s(:x2; k = 8, bs = :cr)])
        @test length(Dbp.byblocks) == 1
        @test length(Dbp.blocks) == 1

        # Mixed with a tensor: both engage. The tensor uses variables the
        # by-smooth does not, because sharing a variable makes `setup_gam`
        # apply side constraints and fall back wholesale — a real property of
        # the dense path, not a limitation of this representation.
        Dbt, = _design([GAM.s(:x1; k = 8, bs = :cr, by = :f),
                        GAM.te(:x2, :sl; k = 4, bs = [:cr, :cr])])
        @test length(Dbt.byblocks) == 1
        @test length(Dbt.tblocks) == 1
    end

    # ------------------------------------------------------------------
    # 2. Unsupported paths fall back DELIBERATELY.
    #
    # These are dense today. Pinning that means implementing one of them makes
    # this test fail, prompting an update, rather than passing silently and
    # leaving nobody aware the representation is live.
    # ------------------------------------------------------------------
    @testset "unsupported paths fall back deliberately" begin
        # discrete=false is always dense, whatever the terms.
        Dd, = _design([GAM.s(:x1; k = 10, bs = :cr)]; discrete = false)
        @test Dd isa GAM.DenseDesign

        # `t2` is out of scope: its per-marginal reparameterisations mean the
        # post-hoc constraint identity `te` relies on does not apply.
        Dt2, = _design([GAM.t2(:x1, :x2; k = 5, bs = [:cr, :cr])])
        @test Dt2 isa GAM.DenseDesign

        # A NUMERIC `by=` is still skipped — it is a per-row multiplier, not a
        # level replication, so the block-offset kernel does not cover it. The
        # plain sibling is still taken.
        Db, = _design([GAM.s(:x1; k = 10, bs = :cr),
                       GAM.s(:x2; k = 8, bs = :cr, by = :sl)])
        @test Db isa GAM.DiscreteDesign
        @test length(Db.blocks) == 1
        @test isempty(Db.byblocks)
        @test Db.blocks[1].label == "s(x1,bs=cr)"

        # `ti` itself stays dense; its 1-D marginals are still discretised, so
        # this is a partial fallback and the block count distinguishes it.
        Dti, = _design([GAM.s(:x1; k = 6, bs = :cr), GAM.s(:x2; k = 6, bs = :cr),
                        GAM.ti(:x1, :x2; k = 4, bs = [:cr, :cr])])
        @test Dti isa GAM.DiscreteDesign
        @test length(Dti.blocks) == 2      # the two marginals
        @test isempty(Dti.tblocks)         # the interaction is NOT discretised
        @test length(Dti.dense_cols) > 1   # intercept + the ti columns
    end

    # ------------------------------------------------------------------
    # 3. The compact block really encodes the dense columns.
    #
    # Structure alone does not prove correctness: a block can exist and encode
    # the WRONG matrix. Reconstructing `scale[i] * Xd[k[i], :]` and comparing
    # against the dense design's own columns is the direct check, and it is
    # what distinguishes "compact" from "compact but lossy".
    # ------------------------------------------------------------------
    @testset "compact blocks reconstruct the dense columns" begin
        for (label, specs) in (
            ("1-D cr",       [GAM.s(:x1; k = 10, bs = :cr)]),
            ("two 1-D",      [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 10, bs = :cr)]),
            ("re factor",    [GAM.s(:g; bs = :re)]),
            ("re slope",     [GAM.s(:sl, :g; bs = :re)]),
            ("1-D + re",     [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:g; bs = :re)]),
        )
            D, X = _design(specs)
            @test D isa GAM.DiscreteDesign
            for blk in D.blocks
                recon = _reconstruct(blk, size(X, 1))
                @test size(recon, 2) == length(blk.cols)
                @test maximum(abs.(recon .- X[:, blk.cols])) < 1e-12
            end
        end
    end

    # ------------------------------------------------------------------
    # 4. `discrete = true` is never a pure loss.
    #
    # Asserted on ALLOCATIONS, not wall time: timings on a loaded machine are
    # not reproducible, and a previous agent correctly refused to claim a
    # speedup when the untouched control moved by the same factor.
    #
    # Per-operation, not just end to end. During failure mode 1 the end-to-end
    # memory ratio was only ~1.07x while `design_finalize` was 85x — so an
    # end-to-end guard alone would have passed. Both are asserted here.
    # ------------------------------------------------------------------
    @testset "discrete is not a pure loss" begin
        # -- per-operation, tensor path (where the regression happened) --
        Dd, X = _design([GAM.te(:x1, :x2; k = 5, bs = [:cr, :cr])]; discrete = false)
        Dk, _ = _design([GAM.te(:x1, :x2; k = 5, bs = [:cr, :cr])])
        n, p = GAM.nrows(Dd), GAM.ncols(Dd)
        w = ones(n); z = collect(_rep_df.y); beta = randn(StableRNG(7), p)
        A = zeros(p, p); bv = zeros(p); eta = zeros(n)

        function _finalize_alloc(D)
            B = zeros(p, p)
            GAM.accumulate_XtWX!(B, D, w)
            ch = cholesky(Symmetric(B + 1e-6I))
            GAM.design_finalize(D, w, B, ch)              # warm: first call compiles
            GC.gc()
            return @allocated GAM.design_finalize(D, w, B, ch)
        end
        fd = _finalize_alloc(Dd)
        fk = _finalize_alloc(Dk)
        # Measured at n=2400: discrete ~1.8x dense. During the regression it
        # was 85x. A generous ceiling still catches that by a wide margin.
        @test fk < 10 * fd

        # `mul_eta!` must not materialise the row tensor.
        GAM.mul_eta!(eta, Dk, beta); GC.gc()
        @test (@allocated GAM.mul_eta!(eta, Dk, beta)) < 2^20   # < 1 MiB

        # -- end to end --
        gfe = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 12, bs = :cr), GAM.s(:x2; k = 12, bs = :cr)])
        bam(gfe, _rep_df); bam(gfe, _rep_df; discrete = true)      # warm both
        GC.gc(); ad = @allocated bam(gfe, _rep_df)
        GC.gc(); ak = @allocated bam(gfe, _rep_df; discrete = true)
        # Measured ratio 0.84-0.93 across families. Gaussian is the neutral
        # case (accumulation runs once, so binning is pure overhead) and was
        # 0.95x end to end; the ceiling only has to exclude a real regression.
        @test ak < 1.5 * ad
    end

    # ------------------------------------------------------------------
    # 5. Dropping `X` on the bam path keeps `model_matrix` bitwise.
    #
    # This is where the measured 587.5 -> 7.6 MiB saving lives, and it works
    # only because `m.X` duplicates the retained per-smooth blocks exactly.
    # ------------------------------------------------------------------
    @testset "retain_X=false keeps model_matrix bitwise (bam)" begin
        gfx = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 10, bs = :cr)])
        mr = bam(gfx, _rep_df)
        md = bam(gfx, _rep_df; retain_X = false)

        @test size(mr.X, 1) == _rep_n
        @test isempty(md.X)                     # dropped
        @test size(md.X_par, 1) == _rep_n       # parametric block kept
        @test GAM.model_matrix(md) == mr.X      # bitwise, not approximate
        @test coef(mr) == coef(md)
        @test fitted(mr) == fitted(md)
        @test sizeof(md.X_par) < sizeof(mr.X)   # the saving is real
    end

    # ------------------------------------------------------------------
    # 5b. `discrete=true` never ASSEMBLES the model matrix.
    #
    # Dropping X after building it saves retention but not peak: the n x p
    # assembly sat in peak RSS either way, which is what masked design-side
    # wins of 53.66x (factor-by), 421x (tensor) and 127x (random effect).
    # Under `discrete` the matrix is now never formed, so `retain_X` defaults
    # to false there. Asserting on the DESIGN rather than on fitted values,
    # because a fit is identical either way and would not catch a regression
    # back to assembling it.
    # ------------------------------------------------------------------
    @testset "discrete=true does not assemble X" begin
        gfx = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 10, bs = :cr)])
        mq = bam(gfx, _rep_df; discrete = true)
        mk = bam(gfx, _rep_df; discrete = true, retain_X = true)
        mr = bam(gfx, _rep_df)                   # dense default: retained

        @test isempty(mq.X)                      # never assembled
        @test size(mq.X_par, 1) == _rep_n        # parametric block available
        @test size(mk.X, 1) == _rep_n            # explicit opt-in still builds
        @test size(mr.X, 1) == _rep_n            # dense default unchanged

        # The X-free path must produce the same fit as the X-holding one,
        # exactly — the design is assembled from the same blocks either way.
        @test coef(mq) == coef(mk)
        @test fitted(mq) == fitted(mk)
        @test GAM.model_matrix(mq) == mk.X       # bitwise reassembly

        # And it must still actually discretise, not fall back to dense.
        D, = _design([GAM.s(:x1; k = 10, bs = :cr), GAM.s(:x2; k = 10, bs = :cr)])
        @test D isa GAM.DiscreteDesign
        @test length(D.blocks) == 2
    end

    # ------------------------------------------------------------------
    # 6. The compact representations agree with dense end to end.
    #
    # Reconstruction fidelity (3) proves the encoding; this proves the fit
    # built on it lands in the same place.
    # ------------------------------------------------------------------
    @testset "compact representations agree with dense end to end" begin
        for (label, specs, tol) in (
            ("re factor", [GAM.s(:g; bs = :re)], 1e-10),
            ("re slope",  [GAM.s(:sl, :g; bs = :re)], 1e-10),
            ("1-D + re",  [GAM.s(:x1; k = 10, bs = :cr), GAM.s(:g; bs = :re)], 1e-8),
            ("te",        [GAM.te(:x1, :x2; k = 5, bs = [:cr, :cr])], 1e-8),
        )
            gf = GAM.GamFormula(:y, Symbol[], true, collect(specs))
            md = bam(gf, _rep_df)
            mk = bam(gf, _rep_df; discrete = true)
            scale = max(maximum(abs, coef(md)), 1.0)
            @test maximum(abs.(coef(md) .- coef(mk))) / scale < tol
            @test maximum(abs.(fitted(md) .- fitted(mk))) < 1e-8
        end
    end

    # ------------------------------------------------------------------
    # 7. REGRESSION GUARD — `s(g, bs=:re, by=<numeric>)` must fall back to dense.
    #
    # This was a live wrong-answer bug for the two hours between the random
    # effect being wired into `bam_design` and this test finding it. The RE
    # branch precedes the `by` guard in the block-selection loop, so it built
    # an UNSCALED compact block: the dense columns carry the by-variable, the
    # block reconstructed to pure {0,1} indicators. It did not fall back — it
    # returned a wrong fit. Measured on a genuine random slope:
    #     max|reconstructed - X[:, cols]| = 1.0   (a full unit, not rounding)
    #     relative coefficient error vs dense     = 0.49
    #     max|Δfitted| = 2.34 on a range of 7.56  (31% of range)
    #
    # A probe with a noise-only response showed just 3.0e-3, because every
    # coefficient sat near zero — that is how this evaded casual checking, and
    # why the assertion below is on the DESIGN rather than on a fitted value.
    # ------------------------------------------------------------------
    @testset "re with numeric by= falls back to dense" begin
        D, _ = _design([GAM.s(:g; bs = :re, by = :sl)])
        # `by=` multiplies every row and the compact block cannot encode it,
        # so the whole design must stay dense rather than drop the multiplier.
        @test D isa GAM.DenseDesign
    end

    # ------------------------------------------------------------------
    # 8. A factor `by` smooth is stored REDUCED, not just compactly designed.
    #
    # The `ByBlock` design was compact well before the smooth was: a by
    # smooth's `ConstructedSmooth.X` stayed `n x (kb*L)` (427.25 MiB at
    # n=5e5, L=8, k=15) because `_reduced_smooth` rejected `by`. Only the
    # ASSEMBLY was avoided, which is why `by` gained 1.28x where a plain `cr`
    # smooth gained 2.32x.
    #
    # Its distinct rows are the distinct observed (cell, level) pairs, so it
    # uses the same `rowmap` contract as every other reduced smooth. Assert
    # BOTH halves: the smooth is reduced AND the ByBlock still engages — a
    # silent fallback in either direction would still fit correctly.
    # ------------------------------------------------------------------
    @testset "factor by= smooth is stored reduced" begin
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x1; k = 8, bs = :cr, by = :f)])
        mq = bam(gf, _rep_df; discrete = true)
        sm = only(mq.smooths)
        @test GAM.is_reduced(sm)
        # One row per observed (cell, level) pair, far fewer than n.
        @test size(sm.X, 1) < _rep_n
        @test length(sm.rowmap) == _rep_n
        # `smooth_matrix` scatters back to the dense block.
        @test size(GAM.smooth_matrix(sm), 1) == _rep_n

        # The design still uses the compact by representation.
        D, = _design([GAM.s(:x1; k = 8, bs = :cr, by = :f)])
        @test length(D.byblocks) == 1

        # And the fit matches the dense one — this data is exactly binnable,
        # so any difference beyond floating-point noise is a bug.
        md = bam(gf, _rep_df)
        @test maximum(abs.(coef(mq) .- coef(md))) /
              max(maximum(abs, coef(md)), 1.0) < 1e-10
        @test maximum(abs.(fitted(mq) .- fitted(md))) < 1e-8
    end
end
