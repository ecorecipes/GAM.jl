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

        # A `by=` smooth is skipped; a plain sibling is still taken. Reaching
        # O(m*k) for factor-`by` needs a block-offset kernel, not the scale
        # field — see the note in bam_design.jl.
        Db, = _design([GAM.s(:x1; k = 10, bs = :cr),
                       GAM.s(:x2; k = 8, bs = :cr, by = :sl)])
        @test Db isa GAM.DiscreteDesign
        @test length(Db.blocks) == 1
        @test Db.blocks[1].label == "s(x1,bs=cr)"

        Dbf, = _design([GAM.s(:x1; k = 8, bs = :cr, by = :f)])
        @test Dbf isa GAM.DenseDesign

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
end
