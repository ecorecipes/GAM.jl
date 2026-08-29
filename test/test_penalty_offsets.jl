# Per-sub-penalty offsets in PenaltyBlock.
#
# Two things are pinned here.
#
# 1. A sub-penalty narrower than its block used to be accepted and silently
#    miscomputed: `total_penalty` accumulated over the FULL block width with an
#    `@inbounds` loop, so a k x k matrix in a kL-wide block read past the end of
#    `Si` and returned a finite, wrong penalty with no exception. The
#    `PenaltyBlock` constructor now rejects that unless an offset says where the
#    sub-penalty sits.
#
# 2. With offsets, a structured penalty need not be materialised. A factor-`by`
#    smooth's penalty is `I_L (x) S_k`, storable as L copies of the narrow `S_k`
#    at offsets 0, k, 2k, ... — O(L*k^2) instead of O(L^3*k^2).

@testset "PenaltyBlock offsets" begin

    @testset "narrow sub-penalty is rejected without an offset" begin
        # This is the case that previously produced silent garbage.
        k, L = 3, 2
        S_k = Matrix{Float64}(I, k, k)
        narrow = [copy(S_k) for _ in 1:L]
        @test_throws DimensionMismatch GAM.PenaltyBlock(narrow, k * L, 1, k * L)

        # The message must say what to do about it, not just that it failed.
        err = try
            GAM.PenaltyBlock(narrow, k * L, 1, k * L)
        catch e
            e
        end
        @test occursin("offset", sprint(showerror, err))
    end

    @testset "constructor validates shape, offset sign and fit" begin
        S3 = Matrix{Float64}(I, 3, 3)
        # Non-square.
        @test_throws DimensionMismatch GAM.PenaltyBlock([randn(3, 4)], 3, 1, 4, [0])
        # Negative offset.
        @test_throws ArgumentError GAM.PenaltyBlock([copy(S3)], 3, 1, 6, [-1])
        # Runs off the end of the block.
        @test_throws DimensionMismatch GAM.PenaltyBlock([copy(S3)], 3, 1, 6, [4])
        # Offset/penalty count mismatch.
        @test_throws DimensionMismatch GAM.PenaltyBlock([copy(S3), copy(S3)], 3, 1, 6, [0])
        # Exactly filling the block is fine.
        @test GAM.PenaltyBlock([copy(S3)], 3, 1, 6, [3]) isa GAM.PenaltyBlock
    end

    @testset "the 4-argument form still works and offsets default to zero" begin
        S = [Matrix{Float64}(I, 5, 5), Matrix{Float64}(2.0I, 5, 5)]
        b = GAM.PenaltyBlock(S, 5, 2, 6)
        @test b.offsets == [0, 0]
        @test b.start == 2 && b.stop == 6
    end

    @testset "offsets reproduce I_L (x) S_k elementwise" begin
        # The payoff: L narrow copies must give exactly the materialised form.
        for (k, L) in ((3, 2), (4, 3), (15, 8), (2, 5))
            S_k = randn(k, k)
            S_k = S_k'S_k + I                       # symmetric positive definite
            width = k * L

            narrow = [copy(S_k) for _ in 1:L]
            offs = [(l - 1) * k for l in 1:L]
            b_narrow = GAM.PenaltyBlock(narrow, width, 1, width, offs)

            wide = [zeros(width, width) for _ in 1:L]
            for l in 1:L
                cols = ((l - 1) * k + 1):(l * k)
                wide[l][cols, cols] .= S_k
            end
            b_wide = GAM.PenaltyBlock(wide, width, 1, width)

            for lsp in (zeros(L), randn(L), fill(-3.0, L))
                sn = GAM.total_penalty(GAM.PenaltySetup([b_narrow], copy(lsp),
                        falses(L)), lsp, width)
                sw = GAM.total_penalty(GAM.PenaltySetup([b_wide], copy(lsp),
                        falses(L)), lsp, width)
                @test sn == sw                       # elementwise, not approx
            end

            # And at equal smoothing parameters it is exactly the Kronecker form.
            st = GAM.total_penalty(
                GAM.PenaltySetup([b_narrow], zeros(L), falses(L)), zeros(L), width)
            @test st ≈ kron(Matrix{Float64}(I, L, L), S_k)
        end
    end

    @testset "in-place and allocating forms agree" begin
        k, L = 4, 3
        S_k = Matrix{Float64}(I, k, k) .+ 0.5
        width = k * L
        b = GAM.PenaltyBlock([copy(S_k) for _ in 1:L], width, 1, width,
            [(l - 1) * k for l in 1:L])
        setup = GAM.PenaltySetup([b], zeros(L), falses(L))
        lsp = [0.3, -1.2, 2.0]

        S1 = GAM.total_penalty(setup, lsp, width)
        S2 = zeros(width, width)
        GAM.total_penalty!(S2, setup, lsp, width)
        @test S1 == S2
    end

    @testset "offsets place a sub-penalty inside a larger coefficient vector" begin
        # block.start offsets into the full coefficient vector; block.offsets
        # offsets within the block. Both must compose.
        k, L, p = 3, 2, 12
        S_k = Matrix{Float64}(I, k, k)
        start = 5
        b = GAM.PenaltyBlock([copy(S_k) for _ in 1:L], k * L, start,
            start + k * L - 1, [0, k])
        St = GAM.total_penalty(GAM.PenaltySetup([b], zeros(L), falses(L)),
            zeros(L), p)
        expect = zeros(p, p)
        expect[start:(start + k * L - 1), start:(start + k * L - 1)] .=
            kron(Matrix{Float64}(I, L, L), S_k)
        @test St == expect
    end

    @testset "memory: narrow storage is O(L k^2), not O(L^3 k^2)" begin
        # The case the plan quantified: L=50, k=15.
        k, L = 15, 50
        S_k = Matrix{Float64}(I, k, k)
        width = k * L

        narrow_bytes = L * sizeof(Float64) * k * k
        wide_bytes = L * sizeof(Float64) * width * width
        @test narrow_bytes * 100 < wide_bytes          # >100x smaller
        @test wide_bytes > 200 * 1024^2                # the 225 MiB figure
        @test narrow_bytes < 1024^2                    # under 1 MiB

        # And it actually constructs at that size, which the dense form
        # would need 225 MiB to do.
        b = GAM.PenaltyBlock([copy(S_k) for _ in 1:L], width, 1, width,
            [(l - 1) * k for l in 1:L])
        @test length(b.S) == L
        @test size(b.S[1]) == (k, k)
    end
end

@testset "every penalty consumer honours sub-penalty offsets" begin
    # A narrow sub-penalty inside a wider block is what a factor-`by` penalty
    # is: `I_L ⊗ S_k` stored as L copies of `S_k` at disjoint offsets. Several
    # consumers looped over the BLOCK width while indexing `Si[j,k]`, under
    # `@inbounds`, so the read ran past the end of `Si` and returned a
    # plausible, finite, WRONG answer. Measured before the fix, on the
    # 4-level k=3 block below:
    #
    #   _log_penalty_det       8.81502  against the correct 12.09240
    #   _block_logdet_derivs   [0.365, 0.090, 2.00]  against [3, 3, 3]
    #   _penalty_range_basis   raised on Inf/NaN from garbage reads
    #
    # Each case here compares the narrow form against the same penalty
    # materialised block-width, which is the ground truth.
    _mk(k, L, seed) = begin
        rng = StableRNG(seed)
        B = randn(rng, k, k)
        Sk = B'B + 0.5I
        w = k * L
        narrow = [Matrix{Float64}(Sk) for _ in 1:L]
        offs = [(l - 1) * k for l in 1:L]
        wide = [begin
                    Z = zeros(w, w)
                    Z[((l - 1) * k + 1):(l * k), ((l - 1) * k + 1):(l * k)] .= Sk
                    Z
                end for l in 1:L]
        (GAM.PenaltyBlock(narrow, k * L, 1, w, offs),
         GAM.PenaltyBlock(wide, k * L, 1, w), w)
    end

    @testset "disjoint narrow block matches the materialised form" begin
        bn, bw, w = _mk(3, 4, 4242)
        L = length(bn.S)
        pn = GAM.PenaltySetup([bn], zeros(L), falses(L))
        pw = GAM.PenaltySetup([bw], zeros(L), falses(L))
        lsp = [0.3, -1.1, 2.0, 0.7]
        beta = randn(StableRNG(11), w)

        # Exact, not approximate: same additions in the same order.
        @test GAM.total_penalty(pn, lsp, w) == GAM.total_penalty(pw, lsp, w)

        # These three were the silent ones.
        Yn = GAM._penalty_range_basis(pn, w)
        Yw = GAM._penalty_range_basis(pw, w)
        @test Yn * Yn' ≈ Yw * Yw' rtol = 1e-10
        @test GAM._log_penalty_det(pn, lsp) ≈ GAM._log_penalty_det(pw, lsp) rtol = 1e-10
        @test GAM._block_logdet_derivs(bn, lsp) ≈
              GAM._block_logdet_derivs(bw, lsp) rtol = 1e-10

        # For pairwise-disjoint sub-penalties the derivative is exactly the
        # rank of each, with no dependence on any smoothing parameter.
        @test GAM._block_logdet_derivs(bn, lsp) ≈ fill(3.0, 4) rtol = 1e-10
        @test GAM._block_logdet_derivs(bn, [5.0, -3.0, 0.0, 1.0]) ≈
              GAM._block_logdet_derivs(bn, lsp) rtol = 1e-10

        # ... and the log-determinant is separable.
        @test GAM._log_penalty_det(pn, lsp) ≈
              sum(3 * lsp[l] for l in 1:4) + 4 * logdet(Matrix(bn.S[1])) rtol = 1e-10

        # Single-penalty helpers.
        on = zeros(w); ow = zeros(w)
        GAM._accumulate_penalty_j!(on, pn, lsp, 2, beta)
        GAM._accumulate_penalty_j!(ow, pw, lsp, 2, beta)
        @test on ≈ ow rtol = 1e-10
        @test GAM._penalty_block_j(pn, lsp, 3, w) ≈
              GAM._penalty_block_j(pw, lsp, 3, w) rtol = 1e-10
        @test GAM._bSb_j(pn, lsp, 2, beta) ≈ GAM._bSb_j(pw, lsp, 2, beta) rtol = 1e-10
    end

    @testset "mixed widths (the select=true shape)" begin
        # `select = true` appends a BLOCK-WIDTH null-space penalty beside the
        # narrow by-penalties, so the block is neither all-narrow nor
        # disjoint. That combination raised DimensionMismatch in the
        # similarity-transform path until it widened its inputs.
        k, L = 3, 4
        w = k * L
        rng = StableRNG(99)
        B = randn(rng, k, k); Sk = B'B + 0.5I
        C = randn(rng, w, w); Snull = C'C .* 1e-3 + 1e-4I
        mixed = vcat([Matrix{Float64}(Sk) for _ in 1:L], [Matrix{Float64}(Snull)])
        moffs = vcat([(l - 1) * k for l in 1:L], [0])
        wide = vcat([begin
                         Z = zeros(w, w)
                         Z[((l - 1) * k + 1):(l * k), ((l - 1) * k + 1):(l * k)] .= Sk
                         Z
                     end for l in 1:L], [Matrix{Float64}(Snull)])
        bm = GAM.PenaltyBlock(mixed, k * L, 1, w, moffs)
        bw = GAM.PenaltyBlock(wide, k * L, 1, w)
        pm = GAM.PenaltySetup([bm], zeros(L + 1), falses(L + 1))
        pw = GAM.PenaltySetup([bw], zeros(L + 1), falses(L + 1))
        lsp = [0.3, -1.1, 2.0, 0.7, -0.5]

        @test !GAM._penalties_disjoint(bm)   # the wide one overlaps all others
        @test GAM.total_penalty(pm, lsp, w) == GAM.total_penalty(pw, lsp, w)
        @test GAM._log_penalty_det(pm, lsp) ≈ GAM._log_penalty_det(pw, lsp) rtol = 1e-10
        Ym = GAM._penalty_range_basis(pm, w); Yw = GAM._penalty_range_basis(pw, w)
        @test Ym * Ym' ≈ Yw * Yw' rtol = 1e-10
    end

    @testset "block not starting at index 1" begin
        # Parametric columns sit ahead of the block, so `start != 1` and the
        # absolute index is `start + offset + j - 1`.
        np, k, L = 3, 4, 3
        w = k * L; p = np + w
        rng = StableRNG(77)
        B = randn(rng, k, k); Sk = B'B + 0.4I
        narrow = [Matrix{Float64}(Sk) for _ in 1:L]
        wide = [begin
                    Z = zeros(w, w)
                    Z[((l - 1) * k + 1):(l * k), ((l - 1) * k + 1):(l * k)] .= Sk
                    Z
                end for l in 1:L]
        bn = GAM.PenaltyBlock(narrow, k * L, np + 1, np + w, [(l - 1) * k for l in 1:L])
        bw = GAM.PenaltyBlock(wide, k * L, np + 1, np + w)
        pn = GAM.PenaltySetup([bn], zeros(L), falses(L))
        pw = GAM.PenaltySetup([bw], zeros(L), falses(L))
        lsp = [0.9, -0.4, 1.3]
        beta = randn(StableRNG(6), p)

        @test GAM.total_penalty(pn, lsp, p) == GAM.total_penalty(pw, lsp, p)
        @test GAM._log_penalty_det(pn, lsp) ≈ GAM._log_penalty_det(pw, lsp) rtol = 1e-10
        @test GAM._penalty_block_j(pn, lsp, 2, p) ≈
              GAM._penalty_block_j(pw, lsp, 2, p) rtol = 1e-10
        @test GAM._bSb_j(pn, lsp, 3, beta) ≈ GAM._bSb_j(pw, lsp, 3, beta) rtol = 1e-10
        # The penalty must land at the right absolute columns, not at 1:w.
        St = GAM.total_penalty(pn, lsp, p)
        @test all(iszero, St[1:np, :])
        @test all(iszero, St[:, 1:np])
    end

    @testset "disjointness detection" begin
        k, L = 3, 4
        bn, bw, w = _mk(k, L, 1234)
        @test GAM._penalties_disjoint(bn)      # L narrow at (l-1)k
        @test !GAM._penalties_disjoint(bw)     # L block-width, all overlapping
        # A single-penalty block is never "disjoint" in the sense used here:
        # it has no siblings and takes the exact `block.rank` path instead.
        one = GAM.PenaltyBlock([zeros(4, 4) + I], 4, 1, 4)
        @test !GAM._penalties_disjoint(one)
    end
end

# ---------------------------------------------------------------------------
# The four sites the FIRST offset pass missed.
#
# `_sub_penalty_idx` was applied to nine consumers, and that was reported as
# "the penalty layer is now ready". An audit proved otherwise: five more sites
# still derived their index range from `block.start:block.stop`. Four are live
# (`gam`/`bam`/`scam` EFS and the REML gradient), so a narrow block would have
# crashed a fit; the fifth was dead code, since removed.
#
# Verified against a worktree at 9520a08, on the L=4, k=3 block below:
#
#   _efs_sp_update        RAISED DimensionMismatch   ->  OK
#   REML gradient loop    RAISED DimensionMismatch   ->  OK
#   _total_range_basis    silently read past the end ->  bounded per component
#
# None of these was covered by the testset above, which is why they survived a
# pass that was believed complete.
# ---------------------------------------------------------------------------
@testset "sites missed by the first offset pass" begin

    # Matched narrow/wide pair: L copies of a k x k penalty at disjoint
    # offsets, against the same thing materialised block-width. The wide form
    # is the ground truth — both describe the identical penalty.
    _pair(k, L, seed) = begin
        rng = StableRNG(seed)
        B = randn(rng, k, k)
        Sk = B'B + 0.5I
        w = k * L
        narrow = [Matrix{Float64}(Sk) for _ in 1:L]
        offs = [(l - 1) * k for l in 1:L]
        wide = [begin
                    Z = zeros(w, w)
                    r = ((l - 1) * k + 1):(l * k)
                    Z[r, r] .= Sk
                    Z
                end for l in 1:L]
        (GAM.PenaltyBlock(narrow, k * L, 1, w, offs),
         GAM.PenaltyBlock(wide, k * L, 1, w), w)
    end

    @testset "_efs_sp_update accepts a narrow block" begin
        k, L = 3, 4
        bn, bw, w = _pair(k, L, 90210)
        pn = GAM.PenaltySetup([bn], zeros(L), falses(L))
        pw = GAM.PenaltySetup([bw], zeros(L), falses(L))

        rng = StableRNG(7)
        beta = randn(rng, w)
        M = randn(rng, w, w)
        Ainv = Matrix(Symmetric(M'M + w * I))
        lsp = [0.4, -0.9, 1.3, 0.1]

        # Raised `DimensionMismatch` before the fix: `beta_block` was a
        # block-width view and `dot(beta_block, Si, beta_block)` mismatched.
        upd_n = GAM._efs_sp_update(lsp, beta, Ainv, pn, 1.0, 1.0)
        upd_w = GAM._efs_sp_update(lsp, beta, Ainv, pw, 1.0, 1.0)
        # Not merely "does not throw" — it must agree with the ground truth.
        @test upd_n ≈ upd_w rtol = 1e-10
        @test length(upd_n) == L
        @test all(isfinite, upd_n)
    end

    @testset "REML gradient accepts a narrow block" begin
        k, L = 3, 3
        bn, bw, w = _pair(k, L, 31337)
        pn = GAM.PenaltySetup([bn], zeros(L), falses(L))
        pw = GAM.PenaltySetup([bw], zeros(L), falses(L))

        rng = StableRNG(11)
        n = 60
        X = randn(rng, n, w)
        y = randn(rng, n)
        wv = ones(n)
        lsp = [0.2, -0.5, 0.8]

        # Gaussian identity so the gradient is exercised without needing a
        # converged PIRLS state; the narrow-vs-wide comparison is what matters.
        _grad(pen) = begin
            St = GAM.total_penalty(pen, lsp, w)
            A = Symmetric(X' * X + St)
            Ach = cholesky(A)
            beta = Ach \ (X' * y)
            mu = X * beta
            dev = sum(abs2, y .- mu)
            GAM._reml_gradient(X, wv, Matrix(St), Ach, beta, mu, y, pen, lsp,
                dev, 1.0, n, w, :REML, 1.0, Normal(), IdentityLink(), wv)
        end

        # Raised `DimensionMismatch` before the fix: `dS[idx, idx] .= λ .* Si`
        # used a block-width `idx` against a narrow `Si`.
        gn = _grad(pn)
        gw = _grad(pw)
        @test gn ≈ gw rtol = 1e-8
        @test all(isfinite, gn)
        @test length(gn) == L
    end

    @testset "_total_range_basis is bounded by each component, not the first" begin
        # Ragged input: the old loop took its bounds from `Ss[1]` and
        # `@inbounds`-indexed every other component, so a narrower one read
        # past its end. It returned the right SHAPE with garbage in it, which
        # is why a size check would not have caught this.
        rng = StableRNG(4242)
        A = randn(rng, 3, 3); S3 = A'A + 0.5I
        B = randn(rng, 2, 2); S2 = B'B + 0.5I

        # Placing narrow components at offsets must equal widening them first.
        wide3 = zeros(5, 5); wide3[1:3, 1:3] .= S3
        wide2 = zeros(5, 5); wide2[4:5, 4:5] .= S2
        got = GAM._total_range_basis([Matrix{Float64}(S3), Matrix{Float64}(S2)], [0, 3])
        ref = GAM._total_range_basis([wide3, wide2])
        # Compare the projectors: an orthonormal basis is defined up to
        # rotation within the range, the subspace is not.
        @test size(got) == size(ref)
        @test got * got' ≈ ref * ref' atol = 1e-10

        # Equal-sized components at offset 0 must be untouched by the change.
        eq = [Matrix{Float64}(S3), Matrix{Float64}(A'A + 0.25I)]
        @test GAM._total_range_basis(eq) == GAM._total_range_basis(eq, [0, 0])
    end
end

@testset "narrow factor-by storage end to end" begin
    # Stage 3 of the narrow-storage plan: `_apply_by_variable!` emits L narrow
    # k×k copies plus `sm.S_offsets`, `setup_penalties` forwards them into
    # `PenaltyBlock.offsets`, and every consumer computes the same quantities.
    # The case is COMPUTE as much as memory: each penalty hot path bounds its
    # loops by `size(Si,1)`, so narrow storage takes the same L² off
    # `total_penalty!` and `_log_penalty_det` per sp-iteration (measured warm:
    # 700×/264× at L=50, k=14 — and 2500× on stored bytes).
    rng_nb = StableRNG(2468)
    n = 600
    dfn = DataFrame(
        x = rand(rng_nb, n),
        x2 = rand(rng_nb, n),
        f = string.(rand(rng_nb, ["a", "b", "c", "d"], n)),
    )
    dfn.y = sin.(2π .* dfn.x) .+ 0.2 .* randn(rng_nb, n)
    L = 4

    @testset "storage is narrow and the block inherits the offsets" begin
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 8, bs = :ps, by = :f)])
        m = gam(gf, dfn)
        sm = m.smooths[1]
        k_eff = size(sm.S[1], 1)
        @test k_eff * L == size(sm.X, 2)             # genuinely narrow
        @test sm.S_offsets == [(l - 1) * k_eff for l in 1:L]
        blk = m.penalty.blocks[1]
        @test blk.offsets == sm.S_offsets            # forwarded, not rebuilt
        @test size(blk.S[1], 1) == k_eff
        # total_penalty over the narrow block == the materialised reference.
        p = size(m.X, 2)
        lsp = m.sp
        Sn = GAM.total_penalty(m.penalty, lsp, p)
        wide = GAM.penalty_matrices(sm)
        blk_w = GAM.PenaltyBlock(wide, blk.rank, blk.start, blk.stop)
        pw = GAM.PenaltySetup([blk_w], copy(m.penalty.sp), copy(m.penalty.fixed))
        @test Sn == GAM.total_penalty(pw, lsp, p)    # elementwise ==, not ≈

        # The discrete path builds its smooths through the same producer, so
        # it inherits narrow storage for free — assert it rather than assume.
        mq = bam(gf, dfn; discrete = true)
        smq = mq.smooths[1]
        @test !isempty(smq.S_offsets)
        @test size(smq.S[1], 1) == k_eff
    end

    @testset "select=true mixes narrow by-penalties with a wide null penalty" begin
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 8, bs = :ps, by = :f)])
        m = gam(gf, dfn; select = true)
        blk = m.penalty.blocks[1]
        widths = unique(size.(blk.S, 1))
        @test length(blk.S) == L + 1                 # L narrow + 1 null-space
        @test length(widths) == 2                    # genuinely mixed widths
        @test blk.offsets[end] == 0                  # null penalty at offset 0
        @test size(blk.S[end], 1) == blk.stop - blk.start + 1
        @test m.converged
    end

    @testset "vector sp= still validates against the penalty count" begin
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 8, bs = :ps, by = :f, sp = [1.0, 2.0])])
        @test_throws ArgumentError gam(gf, dfn)
    end

    @testset "side_constrain! null-dim is correct on narrow storage" begin
        # THE silent-wrong site: `St .+= S[j]` succeeds on same-size narrow
        # matrices — every one is k×k — and returned the SUB-BLOCK's null
        # dimension (measured: 1 where the answer is 4). Materialising before
        # the recompute makes it correct by construction.
        #
        # Fixture note: side constraints compare a smooth only against
        # strictly LOWER-dimensional smooths sharing a variable name, and the
        # `by` variable is baked into that name — so s(x) never constrains
        # s(x, by=f), and two same-dimension by-smooths never constrain each
        # other (probed: del_index = [] for both). The natural trigger is a
        # 1-D by-smooth nested under a te with the SAME by — mgcv's gam.side
        # nesting semantics — which fires on the ordinary with_pen path.
        smA = smooth_construct(GAM.s(:x; k = 6, bs = :cr, by = :f), dfn)
        smB = smooth_construct(GAM.te(:x, :x2; k = 4, bs = [:cr, :cr], by = :f), dfn)
        @test !isempty(smB.S_offsets)                # narrow going in
        smooths = GAM.ConstructedSmooth[smA, smB]
        GAM._assign_smooth_indices!(smooths, 1)
        # Widened control: identical smooths, penalties materialised up front.
        smooths_w = GAM.ConstructedSmooth[deepcopy(smA), deepcopy(smB)]
        for smw in smooths_w
            smw.S = GAM.penalty_matrices(smw)
            smw.S_offsets = Int[]
        end
        Xp = ones(nrow(dfn), 1)
        mod_n = GAM.side_constrain!(smooths, Xp)
        mod_w = GAM.side_constrain!(smooths_w, Xp)
        # Fixture-validity gate: if the removal branch stops firing this test
        # must FAIL loudly rather than pass vacuously.
        @test mod_n && mod_w
        @test !isempty(smooths[2].del_index)
        @test smooths[2].del_index == smooths_w[2].del_index
        # Post-removal: offsets cleared, penalties at (reduced) block width…
        @test isempty(smooths[2].S_offsets)
        @test all(size(Si, 1) == size(smooths[2].X, 2) for Si in smooths[2].S)
        # …and the null-dim recompute agrees with the widened control — this
        # is the assertion that was silently wrong (1 instead of 4) before.
        @test smooths[2].null_dim == smooths_w[2].null_dim
        @test smooths[2].rank == smooths_w[2].rank
        @test length(smooths[2].S) == length(smooths_w[2].S)
        @test all(smooths[2].S[i] == smooths_w[2].S[i]
                  for i in eachindex(smooths[2].S))
    end

    @testset "smooth2random on narrow storage matches the widened form" begin
        gf = GAM.GamFormula(:y, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x; k = 6, bs = :ps, by = :f)])
        m = gam(gf, dfn)
        sm = m.smooths[1]
        @test !isempty(sm.S_offsets)                 # narrow going in
        mm = GAM.smooth2random(sm)
        # Reference: identical smooth with penalties materialised up front.
        smw = deepcopy(sm)
        smw.S = GAM.penalty_matrices(sm)
        smw.S_offsets = Int[]
        mw = GAM.smooth2random(smw)
        # Entries are NOT bitwise equal, and the reason is worth recording:
        # `sum_S` is the same set of values summed in a different association
        # order (25 dense entries vs 400 mostly-zero ones under pairwise
        # reduction), so it differs at the last ulp — measured 7.1e-15 — and a
        # factor-`by` penalty has L-fold DEGENERATE eigenvalues (four
        # identical blocks), so that ulp rotates the eigenvectors within each
        # degenerate subspace by O(1) while the eigenvalues agree to 4e-14.
        # The implied mixed model is invariant to that rotation (i.i.d.
        # normal effects; the fixed part is span-identified), so assert the
        # invariants, not the basis realisation.
        @test length(mm.Zs) == length(mw.Zs)
        for i in eachindex(mm.Zs)
            # λ·Z·Zᵀ is the implied covariance contribution — rotation-invariant.
            @test mm.Zs[i] * mm.Zs[i]' ≈ mw.Zs[i] * mw.Zs[i]' rtol = 1e-8
        end
        # Fixed part spans the same space: each column of one reproduces from
        # the other by least squares.
        if size(mm.Xf, 2) > 0
            resid = mm.Xf .- mw.Xf * (mw.Xf \ mm.Xf)
            @test maximum(abs, resid) < 1e-8
        end
        # trans_D carries eigenvalue-derived scalings of the ulp-perturbed
        # sum_S: one-ulp difference measured (5.6e-17), so ≈ not ==.
        @test mm.trans_D ≈ mw.trans_D atol = 1e-13
        @test mm.pen_ind == mw.pen_ind
    end
end
