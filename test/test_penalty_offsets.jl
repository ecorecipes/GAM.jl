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
