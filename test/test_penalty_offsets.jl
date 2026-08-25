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
