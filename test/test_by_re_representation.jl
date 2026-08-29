# Non-replicating (marginal) representations for `by=` and random effects.
#
# mgcv's discrete path stores a factor `by` as a leading one-column marginal
# plus a per-row index (R/bam.r:2470-2483) rather than replicating the basis
# per level, and keeps the level map for `bs=:re` rather than spending it into
# an n x k indicator. These tests pin the compact forms and, crucially, that
# they reconstruct the dense matrices EXACTLY — a later `bam_design` wiring
# relies on that, and a silent drift between the two forms would show up as a
# plausible-but-wrong fit rather than a failure.
#
# The dense DESIGN path is untouched: `_apply_by_variable!` still replicates
# `X` into `n x (k*L)`. Its PENALTY storage is now narrow — L copies of the
# k x k S_k plus `sm.S_offsets` — which the penalty testsets below pin.

@testset "marginal by= / re representations" begin
    rng_m = StableRNG(20250825)
    n = 400
    xs = collect(range(0, 1; length = n))
    fs_lev = ["a", "b", "c", "d"]
    df = DataFrame(
        x = xs,
        z = 0.5 .+ 2.0 .* xs,                        # numeric by
        f = [fs_lev[mod1(i, length(fs_lev))] for i in 1:n],
        g = string.("g", mod1.(1:n, 25)),            # 25 RE levels
        h = string.("h", mod1.(1:n, 5)),             # 5 RE levels (interaction)
        v = randn(rng_m, n),                          # RE random slope
        y = sin.(2π .* xs) .+ 0.1 .* randn(rng_m, n),
    )

    @testset "factor by reconstructs the replicated basis exactly" begin
        for (bs, k) in ((:cr, 8), (:tp, 10), (:ps, 9))
            spec_dense = GAM.s(:x; k = k, bs = bs, by = :f)
            sm_dense = smooth_construct(spec_dense, df)

            spec_marg = GAM.s(:x; k = k, bs = bs, by = :f)
            base_sm, rep = GAM.by_marginal_representation(spec_marg, df)

            @test rep.nblocks == length(fs_lev)
            @test rep.kbase == size(base_sm.X, 2)
            @test GAM.dense_width(rep) == size(sm_dense.X, 2)
            # The base is unreplicated: L times narrower than the dense form.
            @test size(base_sm.X, 2) * rep.nblocks == size(sm_dense.X, 2)

            Xrec = GAM.reconstruct_dense(rep, base_sm.X)
            @test size(Xrec) == size(sm_dense.X)
            @test maximum(abs, Xrec .- sm_dense.X) < 1e-15
            # Every row lands in exactly one block, so the sparsity pattern
            # must agree too — not just the values.
            @test (Xrec .!= 0.0) == (sm_dense.X .!= 0.0)
        end
    end

    @testset "numeric by reconstructs the scaled basis exactly" begin
        spec_dense = GAM.s(:x; k = 8, bs = :cr, by = :z)
        sm_dense = smooth_construct(spec_dense, df)

        spec_marg = GAM.s(:x; k = 8, bs = :cr, by = :z)
        base_sm, rep = GAM.by_marginal_representation(spec_marg, df)

        @test rep.nblocks == 1                       # numeric by: no replication
        @test rep.scale == Float64.(df.z)
        Xrec = GAM.reconstruct_dense(rep, base_sm.X)
        @test Xrec == sm_dense.X                     # same multiply, bitwise
    end

    @testset "level ordering matches _apply_by_variable!" begin
        # The reconstruction is only faithful because both use
        # `sort!(unique(...))`. Pin it: shuffled input, same level order.
        perm = shuffle(StableRNG(7), 1:n)
        dfp = df[perm, :]
        sm_dense = smooth_construct(GAM.s(:x; k = 8, bs = :cr, by = :f), dfp)
        base_sm, rep = GAM.by_marginal_representation(
            GAM.s(:x; k = 8, bs = :cr, by = :f), dfp)
        @test maximum(abs, GAM.reconstruct_dense(rep, base_sm.X) .- sm_dense.X) < 1e-15
    end

    @testset "unseen level yields a zero row" begin
        rep = GAM.MarginalBlockIndex(Int32[1, 0, 2], [1.0, 1.0, 1.0], 2, 2)
        base = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        X = GAM.reconstruct_dense(rep, base)
        @test X[1, :] == [1.0, 2.0, 0.0, 0.0]
        @test all(iszero, X[2, :])                   # index 0 -> zero row
        @test X[3, :] == [0.0, 0.0, 5.0, 6.0]
    end

    @testset "bs=:re reconstructs the indicator bitwise" begin
        for vars in ((:g,), (:h,), (:g, :h), (:g, :v))
            spec = GAM.s(vars...; bs = :re)
            sm = smooth_construct(spec, df)
            rep, levels_list = GAM.re_marginal_representation(spec, df)

            @test rep.kbase == 1                     # implicit one-column basis
            @test rep.nblocks == size(sm.X, 2)
            Xrec = GAM.reconstruct_dense(rep, nothing)
            @test Xrec == sm.X                       # exact, not approximate
            @test all(1 .<= rep.index .<= rep.nblocks)
        end
    end

    @testset "bs=:re random slope is carried in `scale`, not the basis" begin
        # A numeric column enters as a linear (random-slope) term, matching
        # mgcv. The compact form must put that in `scale`, leaving the block
        # structure a pure indicator.
        spec = GAM.s(:g, :v; bs = :re)
        rep, _ = GAM.re_marginal_representation(spec, df)
        @test rep.scale ≈ Float64.(df.v)
    end

    @testset "compact form is O(n), dense is O(n*k)" begin
        spec = GAM.s(:g; bs = :re)
        sm = smooth_construct(spec, df)
        rep, _ = GAM.re_marginal_representation(spec, df)
        dense_bytes = sizeof(sm.X)
        compact_bytes = sizeof(rep.index) + sizeof(rep.scale)
        @test compact_bytes * 4 < dense_bytes        # 25 levels: ~4x here
        @test dense_bytes == 8 * n * 25
    end

    # ── Penalty bookkeeping ─────────────────────────────────────────────────
    # The plan proposed storing the replicated penalty as `L` copies of the
    # k x k `S_k` and called that "a change in penalty.jl bookkeeping, not in
    # total_penalty". These tests establish that the storage IS narrow now and
    # that the widened view reproduces I_L ⊗ S_k exactly.
    @testset "factor-by penalty stored as L narrow copies plus offsets" begin
        L = length(fs_lev)
        spec_dense = GAM.s(:x; k = 8, bs = :cr, by = :f)
        sm_dense = smooth_construct(spec_dense, df)
        base_sm, _ = GAM.by_marginal_representation(
            GAM.s(:x; k = 8, bs = :cr, by = :f), df)
        k_eff = size(base_sm.X, 2)

        @test length(sm_dense.S) == L * length(base_sm.S)
        # NARROW storage: L copies of the k x k S_k plus per-sub-penalty
        # offsets — O(L·k²), not the old O(L³k²) materialised form. This is
        # also mgcv's own layout (R/smooth.r:3980 replicates the smooth per
        # level, each keeping its k x k S).
        @test all(size(Si) == (k_eff, k_eff) for Si in sm_dense.S)
        @test sm_dense.S_offsets == [(l - 1) * k_eff for l in 1:L]
        # The widened view — what consumers that need block width see —
        # reproduces the materialised penalty exactly.
        Sw = GAM.penalty_matrices(sm_dense)
        @test all(size(Si) == (k_eff * L, k_eff * L) for Si in Sw)
        @test sum(Sw) ≈ kron(Matrix{Float64}(I, L, L), base_sm.S[1])
        # And the exported accessor keeps its documented block-width contract
        # (it silently returned narrow matrices before being routed).
        @test all(size(Si) == (k_eff * L, k_eff * L)
                  for Si in GAM.penalty_matrix(sm_dense))
    end

    @testset "total_penalty tiles narrow sub-penalties via offsets" begin
        # `PenaltyBlock` carries a per-sub-penalty `offsets` field, so
        # `I_L ⊗ S_k` can be stored as L copies of a narrow `S_k` instead of L
        # full-width blocks — O(L·k²) rather than O(L³k²), measured at 0.101
        # against 214.86 MiB for L = 50, k = 15.
        @test fieldnames(GAM.PenaltyBlock) == (:S, :rank, :start, :stop, :offsets)

        # Positive control: full-width sub-penalties give exactly I_L ⊗ S_k.
        k_small, L = 3, 2
        S_k = Matrix{Float64}(I, k_small, k_small)
        wide = [zeros(k_small * L, k_small * L) for _ in 1:L]
        for l in 1:L
            cols = ((l - 1) * k_small + 1):(l * k_small)
            wide[l][cols, cols] .= S_k
        end
        block = GAM.PenaltyBlock(wide, k_small * L, 1, k_small * L)
        setup = GAM.PenaltySetup([block], zeros(L), falses(L))
        @test GAM.total_penalty(setup, zeros(L), k_small * L) ≈
              kron(Matrix{Float64}(I, L, L), S_k)

        # Narrow sub-penalties without offsets used to be accepted silently:
        # the accumulation loop is `@inbounds`, so it read past the end of
        # `Si` and returned plausible garbage rather than raising. That is now
        # a construction-time error, so it is assertable rather than UB.
        narrow = [copy(S_k) for _ in 1:L]
        @test_throws DimensionMismatch GAM.PenaltyBlock(
            narrow, k_small * L, 1, k_small * L)

        # With explicit offsets the narrow form is exact, not approximate:
        # elementwise equal to the materialised I_L ⊗ S_k.
        offs = [(l - 1) * k_small for l in 1:L]
        blk_narrow = GAM.PenaltyBlock(narrow, k_small * L, 1, k_small * L, offs)
        setup_narrow = GAM.PenaltySetup([blk_narrow], zeros(L), falses(L))
        @test GAM.total_penalty(setup_narrow, zeros(L), k_small * L) ==
              GAM.total_penalty(setup, zeros(L), k_small * L)
    end
end
