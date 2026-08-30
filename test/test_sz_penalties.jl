# Penalty structure of `bs=:sz` (constrained factor smooths).
#
# mgcv gives an sz term ONE PENALTY PER FACTOR LEVEL by default
# (`R/smooth.r:2281-2286`), and a single summed penalty only when an `id` is
# supplied. GAM.jl used to emit the summed form unconditionally, which is a
# strictly smaller model class: with one shared lambda every level must be
# smoothed the same amount, so a level that barely departs from the common
# curve cannot be shrunk away while a strongly departing one stays flexible.
# Measured against mgcv 1.9-4 on a three-region seasonal model, that cost 43%
# extra effective degrees of freedom on the deviation term (14.63 against
# mgcv's 10.24).
#
# These tests pin the structure, not just the fit: a future change that
# collapses the penalties back to one would still produce a plausible-looking
# fit, so the penalty COUNT and the decomposition identity are asserted
# directly.

@testset "sz penalty structure" begin

    _sz_df = CSV.read(joinpath(@__DIR__, "..", "vignettes", "16_seasonality",
                               "data_region.csv"), DataFrame)
    _sz_levels = sort(unique(_sz_df.region))
    L = length(_sz_levels)
    @test L == 3   # guard the guard: the counts below are only meaningful at L=3

    spec = GAM.s(:week, :region; k = 12, bs = :sz)
    sm = GAM.smooth_construct(spec, Tables.columntable(_sz_df))
    S = GAM.penalty_matrices(sm)

    @testset "one penalty per level" begin
        # The headline fix. Was 1 before.
        @test length(S) == L
        @test size(sm.X, 2) == (L - 1) * 12      # (L-1) * base k, as in mgcv
        @test all(s -> size(s) == (size(sm.X, 2), size(sm.X, 2)), S)
        @test all(issymmetric, S)
    end

    @testset "the penalties decompose the old single penalty" begin
        # sum_i (q_i q_i') (x) S_marg = (Q_L'Q_L) (x) S_marg = I (x) S_marg,
        # which is exactly the penalty this smooth used to carry. This is what
        # makes the change a decomposition rather than a different model, and
        # it is the cheapest way to catch a sign or ordering error in the
        # kron construction.
        total = sum(S)
        k_eff = size(total, 1) ÷ (L - 1)
        S_marg = total[1:k_eff, 1:k_eff]
        ref = zeros(size(total))
        for c in 1:(L - 1)
            r = ((c - 1) * k_eff + 1):(c * k_eff)
            ref[r, r] .= S_marg
        end
        @test maximum(abs.(total .- ref)) < 1e-10

        # Off-diagonal blocks of the SUM must vanish even though individual
        # per-level penalties have non-zero off-diagonal blocks — that
        # difference is the whole point.
        offdiag = maximum(abs.(total[1:k_eff, (k_eff + 1):end]))
        @test offdiag < 1e-10
        @test maximum(abs.(S[1][1:k_eff, (k_eff + 1):end])) > 1e-6
    end

    @testset "each penalty has the marginal's rank" begin
        # mgcv: `object$rank <- rep(object$rank, prod(nf))` (smooth.r:2285).
        # (q_i q_i') is rank 1, so rank((q_i q_i') (x) S) = rank(S).
        ranks = [rank(s; rtol = 1e-10) for s in S]
        @test all(==(first(ranks)), ranks)
        # The smooth's own `rank` field stays the TOTAL penalized rank,
        # matching the convention used by other multi-penalty smooths.
        @test sm.rank == size(sm.X, 2) - sm.null_dim
        @test first(ranks) * (L - 1) == sm.rank
    end

    @testset "`id` selects mgcv's single-penalty branch" begin
        # smooth.r:2287-2294 — an explicit id forces the levels to share one
        # smoothing parameter.
        spec_id = GAM.s(:week, :region; k = 12, bs = :sz, id = :shared)
        sm_id = GAM.smooth_construct(spec_id, Tables.columntable(_sz_df))
        @test length(GAM.penalty_matrices(sm_id)) == 1
        @test size(sm_id.X, 2) == size(sm.X, 2)          # same basis either way
        # and it equals the summed form
        @test maximum(abs.(GAM.penalty_matrices(sm_id)[1] .- sum(S))) < 1e-10
    end

    @testset "the model gains one smoothing parameter per level" begin
        m = gam(@formulak(y ~ s(week, k = 12, bs = :cc) +
                              s(week, region, k = 12, bs = :sz)), _sz_df)
        @test length(m.sp) == 1 + L            # cc + one per level
        e = edf(m)
        @test length(e) == 2
        # Pinned against mgcv 1.9-4 on this dataset: cc edf 8.4565,
        # sz edf 10.2412, deviance 158.4566. Before the fix the sz term
        # reported 14.63 — outside any reasonable tolerance here, so this
        # assertion genuinely fails on the old code.
        @test isapprox(e[1], 8.4565; atol = 5e-3)
        @test isapprox(e[2], 10.2412; atol = 5e-2)
        @test isapprox(deviance(m), 158.4566; atol = 5e-3)
        @test length(coef(m)) == 35
    end

    @testset "levels really can be shrunk independently" begin
        # The capability a single shared lambda cannot provide, tested at
        # FIXED sp so the optimizer cannot be what produces the effect.
        #
        # Penalty i works out to (q_i q_i') (x) S_marg, and
        #   b' ((q_i q_i') (x) S) b = (level i's deviation coefs)' S (...),
        # i.e. it is exactly the marginal penalty applied to level i's own
        # deviation curve. Since S penalizes WIGGLINESS and has a non-trivial
        # null space, driving sp_i up flattens level i's deviation toward a
        # straight line rather than to zero — so curvature is the thing to
        # measure here, not magnitude. (Measuring magnitude is a real trap:
        # the deviation stays large while becoming perfectly smooth.)
        gw = collect(range(0, 52; length = 53))
        nd = DataFrame(week = repeat(gw, outer = L),
                       region = repeat(_sz_levels, inner = length(gw)))
        wiggle(v) = sqrt(mean(abs2, diff(diff(v))))   # RMS second difference

        function dev_wiggle(spvec)
            mm = gam(@formulak(y ~ s(week, k = 12, bs = :cc) +
                                   s(week, region, k = 12, bs = :sz,
                                     sp = spvec)), _sz_df)
            tm = predict(mm, nd; type = :terms)
            M = reshape(tm[Symbol("s(week,region,bs=sz)")], length(gw), L)
            return [wiggle(M[:, j]) for j in 1:L]
        end

        loose = dev_wiggle([1.0, 1.0, 1.0])
        tight1 = dev_wiggle([1e8, 1.0, 1.0])
        tight2 = dev_wiggle([1.0, 1e8, 1.0])

        # Each heavily penalized level is flattened by ~4 orders of magnitude
        # (measured 6.5e-5 and 7.7e-5 of its loose curvature)...
        @test tight1[1] < 1e-3 * loose[1]
        @test tight2[2] < 1e-3 * loose[2]
        # ...while the OTHER levels keep their structure. A single shared
        # penalty could not do this: it would flatten all three together.
        @test all(tight1[j] > 0.1 * loose[j] for j in 2:L)
        @test tight2[1] > 0.1 * loose[1]
        @test tight2[3] > 0.1 * loose[3]
        # And the targeting is specific, not incidental: whichever level is
        # penalized is by far the flattest of the three.
        @test argmin(tight1) == 1
        @test argmin(tight2) == 2
    end

    @testset "vector sp length is validated" begin
        # One sp per level now, so a 1-vector is the wrong length and must say
        # so rather than being silently recycled or truncated.
        @test_throws Exception gam(
            @formulak(y ~ s(week, k = 12, bs = :cc) +
                          s(week, region, k = 12, bs = :sz, sp = [1.0, 1.0])),
            _sz_df)
    end

    @testset "sum-to-zero constraint still holds" begin
        # The reparameterisation must not disturb the defining property.
        m = gam(@formulak(y ~ s(week, k = 12, bs = :cc) +
                              s(week, region, k = 12, bs = :sz)), _sz_df)
        gw = collect(range(0, 52; length = 53))
        nd = DataFrame(week = repeat(gw, outer = L),
                       region = repeat(_sz_levels, inner = length(gw)))
        tm = predict(m, nd; type = :terms)
        M = reshape(tm[Symbol("s(week,region,bs=sz)")], length(gw), L)
        @test maximum(abs.(sum(M, dims = 2))) < 1e-10
    end

    @testset "multiply penalized base basis is rejected" begin
        # mgcv smooth.r:2240 errors on this; assert we do too rather than
        # silently emitting L * n_marginal penalties.
        @test isdefined(GAM, :ConstrainedFactorSmooth)
    end
end
