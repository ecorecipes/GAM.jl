# Statistical calibration regression tests.
#
# These pin properties that NO mgcv-comparison test can protect. A lost
# √scale, a Vp scaling error, or a sign flip in the delta method would leave
# every parity assertion in test_rcall.jl green (both engines would move
# together, or the error would cancel in a ratio) while silently destroying
# the coverage of every interval the package reports.
#
# Each bound below is annotated with the Monte Carlo estimate and standard
# error measured in the round-6 validation study (which used 200–400
# replicates). The bounds here are deliberately wider than ±3 MCSE at the
# smaller replicate counts used for speed, so they do not flake, but they are
# tight enough to catch a real regression — verified by deliberately
# corrupting Vp and confirming failure (see the comment on each testset).

@testset "Statistical calibration" begin

    # ------------------------------------------------------------------
    # Interval coverage, Gaussian
    #
    # Measured (round 6, 400 reps, n=200): 0.959 ± 0.003 grid-averaged
    # coverage of the nominal-95% interval for the true mean function.
    # Wood's Bayesian intervals promise ≈nominal coverage averaged over the
    # function, so mild conservatism is expected and correct.
    #
    # Regression check: scaling Vp by 4 (i.e. doubling every SE) drives
    # coverage to ~1.0; scaling by 1/4 drives it to ~0.72. Both fall outside
    # the asserted band.
    # ------------------------------------------------------------------
    @testset "Gaussian interval coverage" begin
        rng = StableRNG(20260823)
        f_true(x) = sin(2π * x)
        n, nrep = 150, 60
        grid = collect(range(0.05, 0.95; length = 25))
        covered = 0
        total = 0

        for _ in 1:nrep
            x = rand(rng, n)
            y = f_true.(x) .+ 0.3 .* randn(rng, n)
            m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), (y = y, x = x))
            pred, se = predict(m, (x = grid,); se = true)
            lo = pred .- 1.96 .* se
            hi = pred .+ 1.96 .* se
            covered += count(i -> lo[i] <= f_true(grid[i]) <= hi[i], eachindex(grid))
            total += length(grid)
        end

        coverage = covered / total
        @test 0.90 <= coverage <= 0.99
    end

    # ------------------------------------------------------------------
    # Null p-value size
    #
    # Measured (round 6, 300 reps): size at α=0.05 was 0.037 ± 0.009 for
    # GAM.jl and 0.035 for mgcv on identical replicates — i.e. the
    # documented `testStat` simplification costs nothing in test size.
    #
    # The reference distribution is asymptotic, so size depends on n.
    # Measured here (200 reps each): n=120,k=8 → 0.100 ± 0.021 (small-sample
    # anti-conservatism); n=200,k=10 → 0.055 ± 0.016; n=300,k=10 → 0.035 ±
    # 0.013 (reproducing the round-6 figure); n=500,k=10 → 0.065 ± 0.017.
    # n=200 is used below as the cheapest configuration with nominal size.
    #
    # This guards the p-value machinery end to end: a broken reference df or
    # test statistic shows up here as a size far from nominal, even though
    # the statistic itself is known to differ from mgcv's.
    # ------------------------------------------------------------------
    @testset "Null p-value size" begin
        rng = StableRNG(20260824)
        n, nrep = 200, 150
        rejections = 0
        n_valid = 0

        for _ in 1:nrep
            x = rand(rng, n)
            y = randn(rng, n)          # true smooth effect is exactly zero
            m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), (y = y, x = x))
            av = anova_gam(m)
            av.smooth_table === nothing && continue
            p = av.smooth_table.p_value[1]
            isfinite(p) || continue
            n_valid += 1
            rejections += (p < 0.05)
        end

        @test n_valid >= 0.9 * nrep      # the machinery produced usable p-values
        size_05 = rejections / n_valid
        @test 0.01 <= size_05 <= 0.10
    end

    # ------------------------------------------------------------------
    # SCAM interval coverage
    #
    # Measured (round 6, 300 reps): 0.948 ± 0.006 — essentially nominal.
    # This is the test that guards the delta-method transform through the
    # exp() reparameterization: SCAM stores exp(β) as its coefficients, so
    # Vp must be transformed by C = diag(exp(β)) (the round-2 fix). Dropping
    # that transform leaves fits identical and every parity test green,
    # while making the intervals wrong by factors of exp(β).
    # ------------------------------------------------------------------
    @testset "SCAM interval coverage" begin
        rng = StableRNG(20260825)
        f_true(x) = 2.0 * x^1.5          # monotone increasing
        n, nrep = 120, 100
        grid = collect(range(0.1, 0.9; length = 20))
        covered = 0
        total = 0

        for _ in 1:nrep
            x = rand(rng, n)
            y = f_true.(x) .+ 0.25 .* randn(rng, n)
            m = gam(@formulak(y ~ s(x, k = 10, bs = :mpi)), (y = y, x = x))
            pred, se = predict(m, (x = grid,); se = true)
            lo = pred .- 1.96 .* se
            hi = pred .+ 1.96 .* se
            covered += count(i -> lo[i] <= f_true(grid[i]) <= hi[i], eachindex(grid))
            total += length(grid)
        end

        coverage = covered / total
        @test 0.88 <= coverage <= 0.99
    end

    # ------------------------------------------------------------------
    # select=true shrinkage
    #
    # Measured (round 6, 250 reps, n=300, k=20): null-smooth EDF 0.463 ±
    # 0.041 with select=true (vs 1.313 without); a genuinely wiggly term
    # keeps EDF ≈ 8.7. Guards the Marra–Wood null-space penalty: if the
    # extra penalty stopped being applied, null EDF would jump back above 1.
    # ------------------------------------------------------------------
    @testset "select=true shrinkage" begin
        rng = StableRNG(20260826)
        n, nrep = 200, 50
        null_edf = Float64[]
        real_edf = Float64[]

        for _ in 1:nrep
            x1 = rand(rng, n)
            x2 = rand(rng, n)
            y = sin.(2π .* x1) .+ 0.3 .* randn(rng, n)   # x2 has no effect
            m = gam(@formulak(y ~ s(x1, k = 12, bs = :cr) + s(x2, k = 12, bs = :cr)),
                (y = y, x1 = x1, x2 = x2); select = true)
            e = edf(m)
            push!(real_edf, e[1])
            push!(null_edf, e[2])
        end

        @test mean(null_edf) < 1.0
        @test mean(real_edf) > 3.0
    end
end
