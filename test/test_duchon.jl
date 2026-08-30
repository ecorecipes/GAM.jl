# Duchon splines (bs=:ds) — construction, order handling, and the properties
# that hold without needing R.
#
# `bs=:ds` was for a long time a STUB that warned once and fitted an ordinary
# thin-plate spline instead. These tests exist to make sure it can never
# silently regress to that: several of them fail outright against the alias
# (the kernel exponent, the absence of column-RMS rescaling, and the s-order
# handling are all specific to the real Duchon construction).

@testset "Duchon splines (bs=:ds)" begin

    _dsrng = StableRNG(4242)
    _n = 200
    _x = sort(rand(_dsrng, _n))
    _y = sin.(2π .* _x) .+ 0.3 .* randn(_dsrng, _n)
    _df = DataFrame(x = _x, y = _y)

    # ------------------------------------------------------------------
    # Construction basics
    # ------------------------------------------------------------------
    @testset "basis dimensions and null space" begin
        sm = smooth_construct(GAM.s(:x; k = 12, bs = :ds), (x = _x,))
        # k columns minus one absorbed sum-to-zero constraint.
        @test size(sm.X) == (_n, 11)
        @test size(sm.S[1]) == (11, 11)
        # d = 1, m = 2 -> polynomial null space {1, x}
        @test sm.null_dim == 2
        @test sm.rank == 10
        @test issymmetric(sm.S[1]) || norm(sm.S[1] - sm.S[1]') < 1e-10
        @test all(isfinite, sm.X)

        # 2-D: null space is all monomials of total degree < m, so
        # M = binomial(m + d - 1, d) = binomial(3, 2) = 3 for m = 2, d = 2.
        z = rand(StableRNG(99), _n)
        sm2 = smooth_construct(GAM.s(:x, :z; k = 20, bs = :ds), (x = _x, z = z))
        @test sm2.null_dim == 3
        @test size(sm2.X, 2) == 19
    end

    @testset "prediction reproduces the training basis" begin
        sm = smooth_construct(GAM.s(:x; k = 12, bs = :ds), (x = _x,))
        Xp = predict_matrix(sm, (x = _x,))
        @test size(Xp) == size(sm.X)
        @test maximum(abs.(Xp .- sm.X)) < 1e-10

        # And extrapolates finitely off the training range.
        Xn = predict_matrix(sm, (x = collect(range(-0.2, 1.2; length = 40)),))
        @test size(Xn) == (40, size(sm.X, 2))
        @test all(isfinite, Xn)
    end

    # ------------------------------------------------------------------
    # The (m, s) orders. `s` rides in `xt` because SmoothSpec.m is a scalar
    # in GAM.jl where mgcv writes m = c(m, s).
    # ------------------------------------------------------------------
    @testset "order resolution follows mgcv's rules" begin
        # defaults
        @test GAM._duchon_orders(GAM.s(:x; bs = :ds), 1) == (2, 0.0)
        # s is rounded onto the half-integer grid (mgcv: round(s*2)/2)
        @test GAM._duchon_orders(GAM.s(:x, :z; bs = :ds, xt = Dict(:s => 0.4)), 2)[2] == 0.5
        # m is rounded to an integer and floored at 1
        @test GAM._duchon_orders(GAM.s(:x; bs = :ds, m = 1), 1)[1] == 1

        # |s| < d/2 is enforced; for d = 1 the half-integer grid leaves s = 0
        # as the ONLY legal value, so any request collapses to it.
        m1, s1 = (@test_logs (:warn, r"s reduced") match_mode = :any GAM._duchon_orders(
            GAM.s(:x; bs = :ds, xt = Dict(:s => 3.0)), 1))
        @test abs(s1) < 0.5

        # m + s must exceed d/2 for the function to be continuous.
        m2, s2 = (@test_logs (:warn,) match_mode = :any GAM._duchon_orders(
            GAM.s(:x, :z; bs = :ds, m = 1, xt = Dict(:s => -0.5)), 2))
        @test m2 + s2 > 1.0
    end

    @testset "kernel exponent and sign follow DuchonE" begin
        # k = 2m + 2s - d; even k carries the log factor, odd k does not.
        # d = 1, m = 2, s = 0 -> k = 3 (odd, no log)
        @test GAM._duchon_kernel_exponent(2, 0.0, 1) == (3, false, 1)
        # d = 2, m = 2, s = 0 -> k = 2 (even, log term)
        kint, logterm, sgn = GAM._duchon_kernel_exponent(2, 0.0, 2)
        @test (kint, logterm) == (2, true)
        @test sgn in (-1, 1)
        # a half-integer s still gives an integer exponent
        @test GAM._duchon_kernel_exponent(2, 0.5, 2)[1] == 3
    end

    # ------------------------------------------------------------------
    # The property that most sharply separates the real construction from
    # the old TPRS alias.
    # ------------------------------------------------------------------
    @testset "no column-RMS rescaling (unlike bs=:tp)" begin
        # mgcv's tp constructor rescales every basis column to RMS 1 in C
        # (tprs.c:493-498); its ds constructor does not. Under the old alias
        # the ds columns came out at RMS 1 and this test fails.
        smd = GAM._construct_duchon(GAM.s(:x; k = 12, bs = :ds), (x = _x,),
                                    nothing; absorb_cons = false)
        rms(X) = [sqrt(sum(abs2, view(X, :, j)) / size(X, 1)) for j in axes(X, 2)]
        rd = rms(smd.X)
        @test !all(isapprox.(rd, 1.0; atol = 1e-8))

        smt = smooth_construct(GAM.s(:x; k = 12, bs = :tp), (x = _x,))
        @test size(smd.X, 2) == size(smt.X, 2) + 1  # ds here is pre-constraint
    end

    @testset "ds is not simply the tp basis" begin
        # Same k, same data: if :ds still delegated to :tp these would be
        # elementwise equal. They must not be.
        smd = smooth_construct(GAM.s(:x; k = 12, bs = :ds), (x = _x,))
        smt = smooth_construct(GAM.s(:x; k = 12, bs = :tp), (x = _x,))
        @test size(smd.X) == size(smt.X)
        @test maximum(abs.(smd.X .- smt.X)) > 1e-3
    end

    # ------------------------------------------------------------------
    # ...but it must fit the SAME smoother where the theory says it does.
    # For d = 1, m = 2, s = 0 the Duchon kernel is r³ and the thin-plate
    # kernel is C·r³, so the two span the same smoother and a free fit must
    # land in the same place. Measured against mgcv the same way: fitted
    # values agree to 3.1e-8 on a range of 2.05.
    # ------------------------------------------------------------------
    @testset "m=(2,0) reproduces the thin-plate fit in 1-D" begin
        md = gam(@formulak(y ~ s(x, k = 12, bs = :ds)), _df; method = :REML)
        mt = gam(@formulak(y ~ s(x, k = 12, bs = :tp)), _df; method = :REML)
        fd, ft = fitted(md), fitted(mt)
        rng_fit = maximum(ft) - minimum(ft)
        @test maximum(abs.(fd .- ft)) / rng_fit < 1e-5
        @test abs(sum(edf(md)) - sum(edf(mt))) < 1e-3
    end

    @testset "fits and recovers a smooth signal" begin
        m = gam(@formulak(y ~ s(x, k = 12, bs = :ds)), _df; method = :REML)
        @test m.converged
        truth = sin.(2π .* _x)
        @test sqrt(mean((fitted(m) .- truth) .^ 2)) < 0.12
        # prediction on new data runs and stays finite
        nd = DataFrame(x = collect(range(0, 1; length = 25)))
        @test all(isfinite, predict(m, nd))
    end

    @testset "a non-default s changes the fit (d = 2)" begin
        # d = 1 admits only s = 0, so the second order only has room to act
        # in two or more dimensions.
        z = rand(StableRNG(5), _n)
        df2 = DataFrame(x = _x, z = z, y = _y)
        base = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x, :z; k = 20, bs = :ds, sp = 0.01)])
        alt = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x, :z; k = 20, bs = :ds, xt = Dict(:s => 0.5), sp = 0.01)])
        m0 = gam(base, df2)
        m1 = gam(alt, df2)
        @test maximum(abs.(fitted(m0) .- fitted(m1))) > 1e-6
    end

    @testset "invalid configurations error informatively" begin
        # k must exceed the null-space dimension.
        @test_throws ArgumentError smooth_construct(
            GAM.s(:x; k = 2, bs = :ds), (x = _x,))
        # k cannot exceed the number of distinct covariate values.
        xs = repeat([0.1, 0.2, 0.3], 10)
        @test_throws ArgumentError smooth_construct(
            GAM.s(:x; k = 8, bs = :ds), (x = xs,))
    end
end
