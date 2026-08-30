# Multi-dimensional Gaussian-process smooths (`bs=:gp`).
#
# GP smooths were 1-D only until this suite existed; `_smooth_construct`
# raised `ArgumentError("GP smooths currently support 1d only")`. Since a GP
# smooth is primarily a *spatial* tool, that excluded its main use case.
#
# Reference values are from mgcv 1.9-4 and pinned here so the suite needs no
# R. They were produced by writing the datasets built below to CSV and running:
#
#     m <- gam(y ~ s(x, z, bs = "gp", k = 30), data = d2, method = "REML")
#     m <- gam(y ~ s(x, z, w, bs = "gp", k = 25, m = <m>), data = d3, method = "REML")
#
# As in test_gp_parity.jl, edf is compared at mgcv's OWN selected sp, which
# removes the smoothing-parameter optimizer from the comparison and tests the
# basis and penalty alone.
#
# mgcv's construction, generalised to d covariates (mgcv 1.9-4 `R/smooth.r`,
# `smooth.construct.gp.smooth.spec`, `gpE`, `gpT`):
#
#   * knots are the unique covariate *combinations* (`uniquecombs`, unique
#     ROWS), not the unique values of each column;
#   * covariates are centred by `colMeans(x)` and NOT scaled, so the kernel is
#     isotropic in the covariates' own units;
#   * distances are Euclidean: `sqrt(rowSums((x - xk)^2))`;
#   * the range defaults to the largest pairwise knot distance;
#   * the unpenalized null space is `cbind(1, x)`, so `null.space.dim` is
#     `d + 1` before the identifiability constraint is absorbed.

@testset "GP smooth (bs=:gp) — multi-dimensional" begin

    # ---- shared datasets -------------------------------------------------
    _rng2 = StableRNG(4242)
    _n2 = 250
    _x2 = rand(_rng2, _n2)
    _z2 = rand(_rng2, _n2)
    _e2 = randn(_rng2, _n2)
    _f2 = sin.(2π .* _x2) .* cos.(π .* _z2)
    df2 = DataFrame(x = _x2, z = _z2, y = _f2 .+ 0.3 .* _e2)

    _rng3 = StableRNG(99)
    _n3 = 200
    _x3 = rand(_rng3, _n3)
    _z3 = rand(_rng3, _n3)
    _w3 = rand(_rng3, _n3)
    _e3 = randn(_rng3, _n3)
    _f3 = sin.(2π .* _x3) .* cos.(π .* _z3) .+ 0.5 .* _w3
    df3 = DataFrame(x = _x3, z = _z3, w = _w3, y = _f3 .+ 0.3 .* _e3)

    t2 = Tables.columntable(df2)
    t3 = Tables.columntable(df3)

    # ------------------------------------------------------------------
    # 1. Basis structure matches mgcv exactly.
    # ------------------------------------------------------------------
    @testset "2-D basis structure" begin
        sm = smooth_construct(s(:x, :z; bs = :gp, k = 30), t2)
        # mgcv: length(coef(m)) - 1 == 29 after the centring constraint.
        @test size(sm.X, 2) == 29
        # null.space.dim = ncol(knt) + 1 = 3 BEFORE the constraint; mgcv
        # reports 2 post-constraint, and its rank confirms 30 - 3 = 27.
        @test sm.null_dim == 3
        @test sm.rank == 27
        # rho = largest pairwise distance among the (centred) knots.
        @test sm.predict_cache.rho ≈ 1.3052328085927 atol = 1e-11
        # shift is per covariate.
        @test length(sm.predict_cache.shift) == 2
        @test sm.predict_cache.shift ≈ [mean(_x2), mean(_z2)] atol = 1e-12
        # knots are stored as an nk × d matrix of unique covariate rows.
        @test size(sm.predict_cache.knots, 2) == 2
        @test size(sm.predict_cache.knots, 1) == _n2   # all rows unique here
    end

    @testset "3-D basis structure" begin
        sm = smooth_construct(s(:x, :z, :w; bs = :gp, k = 25), t3)
        @test size(sm.X, 2) == 24
        @test sm.null_dim == 4          # d + 1
        @test sm.rank == 21             # k - null_dim
        @test sm.predict_cache.rho ≈ 1.5479349307158 atol = 1e-11
        @test size(sm.predict_cache.knots, 2) == 3

        # The stationary variant (negative m) drops to an intercept-only null
        # space in any dimension, so nothing is left unpenalized but the mean.
        sms = smooth_construct(s(:x, :z, :w; bs = :gp, k = 25, m = -3), t3)
        @test sms.null_dim == 1
        @test sms.rank == 24
        @test sms.predict_cache.stationary
    end

    # ------------------------------------------------------------------
    # 2. edf against mgcv, at mgcv's own smoothing parameter.
    # ------------------------------------------------------------------
    @testset "edf matches mgcv at mgcv's sp" begin
        m2 = gam(@formulak(y ~ s(x, z, bs = :gp, k = 30, sp = 0.0013987283641902)), df2)
        @test sum(edf(m2)) ≈ 21.5126562628 atol = 1e-6

        for (mm, sp_r, edf_r) in ((2, 0.089770186795593, 18.0389950184),
                                  (5, 0.00052472617017048, 10.3329370824),
                                  (-3, 0.014925390987616, 15.8620904756))
            spec = s(:x, :z, :w; bs = :gp, k = 25, m = mm, sp = sp_r)
            gf = GAM.GamFormula(:y, Symbol[], true, [spec])
            m3 = gam(gf, df3)
            @test sum(edf(m3)) ≈ edf_r atol = 1e-6
        end
    end

    # ------------------------------------------------------------------
    # 3. It actually recovers a known surface — running is not the same as
    #    working.
    # ------------------------------------------------------------------
    @testset "recovers a known 2-D surface" begin
        m = gam(@formulak(y ~ s(x, z, bs = :gp, k = 30)), df2)
        rmse = sqrt(mean((fitted(m) .- _f2) .^ 2))
        # Noise sd is 0.30, so an RMSE against the TRUTH well below that is
        # the smooth doing its job rather than interpolating the noise.
        @test rmse < 0.15
        @test cor(fitted(m), _f2) > 0.95
    end

    # ------------------------------------------------------------------
    # 4. Prediction.
    # ------------------------------------------------------------------
    @testset "prediction" begin
        m = gam(@formulak(y ~ s(x, z, bs = :gp, k = 30)), df2)
        # Predicting at the training rows must reproduce the fit exactly: the
        # cached shift/knots/UZ are re-used rather than re-derived.
        @test maximum(abs.(predict(m, df2) .- fitted(m))) < 1e-10

        nd = DataFrame(x = [0.1, 0.5, 0.9], z = [0.2, 0.5, 0.8])
        p = predict(m, nd)
        @test length(p) == 3
        @test all(isfinite, p)

        # A covariate of the wrong length is an error, not a silent recycle.
        # Passed as a NamedTuple because DataFrame would reject it first.
        @test_throws ArgumentError predict(m, (x = [0.1, 0.2], z = [0.5]))
    end

    # ------------------------------------------------------------------
    # 5. Two properties that follow from mgcv's construction and that users
    #    get wrong. Both are asserted so they cannot drift silently.
    # ------------------------------------------------------------------
    @testset "isotropic in the covariates, and in their own units" begin
        # (a) Euclidean distance is symmetric in the covariates, so the term
        #     order cannot matter: s(x, z) and s(z, x) span the same model
        #     space. Compared at FIXED sp, because the two orderings give the
        #     REML optimizer a differently-parameterised but equivalent
        #     problem, and on this flat optimum it stops ~2% apart in sp
        #     (0.0013988 vs 0.0014301), which moves fitted values by ~1.5e-3.
        #     That is the optimizer, not the basis; at fixed sp the two agree
        #     to machine precision.
        sp_fix = 0.0013987283641902
        m_xz = gam(@formulak(y ~ s(x, z, bs = :gp, k = 30, sp = sp_fix)), df2)
        m_zx = gam(@formulak(y ~ s(z, x, bs = :gp, k = 30, sp = sp_fix)), df2)
        @test maximum(abs.(fitted(m_xz) .- fitted(m_zx))) < 1e-10
        @test abs(sum(edf(m_xz)) - sum(edf(m_zx))) < 1e-10

        # (b) mgcv centres the covariates but does NOT scale them, so the
        #     kernel is isotropic in whatever units the covariates carry.
        #     Rescaling one covariate is therefore a DIFFERENT model — which
        #     is why covariates on wildly different scales should be
        #     standardised by the user before fitting.
        df_scaled = DataFrame(x = df2.x, z = df2.z .* 100, y = df2.y)
        m_scaled = gam(@formulak(y ~ s(x, z, bs = :gp, k = 30, sp = sp_fix)), df_scaled)
        @test maximum(abs.(fitted(m_xz) .- fitted(m_scaled))) > 1e-6
    end

    # ------------------------------------------------------------------
    # 6. The 1-D path is unchanged. test_gp_parity.jl is the real guard for
    #    this; these assertions make the intent explicit at the point where
    #    multi-D support was added.
    # ------------------------------------------------------------------
    @testset "1-D path unchanged" begin
        n = 200
        x = (1:n) ./ (n + 1)
        sm = smooth_construct(s(:x; bs = :gp, k = 10), (x = collect(x),))
        @test sm.null_dim == 2          # d + 1 with d == 1
        @test sm.rank == 8
        @test size(sm.predict_cache.knots, 2) == 1
        @test length(sm.predict_cache.shift) == 1
        # 1-D distances are computed as abs(x - k), not sqrt((x - k)^2), so
        # the pinned parity values stay bit-stable.
        @test sm.predict_cache.rho ≈ (maximum(x) - minimum(x)) atol = 1e-15
    end

    # ------------------------------------------------------------------
    # 7. Errors name what IS supported.
    # ------------------------------------------------------------------
    @testset "informative errors" begin
        # k below the minimum for the null space (mgcv: ncol(knt) + 2).
        @test_throws ArgumentError smooth_construct(s(:x, :z; bs = :gp, k = 3), t2)
        # User knots are 1-D only, and the message must say so.
        err = try
            smooth_construct(s(:x, :z; bs = :gp, k = 30), t2, [0.1, 0.5, 0.9])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("1-D", sprint(showerror, err))
    end
end
