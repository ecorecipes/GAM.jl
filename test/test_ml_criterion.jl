# ML criterion parity with mgcv 1.9-4.
#
# `method = :ML` differs from `:REML` in TWO ways, not one. Before this was
# fixed GAM.jl applied only the first, and scored 1–8% away from mgcv:
#
#   1. The `Mp` null-space term is dropped (`remlInd = 0`, R/gam.fit3.r:545-546).
#      This part was already right.
#
#   2. The log-determinant is taken over the RANGE SPACE of the total penalty,
#      not over all p coefficients. mgcv passes the `REML` flag into `gdi1`
#      (R/gam.fit3.r:563), which branches on its sign (src/gdi.c:2729-2741) and
#      for ML calls `MLpenalty1` — documented at src/gdi.c:1536-1551 as
#      obtaining "the version of log|X'WX+S| that applies to ML" by projecting
#      the Hessian "into the range space of the penalty". It drops the
#      null-space columns of the R factor (src/gdi.c:1560-1570) and accumulates
#      `ldetXWXS = 2*Σ log|R̃_ii|` (src/gdi.c:1658-1660).
#
#   3. The scale is PROFILED, and the profiling equation is criterion-specific.
#      From R/gam.fit3.r:629 with remlInd = 0, Gaussian gives φ̂ = Dp/n where
#      REML gives φ̂ = Dp/(n − Mp). Note mgcv's `b$scale` REPORTS the
#      REML-style scale even for an ML fit, so it cannot be used to infer the
#      scale the ML score was evaluated at.
#
# Reference values below are mgcv 1.9-4's `b$gcv.ubre`, taken at mgcv's own
# selected `sp` so that the CRITERION is compared rather than the optimizer.
# Regenerate with, e.g.:
#     b  <- gam(y~s(x,bs='cr',k=10), data=df, method='ML')
#     bf <- gam(y~s(x,bs='cr',k=10), data=df, method='ML', sp=b$sp)
#     bf$gcv.ubre

@testset "ML criterion vs mgcv" begin
    # Deterministic data — no RNG, so these are exactly reproducible.
    n = 200
    xg = collect(range(0, 1; length = n))
    fg = sin.(2π .* xg) .+ 0.5 .* cos.(5.0 .* xg)
    yg = fg .+ 0.3 .* sin.(29.0 .* (1:n))

    df = DataFrame(x = xg, y = yg)
    gf(sp) = GAM.GamFormula(:y, Symbol[], true,
        GAM.SmoothSpec[GAM.s(:x; k = 10, bs = :cr, sp = sp)])

    @testset "range-space determinant is used for ML, not the full one" begin
        m = gam(gf(nothing), df; method = :ML)
        p = size(m.X, 2)
        pen = m.penalty
        Y = GAM._penalty_range_basis(pen, p)

        # The basis must drop exactly the null-space directions.
        Mp = p - sum(b.rank for b in pen.blocks; init = 0)
        @test size(Y, 2) == p - Mp
        @test Y' * Y ≈ I(p - Mp) atol = 1e-10

        S = GAM.total_penalty(pen, m.sp, p)
        pr = GAM.pirls(m.X, df.y, S, Normal(), IdentityLink())
        A = m.X' * Diagonal(pr.working_weights) * m.X + S

        ld_full = logdet(cholesky(Symmetric(A)))
        ld_rng = logdet(cholesky(Symmetric(Y' * A * Y)))
        # The range-space determinant is strictly smaller; the difference is
        # what the old implementation was missing.
        @test ld_rng < ld_full
    end

    @testset "range basis is smoothing-parameter independent" begin
        # mgcv computes `totalPenaltySpace` once in estimate.gam and reuses it
        # at every trial ρ (R/mgcv.r:1921-1924), because each block is
        # normalized by its own Frobenius norm with no `sp` involved.
        m = gam(gf(nothing), df; method = :ML)
        p = size(m.X, 2)
        Y1 = GAM._penalty_range_basis(m.penalty, p)
        # Span is what the determinant depends on; compare projectors, which
        # are invariant to rotation/sign within the range space.
        P1 = Y1 * Y1'
        m2 = gam(gf(1000.0), df; method = :ML)
        Y2 = GAM._penalty_range_basis(m2.penalty, p)
        @test Y2 * Y2' ≈ P1 atol = 1e-10
    end

    @testset "Gaussian ML matches mgcv at mgcv's own sp" begin
        # mgcv 1.9-4: b$sp, then bf$gcv.ubre at that sp.
        mgcv_sp = 4.0833751910
        mgcv_ml = -10.173800
        m = gam(gf(mgcv_sp), df; method = :ML)
        @test isapprox(m.reml, mgcv_ml; rtol = 1e-5)
    end

    @testset "ML profiled scale follows mgcv's criterion-specific rule" begin
        # Gaussian: ML profiles φ̂ = Dp/n, REML φ̂ = Dp/(n − Mp). The two must
        # differ, and `_ml_profiled_scale` must return the closed form exactly.
        m = gam(gf(nothing), df; method = :ML)
        p = size(m.X, 2)
        S = GAM.total_penalty(m.penalty, m.sp, p)
        pr = GAM.pirls(m.X, df.y, S, Normal(), IdentityLink())
        Dp = pr.deviance + dot(pr.coefficients, S * pr.coefficients)
        Mp = p - sum(b.rank for b in m.penalty.blocks; init = 0)

        phi_ml = GAM._ml_profiled_scale(Normal(), df.y, ones(n), Dp, n, 1.0)
        @test phi_ml ≈ Dp / n
        @test !isapprox(phi_ml, Dp / (n - Mp); rtol = 1e-8)

        # For a family with digamma terms in `ls` there is no closed form, so
        # the numeric solve must still land on a stationary point of the ML
        # criterion w.r.t. log φ.
        yg_pos = 0.5 .+ abs.(yg)
        phi_g = GAM._ml_profiled_scale(Gamma(), yg_pos, ones(n), Dp, n, Dp / n)
        @test isfinite(phi_g) && phi_g > 0
        dls(phi) = (GAM._log_saturated_likelihood(Gamma(), yg_pos, ones(n), phi + 1e-7) -
                    GAM._log_saturated_likelihood(Gamma(), yg_pos, ones(n), phi - 1e-7)) / 2e-7
        @test abs(-Dp / (2 * phi_g) - phi_g * dls(phi_g)) < 1e-4 * max(1.0, Dp / phi_g)
    end

    @testset "REML is unaffected by the ML fix" begin
        # Hard requirement: the ML work must not perturb the default path.
        # mgcv 1.9-4 REML on the same data, at mgcv's own REML sp.
        mgcv_sp = 4.0998963089
        mgcv_reml = -5.069955
        m = gam(gf(mgcv_sp), df; method = :REML)
        @test isapprox(m.reml, mgcv_reml; rtol = 1e-5)

        # REML must still use pearson/(n − edf), NOT the ML profiling scale.
        p = size(m.X, 2)
        S = GAM.total_penalty(m.penalty, m.sp, p)
        pr = GAM.pirls(m.X, df.y, S, Normal(), IdentityLink())
        Dp = pr.deviance + dot(pr.coefficients, S * pr.coefficients)
        @test !isapprox(m.scale, Dp / n; rtol = 1e-8)
    end

    @testset "ML score is strictly below REML on the same fit" begin
        # ML drops the +Mp/2·log(2πφ) term and uses a smaller determinant, so
        # it must sit below REML. A regression that silently reverted the
        # determinant change would shrink this gap toward the Mp term alone.
        sp = 0.002
        m_r = gam(gf(sp), df; method = :REML)
        m_m = gam(gf(sp), df; method = :ML)
        @test m_m.reml < m_r.reml
        p = size(m_r.X, 2)
        Mp = p - sum(b.rank for b in m_r.penalty.blocks; init = 0)
        # The gap is much larger than the Mp term alone — that discrepancy
        # (0.63 observed vs 4.53 expected) is what exposed the bug.
        mp_term_only = 0.5 * Mp * log(2π * m_r.scale)
        @test (m_r.reml - m_m.reml) > 1.5 * abs(mp_term_only)
    end
end
