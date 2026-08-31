# Model-facing API coverage for `bam(...; discrete = true)` fits.
#
# `test_representation.jl` guards the DESIGN side — that the compact blocks
# exist and encode the right matrix. This file guards the other half: that
# every public consumer of a fitted model still works, and agrees with the
# dense fit, when the model was fitted on a compact design.
#
# It exists because the full suite once passed at 8014/8014 with
# `partial_residuals` throwing `DimensionMismatch` on every discrete fit.
# Nothing exercised that path; it surfaced only when an unrelated change
# altered the representation underneath it. Four other regressions in the same
# branch had that shape — a feature silently stopped doing its job while every
# correctness assertion still passed.
#
# On exactly-binnable data the binning is lossless, so a discrete fit and a
# dense fit are the SAME model and any difference is a defect, not an
# approximation. Tolerances are set from measurement, not guessed:
#
#   representation   rel. coef diff   rel. Vp diff   Δsp
#   1-D              6.6e-16          9.5e-16        3.8e-15
#   bs=:re factor    8.9e-16          8.6e-16        3.3e-15
#   factor-by        2.9e-15          2.6e-15        2.8e-15
#   te               7.9e-12          6.8e-11        2.1e-06
#
# `te` is looser for a recorded reason: its smoothing parameter lands 2.1e-6
# away on a flat REML ridge and everything downstream inherits that. It is
# optimiser noise, not representation error — at a FIXED sp the comparison is
# ~1e-15.

@testset "Discrete-fit consumers" begin

    # ── fixtures ────────────────────────────────────────────────────────
    # Few enough distinct covariate values that discretisation rounds nothing.
    _dc_rng = StableRNG(4242)
    _dc_n = 600
    _dcbin(v, m) = round.(v .* (m - 1)) ./ (m - 1)
    _dc_df = DataFrame(
        x1 = _dcbin(rand(_dc_rng, _dc_n), 30),
        x2 = _dcbin(rand(_dc_rng, _dc_n), 30),
        sl = _dcbin(rand(_dc_rng, _dc_n), 15),
        g  = string.(repeat(1:10, inner = div(_dc_n, 10))),
        f  = string.(repeat(1:4, inner = div(_dc_n, 4))),
    )
    _dc_df.y = sin.(2π .* _dc_df.x1) .+ 0.5 .* _dc_df.x2 .+
               0.3 .* randn(_dc_rng, _dc_n)
    _dc_df.cnt = Float64.([rand(_dc_rng, Poisson(exp(0.4 * sin(2π * xi))))
                           for xi in _dc_df.x1])

    _dcf(specs) = GAM.GamFormula(:y, Symbol[], true, GAM.SmoothSpec[specs...])

    # Scale-relative comparison — `maximum(abs, a - b) / maximum(abs, b)`,
    # matching how the reference table above was measured.
    function _dcrel(a, b)
        va = vec(collect(Float64.(a)))
        vb = vec(collect(Float64.(b)))
        length(va) == length(vb) || return Inf
        isempty(va) && return 0.0
        scale = maximum(abs, vb)
        d = maximum(abs, va .- vb)
        return scale > 0 ? d / scale : d
    end

    # NaN-aware elementwise agreement. `k_check`'s `k_index` is NaN for a
    # random effect — in the DENSE fit too — since there is no covariate
    # ordering to compute a neighbour correlation over.
    _dcsame(a, b) = length(a) == length(b) && all(
        (isnan(x) && isnan(y)) || isapprox(x, y; rtol = 1e-10)
        for (x, y) in zip(a, b))

    # `gratia` lists which of the gratia-style consumers this configuration
    # supports AT ALL. The excluded ones fail identically on the dense fit —
    # verified, and asserted below — so they are pre-existing gaps in that
    # surface for smooths carrying a string-valued covariate in `term_vars`
    # (the `re` configurations, where the grid builder cannot make a numeric
    # range out of a factor), not discretisation defects:
    #   :se = smooth_estimates, :pr = partial_residuals, :dv = derivatives
    # `ctol` is the concurvity tolerance, separate because concurvity is
    # computed from `qr(model_matrix(m)).R` and inherits the design's
    # conditioning. For `re factor` that design is numerically singular —
    # cond(X) = 2.2e15, since a 10-level random effect is perfectly confounded
    # with the intercept (its concurvity IS 1.0) — so the ONE-ULP difference
    # between the dense and compact model matrices (2.2e-16) is amplified
    # without a usable bound. `ctol` is `nothing` there, meaning "do not
    # assert cross-fit agreement", because no finite value is defensible:
    # cond(X)*eps is ~0.48, and the same ulp amplifies to 3.8e-3 here,
    # 1.0e-2 under `Pkg.test()`'s --check-bounds=yes, and 7.4e-2 on CI. A
    # tolerance picked from one machine's 3.8e-3 is what turned CI red. Note
    # the divergence is entirely in the *smooth's* concurvity (0.0598 vs
    # 0.0635); the random effect reads exactly 1.0 in both fits, which is the
    # stable fact worth asserting, and is what the branch below checks. The
    # discretisation itself is sound: the model matrices agree to 2.2e-16 and
    # ~40 other assertions in this case compare them at 1e-12.
    _dc_cfgs = [
        ("1-D",       _dcf([GAM.s(:x1; k = 8, bs = :cr)]),
                      1e-12, [:se, :pr, :dv], 1e-10),
        ("1-D x2",    _dcf([GAM.s(:x1; k = 8, bs = :cr),
                            GAM.s(:x2; k = 8, bs = :cr)]),
                      1e-12, [:se, :pr, :dv], 1e-10),
        ("te",        _dcf([GAM.te(:x1, :x2; k = 4, bs = [:cr, :cr])]),
                      1e-7,  [:se, :pr, :dv], 1e-10),
        ("re factor", _dcf([GAM.s(:x1; k = 8, bs = :cr), GAM.s(:g; bs = :re)]),
                      1e-12, Symbol[],        nothing),
        ("re slope",  _dcf([GAM.s(:x1; k = 8, bs = :cr),
                            GAM.s(:sl, :g; bs = :re)]),
                      1e-12, [:pr, :dv],      1e-10),
        # `:se`/`:dv` joined this list when `_make_smooth_grid` learned to
        # put the `by` column on the grid (it built only `term_vars`, so
        # `predict_matrix` threw `FieldError` for every `by=` smooth). The
        # grid is now repeated once per factor level, so these compare 3x the
        # rows of a plain smooth — dense and discrete share that layout.
        ("factor-by", _dcf([GAM.s(:x1; k = 6, bs = :cr, by = :f)]),
                      1e-12, [:se, :pr, :dv], 1e-10),
    ]

    # ── 1. deterministic consumers agree with the dense fit ─────────────
    for (label, gf, tol, gratia, ctol) in _dc_cfgs
        md = bam(gf, _dc_df)
        mq = bam(gf, _dc_df; discrete = true)

        @testset "$label" begin
            # The fit itself.
            @test _dcrel(coef(mq), coef(md)) < tol
            @test _dcrel(fitted(mq), fitted(md)) < tol
            @test _dcrel(GAM.vcov(mq), GAM.vcov(md)) < tol
            @test isapprox(deviance(mq), deviance(md); rtol = tol)
            @test isapprox(mq.edf_total, md.edf_total; rtol = tol)
            @test isapprox(mq.scale, md.scale; rtol = tol)
            @test nobs(mq) == nobs(md)

            # Likelihood summaries. `dof` resolves `edf2`, so this also
            # exercises the lazy smoothing-parameter-uncertainty path.
            @test isapprox(loglikelihood(mq), loglikelihood(md); rtol = tol)
            @test isapprox(aic(mq), aic(md); rtol = tol)
            @test isapprox(bic(mq), bic(md); rtol = tol)
            @test isapprox(dof(mq), dof(md); rtol = tol)
            @test isapprox(dof_residual(mq), dof_residual(md); rtol = tol)
            @test isapprox(conditional_aic(mq), conditional_aic(md); rtol = tol)
            @test isapprox(GAM.deviance_explained(mq),
                GAM.deviance_explained(md); rtol = tol)

            # Every residual type.
            for rt in (:deviance, :pearson, :working, :response)
                @test _dcrel(residuals(mq; type = rt),
                    residuals(md; type = rt)) < tol
            end

            # Prediction: in-sample, on newdata, both scales, with SEs.
            @test _dcrel(predict(mq; type = :link), predict(md; type = :link)) < tol
            @test _dcrel(predict(mq, _dc_df; type = :link),
                predict(md, _dc_df; type = :link)) < tol
            @test _dcrel(predict(mq, _dc_df; type = :response),
                predict(md, _dc_df; type = :response)) < tol
            let (fq, sq) = predict(mq, _dc_df; type = :link, se = true),
                (fd, sd) = predict(md, _dc_df; type = :link, se = true)
                @test _dcrel(fq, fd) < tol
                @test _dcrel(sq, sd) < tol
            end
            # `type = :terms` returns a NamedTuple keyed by term label.
            let tq = predict(mq, _dc_df; type = :terms),
                td = predict(md, _dc_df; type = :terms)
                @test propertynames(tq) == propertynames(td)
                @test all(_dcrel(getproperty(tq, k), getproperty(td, k)) < tol
                          for k in propertynames(td))
            end

            # Model matrices. `m.X` is empty under `discrete` (retain_X
            # defaults false there), so this exercises the reassembly path.
            @test _dcrel(GAM.model_matrix(mq), GAM.model_matrix(md)) < tol
            @test _dcrel(lpmatrix(mq, _dc_df), lpmatrix(md, _dc_df)) < tol

            # Degrees of freedom and the smooth table.
            @test _dcrel(edf(mq), edf(md)) < tol
            @test isapprox(model_edf(mq), model_edf(md); rtol = tol)
            @test _dcrel(edf2(mq), edf2(md)) < tol
            @test _dcrel(ref_df(mq), ref_df(md)) < tol
            @test has_vc(mq) == has_vc(md)
            @test _dcrel(overview(mq).edf, overview(md).edf) < tol

            # Diagnostics. Concurvity is compared ABSOLUTELY: it lives in
            # [0, 1], and for a single-smooth model it is legitimately ~1e-31
            # (nothing to be concurved with), where a relative comparison
            # would divide by noise.
            @test _dcrel(leverage(mq), leverage(md)) < tol
            @test _dcrel(cooksdistance(mq), cooksdistance(md)) < tol
            let cq = concurvity(mq; full = true), cd = concurvity(md; full = true)
                if ctol === nothing
                    # Singular design: cross-fit agreement is not assertable
                    # at any useful tolerance (see the `_dc_cfgs` note). Assert
                    # the part that IS well defined and IS a representation
                    # check — both fits must still see the random effect as
                    # fully confounded with the intercept.
                    @test cq.worst[2] ≈ 1.0 atol = 1e-10
                    @test cd.worst[2] ≈ 1.0 atol = 1e-10
                else
                    @test maximum(abs, cq.worst .- cd.worst) < ctol
                    @test maximum(abs, cq.observed .- cd.observed) < ctol
                    @test maximum(abs, cq.estimate .- cd.estimate) < ctol
                end
            end
            if ctol !== nothing
                @test maximum(abs, concurvity(mq; full = false) .-
                    concurvity(md; full = false)) < ctol
            end

            # k_check seeds by default, so it is reproducible across fits.
            let kq = k_check(mq), kd = k_check(md)
                @test length(kq) == length(kd)
                @test _dcsame([r.edf for r in kq], [r.edf for r in kd])
                @test _dcsame([r.k_index for r in kq], [r.k_index for r in kd])
            end

            # anova_gam. `p_value` and `edf` are what users read and both are
            # stable. The test STATISTIC is not: it goes through a
            # rank-truncated pseudo-inverse whose eigenvalue cut can move on a
            # 1e-15 input perturbation, measured at 5.7e-4 relative for two
            # 1-D smooths and 7.2e-2 for factor-by, while the corresponding
            # p-values agree to 1e-37 and 1e-183. Asserted loosely so gross
            # breakage still fails, with the instability recorded rather than
            # hidden behind a tight tolerance that would have to be waived.
            let aq = anova_gam(mq).smooth_table, ad = anova_gam(md).smooth_table
                @test aq.label == ad.label
                @test _dcrel(aq.edf, ad.edf) < tol
                @test maximum(abs, aq.p_value .- ad.p_value) < 1e-8
                @test _dcrel(aq.statistic, ad.statistic) < 0.2
            end

            # gratia-style surface, where this configuration supports it.
            if :se in gratia
                let sq = smooth_estimates(mq; n = 20),
                    sd = smooth_estimates(md; n = 20)
                    @test _dcrel(sq.estimate, sd.estimate) < tol
                    @test _dcrel(sq.se, sd.se) < tol
                end
            end
            if :pr in gratia
                # The consumer that was silently broken on discrete fits.
                let pq = GAM.partial_residuals(mq), pd = GAM.partial_residuals(md)
                    @test length(pq.residual) == length(pd.residual)
                    @test _dcrel(pq.residual, pd.residual) < tol
                end
            end
            if :dv in gratia
                let dq = GAM.derivatives(mq; n = 20),
                    dd = GAM.derivatives(md; n = 20)
                    # Finite-difference based, so a few orders above the
                    # coefficient floor: 8.3e-11 where coef agrees at 1e-15.
                    @test _dcrel(dq.derivative, dd.derivative) < max(tol, 1e-9)
                end
            end

            # Whatever this configuration does NOT support must fail the same
            # way dense does. Asserting this keeps an unsupported combination
            # a documented decision rather than an accident, and makes the
            # test fail loudly if someone implements it.
            for (sym, f) in ((:se, m -> smooth_estimates(m; n = 20)),
                             (:pr, m -> GAM.partial_residuals(m)),
                             (:dv, m -> GAM.derivatives(m; n = 20)))
                sym in gratia && continue
                dense_threw = try (f(md); false) catch; true end
                disc_threw  = try (f(mq); false) catch; true end
                @test dense_threw            # pre-existing gap, not ours
                @test disc_threw == dense_threw
            end

            # `show` must not throw on a model with no materialised X.
            @test length(sprint(show, mq)) > 0
        end
    end

    # ── 2. sampling consumers ───────────────────────────────────────────
    #
    # NOT asserted elementwise equal to the dense fit, deliberately. `Vp`
    # differs in its last bits (9.5e-16 relative), so its Cholesky factor
    # differs in the last bits too — and for the same normal draws `z`, a
    # different factor yields a different, equally valid sample. Measured
    # elementwise gap: 0.02–0.23. Asserting equality would be asserting that
    # two correct implementations produce identical randomness.
    #
    # What IS assertable: reproducible under a seed, correctly shaped, finite,
    # and centred on the same posterior.
    @testset "sampling consumers" begin
        gf = _dcf([GAM.s(:x1; k = 8, bs = :cr)])
        md = bam(gf, _dc_df)
        mq = bam(gf, _dc_df; discrete = true)

        for (nm, f) in (
            ("posterior_samples", (m, s) -> GAM.posterior_samples(m; n = 200, seed = s)),
            ("fitted_samples",    (m, s) -> GAM.fitted_samples(m; n = 200, seed = s)),
            ("predicted_samples", (m, s) -> GAM.predicted_samples(m; n = 200, seed = s)),
        )
            a = f(mq, 11)
            @test all(isfinite, a)
            @test f(mq, 11) == a                 # reproducible under a seed
            @test f(mq, 12) != a                 # the seed is actually threaded
            @test size(a) == size(f(md, 11))     # same shape as dense

            # Centred on the same posterior: column means within Monte Carlo
            # error of the dense fit's.
            b = f(md, 11)
            ma = vec(mean(a; dims = 1))
            mb = vec(mean(b; dims = 1))
            sb = vec(std(b; dims = 1)) ./ sqrt(size(b, 1))
            @test all(abs.(ma .- mb) .<= 6 .* sb .+ 1e-8)
        end

        let sq = GAM.smooth_samples(mq; n = 50, seed = 11)
            @test sq !== nothing
        end
        let aq = GAM.appraise(mq; n_sim = 5, seed = 11),
            ad = GAM.appraise(md; n_sim = 5, seed = 11)
            @test all(isfinite, aq.residuals_deviance)
            @test _dcrel(aq.residuals_deviance, ad.residuals_deviance) < 1e-12
            @test _dcrel(aq.fitted, ad.fitted) < 1e-12
        end
    end

    # ── 3. the X-free state itself ──────────────────────────────────────
    @testset "consumers with no materialised X" begin
        gf = _dcf([GAM.s(:x1; k = 8, bs = :cr), GAM.s(:x2; k = 8, bs = :cr)])
        md = bam(gf, _dc_df)
        mq = bam(gf, _dc_df; discrete = true)                 # retain_X = false
        mk = bam(gf, _dc_df; discrete = true, retain_X = true)

        # Under `discrete` the model matrix is not retained at all.
        @test size(mq.X, 1) == 0
        @test !has_model_matrix(mq)
        @test has_model_matrix(mk)
        @test size(mk.X, 1) == _dc_n

        # And it reassembles bitwise from the per-smooth blocks.
        @test GAM.model_matrix(mq) == mk.X
        @test GAM.model_matrix(mq) == GAM.model_matrix(mk)

        # Retaining X must not change the fit.
        @test coef(mq) == coef(mk)
        @test fitted(mq) == fitted(mk)

        # Consumers needing the whole matrix work in the X-free state.
        @test all(isfinite, concurvity(mq; full = true).worst)
        @test all(isfinite, leverage(mq))
        @test all(isfinite, GAM.partial_residuals(mq).residual)
        @test all(isfinite, lpmatrix(mq, _dc_df))
        @test all(isfinite, GAM.model_matrix(mq))
    end

    # ── 4. a non-Gaussian family ────────────────────────────────────────
    # Exercises the IRLS path, where accumulation runs every outer iteration
    # rather than once, and `rootogram`, which is Poisson-only.
    @testset "Poisson consumers" begin
        gf = GAM.GamFormula(:cnt, Symbol[], true,
            GAM.SmoothSpec[GAM.s(:x1; k = 8, bs = :cr)])
        md = bam(gf, _dc_df; family = Poisson())
        mq = bam(gf, _dc_df; family = Poisson(), discrete = true)
        tol = 1e-8   # IRLS convergence, not representation, sets this floor

        @test _dcrel(coef(mq), coef(md)) < tol
        @test _dcrel(fitted(mq), fitted(md)) < tol
        @test isapprox(deviance(mq), deviance(md); rtol = tol)
        @test isapprox(aic(mq), aic(md); rtol = tol)
        @test _dcrel(residuals(mq; type = :deviance),
            residuals(md; type = :deviance)) < tol
        @test _dcrel(predict(mq, _dc_df; type = :response),
            predict(md, _dc_df; type = :response)) < tol
        let rq = GAM.rootogram(mq), rd = GAM.rootogram(md)
            @test _dcrel(rq.expected, rd.expected) < tol
        end
    end
end
