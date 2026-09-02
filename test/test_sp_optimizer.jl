# Smoothing-parameter optimizer: `:efs` (default) vs `:newton`.
#
# `:newton` had no test coverage at all before this file — a single incidental
# mention in test_scasm.jl's error-message assertion. It also crashed outright
# on roughly a third of model classes, which no test caught.
#
# What is pinned here:
#
#   1. `:efs` remains the default. Newton can still degrade to EFS (a
#      non-finite Hessian on some `bs=:re` fits), so a Newton default would
#      still mean the optimizer changes strategy with model structure.
#   2. Where Newton runs, it reaches a criterion at least as good as EFS's, and
#      on the shrinkage bases it is materially better — those need very large
#      log-lambda, which is where EFS's fixed point is least accurate.
#   3. Newton now RUNS on multi-penalty blocks: te/ti/t2, bs=:ad, bs=:fs and
#      `select=true`. It used to throw `MethodError: no method matching
#      Float64(::ForwardDiff.Dual...)` from `_stable_penalty_factor` (reml.jl)
#      and fall back. The reparameterization is still not differentiable — see
#      test_penalty_det.jl — but its log-determinant supplies analytic first
#      and second derivatives for ForwardDiff to chain through.
#   4. Where Newton cannot run it still FALLS BACK to EFS with a warning
#      rather than throwing (`bs=:re`, non-finite Hessian).
#
# The criterion is what the optimizer minimizes, so it — not edf — is the
# yardstick used below. Lower is better.

@testset "Smoothing-parameter optimizer (:efs vs :newton)" begin

    _opt_rng = StableRNG(20260901)
    _opt_n = 250
    _opt_df = DataFrame(
        x = sort(rand(_opt_rng, _opt_n)),
        z = rand(_opt_rng, _opt_n),
        g = string.(rand(_opt_rng, 1:4, _opt_n)),
    )
    _opt_df.y = sin.(2π .* _opt_df.x) .+ 0.6 .* _opt_df.z .+
                0.3 .* randn(_opt_rng, _opt_n)

    fit_with(form, opt; kw...) = gam(form, _opt_df;
        control = gam_control(sp_optimizer = opt), kw...)

    @testset "default is :efs" begin
        # Newton is not the default and must not become one silently: it still
        # degrades to EFS on models whose Hessian comes back non-finite, so a
        # Newton default would optimize two terms of one model by different
        # methods in the same run.
        @test gam_control().sp_optimizer === :efs
        @test_throws ArgumentError gam_control(sp_optimizer = :bogus)
    end

    @testset "Newton runs on single-penalty smooths and matches EFS" begin
        # Written out rather than looped: `@formulak` is a macro and does not
        # take an interpolated basis symbol.
        forms = [GAM.@formulak(y ~ s(x, k = 12, bs = :cr)),
                 GAM.@formulak(y ~ s(x, k = 12, bs = :tp)),
                 GAM.@formulak(y ~ s(x, k = 12, bs = :ps))]
        for f in forms
            m_e = fit_with(f, :efs)
            m_n = fit_with(f, :newton)
            @test m_e.converged
            @test m_n.converged
            # Same optimum to well within any modelling tolerance.
            @test isapprox(sp_criterion(m_e), sp_criterion(m_n); rtol = 1e-5)
            @test isapprox(sum(edf(m_e)), sum(edf(m_n)); atol = 1e-2)
        end
    end

    @testset "Newton is better on shrinkage bases" begin
        # `:ts`/`:cs` penalise the null space, so dropping a term needs a very
        # large log-lambda. EFS's fixed point stops short of it; Newton does
        # not. Measured: REML 126.2417 (efs) vs 126.2084 (newton), a real
        # 3.3e-2 improvement in the objective, not numerical noise.
        f = GAM.@formulak(y ~ s(x, k = 15, bs = :ts))
        c_e = sp_criterion(fit_with(f, :efs))
        c_n = sp_criterion(fit_with(f, :newton))
        @test c_n <= c_e + 1e-8          # never worse
        @test c_e - c_n > 1e-3           # and materially better here
    end

    @testset "Newton runs on multi-penalty blocks" begin
        # Each of these took the reparameterization path that used to throw
        # `MethodError(Float64, ::ForwardDiff.Dual)`. The contract now is that
        # Newton actually optimizes them — no warning, no EFS step — and lands
        # on a criterion no worse than EFS's.
        cases = Any[
            ("te",     GAM.@formulak(y ~ te(x, z, k = 5, bs = [:cr, :cr])), NamedTuple()),
            ("ti",     GAM.@formulak(y ~ s(x) + s(z) + ti(x, z, k = 5)),    NamedTuple()),
            ("t2",     GAM.@formulak(y ~ t2(x, z, k = 5, bs = [:cr, :cr])), NamedTuple()),
            ("ad",     GAM.@formulak(y ~ s(x, k = 20, bs = :ad)),           NamedTuple()),
            ("fs",     GAM.@formulak(y ~ s(x, g, k = 8, bs = :fs)),         NamedTuple()),
            ("select", GAM.@formulak(y ~ s(x, k = 12, bs = :cr)),           (select = true,)),
        ]
        for (label, f, kw) in cases
            m_n = @test_logs(min_level = Base.CoreLogging.Warn,
                             fit_with(f, :newton; kw...))
            @test m_n isa GamModel
            @test all(isfinite, coef(m_n))
            m_e = fit_with(f, :efs; kw...)
            # Newton is a genuine optimizer here, not a relabelled EFS step:
            # it must never do worse, and on these it does slightly better
            # (measured 1e-4 to 3e-3 in the criterion).
            @test sp_criterion(m_n) <= sp_criterion(m_e) + 1e-8
        end
    end

    @testset "Newton still falls back instead of throwing" begin
        # Not every model is safe: a `bs=:re` term can drive a smoothing
        # parameter far enough that the ForwardDiff Hessian comes back
        # non-finite. The fit must still succeed, as an EFS fit.
        f = GAM.@formulak(y ~ s(x, k = 10) + s(g, bs = :re))
        m_n = fit_with(f, :newton)
        @test m_n isa GamModel
        @test all(isfinite, coef(m_n))
        @test isapprox(sp_criterion(m_n), sp_criterion(fit_with(f, :efs));
                       rtol = 1e-6)
    end

    @testset "Newton respects fixed smoothing parameters" begin
        # `sp=` pins a term; the Newton step zeroes those coordinates. If that
        # guard regressed, the pinned sp would drift.
        f = GAM.@formulak(y ~ s(x, k = 12, bs = :cr, sp = 0.5))
        m = fit_with(f, :newton)
        @test isapprox(exp(m.sp[1]), 0.5; rtol = 1e-8)
    end
end
