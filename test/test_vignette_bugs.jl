# Regression tests for bugs surfaced by writing the vignettes.
#
# Each of these was found by exercising a *combination* of features that the
# rest of the suite covers only separately, which is why they survived a full
# green suite. They are grouped here rather than scattered so the provenance
# stays visible.

@testset "Vignette-surfaced bugs" begin

    # ------------------------------------------------------------------
    # 1. Extended-family extra parameters must survive the null-deviance fit.
    #
    # `_null_deviance` fits an intercept-plus-offset model when an offset is
    # present. It used to pass the caller's own `family` object, and
    # `pirls_extended` re-estimates the family's extra parameter IN PLACE, so
    # the intercept-only fit silently overwrote the extra parameter the real
    # model had just converged to. Only offsets triggered it: the offset-free
    # branch returns a closed-form null deviance without fitting anything.
    #
    # A CONSTANT offset is the clean probe — it is absorbable into the
    # intercept, so it cannot change the fitted means, and any change in the
    # reported extra parameter is therefore spurious. Before the fix, NB theta
    # went 2.094 -> 1.139 on this data (truth 2.0).
    # ------------------------------------------------------------------
    @testset "extra parameter is offset-invariant" begin
        _rng = StableRNG(20260831)
        n = 400
        θ_true = 2.0
        x = rand(_rng, n)
        μ = exp.(2.5 .+ 0.9 .* sin.(2π .* x))
        y_nb = [rand(_rng, NegativeBinomial(θ_true, θ_true / (θ_true + m)))
                for m in μ]
        df = DataFrame(x = x, y = Float64.(y_nb))
        f = GAM.@formulak(y ~ s(x, k = 15, bs = :cr))

        m_plain = gam(f, df; family = NegBinFamily(), link = LogLink())
        m_off = gam(f, df; family = NegBinFamily(), link = LogLink(),
            offset = fill(1.0, n))

        # The headline assertion: a constant offset must not move theta.
        # Tolerance is 1e-3 relative — far tighter than the 45% error the bug
        # produced, and loose enough for the alternating theta/beta scheme,
        # which lands within ~2e-5 relative here.
        @test isapprox(m_plain.family.theta, m_off.family.theta; rtol = 1e-3)
        # And theta is still near the simulation truth, so the test fails if a
        # future change makes BOTH wrong in the same way.
        @test isapprox(m_plain.family.theta, θ_true; rtol = 0.15)
        @test isapprox(m_off.family.theta, θ_true; rtol = 0.15)

        # With theta pinned, the fits themselves are offset-invariant to
        # near machine precision — this separates "the fit handles offsets"
        # from "the reported extra parameter survives".
        θ̂ = m_plain.family.theta
        m2 = gam(f, df;
            family = NegBinFamily(theta = θ̂, estimate_theta = false),
            link = LogLink())
        m3 = gam(f, df;
            family = NegBinFamily(theta = θ̂, estimate_theta = false),
            link = LogLink(), offset = fill(1.0, n))
        # Measured 7.0e-6 here; 1e-4 leaves room for optimizer wobble while
        # staying three orders below the 0.37% that a mis-estimated theta
        # induces through the smoothing-parameter selection.
        @test maximum(abs.(fitted(m2) .- fitted(m3))) /
              maximum(fitted(m2)) < 1e-4

        # Same defect class, other families: the null fit mutated whichever
        # extra parameter the family carries, so Tweedie p and Beta phi were
        # equally exposed even though only NB was reported.
        y_tw = [rand(_rng, Poisson(m / 3)) * 3.0 for m in μ]
        df_tw = DataFrame(x = x, y = y_tw)
        t_plain = gam(f, df_tw;
            family = TweedieFamily(p = 1.5, estimate_p = true), link = LogLink())
        t_off = gam(f, df_tw;
            family = TweedieFamily(p = 1.5, estimate_p = true), link = LogLink(),
            offset = fill(1.0, n))
        @test isapprox(t_plain.family.p, t_off.family.p; rtol = 1e-3)

        u = rand(_rng, n)
        ηb = 0.4 .* sin.(2π .* x)
        μb = 1.0 ./ (1.0 .+ exp.(-ηb))
        φ_true = 12.0
        y_b = [clamp(rand(_rng, Beta(m * φ_true, (1 - m) * φ_true)),
                     1e-6, 1 - 1e-6) for m in μb]
        df_b = DataFrame(x = x, y = y_b)
        b_plain = gam(f, df_b; family = BetaFamily(), link = LogitLink())
        b_off = gam(f, df_b; family = BetaFamily(), link = LogitLink(),
            offset = fill(0.5, n))
        # Beta's phi is the least well-determined of the three, so the band is
        # wider — but it is measured, not guessed: on this data the bug gave a
        # 19.9% discrepancy and the fix leaves 1.2%, so 3% fails loudly if the
        # null-deviance fit ever starts mutating the family again.
        @test isapprox(b_plain.family.phi, b_off.family.phi; rtol = 0.03)
    end

    # ------------------------------------------------------------------
    # 2. Shrinkage bases must be able to shrink a term all the way out.
    #
    # `bs=:ts`/`:cs` penalize the null space too, so a useless term can be
    # driven to edf ~ 0 — but only if the optimizer can reach the required
    # lambda. The bound was log lambda <= 15, and `:cs` cascades its null-space
    # eigenvalues down to shrink^2, needing ~100x more lambda than `:ts` for
    # the same shrinkage. Both pinned exactly at the bound; mgcv's own optima
    # here are 16.8 (`ts`) and 22.61 (`cs`), both above it.
    # ------------------------------------------------------------------
    @testset "shrinkage bases can shrink a term out" begin
        _rng = StableRNG(20260830)
        n = 400
        x = rand(_rng, n)
        z = rand(_rng, n)              # irrelevant: enters no part of the DGP
        y = sin.(2π .* x) .+ 0.4 .* randn(_rng, n)
        df = DataFrame(x = x, z = z, y = y)

        m_ts = gam(GAM.@formulak(y ~ s(x, k = 15, bs = :ts) +
                                     s(z, k = 15, bs = :ts)), df)
        m_cs = gam(GAM.@formulak(y ~ s(x, k = 15, bs = :cs) +
                                     s(z, k = 15, bs = :cs)), df)

        # The irrelevant term is shrunk essentially to nothing. Before the
        # bound was raised, `:cs` stalled at edf = 0.286 — 5000x larger than
        # the 5e-5 it reaches now, and visibly a term still in the model.
        @test edf(m_ts)[2] < 1e-3
        @test edf(m_cs)[2] < 1e-3
        # ...while the real term keeps its structure, so the test fails if a
        # future change simply shrinks everything.
        @test edf(m_ts)[1] > 5.0
        @test edf(m_cs)[1] > 5.0

        # Non-shrinkage bases are deliberately unaffected: their penalties
        # leave the null space unpenalized, so the linear trend in an
        # irrelevant covariate survives however large lambda grows. This is
        # the contrast that makes the shrinkage bases worth having, and it
        # guards against "fixing" shrinkage by over-penalizing everything.
        m_tp = gam(GAM.@formulak(y ~ s(x, k = 15, bs = :tp) +
                                     s(z, k = 15, bs = :tp)), df)
        @test edf(m_tp)[2] > 0.9

        # The bound itself is shared by every optimizer, so it cannot drift
        # between methods and leave a term shrinkable under one and not
        # another.
        @test GAM.LOG_SP_BOUND >= 25.0
    end

    # ------------------------------------------------------------------
    # 3. Grid-based gratia entry points must work on `by=` smooths.
    #
    # `_make_smooth_grid` built only the smooth's own term_vars, so
    # `predict_matrix` — which reads the by column to reapply the by
    # transform — died with `FieldError: type NamedTuple has no field ...`.
    # That took out smooth_estimates, derivatives, partial_residuals and
    # data_slice for BOTH numeric and factor `by`, i.e. every grid-based way
    # of inspecting a headline feature.
    # ------------------------------------------------------------------
    @testset "by= smooths work on the gratia grid" begin
        _rng = StableRNG(4242)
        n = 300
        x = rand(_rng, n)
        z = randn(_rng, n)
        g = repeat(["a", "b", "c"], inner = n ÷ 3)

        # --- numeric by: contribution is z * f(x), so the grid fixes z = 1
        #     and the returned curve IS f(x). Verify against the known truth.
        y_num = z .* sin.(2π .* x) .+ 0.2 .* randn(_rng, n)
        df_num = DataFrame(x = x, z = z, y = y_num)
        m_num = gam(GAM.@formulak(y ~ s(x, k = 10, by = z)), df_num)

        se_num = smooth_estimates(m_num; select = 1)
        @test length(se_num.estimate) == 100
        @test all(isnothing, se_num.by_level)     # numeric by has no levels
        xs = se_num.covariates[:x]
        @test cor(se_num.estimate, sin.(2π .* xs)) > 0.99

        # --- factor by: one curve per level, grid repeated per level
        amp = Dict("a" => 2.0, "b" => 1.0, "c" => 0.3)
        y_fac = [amp[g[i]] * sin(2π * x[i]) for i in 1:n] .+
                0.2 .* randn(_rng, n)
        df_fac = DataFrame(x = x, g = g, y = y_fac)
        m_fac = gam(GAM.@formulak(y ~ g + s(x, k = 10, by = g)), df_fac)

        se_fac = smooth_estimates(m_fac; select = 1)
        levs = sort(unique(String.(se_fac.by_level)))
        @test levs == ["a", "b", "c"]
        @test length(se_fac.estimate) == 300        # 3 levels x 100 points

        # Each level's curve must recover ITS OWN amplitude — this is what
        # makes the level labelling load-bearing rather than cosmetic. If the
        # rows were mislabelled or the levels concatenated wrongly, the
        # ordering below breaks.
        rng_of(l) = begin
            msk = se_fac.by_level .== l
            maximum(se_fac.estimate[msk]) - minimum(se_fac.estimate[msk])
        end
        @test rng_of("a") > rng_of("b") > rng_of("c")

        # The other three grid entry points used to die the same way.
        @test derivatives(m_fac; select = 1) isa GAM.DerivativeEstimates
        @test partial_residuals(m_fac) isa GAM.PartialResiduals
        @test data_slice(m_fac; var = :x, n = 20) isa NamedTuple
        @test derivatives(m_num; select = 1) isa GAM.DerivativeEstimates
        @test data_slice(m_num; var = :x, n = 20) isa NamedTuple
    end

    # ------------------------------------------------------------------
    # 4. `knots=` must reach the basis, so a cyclic period can be set.
    #
    # `setup_gam` called `smooth_construct(spec, t)` with no knots, so mgcv's
    # `knots = list(week = c(0, 52))` was unreachable and a `:cc` smooth always
    # took its period from the OBSERVED range. On weeks 0..51 of an exactly
    # periodic signal that wrapped over a 51-week year: f(0) != f(52).
    # ------------------------------------------------------------------
    @testset "knots= sets the cyclic period" begin
        _rng = StableRNG(7)
        wk = repeat(0:51, outer = 8)
        n = length(wk)
        # Exactly periodic with period 52, so week 0 and week 52 are the SAME
        # point of the cycle and a correct cyclic fit must agree there.
        y = 2.0 .+ sin.(2π .* wk ./ 52) .+ 0.2 .* randn(_rng, n)
        df = DataFrame(week = Float64.(wk), y = y)
        nd = DataFrame(week = [0.0, 52.0])

        m_auto = gam(GAM.@formulak(y ~ s(week, k = 10, bs = :cc)), df)
        p_auto = predict(m_auto, nd)
        # The bug: period taken from the data range (0..51), so the ends miss.
        @test abs(p_auto[1] - p_auto[2]) > 1e-3

        m_kn = gam(GAM.@formulak(y ~ s(week, k = 10, bs = :cc)), df;
            knots = Dict(:week => [0.0, 52.0]))
        p_kn = predict(m_kn, nd)
        # With the period set, the wrap is exact by construction.
        @test abs(p_kn[1] - p_kn[2]) < 1e-10

        # Two knots on a cyclic basis are the PERIOD ENDPOINTS (mgcv's
        # convention) and the interior is filled in — the basis keeps its
        # requested dimension rather than collapsing to 2 knots.
        @test length(coef(m_kn)) == length(coef(m_auto))

        # Validation of the mapping itself.
        @test_throws ArgumentError gam(
            GAM.@formulak(y ~ s(week, k = 10, bs = :cc)), df;
            knots = Dict(:week => [52.0, 0.0]))     # not sorted
        @test_throws ArgumentError gam(
            GAM.@formulak(y ~ s(week, k = 10, bs = :cc)), df;
            knots = Dict(:week => [26.0]))          # too few
    end

    # ------------------------------------------------------------------
    # 5. A vector `m` gets an actionable error, not a bare MethodError.
    #
    # mgcv writes some orders as vectors (`bs="bs"` -> m = c(3, 2)), so the
    # form arrives from anyone porting R code. It used to die with
    # `MethodError: no method matching Int64(::Vector{Int64})`.
    # ------------------------------------------------------------------
    @testset "vector m is rejected with an explanation" begin
        for f in (GAM.s, GAM.te, GAM.ti, GAM.t2)
            err = try
                f === GAM.s ? f(:x; bs = :bs, m = [3, 2]) :
                              f(:x, :z; bs = :cr, m = [3, 2])
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            # The message must name the convention, not just reject.
            @test occursin("scalar", err.msg)
            @test occursin("m = 2", err.msg)
        end
        # Scalar m keeps working, and `sp` still accepts a vector — the
        # asymmetry the message calls out.
        @test GAM.s(:x; bs = :bs, m = 2).m == 2
        @test GAM.s(:x; bs = :ad, k = 10, sp = [1.0, 2.0]).sp !== nothing
    end

end
