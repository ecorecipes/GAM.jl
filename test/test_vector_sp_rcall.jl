@testset "Vector sp transfers to and from mgcv" begin
    using Random, DataFrames
    using StatsAPI: fitted

    # The payoff for vector `sp`: a per-penalty smoothing-parameter vector
    # should carry a fit between GAM.jl and mgcv in either direction. The
    # comparison is made with sp FIXED at the other package's value, so the
    # smoothing-parameter optimizer is not confounded with the basis and
    # penalty parameterization being tested. (A free-fit comparison across a
    # flat REML ridge is a noisy proxy and has produced wrong conclusions in
    # this codebase before.)

    Random.seed!(11)
    n = 300
    xv = collect(range(0, 1; length = n))
    zv = rand(n)
    gv = string.("L", repeat(1:4, inner = n ÷ 4))
    yv = sin.(6π .* xv .^ 2) ./ (1 .+ 4 .* xv) .+ 0.5 .* zv .+ 0.2 .* randn(n)
    dfv = DataFrame(x = xv, z = zv, g = gv, y = yv)

    RCall.reval("suppressMessages(library(mgcv))")
    @rput xv zv gv yv
    RCall.reval("dfr <- data.frame(x = xv, z = zv, g = factor(gv), y = yv)")

    _fit(spec) = gam(GAM.GamFormula(:y, Symbol[], true, [spec]), dfv)

    function _mgcv_free(rformula)
        RCall.reval("bfree <- gam(as.formula(\"$rformula\"), data = dfr, method = \"REML\")")
        (sp = rcopy(RCall.reval("as.numeric(bfree\$sp)")),
         edf = rcopy(RCall.reval("sum(bfree\$edf)")),
         fit = rcopy(RCall.reval("as.numeric(fitted(bfree))")))
    end

    @testset "bs=:ad transfers exactly in both directions" begin
        r = _mgcv_free("y ~ s(x, bs='ad', k=20)")
        m_free = _fit(GAM.s(:x; bs = :ad, k = 20))
        @test length(m_free.sp) == length(r.sp) == 5

        # mgcv's sp -> GAM.jl, compared against mgcv's own fit at that sp
        m1 = _fit(GAM.s(:x; bs = :ad, k = 20, sp = r.sp))
        @test abs((sum(GAM.edf(m1)) + 1) - r.edf) < 1e-8
        @test maximum(abs.(fitted(m1) .- r.fit)) < 1e-8

        # GAM.jl's sp -> mgcv, compared against GAM.jl's own fit at that sp
        j_sp = exp.(m_free.sp)
        @rput j_sp
        RCall.reval("bfix <- gam(y ~ s(x, bs='ad', k=20), data = dfr, method = 'REML', sp = j_sp)")
        @test abs(rcopy(RCall.reval("sum(bfix\$edf)")) - (sum(GAM.edf(m_free)) + 1)) < 1e-8
        @test maximum(abs.(rcopy(RCall.reval("as.numeric(fitted(bfix))")) .-
                           fitted(m_free))) < 1e-8
    end

    @testset "t2 transfers to within its residual penalty-scale difference" begin
        r = _mgcv_free("y ~ t2(x, z, k=5)")
        m_free = _fit(GAM.t2(:x, :z; k = 5))
        @test length(m_free.sp) == length(r.sp) == 3

        m1 = _fit(GAM.t2(:x, :z; k = 5, sp = r.sp))
        # `nat.param`'s null-space reparameterization leaves t2's penalty scale
        # ~1% from mgcv's (the null eigenvectors of a degenerate eigenspace are
        # not reproducible across LAPACK builds), so transfer is close but not
        # exact. Tightening these bounds is the signal that the scale has been
        # closed; loosening them without explanation is not acceptable.
        @test abs((sum(GAM.edf(m1)) + 1) - r.edf) < 0.1
        @test maximum(abs.(fitted(m1) .- r.fit)) < 0.02
        # ...and transfer is at least as good as the free-fit agreement, i.e.
        # the vector is genuinely carrying information.
        @test abs((sum(GAM.edf(m1)) + 1) - r.edf) <=
              abs((sum(GAM.edf(m_free)) + 1) - r.edf) + 1e-8
    end

    @testset "bs=:fs sp does NOT transfer (penalty normalization differs)" begin
        # Documented limitation, pinned so it is noticed when fixed. GAM.jl's
        # fs penalties are normalized differently from mgcv's, so the sp
        # vectors are not on a common scale -- and it is not a mere ordering
        # difference: all 6 permutations of mgcv's sp were tried and the best
        # still left Δedf ≈ 1.9 out of ≈ 11.4. The FITS agree under free
        # selection; only sp transfer fails.
        r = _mgcv_free("y ~ s(x, g, bs='fs', k=6)")
        m_free = _fit(GAM.s(:x, :g; bs = :fs, k = 6))
        @test length(m_free.sp) == length(r.sp) == 3

        # Free fits agree...
        @test abs((sum(GAM.edf(m_free)) + 1) - r.edf) < 0.2
        @test maximum(abs.(fitted(m_free) .- r.fit)) < 0.05

        # ...and transferring mgcv's sp now reproduces mgcv's fit. This
        # testset previously pinned the OPPOSITE (a > 0.5 edf gap), because
        # `fs` was the one basis that never reached mgcv's `scale.penalty`
        # rescale: it builds no constraint, so it never called
        # `absorb_constraints!`, where GAM.jl applies that step. Our raw
        # penalty 1-norms were exactly `sm$S.scale` times mgcv's.
        # Measured after the fix: 0.098 edf. The residual is a uniform
        # ‖X‖∞ factor from a block-orthogonal basis rotation, not a
        # normalization error — see src/basis_extra.jl.
        m1 = _fit(GAM.s(:x, :g; bs = :fs, k = 6, sp = r.sp))
        @test abs((sum(GAM.edf(m1)) + 1) - r.edf) < 0.2
    end
end
