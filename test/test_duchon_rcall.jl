using Test
using GAM
using RCall
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, deviance

# Direct comparison of bs=:ds against mgcv's bs="ds".
#
# GAM.jl's Duchon construction is a port of mgcv's
# `smooth.construct.ds.smooth.spec` (R/smooth.r) together with `DuchonE` and
# `DuchonT`, so the bases agree elementwise EXCEPT for the signs of individual
# basis columns. That sign difference is not a defect and not observable:
# mgcv's ds constructor rotates with R's `qr()` (LINPACK dqrdc2) where Julia
# uses LAPACK, and the two return Householder factors whose columns may differ
# in sign. Applying a diagonal sign matrix D to the columns sends X -> X·D and
# S -> D·S·D, which leaves fitted values, EDF, penalty eigenvalues, and both
# `opnorm(X, Inf)` and `opnorm(S, 1)` unchanged — so even the smoothing
# parameter stays on mgcv's scale. Hence the elementwise assertions below
# compare absolute values, and everything observable is compared directly.
#
# This differs from bs=:tp, where mgcv's QT factorization DOES have to be
# reproduced exactly (see `_mgcv_qt`): there the column-RMS rescaling happens
# after the rotation, so a different basis changes the column scales and puts
# `sp` on an incompatible scale. The ds constructor applies no such rescaling.

R"suppressMessages(library(mgcv))"

@testset "Duchon splines (bs=:ds) — mgcv comparison" begin

    rng = StableRNG(4242)
    n = 200
    x = sort(rand(rng, n))
    y = sin.(2π .* x) .+ 0.3 .* randn(rng, n)
    df = DataFrame(x = x, y = y)

    @rput x
    @rput y
    R"d1 <- data.frame(x = x, y = y)"

    # ------------------------------------------------------------------
    # The basis itself, before constraint absorption.
    # ------------------------------------------------------------------
    @testset "basis and penalty match elementwise (up to column sign)" begin
        R"""
        sm <- smoothCon(s(x, bs = "ds", k = 12), data = d1,
                        absorb.cons = FALSE, scale.penalty = FALSE)[[1]]
        X_R <- sm$X; S_R <- sm$S[[1]]
        bsdim_R <- sm$bs.dim; null_R <- sm$null.space.dim; rank_R <- sm$rank
        """
        X_R = rcopy(R"X_R")
        S_R = rcopy(R"S_R")

        sm = GAM._construct_duchon(GAM.s(:x; k = 12, bs = :ds), (x = x,),
                                   nothing; absorb_cons = false)

        @test size(sm.X) == size(X_R)
        @test size(sm.S[1]) == size(S_R)
        @test rcopy(R"bsdim_R") == 12
        @test sm.null_dim == rcopy(R"null_R")
        @test sm.rank == rcopy(R"rank_R")

        # Elementwise agreement up to per-column sign.
        #
        # The 1e-6 bound is mgcv's error, not ours, and is measured rather
        # than guessed. mgcv builds the basis from `slanczos(E, k, -1)`, an
        # iterative Lanczos solve whose tolerance is `.Machine$double.eps^0.5`
        # = 1.49e-8. On this problem its eigenVALUES come back accurate to
        # 1.8e-14 but its eigenVECTORS only to 4.4e-7 — the usual Lanczos
        # behaviour, where values converge much faster than vectors. GAM.jl
        # uses a dense symmetric `eigen`, so it is the more accurate side; the
        # residual elementwise difference here is 2.7e-9 (X) and 4.7e-9 (S).
        # A smaller problem (n = 50, k = 10) agrees to 5.5e-14, because the
        # spectrum is better separated and Lanczos converges further.
        @test maximum(abs.(abs.(sm.X) .- abs.(X_R))) < 1e-6
        @test maximum(abs.(abs.(sm.S[1]) .- abs.(S_R))) < 1e-6

        # Eigenvalues are both sign-invariant AND the part slanczos gets
        # essentially exactly, so they are held far tighter.
        @test maximum(abs.(sort(eigvals(Symmetric(sm.S[1]))) .-
                           sort(eigvals(Symmetric(S_R))))) < 1e-10

        # And the sign pattern really is the only difference: recovering it
        # from one column makes the two bases identical.
        nb = size(sm.X, 2)
        sgn = [sign(sm.X[argmax(abs.(view(sm.X, :, j))), j] *
                    X_R[argmax(abs.(view(sm.X, :, j))), j]) for j in 1:nb]
        D = Diagonal(sgn)
        @test maximum(abs.(sm.X * D .- X_R)) < 1e-6
        @test maximum(abs.(D * sm.S[1] * D .- S_R)) < 1e-6
    end

    # ------------------------------------------------------------------
    # `S.scale` — the quantity that gave the old TPRS-alias stub away
    # (it reported the thin-plate value, ~19.6/21.8, against mgcv's ~0.4).
    # mgcv stores the reciprocal of the factor GAM.jl multiplies by.
    # ------------------------------------------------------------------
    @testset "penalty scaling matches mgcv's S.scale" begin
        R"""
        smsc <- smoothCon(s(x, bs = "ds", k = 12), data = d1)[[1]]
        Sscale_R <- smsc$S.scale
        smtp <- smoothCon(s(x, bs = "tp", k = 12), data = d1)[[1]]
        Sscale_tp <- smtp$S.scale
        """
        sm = GAM._construct_duchon(GAM.s(:x; k = 12, bs = :ds), (x = x,),
                                   nothing; absorb_cons = false)
        scale_jl = opnorm(sm.X, Inf)^2 / opnorm(sm.S[1], 1)
        @test isapprox(1 / scale_jl, rcopy(R"Sscale_R"); rtol = 1e-8)

        # It must NOT be the thin-plate value: that is exactly what the old
        # stub reported, and the two differ by more than an order of magnitude.
        @test !isapprox(1 / scale_jl, rcopy(R"Sscale_tp"); rtol = 0.5)
    end

    # ------------------------------------------------------------------
    # Fitted models at FIXED sp, so no optimizer difference can confound.
    # ------------------------------------------------------------------
    @testset "fixed-sp fits agree (1-D)" begin
        for spv in (0.001, 0.1, 10.0)
            @rput spv
            R"""
            mR <- gam(y ~ s(x, bs = "ds", k = 12), data = d1, sp = spv)
            edf_R <- sum(summary(mR)$s.table[, "edf"])
            dev_R <- deviance(mR); fit_R <- as.numeric(fitted(mR))
            """
            gf = GAM.GamFormula(:y, Symbol[], true,
                [GAM.s(:x; k = 12, bs = :ds, sp = spv)])
            m = gam(gf, df)

            @test isapprox(sum(edf(m)), rcopy(R"edf_R"); atol = 1e-6)
            @test isapprox(deviance(m), rcopy(R"dev_R"); rtol = 1e-7)
            @test maximum(abs.(fitted(m) .- rcopy(R"fit_R"))) < 1e-6
        end
    end

    # ------------------------------------------------------------------
    # Two dimensions, and the non-default second order. For d = 1 the
    # half-integer grid plus |s| < d/2 leaves s = 0 as the only legal value,
    # so `s` can only be exercised at d >= 2.
    # ------------------------------------------------------------------
    @testset "fixed-sp fits agree (2-D, incl. non-default m and s)" begin
        rng2 = StableRNG(21)
        n2 = 300
        x2 = rand(rng2, n2)
        z2 = rand(rng2, n2)
        y2 = sin.(2π .* x2) .* cos.(π .* z2) .+ 0.3 .* randn(rng2, n2)
        df2 = DataFrame(x = x2, z = z2, y = y2)
        @rput x2
        @rput z2
        @rput y2
        R"d2 <- data.frame(x = x2, z = z2, y = y2)"

        for (mm, ss) in ((2, 0.0), (2, 0.5), (3, -0.5))
            @rput mm
            @rput ss
            R"""
            mR2 <- gam(y ~ s(x, z, bs = "ds", k = 20, m = c(mm, ss)),
                       data = d2, sp = 0.01)
            smR2 <- smoothCon(s(x, z, bs = "ds", k = 20, m = c(mm, ss)),
                              data = d2)[[1]]
            edf2_R <- sum(summary(mR2)$s.table[, "edf"])
            dev2_R <- deviance(mR2); fit2_R <- as.numeric(fitted(mR2))
            null2_R <- smR2$null.space.dim; Ssc2_R <- smR2$S.scale
            porder_R <- smR2$p.order
            """
            gf = GAM.GamFormula(:y, Symbol[], true,
                [GAM.s(:x, :z; k = 20, bs = :ds, m = mm,
                       xt = Dict(:s => ss), sp = 0.01)])
            m = gam(gf, df2)

            # mgcv's own (m, s) after its adjustment rules — confirms the
            # orders were interpreted identically before anything is fitted.
            po = rcopy(R"porder_R")
            @test GAM._duchon_orders(
                GAM.s(:x, :z; bs = :ds, m = mm, xt = Dict(:s => ss)), 2) ==
                (Int(po[1]), Float64(po[2]))

            smc = GAM._construct_duchon(
                GAM.s(:x, :z; k = 20, bs = :ds, m = mm, xt = Dict(:s => ss)),
                (x = x2, z = z2), nothing; absorb_cons = false)
            @test smc.null_dim == rcopy(R"null2_R")
            @test isapprox(opnorm(smc.S[1], 1) / opnorm(smc.X, Inf)^2,
                           rcopy(R"Ssc2_R"); rtol = 1e-7)

            @test isapprox(sum(edf(m)), rcopy(R"edf2_R"); atol = 1e-5)
            @test isapprox(deviance(m), rcopy(R"dev2_R"); rtol = 1e-7)
            @test maximum(abs.(fitted(m) .- rcopy(R"fit2_R"))) < 1e-5
        end
    end

    # ------------------------------------------------------------------
    # Free fits: sp must transfer, which is the practical test that the
    # penalty is on mgcv's scale rather than merely proportional to it.
    # ------------------------------------------------------------------
    @testset "freely selected sp transfers from mgcv" begin
        R"""
        mfree <- gam(y ~ s(x, bs = "ds", k = 12), data = d1, method = "REML")
        sp_R <- as.numeric(mfree$sp)
        edff_R <- sum(summary(mfree)$s.table[, "edf"])
        fitf_R <- as.numeric(fitted(mfree))
        """
        sp_R = rcopy(R"sp_R")[1]

        # mgcv's sp, fixed in GAM.jl, must reproduce mgcv's fit.
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x; k = 12, bs = :ds, sp = sp_R)])
        mfix = gam(gf, df)
        @test maximum(abs.(fitted(mfix) .- rcopy(R"fitf_R"))) < 1e-5
        @test isapprox(sum(edf(mfix)), rcopy(R"edff_R"); atol = 1e-4)

        # And a free GAM.jl fit lands in the same place.
        mfree = gam(@formulak(y ~ s(x, k = 12, bs = :ds)), df; method = :REML)
        @test isapprox(exp(mfree.sp[1]), sp_R; rtol = 1e-3)
        @test isapprox(sum(edf(mfree)), rcopy(R"edff_R"); atol = 1e-3)
    end
end
