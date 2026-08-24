# P-IRLS step-control invariants
#
# These pin the behaviours that must not change when the five P-IRLS copies
# share step-acceptance policy (`PirlsStepControl` / `pirls_halve!`):
#
#   * gam.fit3 parity — the null-coefficient penalized-deviance baseline and
#     the `10(0.1 + |pdev|)·√eps` divergence threshold that reproduce mgcv's
#     iterates (these are what make the exact-mgcv assertions in
#     test_rcall.jl hold: sp 0.0000, coefficients 8.7e-8, SEs 5.1e-7)
#   * chunked ≡ dense for bam
#   * scasm's feasibility-gated acceptance, including iteration 1
#   * scam's E-matrix positive-definiteness test firing (Fisher fallback)
#   * leverage summing to the effective degrees of freedom
#
# The reference numbers are snapshots of the pre-consolidation fits. They are
# behavioural anchors, not statistical claims: if a refactor changes them, the
# refactor changed the fit.

using Test
using GAM
using GAM: @formulak, leverage
using StatsAPI: fitted, coef
using StableRNGs
using Statistics, LinearAlgebra

@testset "P-IRLS step-control invariants" begin

    # Shared fixtures ------------------------------------------------------
    rng = StableRNG(11)
    n = 300
    x = sort(rand(rng, n))
    z = rand(rng, n)
    y = sin.(2π .* x) .+ 0.5 .* z .+ 0.3 .* randn(rng, n)
    df = (y = y, x = x, z = z)

    @testset "Gaussian reference fit is unchanged" begin
        m = gam(@formulak(y ~ s(x, k = 12, bs = :cr) + s(z, k = 8, bs = :cr)), df)
        @test m.deviance_val ≈ 27.542518077223 rtol = 1e-9
        @test m.edf_total ≈ 11.181589637834 rtol = 1e-9
        @test m.sp[1] ≈ 2.921439327427 rtol = 1e-7
        @test m.sp[2] ≈ 6.881308873934 rtol = 1e-7
        @test coef(m)[1] ≈ 0.185438018157 rtol = 1e-9
        # Leverage sums to EDF (hat-diagonal consistency)
        @test sum(leverage(m)) ≈ m.edf_total rtol = 1e-9
    end

    @testset "Poisson reference fit is unchanged (step halving path)" begin
        rng2 = StableRNG(12)
        yp = Float64[rand(rng2, Poisson(exp(0.5 + sin(2π * xi)))) for xi in x]
        mp = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), (y = yp, x = x);
            family = Poisson(), link = LogLink())
        @test mp.deviance_val ≈ 381.336485934592 rtol = 1e-9
        @test mp.edf_total ≈ 6.957937392715 rtol = 1e-9
        @test mp.sp[1] ≈ 4.177641159555 rtol = 1e-7
        # Iteration count is part of the step-control contract
        @test mp.iterations == 4
    end

    @testset "Gamma reference fit is unchanged" begin
        rng3 = StableRNG(13)
        yg = Float64[rand(rng3, Gamma(4.0, (1.0 + 2.0 * xi) / 4.0)) for xi in x]
        mg = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), (y = yg, x = x);
            family = Gamma(), link = LogLink())
        # Re-pinned once after two related REML-score corrections: profiling the
        # scale for estimated-scale families, and using mgcv's full-Newton
        # (observed-information) weights in `log|X'WX+S|` for NON-canonical pairs
        # such as Gamma+log. Both change the criterion EFS optimises, so the
        # selected sp — and hence these three quantities — move.
        #
        # Verified against mgcv 1.9-4 on this exact fit:
        #            GAM.jl        mgcv        rel
        #   dev     87.866290   87.877797   1.3e-4
        #   edf      3.482381    3.464557   5.1e-3
        #   scale    0.271199    0.271259   2.2e-4
        #   sp    1892.53     1948.25       (flat REML ridge)
        #
        # Previous pins were dev 87.828825, edf 3.541309, scale 0.271007.
        @test mg.deviance_val ≈ 87.866289782686 rtol = 1e-9
        @test mg.edf_total ≈ 3.482380889071 rtol = 1e-9
        @test mg.scale ≈ 0.271199414910 rtol = 1e-9
    end

    @testset "bam: chunked accumulation ≡ dense gam" begin
        m = gam(@formulak(y ~ s(x, k = 12, bs = :cr) + s(z, k = 8, bs = :cr)), df)
        mb = bam(@formulak(y ~ s(x, k = 12, bs = :cr) + s(z, k = 8, bs = :cr)), df)
        @test mb.deviance_val ≈ 27.542314103612 rtol = 1e-8
        @test mb.edf_total ≈ 11.182489027244 rtol = 1e-8
        @test maximum(abs.(fitted(mb) .- fitted(m))) < 1e-4
    end

    @testset "scam reference fit is unchanged" begin
        ysc = 2.0 .* x .+ 0.2 .* randn(StableRNG(14), n)
        msc = scam(@formulak(y ~ s(x, bs = :mpi, k = 10)), (y = ysc, x = x);
            method = :REML)
        # Round-5: the scam outer loop gained the score-based stop the
        # standard and bam loops already had, so it exits a numerically flat
        # criterion ridge at iteration 15 instead of running to outer_maxit
        # (200). The fit is marginally better (lower deviance); these are the
        # post-change anchors.
        @test msc.deviance_val ≈ 12.683978264 rtol = 1e-6
        @test msc.edf_total ≈ 2.64351707 rtol = 1e-6
        @test msc.iterations < 50
        # Monotonicity is the defining constraint
        @test all(diff(fitted(msc)) .>= -1e-8)
    end

    @testset "scasm: feasibility-gated acceptance holds from iteration 1" begin
        ysc = 2.0 .* x .+ 0.2 .* randn(StableRNG(14), n)
        msa = gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 10)), (y = ysc, x = x))
        @test msa.deviance_val ≈ 12.699685917233 rtol = 1e-7
        @test msa.edf_total ≈ 2.000412930824 rtol = 1e-7
        # The constrained solution must satisfy its shape constraint
        @test all(diff(fitted(msa)) .>= -1e-8)
    end

    @testset "scam E-matrix PD test can fire (Fisher fallback reachable)" begin
        # Poisson + monotone constraint on noisy counts drives negative Newton
        # weights, which is what the E-matrix eigenvalue test guards against.
        rng4 = StableRNG(15)
        xc = sort(rand(rng4, 150))
        yc = Float64[rand(rng4, Poisson(exp(0.3 + 1.5 * xi))) for xi in xc]
        mpc = scam(@formulak(y ~ s(x, bs = :mpi, k = 8)), (y = yc, x = xc);
            family = Poisson(), link = LogLink(), method = :REML)
        @test mpc.converged
        @test all(isfinite, fitted(mpc))
        @test all(fitted(mpc) .> 0)
        @test all(diff(fitted(mpc)) .>= -1e-8)
    end

    # ---------------------------------------------------------------------
    # Family-domain invariants across every fitter (mgcv's validmu contract).
    #
    # Gamma/InverseGaussian with their canonical inverse link have mean domain
    # μ > 0, i.e. η > 0. A fitter must either keep the iterates in-domain or
    # report honestly — never converge silently onto clamped means with a
    # nonsense scale.
    # ---------------------------------------------------------------------
    @testset "leverage/EDF identity across fitters and families" begin
        # h_i = w_i · x_i' A⁻¹ x_i, so sum(hat) == tr(F) == edf_total, and every
        # leverage lies in [0, 1]. bam hand-rolled this without the weight
        # factor, which is invisible for unit-weight Gaussian (w ≡ 1) and wrong
        # for every weighted or non-Gaussian fit — silently corrupting
        # leverage() and cooksdistance().
        rngh = StableRNG(4321)
        nh = 300
        xh = sort(rand(rngh, nh))

        function check_leverage(m, label)
            h = leverage(m)
            @test length(h) == nh
            @test all(isfinite, h)
            @test all(h .>= -1e-8)
            @test maximum(h) <= 1.0 + 1e-6          # a leverage cannot exceed 1
            @test sum(h) ≈ m.edf_total rtol = 1e-6  # the defining identity
            d = cooksdistance(m)
            @test all(isfinite, d)
        end

        # Gaussian, unit weights — the only case existing tests covered
        yg = sin.(2π .* xh) .+ 0.3 .* randn(rngh, nh)
        dfg = (y = yg, x = xh)
        @testset "gam Gaussian unit-weight" begin
            check_leverage(gam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfg), "gam")
        end
        @testset "bam Gaussian unit-weight" begin
            check_leverage(bam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfg), "bam")
        end

        # Gaussian with prior weights
        wts = 0.5 .+ rand(rngh, nh)
        @testset "gam Gaussian prior-weight" begin
            check_leverage(gam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfg;
                weights = wts), "gam/w")
        end
        @testset "bam Gaussian prior-weight" begin
            check_leverage(bam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfg;
                weights = wts), "bam/w")
        end

        # Poisson
        yp = Float64[rand(rngh, Poisson(exp(0.5 + sin(2π * t)))) for t in xh]
        dfp = (y = yp, x = xh)
        @testset "gam Poisson" begin
            check_leverage(gam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfp;
                family = Poisson(), link = LogLink()), "gam/pois")
        end
        @testset "bam Poisson" begin
            check_leverage(bam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfp;
                family = Poisson(), link = LogLink()), "bam/pois")
        end

        # Bernoulli
        yb = Float64[rand(rngh) < 1 / (1 + exp(-(2.0 * sin(2π * t)))) ? 1.0 : 0.0
                     for t in xh]
        dfb = (y = yb, x = xh)
        @testset "gam Bernoulli" begin
            check_leverage(gam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfb;
                family = Bernoulli(), link = LogitLink()), "gam/bern")
        end
        @testset "bam Bernoulli" begin
            check_leverage(bam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfb;
                family = Bernoulli(), link = LogitLink()), "bam/bern")
        end

        # Constrained fitters (Gaussian, monotone truth)
        ym = 2.0 .* xh .+ 0.2 .* randn(rngh, nh)
        dfm = (y = ym, x = xh)
        @testset "scam" begin
            check_leverage(scam(@formulak(y ~ s(x, bs = :mpi, k = 10)), dfm;
                method = :REML), "scam")
        end
        @testset "scasm" begin
            check_leverage(gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 10)),
                dfm), "scasm")
        end
    end

    @testset "family-domain invariants across fitters" begin
        rngd = StableRNG(103)
        nd = 200
        xd = sort(rand(rngd, nd))
        mu_true = 1.0 .+ 2.0 .* xd
        yd = Float64[rand(rngd, Gamma(4.0, m / 4.0)) for m in mu_true]
        dfd = (y = yd, x = xd)

        # The contract is honesty, not success: a fitter may fail to keep an
        # inverse-link fit inside the mean domain, but it must then say so
        # rather than report convergence with silently clamped means and a
        # meaningless scale.
        function check_domain(m, label)
            eta = m.linear_predictor
            fit = fitted(m)
            @test all(isfinite, fit)
            @test all(fit .> 0)              # Gamma/IG mean domain
            if m.converged
                @test all(eta .> 0)          # in-domain
                @test isfinite(m.scale)
                @test m.scale < 1e6          # 1.17e15 was the observed failure
            end
        end

        @testset "gam" begin
            m = gam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfd;
                family = Gamma(), link = InverseLink())
            check_domain(m, "gam")
        end

        @testset "bam" begin
            m = bam(@formulak(y ~ s(x, k = 12, bs = :cr)), dfd;
                family = Gamma(), link = InverseLink())
            check_domain(m, "bam")
        end

        @testset "scam" begin
            m = scam(@formulak(y ~ s(x, bs = :mpi, k = 12)), dfd;
                family = Gamma(), link = InverseLink(), method = :REML)
            check_domain(m, "scam")
        end

        @testset "scasm" begin
            m = gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12)), dfd;
                family = Gamma(), link = InverseLink())
            check_domain(m, "scasm")
        end

        @testset "InverseGaussian/inverse via gam" begin
            yig = Float64[rand(StableRNG(104), InverseGaussian(m, 8.0)) for m in mu_true]
            m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), (y = yig, x = xd);
                family = InverseGaussian(), link = InverseLink())
            check_domain(m, "gam/IG")
        end

        @testset "Gamma/identity via gam" begin
            m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), dfd;
                family = Gamma(), link = IdentityLink())
            check_domain(m, "gam/identity")
        end
    end
end
