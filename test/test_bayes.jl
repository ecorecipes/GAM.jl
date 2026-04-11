using Test
using GAM
using DataFrames
using Random
using LinearAlgebra
using Distributions
using Turing
using StatsAPI
using Statistics: mean

@testset "smooth2random" begin
    Random.seed!(42)
    n = 200
    x = sort(randn(n))
    x2 = randn(n)
    y = sin.(2 .* x) .+ 0.3 .* randn(n)
    df = DataFrame(x = x, x2 = x2, y = y)

    @testset "Single-penalty smooths" begin
        # CR spline: null_dim=2 means 1 Xf col (after constraint), 8 Zs cols
        sm = smooth_construct(s(:x, bs = :cr, k = 10), df)
        smm = smooth2random(sm)
        @test !smm.fixed
        @test size(smm.Xf, 2) == sm.null_dim - 1  # minus 1 for absorbed constraint
        @test size(smm.Zs[1], 2) == sm.rank
        @test size(smm.Xf, 2) + size(smm.Zs[1], 2) == size(sm.X, 2)
        @test smm.trans_U !== nothing
        @test length(smm.trans_D) == size(sm.X, 2)
        @test length(smm.Zs) == 1
        @test smm.label == sm.spec.label

        # TP spline
        sm_tp = smooth_construct(s(:x, bs = :tp, k = 10), df)
        smm_tp = smooth2random(sm_tp)
        @test size(smm_tp.Xf, 2) + size(smm_tp.Zs[1], 2) == size(sm_tp.X, 2)
        @test !smm_tp.fixed

        # P-spline
        sm_ps = smooth_construct(s(:x, bs = :ps, k = 10), df)
        smm_ps = smooth2random(sm_ps)
        @test size(smm_ps.Xf, 2) + size(smm_ps.Zs[1], 2) == size(sm_ps.X, 2)

        # Random effect
        sm_re = smooth_construct(s(:x, bs = :re, k = 10), df)
        smm_re = smooth2random(sm_re)
        @test size(smm_re.Xf, 2) == 0  # RE is all penalized
        @test length(smm_re.Zs) == 1
    end

    @testset "Shrinkage smooths (multi-penalty)" begin
        # TS (shrinkage): has 2 penalties (wiggle + null space)
        sm_ts = smooth_construct(s(:x, bs = :ts, k = 10), df)
        smm_ts = smooth2random(sm_ts)
        @test length(smm_ts.Zs) == 2  # one per penalty
        @test !smm_ts.fixed
        @test size(smm_ts.Xf, 1) == n

        # CS (cubic shrinkage): also 2 penalties
        sm_cs = smooth_construct(s(:x, bs = :cs, k = 10), df)
        smm_cs = smooth2random(sm_cs)
        @test length(smm_cs.Zs) >= 1
        @test !smm_cs.fixed
    end

    @testset "Fixed (unpenalized) smooth" begin
        sm_fx = smooth_construct(
            SmoothSpec([:x], CubicSpline(), 10, nothing, nothing, nothing, true, nothing, "s(x,fx)"),
            df
        )
        smm_fx = smooth2random(sm_fx)
        @test smm_fx.fixed
        @test size(smm_fx.Xf) == size(sm_fx.X)
        @test isempty(smm_fx.Zs)
    end

    @testset "Reconstruction accuracy" begin
        sm = smooth_construct(s(:x, bs = :cr, k = 10), df)
        smm = smooth2random(sm)

        β_test = randn(size(sm.X, 2))
        y_orig = sm.X * β_test

        # Transform to mixed-model space
        UD = smm.trans_U * Diagonal(smm.trans_D)
        β_mm = UD \ β_test
        p_rank = length(smm.rind)
        y_recon = smm.Xf * β_mm[(p_rank + 1):end] + smm.Zs[1] * β_mm[1:p_rank]

        @test maximum(abs.(y_orig - y_recon)) < 1e-10
    end

    @testset "Consistency: Xf*β_f + Zs*β_r = X*β_orig" begin
        for bs in [:cr, :tp, :ps]
            sm = smooth_construct(s(:x, bs = bs, k = 10), df)
            smm = smooth2random(sm)

            k_f = size(smm.Xf, 2)
            k_r = size(smm.Zs[1], 2)
            β_f = randn(k_f)
            β_r = randn(k_r)
            y_mm = smm.Xf * β_f + smm.Zs[1] * β_r

            UD = smm.trans_U * Diagonal(smm.trans_D)
            β_orig = UD * vcat(β_r, β_f)
            y_orig = sm.X * β_orig

            @test maximum(abs.(y_mm - y_orig)) < 1e-6
        end
    end
end

@testset "PriorSpec" begin
    @testset "Default construction" begin
        ps = PriorSpec()
        @test ps.b isa Normal
        @test ps.sds isa Truncated
        @test ps.sigma isa Truncated
    end

    @testset "Custom construction" begin
        ps = PriorSpec(sds = Exponential(1.0), sigma = InverseGamma(2, 3))
        @test ps.sds isa Exponential
        @test ps.sigma isa InverseGamma
    end

    @testset "get_prior hierarchical lookup" begin
        ps = PriorSpec(
            sds = Exponential(1.0),
            specific = Dict("sds_s(x2)" => Exponential(0.5))
        )
        # Class default
        p1 = get_prior(ps, :sds, "s(x1)")
        @test p1 isa Exponential
        @test p1.θ ≈ 1.0

        # Specific override
        p2 = get_prior(ps, :sds, "s(x2)")
        @test p2 isa Exponential
        @test p2.θ ≈ 0.5

        # Other classes
        @test get_prior(ps, :b) isa Normal
        @test get_prior(ps, :sigma) isa Truncated
    end

    @testset "default_priors" begin
        ps = default_priors(Normal())
        @test ps isa PriorSpec
    end
end

@testset "gam_smooth convenience" begin
    Random.seed!(42)
    df = DataFrame(x = randn(100), y = randn(100))

    smm = gam_smooth(:x, df; bs = :cr, k = 10)
    @test smm isa SmoothMixedModel
    @test size(smm.Xf, 1) == 100
    @test !smm.fixed
end

@testset "gam_matrices" begin
    Random.seed!(42)
    df = DataFrame(x = randn(100), x2 = randn(100), y = randn(100))

    gf = @formulak(y ~ x + s(x2, bs = :cr, k = 10))
    X_para, sms, labels = gam_matrices(gf, df)

    @test size(X_para, 1) == 100
    @test size(X_para, 2) == 2  # intercept + x
    @test length(sms) == 1
    @test sms[1] isa SmoothMixedModel
    @test length(labels) == 1
end

@testset "BayesGamModel type" begin
    # Stubs are declared as functions (filled by extension)
    @test isdefined(GAM, :_fit_gam_bayes)
    @test isdefined(GAM, :_fit_gamlss_bayes)
    @test isdefined(GAM, :_fit_scam_bayes)
end

@testset "LOOResult compatibility constructor" begin
    l = LOOResult(1.0, 0.5, -2.0, 0.1, [0.2, 0.3], [0.01, 0.02])
    @test l.method == :is
    @test length(l.pareto_k) == 2
    @test all(isnan, l.pareto_k)
    @test length(l.n_eff) == 2
    @test all(isnan, l.n_eff)
end

@testset "smooth2random vs R mgcv" begin
    r_available = try
        @eval using RCall
        @eval RCall.reval("library(mgcv)")
        true
    catch
        false
    end

    if r_available
        Random.seed!(42)
        n = 200
        x = sort(randn(n))
        y = sin.(2 .* x) .+ 0.3 .* randn(n)
        df = DataFrame(x = x, y = y)

        _rcall = @eval RCall
        _reval = _rcall.reval
        _rcopy = _rcall.rcopy

        _rcall.globalEnv[:x_r] = x
        _rcall.globalEnv[:y_r] = y

        for (bs_jl, bs_r) in [(:cr, "cr"), (:tp, "tp")]
            @testset "basis=$bs_r" begin
                sm = smooth_construct(s(:x, bs = bs_jl, k = 10), df)
                smm = smooth2random(sm)

                # Build R decomposition using smoothCon + smooth2random
                _reval("""
                    dat <- data.frame(x = x_r, y = y_r)
                    sm_r <- smoothCon(s(x, k = 10, bs = "$bs_r"), data = dat)[[1]]
                    s2r <- smooth2random(sm_r, "")
                    # Derive Zs and Xf from trans.U, trans.D
                    UD_r <- s2r\$trans.U %*% diag(s2r\$trans.D)
                    X_new_r <- sm_r\$X %*% UD_r
                    p_rank_r <- length(s2r\$rind)
                    Zs_r <- X_new_r[, 1:p_rank_r, drop = FALSE]
                    Xf_r <- s2r\$Xf
                """)
                Zs_r = _rcopy(_reval("Zs_r"))
                Xf_r = _rcopy(_reval("Xf_r"))
                rind_r = Int.(_rcopy(_reval("s2r\$rind")))
                pen_ind_r = Int.(_rcopy(_reval("s2r\$pen.ind")))

                # Random effect dimensions must match exactly
                @test size(smm.Zs[1], 2) == size(Zs_r, 2)
                @test size(smm.Zs[1], 1) == size(Zs_r, 1)

                # Julia absorbs one identifiability constraint into the intercept,
                # so Xf has null_dim-1 columns vs R's null_dim
                @test size(smm.Xf, 2) == size(Xf_r, 2) - 1
                @test length(smm.rind) == length(rind_r)

                # Number of penalized columns must match
                @test count(==(1), smm.pen_ind) == count(==(1), pen_ind_r)

                # Reconstruction accuracy: Julia's decomposition must be exact
                β_test = randn(size(sm.X, 2))
                y_orig = sm.X * β_test
                UD = smm.trans_U * Diagonal(smm.trans_D)
                β_mm = UD \ β_test
                p_rank = length(smm.rind)
                y_recon = smm.Xf * β_mm[(p_rank + 1):end] + smm.Zs[1] * β_mm[1:p_rank]
                @test maximum(abs.(y_orig - y_recon)) < 1e-10
            end
        end
    else
        @test_skip "R/mgcv not available — skipping smooth2random comparison"
    end
end

@testset "Prior specification validation" begin
    @testset "default_priors for all families" begin
        for fam in [Normal(), Poisson(), Bernoulli(), Gamma()]
            ps = default_priors(fam)
            @test ps isa PriorSpec
            @test ps.b isa Distribution
            @test ps.sds isa Distribution
            @test ps.sigma isa Distribution
            @test ps.phi isa Distribution
        end
    end

    @testset "Custom priors propagate — tight vs wide" begin
        Random.seed!(42)
        n = 150
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.3 .* randn(n)
        df = DataFrame(x = x, y = y)

        # Tight prior on fixed-effect coefficients
        ps_tight = PriorSpec(b = Normal(0, 0.01))
        m_tight = gam(@formulak(y ~ s(x, k = 10)), df;
            priors = ps_tight, nsamples = 500, nchains = 1)

        # Wide prior on fixed-effect coefficients
        ps_wide = PriorSpec(b = Normal(0, 100))
        m_wide = gam(@formulak(y ~ s(x, k = 10)), df;
            priors = ps_wide, nsamples = 500, nchains = 1)

        # Tight priors should produce smaller (or equal) coefficient magnitudes on average
        coef_tight = StatsAPI.coef(m_tight)
        coef_wide = StatsAPI.coef(m_wide)
        @test mean(abs.(coef_tight)) <= mean(abs.(coef_wide)) + 0.5
    end

    @testset "get_prior hierarchical lookup with specific overrides" begin
        ps = PriorSpec(
            b = Normal(0, 5),
            sds = Exponential(2.0),
            sigma = truncated(Normal(0, 1.0); lower = 0),
            phi = truncated(Normal(0, 3.0); lower = 0),
            specific = Dict(
                "sds_s(x1)" => Exponential(0.1),
                "b_(Intercept)" => Normal(0, 50),
            ),
        )

        # Specific overrides
        p_sds_x1 = get_prior(ps, :sds, "s(x1)")
        @test p_sds_x1 isa Exponential
        @test p_sds_x1.θ ≈ 0.1

        p_b_int = get_prior(ps, :b, "(Intercept)")
        @test p_b_int isa Normal
        @test std(p_b_int) ≈ 50.0

        # Class defaults (no matching specific override)
        p_sds_x2 = get_prior(ps, :sds, "s(x2)")
        @test p_sds_x2 isa Exponential
        @test p_sds_x2.θ ≈ 2.0

        p_b_default = get_prior(ps, :b, "x_linear")
        @test p_b_default isa Normal
        @test std(p_b_default) ≈ 5.0

        # sigma and phi class defaults
        @test get_prior(ps, :sigma) isa Truncated
        @test get_prior(ps, :phi) isa Truncated

        # Unknown class should error
        @test_throws ErrorException get_prior(ps, :unknown)
    end
end

@testset "priors= kwarg dispatch" begin
    Random.seed!(42)
    df = DataFrame(x = randn(50), y = randn(50))

    # Without priors: works as usual (frequentist)
    m = gam(@formulak(y ~ s(x)), df)
    @test m isa GamModel
end
