using Test
using GAM
using DataFrames
using Random
using LinearAlgebra
using Distributions

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

    gf = @gam_formula(y ~ x + s(x2, bs = :cr, k = 10))
    X_para, sms, labels = gam_matrices(gf, df)

    @test size(X_para, 1) == 100
    @test size(X_para, 2) == 2  # intercept + x
    @test length(sms) == 1
    @test sms[1] isa SmoothMixedModel
    @test length(labels) == 1
end

@testset "BayesGamModel stub" begin
    @test_throws ErrorException GAM._fit_gam_bayes()
    @test_throws ErrorException GAM._fit_gamlss_bayes()
    @test_throws ErrorException GAM._fit_scam_bayes()
end

@testset "priors= kwarg dispatch" begin
    Random.seed!(42)
    df = DataFrame(x = randn(50), y = randn(50))

    # Without priors: works as usual (frequentist)
    m = gam(@gam_formula(y ~ s(x)), df)
    @test m isa GamModel

    # With priors but no Turing: should error informatively
    @test_throws ErrorException gam(@gam_formula(y ~ s(x)), df; priors = PriorSpec())
    @test_throws ErrorException gamlss(@gam_formula(y ~ s(x)), df, Normal(); priors = PriorSpec())
    @test_throws ErrorException scam(@gam_formula(y ~ s(x, bs = :mpi)), df; priors = PriorSpec())
end
