using Test
using GAM
using DataFrames
using Random
using Statistics
using Distributions
using StatsModels
using StatsAPI: coef, vcov, fitted, residuals, nobs, deviance, loglikelihood, dof, response, predict
using LinearAlgebra

@testset "GAMM — Generalized Additive Mixed Models" begin

    # ========================================================================
    # Formula parsing
    # ========================================================================
    @testset "Formula parsing" begin
        @testset "@gamm_formula: random intercept" begin
            gf = @gamm_formula(y ~ s(x) + (1 | group))
            @test length(gf.random_effects) == 1
            re = gf.random_effects[1]
            @test re.grouping == :group
            @test re.has_intercept == true
            @test isempty(re.terms)
            @test re.label == "(1 | group)"
        end

        @testset "@gamm_formula: random slope" begin
            gf = @gamm_formula(y ~ s(x) + (0 + x | group))
            @test length(gf.random_effects) == 1
            re = gf.random_effects[1]
            @test re.grouping == :group
            @test re.has_intercept == false
            @test :x in re.terms
        end

        @testset "@gamm_formula: random intercept + slope" begin
            gf = @gamm_formula(y ~ s(x) + (1 + z | subject))
            @test length(gf.random_effects) == 1
            re = gf.random_effects[1]
            @test re.grouping == :subject
            @test re.has_intercept == true
            @test :z in re.terms
        end

        @testset "@gamm_formula: multiple random effects" begin
            gf = @gamm_formula(y ~ s(x) + (1 | site) + (1 | subject))
            @test length(gf.random_effects) == 2
            @test gf.random_effects[1].grouping == :site
            @test gf.random_effects[2].grouping == :subject
        end

        @testset "@formula with (1|group)" begin
            Random.seed!(1)
            n = 100
            df = DataFrame(x = randn(n), y = randn(n), group = repeat(1:5, 20))
            m = gamm(@formula(y ~ cr(x, 10) + (1 | group)), df)
            @test m isa GammModel
            @test length(m.random_effects) == 1
        end

        @testset "@formula with re(group)" begin
            Random.seed!(1)
            n = 100
            df = DataFrame(x = randn(n), y = randn(n), group = repeat(1:5, 20))
            m = gamm(@formula(y ~ cr(x, 10) + re(group)), df)
            @test m isa GammModel
            @test length(m.random_effects) == 1
        end
    end

    # ========================================================================
    # Z matrix construction
    # ========================================================================
    @testset "Z matrix construction" begin
        @testset "Random intercept Z dimensions" begin
            spec = RandomEffectSpec(:g, Symbol[], true, true, "(1|g)")
            df = DataFrame(g = repeat(1:5, 20), x = randn(100))
            cre = construct_random_effect(spec, df)
            @test cre.n_levels == 5
            @test cre.n_terms == 1
            # With sum-to-zero constraint: n_levels - 1 columns
            @test size(cre.Z, 1) == 100
            @test size(cre.Z, 2) == 4  # 5 - 1 = 4 constrained cols
            @test cre.constraint_basis !== nothing
            @test size(cre.constraint_basis) == (5, 4)
        end

        @testset "Random slope Z" begin
            spec = RandomEffectSpec(:g, [:x], false, true, "(0+x|g)")
            df = DataFrame(g = repeat(1:3, 30), x = randn(90))
            cre = construct_random_effect(spec, df)
            @test cre.n_levels == 3
            @test cre.n_terms == 1
            @test size(cre.Z, 1) == 90
            # Random slope: no intercept, so no sum-to-zero needed — but check it works
            @test size(cre.Z, 2) >= 2  # at least 2 cols (3 - 1 or 3)
        end

        @testset "Prediction Z for new groups" begin
            spec = RandomEffectSpec(:g, Symbol[], true, true, "(1|g)")
            df_train = DataFrame(g = repeat(1:5, 20), x = randn(100))
            cre = construct_random_effect(spec, df_train)

            # Known group
            df_known = DataFrame(g = [1, 2, 3])
            Z_known = predict_re_matrix(cre, df_known)
            @test size(Z_known, 1) == 3
            @test size(Z_known, 2) == size(cre.Z, 2)

            # Unknown group → zeros
            df_new = DataFrame(g = [99, 100])
            Z_new = predict_re_matrix(cre, df_new)
            @test all(Z_new .== 0.0)
        end
    end

    # ========================================================================
    # Gaussian GAMM: random intercept
    # ========================================================================
    @testset "Gaussian GAMM: random intercept" begin
        Random.seed!(42)
        n_groups = 10
        n_per = 50
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) * 0.5
        y = sin.(x) .+ true_re[group] .+ 0.3 .* randn(n)
        df = DataFrame(x = x, y = y, group = group)

        m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)

        @test m isa GammModel
        @test m.gam_model.converged

        # Random effects should correlate well with truth
        re_est = ranef(m)
        est = vec(re_est.group.effects)
        @test length(est) == n_groups
        @test cor(est, true_re) > 0.9

        # Variance component should be in the right ballpark (true σ = 0.5)
        vc = VarCorr(m)
        @test length(vc) == 1
        @test vc[1].std > 0.1
        @test vc[1].std < 1.5

        # StatsAPI methods should work
        @test length(coef(m)) > 0
        @test nobs(m) == n
        @test deviance(m) > 0
        @test length(fitted(m)) == n
        @test length(residuals(m)) == n

        # Show method should not error
        io = IOBuffer()
        show(io, m)
        str = String(take!(io))
        @test occursin("Generalized Additive Mixed Model", str)
        @test occursin("Random Effects", str)
        @test occursin("group", str)
    end

    # ========================================================================
    # Consistency: gamm() vs gam(s(group, bs=:re))
    # ========================================================================
    @testset "GAMM vs s(group, bs=:re) consistency" begin
        Random.seed!(123)
        n_groups = 8
        n_per = 40
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) * 0.4
        y = cos.(x) .+ true_re[group] .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, group = group)

        # Fit GAMM
        m_gamm = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)

        # Fit equivalent GAM with s(group, bs=:re)
        m_gam = gam(@gam_formula(y ~ s(x) + s(group, bs = :re)), df)

        # Both should give similar scale estimates
        @test abs(m_gamm.gam_model.scale - m_gam.scale) / m_gam.scale < 0.5

        # Both should recover similar fixed smooth effects
        re_gamm = ranef(m_gamm)
        est_gamm = vec(re_gamm.group.effects)
        @test cor(est_gamm, true_re) > 0.85

        # GAMM fitted values should be close to GAM fitted values
        fit_gamm = fitted(m_gamm)
        fit_gam = fitted(m_gam)
        @test cor(fit_gamm, fit_gam) > 0.95
    end

    # ========================================================================
    # Poisson GAMM
    # ========================================================================
    @testset "Poisson GAMM" begin
        Random.seed!(42)
        n_groups = 6
        n_per = 80
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) * 0.3
        η = 0.5 .* x .+ true_re[group]
        y = [Float64(rand(Poisson(exp(η[i])))) for i in 1:n]
        df = DataFrame(x = x, y = y, group = group)

        m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df, Poisson())

        @test m isa GammModel{Poisson{Float64}, LogLink}
        @test m.gam_model.converged

        re_est = ranef(m)
        est = vec(re_est.group.effects)
        @test cor(est, true_re) > 0.8

        # Variance component
        vc = VarCorr(m)
        @test vc[1].std > 0.0
        @test vc[1].std < 2.0
    end

    # ========================================================================
    # Binomial GAMM
    # ========================================================================
    @testset "Binomial GAMM" begin
        Random.seed!(42)
        n_groups = 8
        n_per = 60
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) * 0.5
        η = x .+ true_re[group]
        prob = 1.0 ./ (1.0 .+ exp.(-η))
        y = [Float64(rand(Bernoulli(prob[i]))) for i in 1:n]
        df = DataFrame(x = x, y = y, group = group)

        m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df, Bernoulli())

        @test m isa GammModel
        @test m.gam_model.converged

        re_est = ranef(m)
        est = vec(re_est.group.effects)
        @test cor(est, true_re) > 0.5  # Binomial is noisier
    end

    # ========================================================================
    # Prediction
    # ========================================================================
    @testset "Prediction" begin
        Random.seed!(42)
        n_groups = 5
        n_per = 60
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) * 0.4
        y = x .+ true_re[group] .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, group = group)

        m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)

        # Predict on training data
        ŷ = predict(m, df)
        @test length(ŷ) == n
        @test cor(ŷ, y) > 0.9

        # Predict on new data with known groups
        df_new = DataFrame(x = [0.0, 1.0], group = [1, 2])
        ŷ_new = predict(m, df_new)
        @test length(ŷ_new) == 2
        @test all(isfinite.(ŷ_new))

        # Predict on new data with unknown group
        df_unknown = DataFrame(x = [0.0], group = [999])
        ŷ_unk = predict(m, df_unknown)
        @test length(ŷ_unk) == 1
        @test isfinite(ŷ_unk[1])  # should get zero RE contribution
    end

    # ========================================================================
    # Multiple formula paths
    # ========================================================================
    @testset "Multiple formula paths give same result" begin
        Random.seed!(42)
        n = 200
        group = repeat(1:5, 40)
        x = randn(n)
        y = x .+ randn(5)[group] * 0.3 .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, group = group)

        # @gamm_formula path
        m1 = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
        # @formula with (1|group) path
        m2 = gamm(@formula(y ~ cr(x, 10) + (1 | group)), df)
        # @formula with re() path
        m3 = gamm(@formula(y ~ cr(x, 10) + re(group)), df)

        # All should be GammModel
        @test m1 isa GammModel
        @test m2 isa GammModel
        @test m3 isa GammModel

        # All should have 1 random effect
        @test length(m1.random_effects) == 1
        @test length(m2.random_effects) == 1
        @test length(m3.random_effects) == 1

        # Fitted values should be highly correlated
        @test cor(fitted(m2), fitted(m3)) > 0.99
    end

    # ========================================================================
    # Edge cases
    # ========================================================================
    @testset "Edge cases" begin
        @testset "Many groups, few per group" begin
            Random.seed!(42)
            n_groups = 50
            n_per = 5
            n = n_groups * n_per
            group = repeat(1:n_groups, inner = n_per)
            x = randn(n)
            y = sin.(x) .+ randn(n_groups)[group] * 0.3 .+ 0.5 .* randn(n)
            df = DataFrame(x = x, y = y, group = group)

            m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
            @test m isa GammModel
            @test m.gam_model.converged
        end

        @testset "Two groups (minimum)" begin
            Random.seed!(42)
            n = 200
            group = repeat(1:2, 100)
            x = randn(n)
            y = x .+ [0.5, -0.5][group] .+ 0.3 .* randn(n)
            df = DataFrame(x = x, y = y, group = group)

            m = gamm(@gamm_formula(y ~ s(x) + (1 | group)), df)
            @test m isa GammModel
            re = ranef(m)
            est = vec(re.group.effects)
            @test length(est) == 2
            # Should recover direction: group 1 > group 2
            @test est[1] > est[2]
        end
    end

    # ========================================================================
    # RandomEffectSpec show
    # ========================================================================
    @testset "RandomEffectSpec show" begin
        re = RandomEffectSpec(:group, Symbol[], true, true, "(1 | group)")
        io = IOBuffer()
        show(io, re)
        @test String(take!(io)) == "(1 | group)"

        re2 = RandomEffectSpec(:subject, [:x, :z], true, true, "(1 + x + z | subject)")
        io2 = IOBuffer()
        show(io2, re2)
        s = String(take!(io2))
        @test occursin("1", s)
        @test occursin("x", s)
        @test occursin("subject", s)
    end
end
