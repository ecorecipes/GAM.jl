using Test
using GAM
using DataFrames
using Random
using LinearAlgebra
using Statistics
using StatsAPI: deviance, nobs, dof_residual

@testset "anova_gam" begin
    rng = MersenneTwister(123)

    # ========================================================================
    # Shared test data
    # ========================================================================
    n = 300
    x = sort(rand(rng, n)) .* 2π
    z = sort(rand(rng, n)) .* 2π
    y_gauss = sin.(x) .+ 0.5 .* cos.(z) .+ 0.3 .* randn(rng, n)

    # Poisson data
    λ = exp.(0.5 .* sin.(x) .+ 0.3 .* cos.(z))
    y_pois = [rand(rng, Poisson(λ_i)) for λ_i in λ]

    df = DataFrame(x=x, z=z, y=y_gauss, y_pois=Float64.(y_pois))

    # ========================================================================
    # 1. Single-model smooth significance (Gaussian / F-test)
    # ========================================================================
    @testset "Single model - Gaussian (F-test)" begin
        m = gam(@gam_formula(y ~ s(x, k=10, bs=:cr) + s(z, k=10, bs=:cr)), df)

        result = anova_gam(m)

        @test result isa AnovaGamResult
        @test result.test_type == :F
        @test result.model_table === nothing
        @test result.smooth_table !== nothing

        st = result.smooth_table
        @test length(st.label) == 2
        @test length(st.edf) == 2
        @test length(st.ref_df) == 2
        @test length(st.statistic) == 2
        @test length(st.p_value) == 2

        # EDFs should be positive
        @test all(st.edf .> 0)
        # Reference df should be positive
        @test all(st.ref_df .> 0)
        # F statistics should be positive
        @test all(st.statistic .> 0)
        # p-values should be in [0, 1]
        @test all(0 .<= st.p_value .<= 1)

        # With a clear signal, at least one smooth should be significant
        @test minimum(st.p_value) < 0.05

        # Pretty printing should not error
        buf = IOBuffer()
        show(buf, result)
        output = String(take!(buf))
        @test occursin("edf", output)
        @test occursin("Ref.df", output)
        @test occursin("F", output)
    end

    # ========================================================================
    # 2. Single-model smooth significance (Poisson / Chi-sq test)
    # ========================================================================
    @testset "Single model - Poisson (Chi-sq test)" begin
        m_pois = gam(@gam_formula(y_pois ~ s(x, k=10, bs=:cr) + s(z, k=10, bs=:cr)),
                     df, family=Poisson(), link=LogLink())

        result = anova_gam(m_pois)

        @test result.test_type == :Chisq
        @test result.smooth_table !== nothing

        st = result.smooth_table
        @test length(st.label) == 2
        @test all(st.statistic .> 0)
        @test all(0 .<= st.p_value .<= 1)

        # Pretty printing
        buf = IOBuffer()
        show(buf, result)
        output = String(take!(buf))
        @test occursin("Chi.sq", output)
    end

    # ========================================================================
    # 3. Multi-model comparison (Gaussian / F-test)
    # ========================================================================
    @testset "Multi-model comparison - Gaussian" begin
        # Intercept-only (no smooths — use a linear model via fixed smooth)
        m1 = gam(@gam_formula(y ~ s(x, k=10, bs=:cr)), df)
        m2 = gam(@gam_formula(y ~ s(x, k=10, bs=:cr) + s(z, k=10, bs=:cr)), df)

        result = anova_gam(m1, m2)

        @test result isa AnovaGamResult
        @test result.test_type == :F
        @test result.smooth_table === nothing
        @test result.model_table !== nothing

        mt = result.model_table
        @test length(mt.resid_df) == 2
        @test length(mt.resid_dev) == 2

        # Models should be sorted by increasing EDF
        @test mt.resid_df[1] >= mt.resid_df[2]
        # Deviance should decrease with more terms
        @test mt.resid_dev[1] >= mt.resid_dev[2]

        # First row should have NaN for comparative stats
        @test isnan(mt.df[1])
        @test isnan(mt.statistic[1])
        @test isnan(mt.p_value[1])

        # Second row should have positive test statistic
        @test mt.df[2] > 0
        @test mt.statistic[2] > 0
        @test 0 <= mt.p_value[2] <= 1

        # Pretty printing
        buf = IOBuffer()
        show(buf, result)
        output = String(take!(buf))
        @test occursin("Analysis of Deviance", output)
        @test occursin("Resid.Df", output)
    end

    # ========================================================================
    # 4. Multi-model comparison (Poisson / Chi-sq test)
    # ========================================================================
    @testset "Multi-model comparison - Poisson" begin
        # Use stronger signal to ensure deviance decreases
        λ_strong = exp.(1.5 .* sin.(x) .+ 1.0 .* cos.(z))
        y_pois_strong = [rand(rng, Poisson(λ_i)) for λ_i in λ_strong]
        df_p = DataFrame(x=x, z=z, y_pois=Float64.(y_pois_strong))

        m1_p = gam(@gam_formula(y_pois ~ s(x, k=10, bs=:cr)),
                   df_p, family=Poisson(), link=LogLink())
        m2_p = gam(@gam_formula(y_pois ~ s(x, k=10, bs=:cr) + s(z, k=10, bs=:cr)),
                   df_p, family=Poisson(), link=LogLink())

        result = anova_gam(m1_p, m2_p)

        @test result.test_type == :Chisq

        mt = result.model_table
        @test mt.resid_dev[1] >= mt.resid_dev[2]
        @test mt.statistic[2] > 0
        @test 0 <= mt.p_value[2] <= 1
    end

    # ========================================================================
    # 5. Three-model comparison
    # ========================================================================
    @testset "Three-model comparison" begin
        m_small = gam(@gam_formula(y ~ s(x, k=5, bs=:cr)), df)
        m_med = gam(@gam_formula(y ~ s(x, k=10, bs=:cr)), df)
        m_large = gam(@gam_formula(y ~ s(x, k=10, bs=:cr) + s(z, k=10, bs=:cr)), df)

        result = anova_gam(m_small, m_med, m_large)

        mt = result.model_table
        @test length(mt.resid_df) == 3

        # Residual deviance should be monotonically non-increasing
        @test mt.resid_dev[1] >= mt.resid_dev[2]
        @test mt.resid_dev[2] >= mt.resid_dev[3]

        # Comparative stats for rows 2 and 3 should exist
        @test !isnan(mt.statistic[2])
        @test !isnan(mt.statistic[3])
    end
end
