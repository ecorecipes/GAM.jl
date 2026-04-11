using Test
using GAM
using DataFrames
using Distributions
using Random
using Statistics

Random.seed!(123)

@testset "Input Validation" begin

    # ====================================================================
    # Helper data
    # ====================================================================
    n = 100
    x = range(0, 2π; length = n)
    y_good = sin.(collect(x)) .+ 0.1 .* randn(n)
    df_good = DataFrame(x = collect(x), y = y_good)

    # ====================================================================
    # Response validation
    # ====================================================================
    @testset "Response: NaN values" begin
        y_nan = copy(y_good)
        y_nan[5] = NaN
        df_nan = DataFrame(x = collect(x), y = y_nan)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_nan)
    end

    @testset "Response: Inf values" begin
        y_inf = copy(y_good)
        y_inf[10] = Inf
        df_inf = DataFrame(x = collect(x), y = y_inf)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_inf)
    end

    @testset "Response: negative Inf" begin
        y_neginf = copy(y_good)
        y_neginf[3] = -Inf
        df_neginf = DataFrame(x = collect(x), y = y_neginf)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_neginf)
    end

    @testset "Response: Poisson requires non-negative" begin
        y_neg = abs.(y_good)
        y_neg[1] = -1.0
        df_neg = DataFrame(x = collect(x), y = y_neg)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_neg; family = Poisson())
    end

    @testset "Response: Gamma requires positive" begin
        y_zero = abs.(y_good) .+ 0.1
        y_zero[1] = 0.0
        df_zero = DataFrame(x = collect(x), y = y_zero)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_zero; family = Gamma())
    end

    @testset "Response: InverseGaussian requires positive" begin
        y_zero = abs.(y_good) .+ 0.1
        y_zero[1] = -0.5
        df_zero = DataFrame(x = collect(x), y = y_zero)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_zero;
            family = InverseGaussian())
    end

    @testset "Response: Binomial requires [0,1]" begin
        y_bin = rand(n)
        y_bin[1] = 1.5
        df_bin = DataFrame(x = collect(x), y = y_bin)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_bin;
            family = Binomial())
    end

    @testset "Response: Bernoulli requires [0,1]" begin
        y_bern = Float64.(rand([0, 1], n))
        y_bern[1] = -0.1
        df_bern = DataFrame(x = collect(x), y = y_bern)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_bern;
            family = Bernoulli())
    end

    # ====================================================================
    # Formula validation
    # ====================================================================
    @testset "Formula: missing smooth variable" begin
        @test_throws ArgumentError gam(@formulak(y ~ s(z)), df_good)
    end

    @testset "Formula: missing by variable" begin
        @test_throws ArgumentError gam(@formulak(y ~ s(x, by = :nonexistent)), df_good)
    end

    @testset "Formula: missing response" begin
        @test_throws ArgumentError gam(@formulak(z ~ s(x)), df_good)
    end

    @testset "Formula: no smooths warning" begin
        # Verify the warning function works directly
        @test_logs (:warn, r"no smooth terms") GAM._validate_has_smooths(SmoothSpec[])
    end

    # ====================================================================
    # Smooth k validation
    # ====================================================================
    @testset "Smooth: k too small" begin
        @test_throws ArgumentError GAM._validate_smooth_k(2, 100, "s(x)")
        @test_throws ArgumentError GAM._validate_smooth_k(1, 100, "s(x)")
    end

    @testset "Smooth: k >= n" begin
        @test_throws ArgumentError GAM._validate_smooth_k(100, 100, "s(x)")
        @test_throws ArgumentError GAM._validate_smooth_k(150, 100, "s(x)")
    end

    @testset "Smooth: k > n/2 warns" begin
        @test_logs (:warn, r"large relative") GAM._validate_smooth_k(60, 100, "s(x)")
    end

    @testset "Smooth: valid k passes" begin
        # Should not throw or warn
        GAM._validate_smooth_k(10, 100, "s(x)")
    end

    # ====================================================================
    # Smooth data validation
    # ====================================================================
    @testset "Smooth data: NaN in predictor" begin
        x_nan = collect(x)
        x_nan[5] = NaN
        df_xnan = DataFrame(x = x_nan, y = y_good)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_xnan)
    end

    @testset "Smooth data: Inf in predictor" begin
        x_inf = collect(x)
        x_inf[5] = Inf
        df_xinf = DataFrame(x = x_inf, y = y_good)
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df_xinf)
    end

    # ====================================================================
    # GAMM validation
    # ====================================================================
    @testset "GAMM: random effect grouping not in data" begin
        @test_throws ArgumentError GAM._validate_gamm_random_effects(
            [GAM.RandomEffectSpec(:nonexistent, Symbol[], true, true, "(1|nonexistent)")],
            df_good)
    end

    @testset "GAMM: no random effects" begin
        @test_throws ArgumentError GAM._validate_gamm_random_effects(
            GAM.RandomEffectSpec[], df_good)
    end

    @testset "GAMM: numeric grouping variable warns" begin
        df_numgroup = DataFrame(x = collect(x), y = y_good,
            group = randn(n))
        @test_logs (:warn, r"numeric") GAM._validate_gamm_random_effects(
            [GAM.RandomEffectSpec(:group, Symbol[], true, true, "(1|group)")],
            df_numgroup)
    end

    @testset "GAMM: slope variable not in data" begin
        df_group = DataFrame(x = collect(x), y = y_good,
            group = repeat(["a", "b"], n ÷ 2))
        @test_throws ArgumentError GAM._validate_gamm_random_effects(
            [GAM.RandomEffectSpec(:group, [:missing_var], true, true, "(missing_var|group)")],
            df_group)
    end

    # ====================================================================
    # SCAM validation
    # ====================================================================
    @testset "SCAM: no shape constraints warns" begin
        specs = [s(:x)]  # regular smooth, not constrained
        @test_logs (:warn, r"shape-constrained") GAM._validate_scam_has_constraints(specs)
    end

    @testset "SCAM: has shape constraints passes" begin
        specs = [s(:x, bs = :mpi)]
        # Should not warn
        GAM._validate_scam_has_constraints(specs)
    end

    # ====================================================================
    # GAMLSS validation
    # ====================================================================
    @testset "GAMLSS: wrong number of formulas" begin
        gf1 = @formulak(y ~ s(x))
        @test_throws ArgumentError GAM._validate_gamlss_formulas(
            [gf1, gf1, gf1],  # 3 formulas
            GammaLocationScale())  # expects 2
    end

    @testset "GAMLSS: correct number of formulas passes" begin
        gf1 = @formulak(y ~ s(x))
        # Should not throw
        GAM._validate_gamlss_formulas([gf1, gf1], GammaLocationScale())
    end

    @testset "GAMLSS: single formula is replicated (no error)" begin
        gf1 = @formulak(y ~ s(x))
        # Single formula should not throw (it gets replicated internally)
        GAM._validate_gamlss_formulas(gf1, GammaLocationScale())
    end

    # ====================================================================
    # Error message quality
    # ====================================================================
    @testset "Error messages are helpful" begin
        # Poisson with negative y — check message content
        y_neg = abs.(y_good)
        y_neg[1] = -5.0
        try
            GAM._validate_response_family(y_neg, Poisson())
            @test false  # should not reach here
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("non-negative", msg)
            @test occursin("Poisson", msg)
            @test occursin("-5.0", msg)
        end

        # Binomial with out-of-range y — check message content
        y_oob = rand(n)
        y_oob[1] = 2.0
        try
            GAM._validate_response_family(y_oob, Binomial())
            @test false
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("[0, 1]", msg) || occursin("[0,1]", msg)
            @test occursin("Binomial", msg) || occursin("Bernoulli", msg)
        end

        # Missing variable — check message mentions column names
        try
            GAM._validate_smooth_vars_in_data(s(:nonexistent), df_good)
            @test false
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("nonexistent", msg)
            @test occursin("x", msg)  # suggests available columns
        end
    end

    # ====================================================================
    # Integration: good data passes through
    # ====================================================================
    @testset "Valid data passes all validation" begin
        # Normal GAM — should succeed
        m = gam(@formulak(y ~ s(x)), df_good)
        @test m isa GAM.GamModel

        # Poisson GAM
        y_count = Float64.(rand(Poisson(5), n))
        df_count = DataFrame(x = collect(x), y = y_count)
        m2 = gam(@formulak(y ~ s(x)), df_count; family = Poisson())
        @test m2 isa GAM.GamModel

        # Binomial GAM
        y_prop = rand(n)
        df_prop = DataFrame(x = collect(x), y = y_prop)
        m3 = gam(@formulak(y ~ s(x)), df_prop; family = Binomial())
        @test m3 isa GAM.GamModel
    end
end
