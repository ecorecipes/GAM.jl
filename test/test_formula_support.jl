@testset "@formula smooth term support" begin
    using StableRNGs
    rng = StableRNG(123)

    n = 100
    x = range(0, 2π; length = n)
    x2 = randn(rng, n)
    y = sin.(x) .+ 0.3 .* randn(rng, n)
    df = DataFrame(x = collect(x), x2 = x2, y = y)

    @testset "s() in @formula" begin
        # Basic smooth
        m = gam(@formula(y ~ s(x)), df)
        @test length(coef(m)) == 10  # intercept + 9 basis
        @test deviance(m) < sum((y .- mean(y)) .^ 2)  # better than null

        # Identical to @gam_formula
        m_g = gam(@gam_formula(y ~ s(x)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "s(x, k) positional k" begin
        m = gam(@formula(y ~ s(x, 20)), df)
        @test length(coef(m)) == 20

        m_g = gam(@gam_formula(y ~ s(x, k = 20)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "Multiple smooths" begin
        m = gam(@formula(y ~ s(x) + s(x2)), df)
        @test length(coef(m)) == 19  # 1 intercept + 9 + 9

        m_g = gam(@gam_formula(y ~ s(x) + s(x2)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "Parametric + smooth" begin
        m = gam(@formula(y ~ x2 + s(x)), df)
        @test length(coef(m)) == 11  # intercept + x2 + 9 basis
    end

    @testset "Basis-type aliases" begin
        # cr()
        m_cr = gam(@formula(y ~ cr(x)), df)
        m_cr_g = gam(@gam_formula(y ~ s(x, bs = :cr)), df)
        @test isapprox(coef(m_cr), coef(m_cr_g), atol = 1e-10)

        # cr(x, 20)
        m_cr20 = gam(@formula(y ~ cr(x, 20)), df)
        m_cr20_g = gam(@gam_formula(y ~ s(x, bs = :cr, k = 20)), df)
        @test isapprox(coef(m_cr20), coef(m_cr20_g), atol = 1e-10)

        # tp()
        m_tp = gam(@formula(y ~ tp(x, 15)), df)
        m_tp_g = gam(@gam_formula(y ~ s(x, bs = :tp, k = 15)), df)
        @test isapprox(coef(m_tp), coef(m_tp_g), atol = 1e-10)

        # ps()
        m_ps = gam(@formula(y ~ ps(x)), df)
        m_ps_g = gam(@gam_formula(y ~ s(x, bs = :ps)), df)
        @test isapprox(coef(m_ps), coef(m_ps_g), atol = 1e-10)

        # ts() — thin plate with shrinkage
        m_ts = gam(@formula(y ~ ts(x)), df)
        @test length(coef(m_ts)) == 10

        # cs() — cubic with shrinkage
        m_cs = gam(@formula(y ~ cs(x)), df)
        @test length(coef(m_cs)) == 10

        # cc() — cyclic cubic (k-1 coefs due to cyclic constraint)
        m_cc = gam(@formula(y ~ cc(x)), df)
        @test length(coef(m_cc)) == 9
    end

    @testset "te() in @formula" begin
        m = gam(@formula(y ~ te(x, x2)), df)
        m_g = gam(@gam_formula(y ~ te(x, x2)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "Mixed aliases + smooths" begin
        m = gam(@formula(y ~ cr(x) + s(x2)), df)
        @test length(coef(m)) == 19  # 1 + 9 + 9
    end

    @testset "Non-Gaussian families" begin
        df.count = rand.(rng, Poisson.(exp.(0.5 .* sin.(x))))

        m = gam(@formula(count ~ s(x)), df; family = Poisson())
        m_g = gam(@gam_formula(count ~ s(x)), df; family = Poisson())
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "_functionterm_to_smoothspec" begin
        ft = @formula(y ~ s(x)).rhs
        spec = GAM._functionterm_to_smoothspec(ft)
        @test spec isa SmoothSpec
        @test spec.term_vars == [:x]
        @test spec.k == 10  # default

        ft2 = @formula(y ~ s(x, 20)).rhs
        spec2 = GAM._functionterm_to_smoothspec(ft2)
        @test spec2.k == 20

        ft3 = @formula(y ~ cr(x)).rhs
        spec3 = GAM._functionterm_to_smoothspec(ft3)
        @test spec3.basis isa CubicSpline
    end

    @testset "_is_smooth_function" begin
        @test GAM._is_smooth_function(s)
        @test GAM._is_smooth_function(te)
        @test GAM._is_smooth_function(ti)
        @test GAM._is_smooth_function(cr)
        @test GAM._is_smooth_function(tp)
        @test GAM._is_smooth_function(ps)
        @test GAM._is_smooth_function(ts)
        @test GAM._is_smooth_function(cs)
        @test GAM._is_smooth_function(cc)
        @test !GAM._is_smooth_function(sin)
        @test !GAM._is_smooth_function(log)
    end
end
