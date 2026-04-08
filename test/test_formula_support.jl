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

    @testset "No-intercept smooths" begin
        m = gam(@formula(y ~ 0 + s(x)), df)
        m_g = gam(@gam_formula(y ~ 0 + s(x)), df)

        @test m.n_parametric == 0
        @test m_g.n_parametric == 0

        newdf = DataFrame(x = collect(range(0.15, 6.0; length = 7)))
        pred = predict(m, newdf; type = :link)
        pred_g = predict(m_g, newdf; type = :link)

        X_new = predict_matrix(m.smooths[1], newdf)
        X_new_g = predict_matrix(m_g.smooths[1], newdf)

        @test pred ≈ X_new * coef(m) atol = 1e-8
        @test pred_g ≈ X_new_g * coef(m_g) atol = 1e-8
        @test pred ≈ pred_g atol = 1e-8
    end

    @testset "Parametric terms affect prediction" begin
        y_lin = 1 .+ 2 .* x2 .+ sin.(x) .+ 0.05 .* randn(rng, n)
        df_lin = DataFrame(x = collect(x), x2 = x2, y = y_lin)

        m = gam(@formula(y ~ x2 + s(x)), df_lin)
        m_g = gam(@gam_formula(y ~ x2 + s(x)), df_lin)

        newdf = DataFrame(x = fill(π, 3), x2 = [-1.0, 0.0, 1.0])
        pred = predict(m, newdf; type = :link)
        pred_g = predict(m_g, newdf; type = :link)

        X_new = hcat(
            ones(3, 1),
            reshape(Float64.(newdf.x2), :, 1),
            predict_matrix(m.smooths[1], newdf),
        )
        X_new_g = hcat(
            ones(3, 1),
            reshape(Float64.(newdf.x2), :, 1),
            predict_matrix(m_g.smooths[1], newdf),
        )

        @test pred ≈ vec(X_new * coef(m)) atol = 1e-8
        @test pred_g ≈ vec(X_new_g * coef(m_g)) atol = 1e-8
        @test pred[1] != pred[3]
        @test pred ≈ pred_g atol = 1e-8
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

    @testset "ti() in @formula" begin
        m = gam(@formula(y ~ ti(x, x2)), df)
        m_g = gam(@gam_formula(y ~ ti(x, x2)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "t2() in @formula" begin
        m = gam(@formula(y ~ t2(x, x2)), df)
        m_g = gam(@gam_formula(y ~ t2(x, x2)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "te() with positional k" begin
        m = gam(@formula(y ~ te(x, x2, 8)), df)
        m_g = gam(@gam_formula(y ~ te(x, x2, k = 8)), df)
        @test isapprox(coef(m), coef(m_g), atol = 1e-10)
    end

    @testset "Smooth + tensor product" begin
        m = gam(@formula(y ~ s(x) + s(x2)), df)
        @test length(coef(m)) == 19  # intercept + 9 + 9
        @test deviance(m) < sum((y .- mean(y)) .^ 2)

        # te with separate variable set
        m2 = gam(@formula(y ~ te(x, x2)), df)
        @test length(coef(m2)) > 1
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

    @testset "apply_schema pipeline" begin
        # FunctionTerm{typeof(s)} is converted to AppliedSmoothTerm via apply_schema
        f = @formula(y ~ s(x))
        sch = schema(f, df)
        af = apply_schema(f, sch)
        @test af.rhs isa StatsModels.MatrixTerm
        inner = af.rhs.terms[1]
        @test inner isa GAM.AppliedSmoothTerm
        @test inner.spec.term_vars == [:x]

        # modelcols works through the standard pipeline
        y_cols, X_cols = modelcols(af, df)
        @test length(y_cols) == n
        @test size(X_cols) == (n, 9)  # 9 basis functions (k=10 minus constraint)

        # te() also converts
        f_te = @formula(y ~ te(x, x2))
        af_te = apply_schema(f_te, schema(f_te, df))
        @test af_te.rhs.terms[1] isa GAM.AppliedSmoothTerm

        # ti() converts
        f_ti = @formula(y ~ ti(x, x2))
        af_ti = apply_schema(f_ti, schema(f_ti, df))
        @test af_ti.rhs.terms[1] isa GAM.AppliedSmoothTerm

        # t2() converts
        f_t2 = @formula(y ~ t2(x, x2))
        af_t2 = apply_schema(f_t2, schema(f_t2, df))
        @test af_t2.rhs.terms[1] isa GAM.AppliedSmoothTerm

        # cr() alias converts
        f_cr = @formula(y ~ cr(x))
        af_cr = apply_schema(f_cr, schema(f_cr, df))
        @test af_cr.rhs.terms[1] isa GAM.AppliedSmoothTerm
        @test af_cr.rhs.terms[1].spec.basis isa CubicSpline

        # Mixed parametric + smooth
        f_mix = @formula(y ~ x2 + s(x))
        af_mix = apply_schema(f_mix, schema(f_mix, df))
        terms_mix = af_mix.rhs.terms
        @test terms_mix[1] isa ContinuousTerm
        @test terms_mix[2] isa GAM.AppliedSmoothTerm
        _, X_mix = modelcols(af_mix, df)
        @test size(X_mix, 2) == 10  # 1 parametric + 9 basis

        # Multiple smooths
        f_multi = @formula(y ~ s(x) + s(x2))
        af_multi = apply_schema(f_multi, schema(f_multi, df))
        @test all(t -> t isa GAM.AppliedSmoothTerm, af_multi.rhs.terms)
        _, X_multi = modelcols(af_multi, df)
        @test size(X_multi, 2) == 18  # 9 + 9

        # coefnames are generated
        cn = coefnames(af.rhs)
        @test length(cn) == 9
        @test all(c -> startswith(c, "s(x,bs=tp)"), cn)
    end

    @testset "@formula vs @gam_formula fitted values" begin
        m1 = gam(@formula(y ~ s(x)), df)
        m2 = gam(@gam_formula(y ~ s(x)), df)
        @test cor(fitted(m1), fitted(m2)) > 0.999

        m3 = gam(@formula(y ~ s(x, 15)), df)
        m4 = gam(@gam_formula(y ~ s(x, k = 15)), df)
        @test cor(fitted(m3), fitted(m4)) > 0.999

        m5 = gam(@formula(y ~ te(x, x2)), df)
        m6 = gam(@gam_formula(y ~ te(x, x2)), df)
        @test cor(fitted(m5), fitted(m6)) > 0.999

        m7 = gam(@formula(y ~ s(x) + s(x2)), df)
        m8 = gam(@gam_formula(y ~ s(x) + s(x2)), df)
        @test cor(fitted(m7), fitted(m8)) > 0.999
    end

    @testset "GAMM with @formula" begin
        df.group = repeat(1:5, 20)
        m = gamm(@formula(y ~ s(x) + (1 | group)), df)
        @test m isa GammModel
        @test length(m.random_effects) == 1
        @test m.random_effects[1].spec.grouping == :group
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

        ft4 = @formula(y ~ te(x, x2)).rhs
        spec4 = GAM._functionterm_to_smoothspec(ft4)
        @test spec4.term_vars == [:x, :x2]
        @test spec4.basis isa TensorProduct

        ft5 = @formula(y ~ ti(x, x2)).rhs
        spec5 = GAM._functionterm_to_smoothspec(ft5)
        @test spec5.basis isa TensorInteraction

        ft6 = @formula(y ~ t2(x, x2)).rhs
        spec6 = GAM._functionterm_to_smoothspec(ft6)
        @test spec6.basis isa T2TensorProduct
    end

    @testset "_is_smooth_function" begin
        @test GAM._is_smooth_function(s)
        @test GAM._is_smooth_function(te)
        @test GAM._is_smooth_function(ti)
        @test GAM._is_smooth_function(t2)
        @test GAM._is_smooth_function(cr)
        @test GAM._is_smooth_function(tp)
        @test GAM._is_smooth_function(ps)
        @test GAM._is_smooth_function(ts)
        @test GAM._is_smooth_function(cs)
        @test GAM._is_smooth_function(cc)
        @test GAM._is_smooth_function(cps)
        @test !GAM._is_smooth_function(sin)
        @test !GAM._is_smooth_function(log)
    end
end
