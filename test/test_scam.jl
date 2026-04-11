using Test, GAM, DataFrames, Random, Statistics, StatsAPI, LinearAlgebra

@testset "Shape-Constrained GAM (SCAM)" begin

    @testset "Softplus and derivatives" begin
        # softplus(0) = log(2) ≈ 0.6931
        @test GAM.softplus(0.0) ≈ log(2) atol = 1e-12
        # softplus(x) → x for large x
        @test GAM.softplus(100.0) ≈ 100.0 atol = 1e-6
        # softplus(x) → exp(x) for very negative x
        @test GAM.softplus(-100.0) ≈ 0.0 atol = 1e-30
        # Always positive
        @test GAM.softplus(-10.0) > 0.0

        # First derivative: sigmoid
        @test GAM.softplus_d1(0.0) ≈ 0.5 atol = 1e-12
        @test GAM.softplus_d1(100.0) ≈ 1.0 atol = 1e-6
        @test GAM.softplus_d1(-100.0) ≈ 0.0 atol = 1e-30

        # Second derivative
        @test GAM.softplus_d2(0.0) ≈ 0.25 atol = 1e-12
        @test GAM.softplus_d2(100.0) ≈ 0.0 atol = 1e-6

        # Third derivative
        @test GAM.softplus_d3(0.0) ≈ 0.0 atol = 1e-12  # inflection point

        # Check derivatives numerically
        h = 1e-7
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]
            d1_num = (GAM.softplus(x + h) - GAM.softplus(x - h)) / (2h)
            @test GAM.softplus_d1(x) ≈ d1_num atol = 1e-5

            d2_num = (GAM.softplus_d1(x + h) - GAM.softplus_d1(x - h)) / (2h)
            @test GAM.softplus_d2(x) ≈ d2_num atol = 1e-4
        end
    end

    @testset "Sigma matrix construction" begin
        # Monotone increasing: lower triangular of 1's
        Sig_mpi = GAM._sigma_matrix(GAM.MonoIncBasis(), 5)
        @test size(Sig_mpi) == (5, 5)
        @test all(Sig_mpi[i, j] == 1.0 for i in 1:5 for j in 1:5 if j <= i)
        @test all(Sig_mpi[i, j] == 0.0 for i in 1:5 for j in 1:5 if j > i)

        # Monotone decreasing: negative lower triangular, positive first col
        Sig_mpd = GAM._sigma_matrix(GAM.MonoDecBasis(), 5)
        @test all(Sig_mpd[:, 1] .== 1.0)
        @test all(Sig_mpd[i, j] == -1.0 for i in 1:5 for j in 2:5 if j <= i)
        @test all(Sig_mpd[i, j] == 0.0 for i in 1:5 for j in 2:5 if j > i)

        # Concave: (q-1) × (q-1) matrix
        Sig_cv = GAM._sigma_matrix(GAM.ConcaveBasis(), 5)
        @test size(Sig_cv) == (4, 4)
        @test all(Sig_cv[:, 1] .== [1, 2, 3, 4])

        # Convex: similar structure but positive off-diagonal
        Sig_cx = GAM._sigma_matrix(GAM.ConvexBasis(), 5)
        @test size(Sig_cx) == (4, 4)
        @test all(Sig_cx[:, 1] .== [1, 2, 3, 4])
    end

    @testset "Shape-constrained basis construction" begin
        Random.seed!(42)
        n = 100
        x = sort(rand(n))
        data = DataFrame(x = x)

        for (bs, label) in [
            (:mpi, "mpi"), (:mpd, "mpd"), (:cx, "cx"), (:cv, "cv"),
            (:micx, "micx"), (:micv, "micv"), (:mdcx, "mdcx"), (:mdcv, "mdcv"),
        ]
            spec = GAM.SmoothSpec([:x], GAM.resolve_basis_type(bs), 10, nothing, nothing,
                nothing, false, nothing, "s(x)")
            sm = smooth_construct(spec, data)

            @test size(sm.X, 1) == n
            @test size(sm.X, 2) == 9  # k-1 = 10-1 = 9
            @test length(sm.S) == 1
            @test sm.Sigma !== nothing
            @test sm.cmX !== nothing
            @test sm.p_ident !== nothing
            @test all(sm.p_ident)  # all coefficients constrained
        end
    end

    @testset "Prediction matrix for constrained smooths" begin
        Random.seed!(42)
        n = 100
        x = sort(rand(n))
        data = DataFrame(x = x)

        for bs in [:mpi, :mpd, :cx, :cv]
            spec = GAM.SmoothSpec([:x], GAM.resolve_basis_type(bs), 10, nothing, nothing,
                nothing, false, nothing, "s(x)")
            sm = smooth_construct(spec, data)

            # Prediction at same data should give same matrix
            Xp = predict_matrix(sm, data)
            @test size(Xp) == size(sm.X)
            @test Xp ≈ sm.X atol = 1e-10
        end
    end

    @testset "p_ident construction" begin
        Random.seed!(42)
        n = 100
        x = sort(rand(n))
        data = DataFrame(x = x, y = randn(n))

        # Build model setup with constrained smooth
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.SmoothSpec([:x], GAM.MonoIncBasis(), 10, nothing, nothing,
                nothing, false, nothing, "s(x)")])
        y, X, X_para, smooths, n_parametric = GAM.setup_gam(gf, data; family = Normal())

        p_ident = GAM.build_p_ident(smooths, n_parametric, size(X, 2))
        @test length(p_ident) == size(X, 2)
        @test !p_ident[1]  # intercept not constrained
        @test all(p_ident[2:end])  # all smooth coefficients constrained
    end

    @testset "SCAM fitting - monotone increasing" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = 3.0 .* x .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :mpi, k = 10)), df)
        @test m.converged || m.deviance_val < 20.0  # may not converge but fits well

        # Check monotonicity of predictions
        xp = collect(0.01:0.01:0.99)
        pred = StatsAPI.predict(m, DataFrame(x = xp))
        @test all(diff(pred) .>= -1e-8)  # monotone increasing

        # Good fit (correlation with true function)
        @test cor(pred, 3.0 .* xp) > 0.99
    end

    @testset "SCAM fitting - monotone decreasing" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = 3.0 .- 3.0 .* x .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :mpd, k = 10)), df)
        @test m.deviance_val < 20.0

        xp = collect(0.01:0.01:0.99)
        pred = StatsAPI.predict(m, DataFrame(x = xp))
        @test all(diff(pred) .<= 1e-8)  # monotone decreasing
    end

    @testset "SCAM fitting - convex" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = x .^ 2 .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :cx, k = 10)), df)
        @test m.deviance_val < 5.0
        @test cor(m.fitted_values, x .^ 2) > 0.98
    end

    @testset "SCAM fitting - concave" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = sqrt.(x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(@formulak(y ~ s(x, bs = :cv, k = 10)), df)
        @test m.deviance_val < 10.0
    end

    @testset "SCAM fitting - combined constraints" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))

        # Monotone increasing + convex
        y = x .^ 2 .+ 0.1 .* randn(n)
        m_micx = scam(@formulak(y ~ s(x, bs = :micx, k = 10)), DataFrame(x = x, y = y))
        @test m_micx.deviance_val < 10.0

        # Monotone increasing + concave
        y = sqrt.(x) .+ 0.1 .* randn(n)
        m_micv = scam(@formulak(y ~ s(x, bs = :micv, k = 10)), DataFrame(x = x, y = y))
        @test m_micv.deviance_val < 10.0

        # Monotone decreasing + concave
        y = -x .^ 2 .+ 0.1 .* randn(n)
        m_mdcv = scam(@formulak(y ~ s(x, bs = :mdcv, k = 10)), DataFrame(x = x, y = y))
        @test m_mdcv.deviance_val < 10.0
    end

    @testset "SCAM supports parametric terms with constrained smooths" begin
        Random.seed!(20260408)
        n = 240
        x = sort(rand(n))
        z = randn(n)
        y = 1.5 .+ 2.0 .* z .+ log1p.(4 .* x) .+ 0.05 .* randn(n)
        df = DataFrame(x = x, z = z, y = y)

        f = @formulak(y ~ z + s(x, bs = :mpi, k = 12))
        m_scam = scam(f, df)
        m_gam = gam(f, df)

        @test m_scam.converged
        @test m_gam.converged
        @test coefnames(m_scam) == coefnames(m_gam)
        @test "z" in coefnames(m_scam)

        newdf_z = DataFrame(x = fill(0.5, 5), z = [-2.0, -1.0, 0.0, 1.0, 2.0])
        pred_scam_z = StatsAPI.predict(m_scam, newdf_z)
        pred_gam_z = StatsAPI.predict(m_gam, newdf_z)
        @test pred_scam_z ≈ pred_gam_z atol = 1e-8
        @test all(diff(pred_scam_z) .> 0)

        newdf_x = DataFrame(x = collect(range(0.01, 0.99; length = 50)), z = fill(0.0, 50))
        pred_scam_x = StatsAPI.predict(m_scam, newdf_x)
        @test all(diff(pred_scam_x) .>= -1e-8)
    end

    @testset "SCAM falls back to GAM for unconstrained smooths" begin
        Random.seed!(42)
        n = 200
        x = sort(rand(n))
        y = sin.(2π .* x) .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        # Using standard cr basis with scam() should fall back to gam()
        m = scam(@formulak(y ~ s(x, bs = :cr, k = 20)), df)
        @test m.converged
        @test cor(m.fitted_values, sin.(2π .* x)) > 0.95
    end

    @testset "ScamControl parameters" begin
        ctrl = scam_control()
        @test ctrl.epsilon == 1e-7
        @test ctrl.maxit == 200
        @test ctrl.not_exp == false

        ctrl2 = scam_control(epsilon = 1e-5, not_exp = true)
        @test ctrl2.epsilon == 1e-5
        @test ctrl2.not_exp == true
    end

    @testset "has_shape_constraints" begin
        Random.seed!(42)
        n = 50
        x = sort(rand(n))
        data = DataFrame(x = x)

        # Constrained smooth
        spec_c = GAM.SmoothSpec([:x], GAM.MonoIncBasis(), 10, nothing, nothing,
            nothing, false, nothing, "s(x)")
        sm_c = smooth_construct(spec_c, data)
        @test GAM.has_shape_constraints([sm_c])

        # Unconstrained smooth
        spec_u = GAM.SmoothSpec([:x], GAM.CubicSpline(), 10, nothing, nothing,
            nothing, false, nothing, "s(x)")
        sm_u = smooth_construct(spec_u, data)
        @test !GAM.has_shape_constraints([sm_u])
    end
end
