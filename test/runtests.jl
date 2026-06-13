using Test
using GAM
using GAM: @formula
using DataFrames
using Distributions
using StableRNGs
using LinearAlgebra
using Statistics
using StatsAPI: fitted, nobs, deviance, dof, dof_residual, loglikelihood, aic, bic,
    coef, residuals, predict

const rng = StableRNG(42)

function simulate_tweedie(rng, mu::AbstractVector{<:Real}, p::Real, phi::Real)
    1.0 < p < 2.0 || throw(ArgumentError("simulate_tweedie requires 1 < p < 2"))
    phi > 0 || throw(ArgumentError("simulate_tweedie requires phi > 0"))

    alpha = (2.0 - p) / (p - 1.0)
    y = Vector{Float64}(undef, length(mu))
    @inbounds for i in eachindex(mu)
        mui = Float64(mu[i])
        lambda = mui^(2.0 - p) / (phi * (2.0 - p))
        gamma_scale = phi * (p - 1.0) * mui^(p - 1.0)
        n_terms = rand(rng, Poisson(lambda))
        total = 0.0
        if n_terms > 0
            dist = Gamma(alpha, gamma_scale)
            for _ in 1:n_terms
                total += rand(rng, dist)
            end
        end
        y[i] = total
    end
    return y
end

@testset "GAM.jl" begin
    @testset "SmoothSpec construction" begin
        sp = s(:x)
        @test sp isa SmoothSpec
        @test sp.term_vars == [:x]
        @test sp.basis isa ThinPlateSpline
        @test sp.k == 10
        @test sp.fx == false
        @test sp.by === nothing

        sp2 = s(:x, bs = :cr, k = 20)
        @test sp2.basis isa CubicSpline
        @test sp2.k == 20

        sp3 = s(:x, :y)
        @test sp3.term_vars == [:x, :y]
        @test sp3.k == 30  # 2d default for TPRS

        sp4 = s(:x, bs = :ps, k = 15, fx = true)
        @test sp4.basis isa PSpline
        @test sp4.fx == true
        @test sp4.k == 15

        @test_throws ArgumentError s(:x, bs = :nonexistent)
    end

    @testset "Knot placement" begin
        x = collect(1.0:100.0)
        knots = GAM.place_knots(x, 10)
        @test length(knots) == 10
        @test knots[1] ≈ 1.0
        @test knots[end] ≈ 100.0
        @test issorted(knots)

        # With fewer unique values than k
        x_small = [1.0, 2.0, 3.0]
        knots_small = GAM.place_knots(x_small, 10)
        @test length(knots_small) == 3
    end

    @testset "TPRS basis construction" begin
        n = 100
        x = range(0, 2π; length = n)
        data = (x = collect(x),)

        spec = s(:x, k = 10)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 9  # k=10, minus 1 for constraint
        @test length(sm.S) == 1  # single penalty for tp
        @test size(sm.S[1]) == (9, 9)
        @test sm.null_dim == 2  # linear + constant for m=2
        @test issymmetric(sm.S[1]) || norm(sm.S[1] - sm.S[1]') < 1e-10

        Xp = predict_matrix(sm, data)
        @test size(Xp) == size(sm.X)
        @test Xp ≈ sm.X atol = 1e-10
    end

    @testset "Multi-D TPRS prediction" begin
        n = 120
        x = randn(rng, n)
        z = randn(rng, n)
        y_true = sin.(x) .* cos.(z)
        y = y_true .+ 0.1 .* randn(rng, n)
        data = (x = x, z = z)

        spec = s(:x, :z, bs = :tp, k = 20)
        sm = smooth_construct(spec, data)
        Xp = predict_matrix(sm, data)
        @test size(Xp) == size(sm.X)
        @test Xp ≈ sm.X atol = 1e-10

        newdata = (x = randn(rng, 25), z = randn(rng, 25))
        Xp_new = predict_matrix(sm, newdata)
        @test size(Xp_new) == (25, size(sm.X, 2))
        @test all(isfinite, Xp_new)

        df = DataFrame(x = x, z = z, y = y)
        m = gam(@formulak(y ~ s(x, z, bs = :tp, k = 20)), df)
        @test m.converged
        pred = predict(m, df; type = :response)
        @test pred ≈ m.fitted_values atol = 1e-8
    end

    @testset "CR basis construction" begin
        n = 100
        x = range(0, 1; length = n)
        data = (x = collect(x),)

        spec = s(:x, bs = :cr, k = 10)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 9  # k=10, minus 1 for constraint
        @test length(sm.S) == 1
    end

    @testset "P-spline basis construction" begin
        n = 100
        x = range(0, 1; length = n)
        data = (x = collect(x),)

        spec = s(:x, bs = :ps, k = 10)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        @test length(sm.S) == 1
        @test issymmetric(sm.S[1]) || norm(sm.S[1] - sm.S[1]') < 1e-10
    end

    @testset "Penalty setup" begin
        n = 100
        data = (x = collect(range(0, 1; length = n)),)

        spec1 = s(:x, k = 10)
        sm1 = smooth_construct(spec1, data)

        penalty = GAM.setup_penalties([sm1], 1)
        @test length(penalty.blocks) == 1
        @test length(penalty.sp) == 1
        @test penalty.blocks[1].start == 2
        @test penalty.blocks[1].stop == 10  # 1 intercept + 9 smooth params
    end

    @testset "P-IRLS Gaussian" begin
        # Simple linear relationship to verify P-IRLS works
        n = 100
        x = randn(rng, n)
        y = 2.0 .* x .+ 0.5 .* randn(rng, n)
        X = hcat(ones(n), x)

        S_total = zeros(2, 2)  # no penalty
        result = GAM.pirls(X, y, S_total, Normal(), IdentityLink())

        @test result.converged
        @test abs(result.coefficients[2] - 2.0) < 0.2  # slope ≈ 2
        @test result.deviance > 0
    end

    @testset "GAM fit — Gaussian sine curve" begin
        n = 100
        x = range(0, 2π; length = n) |> collect
        y_true = sin.(x)
        y = y_true .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        # Fit with a smooth term specified directly
        spec = s(:x, k = 15)
        data = Tables.columntable(df)
        sm = smooth_construct(spec, data)
        sm.first_para = 2
        sm.last_para = 1 + size(sm.X, 2)

        X = hcat(ones(n), sm.X)
        penalty = GAM.setup_penalties([sm], 1)
        p = size(X, 2)
        S_total = GAM.total_penalty(penalty, penalty.sp, p)

        result = GAM.pirls(X, y, S_total, Normal(), IdentityLink())
        @test result.converged

        # Predictions should roughly follow sine curve
        mu = result.fitted_values
        rmse = sqrt(mean((mu .- y_true) .^ 2))
        @test rmse < 0.5  # reasonable fit
    end

    @testset "GamControl" begin
        ctrl = gam_control()
        @test ctrl.epsilon == 1e-7
        @test ctrl.maxit == 200
        @test ctrl.gamma == 1.0

        ctrl2 = gam_control(epsilon = 1e-8, maxit = 100, trace = true)
        @test ctrl2.epsilon == 1e-8
        @test ctrl2.maxit == 100
        @test ctrl2.trace == true
    end

    @testset "Deviance functions" begin
        y = [1.0, 2.0, 3.0]
        mu = [1.1, 1.9, 3.1]
        wt = [1.0, 1.0, 1.0]

        # Gaussian deviance = RSS
        dev_gauss = GAM._deviance(Normal(), y, mu, wt)
        @test dev_gauss ≈ sum((y .- mu) .^ 2)

        # Poisson deviance
        y_p = [1.0, 5.0, 10.0]
        mu_p = [1.5, 4.0, 11.0]
        dev_pois = GAM._deviance(Poisson(), y_p, mu_p, wt)
        @test dev_pois > 0
        @test isfinite(dev_pois)
    end

    @testset "Absorb constraints" begin
        X = randn(rng, 50, 5)
        S = [Matrix{Float64}(I, 5, 5)]
        X_c, S_c, C, _ = GAM.absorb_constraints!(X, S)

        @test size(X_c, 2) == 4  # one column removed
        @test size(S_c[1]) == (4, 4)
        @test size(C) == (1, 5)
    end

    @testset "Tensor product smooths (te)" begin
        n = 100
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        data = (x1=x1, x2=x2)

        # Basic te() construction
        spec_te = te(:x1, :x2)
        sm_te = smooth_construct(spec_te, data)

        @test sm_te isa ConstructedSmooth
        @test sm_te.spec.basis isa TensorProduct
        @test size(sm_te.X, 1) == n
        @test size(sm_te.X, 2) > 0
        @test length(sm_te.S) >= 2  # one penalty per margin

        # Penalties should be symmetric positive semidefinite
        for S in sm_te.S
            @test size(S, 1) == size(sm_te.X, 2)
            @test norm(S - S') < 1e-10
            @test all(eigvals(Symmetric(S)) .>= -1e-10)
        end

        # Prediction at new data
        newdata = (x1=randn(rng, 50), x2=randn(rng, 50))
        Xp = predict_matrix(sm_te, newdata)
        @test size(Xp) == (50, size(sm_te.X, 2))

        # te() with explicit k
        spec_te_k = te(:x1, :x2, k=16)
        sm_te_k = smooth_construct(spec_te_k, data)
        @test size(sm_te_k.X, 1) == n
        @test size(sm_te_k.X, 2) > 0

        # te() with P-spline margins
        spec_te_ps = te(:x1, :x2, bs=:ps)
        sm_te_ps = smooth_construct(spec_te_ps, data)
        @test size(sm_te_ps.X, 1) == n
        @test length(sm_te_ps.S) >= 2
    end

    @testset "Tensor product interaction (ti)" begin
        n = 100
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        data = (x1=x1, x2=x2)

        spec_te = te(:x1, :x2)
        sm_te = smooth_construct(spec_te, data)

        spec_ti = ti(:x1, :x2)
        sm_ti = smooth_construct(spec_ti, data)

        @test sm_ti isa ConstructedSmooth
        @test sm_ti.spec.basis isa TensorInteraction
        @test size(sm_ti.X, 1) == n
        @test size(sm_ti.X, 2) > 0

        # ti() should have fewer columns than te() (main effects removed)
        @test size(sm_ti.X, 2) < size(sm_te.X, 2)

        # Penalties should be well-formed
        for S in sm_ti.S
            @test size(S, 1) == size(sm_ti.X, 2)
            @test norm(S - S') < 1e-10
        end

        # Prediction
        newdata = (x1=randn(rng, 50), x2=randn(rng, 50))
        Xp_ti = predict_matrix(sm_ti, newdata)
        @test size(Xp_ti) == (50, size(sm_ti.X, 2))
    end

    @testset "Tensor product spec construction" begin
        # te() requires at least 2 variables
        @test_throws ArgumentError te(:x1)

        # ti() requires at least 2 variables
        @test_throws ArgumentError ti(:x1)

        # te() stores correct vars
        spec = te(:x1, :x2, :x3)
        @test spec.term_vars == [:x1, :x2, :x3]
        @test spec.basis isa TensorProduct

        # ti() stores correct vars
        spec_ti = ti(:x1, :x2)
        @test spec_ti.term_vars == [:x1, :x2]
        @test spec_ti.basis isa TensorInteraction

        # Per-margin basis types
        spec_mixed = te(:x1, :x2, bs=[:cr, :ps])
        @test spec_mixed.basis isa TensorProduct
    end

    @testset "Tensor product @formulak integration" begin
        n = 100
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        y = sin.(x1) .* cos.(x2) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x1=x1, x2=x2, y=y)

        # @formulak with te()
        gf = @formulak(y ~ te(x1, x2))
        @test length(gf.smooth_specs) == 1
        @test gf.smooth_specs[1].basis isa TensorProduct

        # @formulak with ti()
        gf_ti = @formulak(y ~ ti(x1, x2))
        @test length(gf_ti.smooth_specs) == 1
        @test gf_ti.smooth_specs[1].basis isa TensorInteraction

        # Full GAM fit with te()
        m = gam(gf, df)
        @test m.converged
        @test m.n_smooth == 1
        @test length(m.edf) == 1

        # Fitted values should be reasonable for a 2d surface
        y_true = sin.(x1) .* cos.(x2)
        rmse = sqrt(mean((m.fitted_values .- y_true) .^ 2))
        @test rmse < 1.0
    end

    @testset "GP smooth basis (bs=:gp)" begin
        n = 100
        x = range(0, 2π; length=n) |> collect
        data = (x=x,)

        spec = s(:x, bs=:gp, k=10)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth
        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 9  # k=10 minus 1 for constraint
        @test length(sm.S) == 1
        @test issymmetric(sm.S[1]) || norm(sm.S[1] - sm.S[1]') < 1e-10

        # Prediction at new data
        newdata = (x=range(0, 2π; length=50) |> collect,)
        Xp = predict_matrix(sm, newdata)
        @test size(Xp) == (50, size(sm.X, 2))
    end

    @testset "GP smooth in GAM fit" begin
        n = 100
        x = range(0, 2π; length=n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, bs=:gp, k=15)), df)
        @test m isa GamModel
        @test m.converged
        @test m.n_smooth == 1

        y_true = sin.(x)
        rmse = sqrt(mean((m.fitted_values .- y_true) .^ 2))
        @test rmse < 0.5
    end
end

@testset "Extended families" begin
    @testset "NegBinFamily type construction" begin
        f = NegBinFamily()
        @test f isa ExtendedFamily
        @test f.theta == 1.0
        @test GAM._default_link(f) isa LogLink
        @test f.estimate_theta == true

        f2 = NegBinFamily(theta=5.0, estimate_theta=false)
        @test f2.theta == 5.0
        @test f2.estimate_theta == false
    end

    @testset "Quasi family type construction" begin
        f = QuasiPoissonFamily()
        @test f isa ExtendedFamily
        @test GAM._default_link(f) isa LogLink
        @test GAM._has_extra_param(f) == false

        f2 = QuasiBinomialFamily()
        @test f2 isa ExtendedFamily
        @test GAM._default_link(f2) isa LogitLink
        @test GAM._has_extra_param(f2) == false
    end

    @testset "TweedieFamily type construction" begin
        f = TweedieFamily()
        @test f isa ExtendedFamily
        @test f.p == 1.5
        @test GAM._default_link(f) isa LogLink
        @test f.estimate_p == false

        f2 = TweedieFamily(p=1.8)
        @test f2.p == 1.8

        f3 = TweedieFamily(p=1.3, estimate_p=true)
        @test f3.p == 1.3
        @test f3.estimate_p == true
    end

    @testset "BetaFamily type construction" begin
        f = BetaFamily()
        @test f isa ExtendedFamily
        @test f.phi == 1.0
        @test GAM._default_link(f) isa LogitLink
        @test f.estimate_phi == true

        f2 = BetaFamily(phi=10.0, estimate_phi=false)
        @test f2.phi == 10.0
    end

    @testset "NegBin variance and deviance" begin
        mu = [1.0, 2.0, 5.0]
        f = NegBinFamily(theta=2.0)

        # Variance: mu + mu^2/theta
        v = GAM._variance(f, mu)
        @test v ≈ mu .+ mu .^ 2 ./ 2.0

        # Deviance should be non-negative
        y = [1.0, 3.0, 4.0]
        wt = ones(3)
        dev = GAM._deviance(f, y, mu, wt)
        @test dev >= 0
        @test isfinite(dev)

        # Deviance at mu=y should be 0
        dev_sat = GAM._deviance(f, mu, mu, wt)
        @test dev_sat ≈ 0.0 atol = 1e-10
    end

    @testset "Quasi variance and deviance" begin
        wt = ones(3)

        mu_count = [1.0, 2.0, 5.0]
        y_count = [1.0, 3.0, 4.0]
        f_count = QuasiPoissonFamily()
        @test GAM._variance(f_count, mu_count) ≈ mu_count
        dev_count = GAM._deviance(f_count, y_count, mu_count, wt)
        @test dev_count >= 0
        @test isfinite(dev_count)
        @test GAM._deviance(f_count, mu_count, mu_count, wt) ≈ 0.0 atol = 1e-10

        mu_bin = [0.2, 0.4, 0.8]
        y_bin = [0.0, 0.5, 1.0]
        f_bin = QuasiBinomialFamily()
        @test GAM._variance(f_bin, mu_bin) ≈ mu_bin .* (1.0 .- mu_bin)
        dev_bin = GAM._deviance(f_bin, y_bin, mu_bin, wt)
        @test dev_bin >= 0
        @test isfinite(dev_bin)
        @test GAM._deviance(f_bin, mu_bin, mu_bin, wt) ≈ 0.0 atol = 1e-10
    end

    @testset "Tweedie variance and deviance" begin
        mu = [1.0, 2.0, 5.0]
        f = TweedieFamily(p=1.5)

        v = GAM._variance(f, mu)
        @test v ≈ mu .^ 1.5

        y = [1.0, 3.0, 4.0]
        wt = ones(3)
        dev = GAM._deviance(f, y, mu, wt)
        @test isfinite(dev)

        # Deviance at y=mu should be approximately 0
        dev_sat = GAM._deviance(f, mu, mu, wt)
        @test abs(dev_sat) < 1e-10
    end

    @testset "Tweedie Dd derivatives" begin
        y = [0.0, 0.35, 1.1, 2.4, 4.2]
        mu = [0.4, 0.8, 1.3, 2.0, 3.1]
        wt = [1.0, 0.7, 1.2, 0.9, 1.5]
        f = TweedieFamily(p=1.4)

        @test GAM._has_Dd(f)

        dd = GAM.tweedie_Dd(f, y, mu, wt; level=0)
        @test haskey(dd, :Dmu)
        @test haskey(dd, :Dmu2)
        @test haskey(dd, :EDmu2)
        @test dd[:EDmu2] ≈ 2 .* wt ./ mu .^ f.p atol = 1e-12
        @test all(dd[:EDmu2] .> 0)

        dev0 = GAM._deviance(f, y, mu, wt)
        @inbounds for i in eachindex(mu)
            h = 1e-5 * max(mu[i], 1.0)

            mu_p = copy(mu)
            mu_m = copy(mu)
            mu_p[i] += h
            mu_m[i] -= h

            dev_p = GAM._deviance(f, y, mu_p, wt)
            dev_m = GAM._deviance(f, y, mu_m, wt)

            Dmu_fd = (dev_p - dev_m) / (2h)
            Dmu2_fd = (dev_p - 2dev0 + dev_m) / (h^2)

            @test dd[:Dmu][i] ≈ Dmu_fd rtol = 1e-7 atol = 1e-8
            @test dd[:Dmu2][i] ≈ Dmu2_fd rtol = 3e-4 atol = 1e-4

            Deta2_i = dd[:Dmu2][i] * mu[i]^2 + dd[:Dmu][i] * mu[i]
            @test Deta2_i > 0
        end
    end

    @testset "Tweedie log density helpers" begin
        y = [0.0, 0.4, 1.2, 2.8]
        mu = [0.5, 0.8, 1.5, 2.2]
        p = 1.4
        phi = 0.7
        wt = ones(length(y))

        ll = [GAM._tweedie_logdensity(yi, mui, p, phi) for (yi, mui) in zip(y, mu)]
        @test all(isfinite, ll)
        @test ll[1] ≈ -mu[1]^(2 - p) / (phi * (2 - p)) atol = 1e-12

        total_ll = GAM._tweedie_total_loglik(y, mu, wt, p, phi)
        @test total_ll ≈ sum(ll) atol = 1e-10
        @test isfinite(GAM._tweedie_total_loglik(y, mu, wt, 1.01, phi))
        @test isfinite(GAM._tweedie_total_loglik(y, mu, wt, 1.99, phi))
    end

    @testset "Tweedie power estimation" begin
        rng_ext = StableRNG(505)
        n = 1500
        x = range(-1.0, 1.0; length = n) |> collect
        mu_vals = exp.(0.3 .+ 0.7 .* x)
        true_p = 1.35
        true_phi = 0.8
        y_tw = simulate_tweedie(rng_ext, mu_vals, true_p, true_phi)

        f = TweedieFamily(p=1.8, estimate_p=true)
        initial_error = abs(f.p - true_p)
        for _ in 1:4
            GAM.estimate_theta!(f, y_tw, mu_vals, ones(n), true_phi)
        end

        @test 1.01 < f.p < 1.99
        @test abs(f.p - true_p) < initial_error
        @test abs(f.p - true_p) < 0.2
    end

    @testset "Beta variance and deviance" begin
        mu = [0.3, 0.5, 0.7]
        f = BetaFamily(phi=5.0)

        v = GAM._variance(f, mu)
        @test v ≈ mu .* (1.0 .- mu) ./ 6.0

        y = [0.2, 0.6, 0.8]
        wt = ones(3)
        dev = GAM._deviance(f, y, mu, wt)
        @test dev >= 0
        @test isfinite(dev)
    end

    @testset "NegBin theta estimation" begin
        rng_ext = StableRNG(123)
        n = 500
        true_theta = 3.0
        mu_vals = exp.(0.5 .+ 0.3 .* randn(rng_ext, n))
        y_nb = Float64[rand(rng_ext, NegativeBinomial(true_theta, true_theta / (true_theta + m))) for m in mu_vals]

        f = NegBinFamily(theta=1.0)
        GAM.estimate_theta!(f, y_nb, mu_vals, ones(n), 1.0)
        # Theta should move toward the true value
        @test f.theta > 0
        @test isfinite(f.theta)
    end

    @testset "Beta phi estimation" begin
        rng_ext = StableRNG(456)
        n = 500
        mu_vals = 1.0 ./ (1.0 .+ exp.(-randn(rng_ext, n)))
        mu_vals = clamp.(mu_vals, 0.01, 0.99)
        # Generate beta-distributed data with known phi
        true_phi = 10.0
        y_beta = Float64[]
        for m in mu_vals
            a = m * true_phi
            b = (1.0 - m) * true_phi
            push!(y_beta, clamp(rand(rng_ext, Distributions.Beta(a, b)), 0.001, 0.999))
        end

        f = BetaFamily(phi=1.0)
        GAM.estimate_theta!(f, y_beta, mu_vals, ones(n), 1.0)
        @test f.phi > 0
        @test isfinite(f.phi)
    end

    @testset "pirls_extended NegBin" begin
        rng_ext = StableRNG(789)
        n = 100
        x = randn(rng_ext, n)
        mu_true = exp.(1.0 .+ 0.5 .* x)
        true_theta = 3.0
        y = Float64[rand(rng_ext, NegativeBinomial(true_theta, true_theta / (true_theta + m))) for m in mu_true]

        X = hcat(ones(n), x)
        S_total = zeros(2, 2)
        family = NegBinFamily(theta=1.0)

        result = GAM.pirls_extended(X, y, S_total, family, LogLink(); control=gam_control(maxit=100))
        @test result.converged
        @test result.deviance >= 0
        @test isfinite(result.deviance)
        # Coefficients should be reasonable
        @test abs(result.coefficients[1] - 1.0) < 1.0
        @test abs(result.coefficients[2] - 0.5) < 1.0
    end

    @testset "pirls_extended Beta" begin
        rng_ext = StableRNG(101)
        n = 100
        x = randn(rng_ext, n)
        eta = 0.0 .+ 0.5 .* x
        mu_true = 1.0 ./ (1.0 .+ exp.(-eta))
        true_phi = 10.0
        y = Float64[]
        for m in mu_true
            a = m * true_phi
            b = (1.0 - m) * true_phi
            push!(y, clamp(rand(rng_ext, Distributions.Beta(a, b)), 0.001, 0.999))
        end

        X = hcat(ones(n), x)
        S_total = zeros(2, 2)
        family = BetaFamily(phi=1.0)

        result = GAM.pirls_extended(X, y, S_total, family, LogitLink(); control=gam_control(maxit=100))
        @test result.converged
        @test result.deviance >= 0
        @test isfinite(result.deviance)
    end

    @testset "GAM fit NegBin with smooth" begin
        rng_ext = StableRNG(202)
        n = 150
        x = range(0, 2π; length=n) |> collect
        mu_true = exp.(1.0 .+ sin.(x))
        true_theta = 2.0
        y = Float64[rand(rng_ext, NegativeBinomial(true_theta, true_theta / (true_theta + m))) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 10)), df; family=NegBinFamily(theta=1.0))
        @test m isa GamModel
        @test m.family isa NegBinFamily
        @test m.converged
        @test m.deviance_val >= 0
        @test m.n_smooth == 1
        @test length(edf(m)) == 1
        @test edf(m)[1] > 1.0

        # Predictions
        mu_hat = fitted(m)
        @test length(mu_hat) == n
        @test all(mu_hat .> 0)

        # Residuals
        r = residuals(m; type=:response)
        @test length(r) == n
        r_p = residuals(m; type=:pearson)
        @test length(r_p) == n
        r_d = residuals(m; type=:deviance)
        @test length(r_d) == n
    end

    @testset "GAM fit Beta with smooth" begin
        rng_ext = StableRNG(303)
        n = 150
        x = range(-2, 2; length=n) |> collect
        eta_true = 0.5 .* sin.(x)
        mu_true = 1.0 ./ (1.0 .+ exp.(-eta_true))
        true_phi = 20.0
        y = Float64[]
        for m_val in mu_true
            a = m_val * true_phi
            b = (1.0 - m_val) * true_phi
            push!(y, clamp(rand(rng_ext, Distributions.Beta(a, b)), 0.001, 0.999))
        end
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 10)), df; family=BetaFamily(phi=1.0))
        @test m isa GamModel
        @test m.family isa BetaFamily
        @test m.converged
        @test m.deviance_val >= 0
        @test m.family.phi > 0
        @test all(0 .< fitted(m) .< 1)
    end

    @testset "Quasi family response validation" begin
        bad_count = DataFrame(x=1:6, y=[0.0, 1.0, -1.0, 2.0, 3.0, 1.0])
        @test_throws ArgumentError gam(@formulak(y ~ s(x, k = 4)), bad_count;
            family=QuasiPoissonFamily())

        bad_prop = DataFrame(x=1:6, y=[0.1, 0.2, 1.2, 0.4, 0.5, 0.6])
        @test_throws ArgumentError gam(@formulak(y ~ s(x, k = 4)), bad_prop;
            family=QuasiBinomialFamily())

        bad_tw = DataFrame(x=1:6, y=[0.0, 0.2, -0.1, 0.5, 1.0, 0.3])
        @test_throws ArgumentError gam(@formulak(y ~ s(x, k = 4)), bad_tw;
            family=TweedieFamily(p=1.4))
    end

    @testset "GAM fit QuasiPoisson with smooth" begin
        rng_ext = StableRNG(304)
        n = 180
        x = range(0, 2π; length=n) |> collect
        mu_true = exp.(0.4 .+ 0.7 .* sin.(x))
        true_theta = 1.5
        y = Float64[rand(rng_ext, NegativeBinomial(true_theta, true_theta / (true_theta + m))) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 10)), df; family=QuasiPoissonFamily())
        @test m isa GamModel
        @test m.family isa QuasiPoissonFamily
        @test m.converged
        @test m.scale > 1.0
        @test all(fitted(m) .> 0)
        @test isfinite(loglikelihood(m))
        @test length(residuals(m; type=:pearson)) == n
    end

    @testset "GAM fit QuasiBinomial with smooth" begin
        rng_ext = StableRNG(305)
        n = 220
        x = range(-2, 2; length=n) |> collect
        eta_true = -0.3 .+ 1.1 .* sin.(x)
        p_true = 1.0 ./ (1.0 .+ exp.(-(eta_true .+ 1.2 .* randn(rng_ext, n))))
        y = Float64.(rand(rng_ext, n) .< p_true)
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 12)), df; family=QuasiBinomialFamily())
        @test m isa GamModel
        @test m.family isa QuasiBinomialFamily
        @test m.converged
        @test m.scale > 1.0
        @test all(0 .< fitted(m) .< 1)
        @test length(residuals(m; type=:deviance)) == n
    end

    @testset "GAM fit Tweedie with smooth" begin
        rng_ext = StableRNG(404)
        n = 150
        x = range(0, 2π; length=n) |> collect
        mu_true = exp.(0.5 .+ 0.5 .* sin.(x))
        # Generate Tweedie-like data using Poisson (p→1 limit)
        y = Float64[max(rand(rng_ext, Poisson(m)), 0.0) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 10)), df; family=TweedieFamily(p=1.5))
        @test m isa GamModel
        @test m.family isa TweedieFamily
        @test m.deviance_val >= 0
        @test all(fitted(m) .> 0)
        @test isfinite(loglikelihood(m))
        @test dof(m) ≈ m.edf_total + 1
        @test isfinite(aic(m))
        @test isfinite(bic(m))
    end

    @testset "GAM fit Tweedie with estimated power" begin
        rng_ext = StableRNG(405)
        n = 220
        x = range(0, 1; length = n) |> collect
        mu_true = exp.(0.2 .+ 0.6 .* sin.(2π .* x))
        true_p = 1.4
        true_phi = 0.7
        y = simulate_tweedie(rng_ext, mu_true, true_p, true_phi)
        df = DataFrame(x = x, y = y)

        family = TweedieFamily(p = 1.8, estimate_p = true)
        m = gam(@formulak(y ~ s(x, k = 12)), df; family = family)

        @test m isa GamModel
        @test m.family isa TweedieFamily
        @test m.converged
        @test 1.01 < m.family.p < 1.99
        @test abs(m.family.p - true_p) < abs(1.8 - true_p)
        @test all(fitted(m) .> 0)
        @test isfinite(loglikelihood(m))
        @test dof(m) ≈ m.edf_total + 2
        @test isfinite(aic(m))
        @test isfinite(bic(m))
    end

    @testset "Extended family show and StatsBase" begin
        rng_ext = StableRNG(505)
        n = 100
        x = randn(rng_ext, n)
        mu_true = exp.(1.0 .+ 0.3 .* x)
        y = Float64[rand(rng_ext, NegativeBinomial(2.0, 2.0 / (2.0 + m))) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 8)), df; family=NegBinFamily())
        @test nobs(m) == n
        @test length(coef(m)) == size(m.X, 2)
        @test deviance(m) >= 0
        @test isfinite(loglikelihood(m))
        @test dof_residual(m) > 0

        # show should not error
        buf = IOBuffer()
        show(buf, MIME"text/plain"(), m)
        out_str = String(take!(buf))
        @test occursin("NegativeBinomial", out_str)
        @test occursin("Theta est.", out_str)

        m_q = gam(@formulak(y ~ s(x, k = 8)), df; family=QuasiPoissonFamily())
        buf_q = IOBuffer()
        show(buf_q, MIME"text/plain"(), m_q)
        out_quasi = String(take!(buf_q))
        @test occursin("QuasiPoisson", out_quasi)
        @test occursin("Scale est.", out_quasi)
    end

    @testset "Extended family predict" begin
        rng_ext = StableRNG(606)
        n = 100
        x = range(0, 2π; length=n) |> collect
        mu_true = exp.(0.5 .+ sin.(x))
        y = Float64[rand(rng_ext, NegativeBinomial(3.0, 3.0 / (3.0 + m))) for m in mu_true]
        df = DataFrame(x=x, y=y)

        m = gam(@formulak(y ~ s(x, k = 10)), df; family=NegBinFamily())

        # Predict on training data
        pred_link = predict(m; type=:link)
        pred_resp = predict(m; type=:response)
        @test length(pred_link) == n
        @test length(pred_resp) == n
        @test all(pred_resp .> 0)

        # Predict on new data
        new_df = DataFrame(x=range(0, 2π; length=50) |> collect)
        pred_new = predict(m, new_df; type=:response)
        @test length(pred_new) == 50
        @test all(pred_new .> 0)
    end
end

# BAM tests
include("test_bam.jl")

# t2() tensor product smooth tests
include("test_t2.jl")

# GINLA tests
include("test_ginla.jl")

# Multi-parameter model tests (evgam)
include("test_multiparameter.jl")

# R integration tests — run when RCall and mgcv are available
# Set GAM_SKIP_RCALL=true to skip these tests
if !parse(Bool, get(ENV, "GAM_SKIP_RCALL", "false"))
    _rcall_available = try
        @eval using RCall
        _ok = @eval RCall.reval("library(mgcv)")
        true
    catch e
        @warn "Skipping R integration tests (RCall/mgcv not available)" exception = e
        false
    end

    if _rcall_available
        @eval include("test_rcall.jl")

        # evgam R comparison tests — need evgam and evd packages
        _evgam_available = try
            @eval RCall.reval("library(evgam)")
            @eval RCall.reval("library(evd)")
            true
        catch e
            @warn "Skipping evgam R comparison tests (evgam/evd not available)" exception = e
            false
        end

        if _evgam_available
            @eval include("test_evgam_rcall.jl")
        end

        # EGPD R comparison tests — need egpd package
        _egpd_available = try
            @eval RCall.reval("library(egpd)")
            true
        catch e
            @warn "Skipping EGPD R comparison tests (egpd not available)" exception = e
            false
        end

        if _egpd_available
            @eval include("test_egpd_rcall.jl")
        end
    end

    # qgam R comparison tests
    if _rcall_available
        _qgam_available = try
            @eval RCall.reval("library(qgam)")
            true
        catch e
            @warn "Skipping qgam R comparison tests (qgam not available)" exception = e
            false
        end

        if _qgam_available
            @eval include("test_qgam_rcall.jl")
        end
    end

    # scam R comparison tests
    if _rcall_available
        _scam_available = try
            @eval RCall.reval("library(scam)")
            true
        catch e
            @warn "Skipping scam R comparison tests (scam not available)" exception = e
            false
        end

     if _scam_available
         @eval include("test_scam_rcall.jl")
     end
    end

    # mgcv::scasm R comparison tests
    if _rcall_available
        _scasm_available = try
            Bool(RCall.rcopy(RCall.reval(
                "library(mgcv); exists('scasm', where = asNamespace('mgcv'), inherits = FALSE)"
            )))
        catch e
            @warn "Skipping scasm R comparison tests (mgcv/scasm not available)" exception = e
            false
        end

        if _scasm_available
            @eval include("test_scasm_rcall.jl")
        end
    end

    # gratia R comparison tests
    if _rcall_available
        _gratia_available = try
            @eval RCall.reval("library(gratia)")
            true
        catch e
            @warn "Skipping gratia R comparison tests (gratia not available)" exception = e
            false
        end

        if _gratia_available
            @eval include("test_gratia_rcall.jl")
        end
    end
end

# EGPD unit tests (no R needed) — must run regardless of GAM_SKIP_RCALL
@eval include("test_egpd.jl")

# Quantile GAM (qgam) unit tests (no R needed)
@eval include("test_qgam.jl")

# SCAM unit tests (no R needed)
@eval include("test_scam.jl")

# SCASM unit tests (no R needed)
@eval include("test_scasm.jl")

# Unified gam() API dispatch tests
@eval include("test_unified_api.jl")

# Adaptive smooth tests
@eval include("test_adaptive.jl")

# General fit (WPS algorithm) tests
@eval include("test_general_fit.jl")

# GAMLSS tests
@eval include("test_gamlss.jl")

# GAMLSS comparison tests against mgcv and R gamlss reference outputs
@eval include("test_gamlss_vs_mgcv.jl")
@eval include("test_gamlss_vs_rgamlss.jl")

# Gratia diagnostics unit tests (no R needed)
@eval include("test_gratia.jl")

# ANOVA / smooth significance tests
@eval include("test_diagnostics.jl")

# Bayesian GAM infrastructure tests (smooth2random, PriorSpec, dispatch)
@eval include("test_bayes.jl")

# Bayesian GAM end-to-end tests (requires Turing.jl extension)
@eval include("test_bayes_e2e.jl")

@eval include("test_formula_support.jl")

@eval include("test_bayes_gamlss_scam.jl")

@eval include("test_gamm.jl")

if !parse(Bool, get(ENV, "GAM_SKIP_RCALL", "false"))
    # Check availability separately so genuine test failures are NOT swallowed:
    # only the load/availability check is wrapped in try/catch; the include
    # (which runs the @testsets) is not.
    _gamm_rcall_ok = try
        @eval using RCall
        @eval RCall.reval("library(nlme)")
        true
    catch e
        @warn "Skipping GAMM R comparison tests (nlme/RCall not available)" exception = e
        false
    end
    _gamm_rcall_ok && @eval include("test_gamm_rcall.jl")
end

@eval include("test_side_constraints.jl")

# Spherical spline (sos) tests
@eval include("test_sos.jl")

# Constrained factor smooth (sz) tests
@eval include("test_sz.jl")

# Factor-smooth interaction (fs) tests
@eval include("test_fs.jl")

# Markov random field (mrf) tests
@eval include("test_mrf.jl")

# Soap film (so) tests
@eval include("test_soap.jl")

if !parse(Bool, get(ENV, "GAM_SKIP_RCALL", "false"))
    _sidecon_rcall_ok = try
        @eval using RCall
        @eval RCall.reval("library(mgcv)")
        true
    catch e
        @warn "Skipping side constraint R comparison tests (RCall/mgcv not available)" exception = e
        false
    end
    _sidecon_rcall_ok && @eval include("test_side_constraints_rcall.jl")
end

@eval include("test_loess.jl")

@eval include("test_fp.jl")

# Cyclic P-spline (cps) tests
@eval include("test_cyclic_ps.jl")

# SPDE Matérn smooth tests
@eval include("test_spde.jl")

# SPDE R comparison tests (uses pre-generated CSV data, not RCall).
# Only the CSV availability check is guarded; the include is not, so test
# failures surface instead of being downgraded to warnings.
_spde_csv_ok = try
    @eval using CSV
    true
catch e
    @warn "Skipping SPDE R comparison tests (CSV not available)" exception = e
    false
end
_spde_csv_ok && @eval include("test_spde_rcall.jl")

# Input validation tests
@eval include("test_validation.jl")

# Regression tests for code-review fixes (no R needed)
@eval include("test_review_fixes_core.jl")
@eval include("test_review_fixes_select_scam.jl")
@eval include("test_review_fixes_basis.jl")
@eval include("test_review_fixes_ti.jl")
@eval include("test_review_fixes_by.jl")
@eval include("test_review_fixes_extended.jl")
@eval include("test_review_fixes_serialization.jl")

# Plotting tests — run only when Plots.jl (a weak dependency) is available.
# Availability is checked separately so plotting test failures are not swallowed.
_plots_available = try
    @eval using Plots
    true
catch
    @info "Skipping plotting tests (Plots.jl not available)"
    false
end
_plots_available && @eval include("test_plots.jl")
