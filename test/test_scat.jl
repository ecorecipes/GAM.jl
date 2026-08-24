using Test
using GAM
using DataFrames
using Random
using Statistics
using LinearAlgebra
using ForwardDiff
using StatsAPI: fitted, coef, predict
using GAM: ScatFamily, scat_Dd, _scat_loglik, _deviance, _deviance_residuals,
           _variance, _dvariance_scalar_mu, _has_Dd, _estimates_scale,
           _default_link, _initialize_mu, _null_deviance, _preinitialize!,
           _use_expected_information, estimate_theta!

# Unit deviance exactly as mgcv's scat()$dev.resids defines it. Every
# derivative in `scat_Dd` is checked against automatic differentiation of this
# one function, so a sign or factor error anywhere shows up immediately.
_scat_unit_dev(yi, mui, wi, ν, σ) = wi * (ν + 1) * log1p(((yi - mui) / σ)^2 / ν)

@testset "Scaled t (scat) family" begin

    # ========================================================================
    # Derivatives: scat_Dd vs ForwardDiff on the unit deviance
    # ========================================================================
    @testset "Dd derivatives match automatic differentiation" begin
        rng = MersenneTwister(11)
        n = 80
        y = randn(rng, n) .* 2.0 .+ 1.0
        mu = randn(rng, n) .* 1.5
        wt = rand(rng, n) .+ 0.5           # non-unit weights on purpose
        min_df = 3.0

        for (ν, σ) in ((5.7, 1.3), (4.0, 0.4), (30.0, 2.2))
            f = ScatFamily(; nu = ν, sigma = σ, min_df = min_df)
            dd = scat_Dd(f, y, mu, wt; level = 1)
            θ0 = [log(ν - min_df), log(σ)]

            for i in 1:n
                g(m) = _scat_unit_dev(y[i], m, wt[i], ν, σ)
                d1(m) = ForwardDiff.derivative(g, m)
                d2(m) = ForwardDiff.derivative(d1, m)

                @test ForwardDiff.derivative(g, mu[i]) ≈ dd[:Dmu][i] rtol = 1e-9
                @test d2(mu[i]) ≈ dd[:Dmu2][i] rtol = 1e-8
                @test ForwardDiff.derivative(d2, mu[i]) ≈ dd[:Dmu3][i] rtol = 1e-8

                # Derivatives w.r.t. the internal parameters θ = (log(ν-min_df), log σ)
                hθ(θ) = _scat_unit_dev(y[i], mu[i], wt[i], exp(θ[1]) + min_df, exp(θ[2]))
                @test ForwardDiff.gradient(hθ, θ0) ≈ dd[:Dth][i, :] rtol = 1e-8

                hμθ(θ) = ForwardDiff.derivative(
                    m -> _scat_unit_dev(y[i], m, wt[i], exp(θ[1]) + min_df, exp(θ[2])), mu[i])
                @test ForwardDiff.gradient(hμθ, θ0) ≈ dd[:Dmuth][i, :] rtol = 1e-8

                hμ2θ(θ) = ForwardDiff.derivative(
                    m -> ForwardDiff.derivative(
                        m2 -> _scat_unit_dev(y[i], m2, wt[i], exp(θ[1]) + min_df, exp(θ[2])), m),
                    mu[i])
                @test ForwardDiff.gradient(hμ2θ, θ0) ≈ dd[:Dmu2th][i, :] rtol = 1e-7
            end
        end
    end

    @testset "EDmu2 is the expected value of Dmu2" begin
        # E[Dmu2] over y ~ μ + σ·t_ν must equal EDmu2 = 2w(ν+1)/(σ²(ν+3)).
        rng = MersenneTwister(5)
        ν, σ = 6.0, 0.8
        f = ScatFamily(; nu = ν, sigma = σ)
        n = 400_000
        # t_ν draws as z / sqrt(χ²_ν / ν)
        tdraw = [randn(rng) / sqrt(sum(abs2, randn(rng, 6)) / ν) for _ in 1:n]
        y = 2.0 .+ σ .* tdraw
        mu = fill(2.0, n)
        dd = scat_Dd(f, y, mu, ones(n); level = 0)
        @test mean(dd[:Dmu2]) ≈ dd[:EDmu2][1] rtol = 0.02
        # And the gradient has mean zero at the true μ.
        @test abs(mean(dd[:Dmu])) < 0.02
    end

    @testset "curvature redescends beyond |y-μ| > σ√ν" begin
        ν, σ = 5.0, 1.0
        f = ScatFamily(; nu = ν, sigma = σ)
        cut = σ * sqrt(ν)
        r = [0.5 * cut, 0.99 * cut, 1.01 * cut, 4.0 * cut]
        dd = scat_Dd(f, r, zeros(4), ones(4); level = 0)
        @test dd[:Dmu2][1] > 0
        @test dd[:Dmu2][2] > 0
        @test dd[:Dmu2][3] < 0          # redescending: outliers lose their weight
        @test dd[:Dmu2][4] < 0
        # Influence |Dmu| is bounded and decays for extreme residuals.
        @test abs(dd[:Dmu][4]) < abs(dd[:Dmu][2])
    end

    # ========================================================================
    # Likelihood, deviance and the family interface
    # ========================================================================
    @testset "deviance equals 2(ℓ_saturated - ℓ)" begin
        rng = MersenneTwister(3)
        n = 50
        y = randn(rng, n)
        mu = randn(rng, n) .* 0.5
        wt = rand(rng, n) .+ 0.5
        f = ScatFamily(; nu = 7.0, sigma = 1.1)
        ll = _scat_loglik(f.nu, f.sigma, y, mu, wt)
        ll_sat = _scat_loglik(f.nu, f.sigma, y, y, wt)
        @test _deviance(f, y, mu, wt) ≈ 2 * (ll_sat - ll) rtol = 1e-12
        # Deviance residuals square (and sum) back to the deviance.
        @test sum(abs2, _deviance_residuals(f, y, mu, wt)) ≈ _deviance(f, y, mu, wt) rtol = 1e-12
    end

    @testset "family interface registration" begin
        f = ScatFamily(; nu = 8.0, sigma = 2.0)
        @test _has_Dd(f)                       # mandatory: V(μ) is constant
        @test !_estimates_scale(f)             # σ is the family's own parameter
        @test !_use_expected_information(f)    # EDF must use the Newton weights
        @test _default_link(f) isa GAM.IdentityLink
        @test _variance(f, 0.0) ≈ 4.0 * 8.0 / 6.0
        @test _variance(f, 100.0) ≈ _variance(f, -100.0)   # constant in μ
        @test _dvariance_scalar_mu(f, 3.0) == 0.0
        @test _variance(f, [1.0, 2.0]) ≈ fill(4.0 * 8.0 / 6.0, 2)
    end

    @testset "constructor and min_df handling" begin
        @test_throws ArgumentError ScatFamily(; nu = 5.0)             # sigma missing
        @test_throws ArgumentError ScatFamily(; nu = -1.0, sigma = 1.0)
        @test_throws ArgumentError ScatFamily(; nu = 5.0, sigma = 0.0)
        # mgcv resets min.df to 0.9ν (with a warning) when ν ≤ min.df
        f = @test_logs (:warn, r"min_df reset") ScatFamily(; nu = 2.0, sigma = 1.0)
        @test f.min_df ≈ 1.8
        @test f.nu ≈ 2.0
        # Unparameterised construction defers to _preinitialize!
        g = ScatFamily()
        @test !g.initialized
        y = randn(MersenneTwister(1), 500) .* 3.0
        _preinitialize!(g, y)
        @test g.initialized
        @test g.sigma ≈ 0.8 * std(y)        # mgcv: Theta = c(1.5, log(0.8 sd(y)))
        @test g.nu ≈ exp(1.5) + 3.0
        # Idempotent: a warm-started refit must not reset an estimated value.
        g.nu = 12.0
        _preinitialize!(g, y)
        @test g.nu ≈ 12.0
    end

    @testset "null deviance minimises over μ" begin
        rng = MersenneTwister(9)
        y = randn(rng, 200) .+ 5.0
        y[1:5] .+= 40.0                     # outliers the mean would chase
        wt = ones(200)
        f = ScatFamily(; nu = 4.0, sigma = 1.0)
        nd = _null_deviance(f, y, wt)
        @test nd <= _deviance(f, y, fill(mean(y), 200), wt)
        for m in range(minimum(y), maximum(y); length = 400)
            @test nd <= _deviance(f, y, fill(m, 200), wt) + 1e-8
        end
    end

    # ========================================================================
    # Parameter estimation
    # ========================================================================
    @testset "estimate_theta! recovers ν and σ" begin
        rng = MersenneTwister(21)
        n = 20_000
        ν_true, σ_true = 6.0, 1.4
        t = [randn(rng) / sqrt(sum(abs2, randn(rng, 6)) / 6.0) for _ in 1:n]
        mu = fill(2.0, n)
        y = mu .+ σ_true .* t
        f = ScatFamily()
        _preinitialize!(f, y)
        estimate_theta!(f, y, mu, ones(n), 1.0)
        @test f.nu ≈ ν_true rtol = 0.20
        @test f.sigma ≈ σ_true rtol = 0.05

        # It must sit at a stationary point of the conditional log-likelihood.
        min_df = f.min_df
        obj(θ) = -_scat_loglik(exp(θ[1]) + min_df, exp(θ[2]), y, mu, ones(n))
        g = ForwardDiff.gradient(obj, [log(f.nu - min_df), log(f.sigma)])
        @test maximum(abs, g) < 1e-4 * n

        # Gaussian data drives ν to the large-ν (Gaussian) limit.
        yg = mu .+ 1.3 .* randn(rng, n)
        fg = ScatFamily()
        _preinitialize!(fg, yg)
        estimate_theta!(fg, yg, mu, ones(n), 1.0)
        @test fg.nu > 20.0
        @test fg.sigma ≈ 1.3 rtol = 0.05
    end

    @testset "estimate_theta! respects estimate_theta = false" begin
        rng = MersenneTwister(4)
        y = randn(rng, 500)
        f = ScatFamily(; nu = 9.0, sigma = 3.0, estimate_theta = false)
        estimate_theta!(f, y, zeros(500), ones(500), 1.0)
        @test f.nu == 9.0
        @test f.sigma == 3.0
    end

    # ========================================================================
    # Fitting: recovery and the robustness property
    # ========================================================================
    @testset "recovers a smooth under heavy-tailed noise" begin
        rng = MersenneTwister(31)
        n = 300
        x = sort(rand(rng, n))
        ft = sin.(2π .* x) .+ 0.4 .* x
        t3 = [randn(rng) / sqrt(sum(abs2, randn(rng, 3)) / 3.0) for _ in 1:n]
        df = DataFrame(x = x, y = ft .+ 0.3 .* t3)

        m = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
        @test m.converged
        @test m.family.nu < 12.0            # heavy tails detected
        @test m.family.sigma ≈ 0.3 rtol = 0.5
        @test cor(fitted(m), ft) > 0.98

        mg = gam(@formula(y ~ s(x, k = 10)), df)
        rmse(p) = sqrt(mean(abs2, p .- ft))
        @test rmse(fitted(m)) < rmse(fitted(mg))
    end

    @testset "gross outliers: scat beats Gaussian by a wide margin" begin
        rng = MersenneTwister(7)
        n = 300
        x = sort(rand(rng, n))
        ft = sin.(2π .* x) .+ 0.4 .* x
        y = ft .+ 0.3 .* randn(rng, n)
        y[[10, 60, 120, 180, 240, 290]] .+= [8.0, -9.0, 7.0, -8.5, 9.5, -7.5]
        df = DataFrame(x = x, y = y)

        m  = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
        mg = gam(@formula(y ~ s(x, k = 10)), df)
        rmse(p) = sqrt(mean(abs2, p .- ft))

        @test m.converged
        # The substantive claim for the family: with six gross outliers in 300
        # points the robust fit tracks the truth several times more closely.
        @test rmse(fitted(m)) < 0.4 * rmse(fitted(mg))
        @test m.family.nu < 6.0             # driven to the heavy-tailed end

        # The outliers must carry near-zero working weight. With the identity
        # link the P-IRLS weight is exactly clamp(Dmu2/2, eps, 1e10), so this
        # reconstructs what the fit actually used at its own solution.
        dd = scat_Dd(m.family, df.y, fitted(m), ones(n); level = 0)
        w = clamp.(0.5 .* dd[:Dmu2], eps(), 1e10)
        outliers = [10, 60, 120, 180, 240, 290]
        @test maximum(w[outliers]) < 0.05 * median(w)
        @test all(dd[:Dmu2][outliers] .< 0)      # past the redescending point
        @test median(w) > 1.0                    # ordinary points keep full weight
    end

    @testset "clean Gaussian data: agrees with a Gaussian fit" begin
        rng = MersenneTwister(13)
        n = 300
        x = sort(rand(rng, n))
        ft = sin.(2π .* x) .+ 0.4 .* x
        df = DataFrame(x = x, y = ft .+ 0.3 .* randn(rng, n))

        m  = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
        mg = gam(@formula(y ~ s(x, k = 10)), df)
        @test m.family.nu > 20.0                       # no heavy tails to find
        @test m.family.sigma ≈ 0.3 rtol = 0.25
        @test cor(fitted(m), fitted(mg)) > 0.999
        @test maximum(abs.(fitted(m) .- fitted(mg))) < 0.05
    end

    @testset "fixed ν and σ are held" begin
        rng = MersenneTwister(17)
        n = 200
        x = sort(rand(rng, n))
        df = DataFrame(x = x, y = sin.(2π .* x) .+ 0.3 .* randn(rng, n))
        m = gam(@formula(y ~ s(x, k = 8)), df;
            family = ScatFamily(; nu = 5.0, sigma = 0.4, estimate_theta = false))
        @test m.family.nu == 5.0
        @test m.family.sigma == 0.4
        @test m.converged
    end

    @testset "fixed sp: (ν, σ) still reach a stationary point" begin
        # With no free smoothing parameter there is no outer EFS loop, so
        # `outer.jl` has to converge the family parameters by alternating
        # refits (the round-5 fix). Check that the alternation actually lands
        # on a stationary point of the conditional log-likelihood, for a range
        # of fixed sp, rather than stopping one step in.
        rng = MersenneTwister(31)
        n = 300
        x = sort(rand(rng, n))
        ft = sin.(2π .* x) .+ 0.4 .* x
        t3 = [randn(rng) / sqrt(sum(abs2, randn(rng, 3)) / 3.0) for _ in 1:n]
        df = DataFrame(x = x, y = ft .+ 0.3 .* t3)

        models = [gam(@formula(y ~ s(x, k = 10, sp = 0.001)), df; family = ScatFamily()),
                  gam(@formula(y ~ s(x, k = 10, sp = 0.1)), df; family = ScatFamily()),
                  gam(@formula(y ~ s(x, k = 10, sp = 10.0)), df; family = ScatFamily())]
        for m in models
            @test m.converged
            md = m.family.min_df
            obj(θ) = -_scat_loglik(exp(θ[1]) + md, exp(θ[2]), df.y, fitted(m), ones(n))
            g = ForwardDiff.gradient(obj, [log(m.family.nu - md), log(m.family.sigma)])
            @test maximum(abs, g) < 1e-6
            # Only domain sanity here — NOT recovery of the true (ν, σ). A fixed
            # *numeric* sp means different amounts of smoothing on different basis
            # parameterizations: at sp = 10 the tprs fit collapses to edf ≈ 3.7
            # while `bs=:cr` is still at edf ≈ 9.6. Over-smoothing pushes the
            # unfitted part of the signal into the noise scale (σ 0.32 → 0.40),
            # which is correct behaviour, not an estimation failure. Recovery is
            # asserted below, under the sp the fitting criterion actually selects.
            @test m.family.nu > md
            @test m.family.sigma > 0
        end
        # Tighter smoothing must cost degrees of freedom.
        @test models[1].edf_total > models[3].edf_total
    end

    @testset "(ν, σ) recovery under criterion-selected sp" begin
        # The meaningful estimation property: with sp chosen by the fitting
        # criterion rather than pinned to an arbitrary value, (ν, σ) recover the
        # truth — and do so on both bases, since a criterion-selected sp adapts
        # to whatever parameterization the basis uses.
        rng = MersenneTwister(31)
        n = 300
        x = sort(rand(rng, n))
        ft = sin.(2π .* x) .+ 0.4 .* x
        t3 = [randn(rng) / sqrt(sum(abs2, randn(rng, 3)) / 3.0) for _ in 1:n]
        df = DataFrame(x = x, y = ft .+ 0.3 .* t3)

        m_tp = gam(@formula(y ~ s(x, k = 10)), df; family = ScatFamily())
        m_cr = gam(@formula(y ~ s(x, k = 10, bs = :cr)), df; family = ScatFamily())
        for m in (m_tp, m_cr)
            @test m.converged
            # True ν = 3, σ = 0.3.
            @test m.family.nu ≈ 3.0 rtol = 0.5
            @test m.family.sigma ≈ 0.3 rtol = 0.15
        end
        # Recovery must not depend on the basis, even though the selected sp does
        # (tprs ≈ 0.17 vs cr ≈ 89 here — the same fit in different coordinates).
        @test m_tp.family.sigma ≈ m_cr.family.sigma rtol = 0.05
        @test m_tp.family.nu ≈ m_cr.family.nu rtol = 0.1
    end

    @testset "prediction and summary work" begin
        rng = MersenneTwister(23)
        n = 200
        x = sort(rand(rng, n))
        df = DataFrame(x = x, y = sin.(2π .* x) .+ 0.3 .* randn(rng, n))
        m = gam(@formula(y ~ s(x, k = 8)), df; family = ScatFamily())
        p, se = predict(m, df; type = :response, se = true)
        @test length(p) == n
        @test all(isfinite, se)
        @test all(>(0), se)
        @test p ≈ fitted(m) rtol = 1e-8
        @test sprint(show, m) isa String
    end
end
