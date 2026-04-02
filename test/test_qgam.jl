using Test, GAM, GLM, Random, Statistics, StatsBase

@testset "Quantile GAM (qgam)" begin

    @testset "log1pexp" begin
        # Basic correctness
        @test GAM.log1pexp(0.0) ≈ log(2.0)
        @test GAM.log1pexp(1.0) ≈ log(1 + exp(1.0))
        @test GAM.log1pexp(-50.0) ≈ exp(-50.0) atol=1e-20
        @test GAM.log1pexp(50.0) ≈ 50.0 atol=1e-10
        @test GAM.log1pexp(100.0) == 100.0

        # Vector version
        x = [-50.0, -10.0, 0.0, 10.0, 50.0]
        out = similar(x)
        GAM.log1pexp!(out, x)
        @test all(out .≈ [GAM.log1pexp(xi) for xi in x])
    end

    @testset "sigmoid_derivs" begin
        D0, D1, D2, D3 = GAM.sigmoid_derivs(0.0)
        @test D0 ≈ 0.5
        @test D1 ≈ 0.25
        @test D2 ≈ 0.0 atol=1e-15
        @test D3 ≈ -0.125

        # Numerical derivative check
        h = 1e-7
        for x in [-2.0, 0.0, 1.5, 5.0]
            D0, D1, D2, D3 = GAM.sigmoid_derivs(x)
            D0p, _, _, _ = GAM.sigmoid_derivs(x + h)
            D0m, _, _, _ = GAM.sigmoid_derivs(x - h)
            @test abs(D1 - (D0p - D0m) / (2h)) < 1e-5

            _, D1p, _, _ = GAM.sigmoid_derivs(x + h)
            _, D1m, _, _ = GAM.sigmoid_derivs(x - h)
            @test abs(D2 - (D1p - D1m) / (2h)) < 1e-5
        end
    end

    @testset "ELFFamily construction" begin
        elf = ELFFamily(qu=0.5, co=0.1)
        @test elf.qu == 0.5
        @test length(elf.co) == 1
        @test elf.theta == 0.0
        @test !elf.estimate_theta

        @test_throws ArgumentError ELFFamily(qu=0.0)
        @test_throws ArgumentError ELFFamily(qu=1.0)
        @test_throws ArgumentError ELFFamily(qu=-0.1)
    end

    @testset "ELF deviance" begin
        y = [1.0, 2.0, 3.0, -1.0, 0.5]
        mu = [0.5, 1.5, 2.5, 0.0, 0.3]
        wt = ones(5)

        elf = ELFFamily(qu=0.5, co=fill(0.2, 5), theta=log(0.5))
        dev = GAM._deviance(elf, y, mu, wt)
        @test dev > 0  # deviance should be positive
        @test isfinite(dev)

        # Deviance at true values should be lower than at offset values
        dev_offset = GAM._deviance(elf, y, mu .+ 2.0, wt)
        @test dev < dev_offset
    end

    @testset "ELF Dd derivatives" begin
        y = [1.0, 2.0, 3.0]
        mu = [0.5, 1.5, 2.5]
        wt = ones(3)

        elf = ELFFamily(qu=0.5, co=fill(0.2, 3), theta=0.0)

        # Level 0
        dd0 = GAM.elf_Dd(elf, y, mu, wt; level=0)
        @test haskey(dd0, :Dmu)
        @test haskey(dd0, :Dmu2)
        @test haskey(dd0, :EDmu2)
        @test length(dd0[:Dmu]) == 3
        @test all(dd0[:Dmu2] .> 0)  # second derivative should be positive

        # Level 1
        dd1 = GAM.elf_Dd(elf, y, mu, wt; level=1)
        @test haskey(dd1, :Dth)
        @test haskey(dd1, :Dmuth)
        @test haskey(dd1, :Dmu3)

        # Level 2
        dd2 = GAM.elf_Dd(elf, y, mu, wt; level=2)
        @test haskey(dd2, :Dmu4)
        @test haskey(dd2, :Dth2)

        # Numerical check: Dmu should match finite differences
        h = 1e-6
        for i in 1:3
            mu_p = copy(mu); mu_p[i] += h
            mu_m = copy(mu); mu_m[i] -= h
            dev_p = GAM._deviance(elf, y, mu_p, wt)
            dev_m = GAM._deviance(elf, y, mu_m, wt)
            numerical_Dmu = (dev_p - dev_m) / (2h)
            @test abs(dd0[:Dmu][i] - numerical_Dmu) < 1e-4
        end
    end

    @testset "ELF log-saturated likelihood" begin
        y = [1.0, 2.0, 3.0]
        wt = ones(3)
        elf = ELFFamily(qu=0.5, co=fill(0.2, 3), theta=0.0)

        ls = GAM.elf_ls(elf, y, wt)
        @test isfinite(ls.ls)
        @test isfinite(ls.lsth1)
        @test isfinite(ls.lsth2)

        # Numerical check: lsth1 matches finite difference
        h = 1e-6
        elf_p = ELFFamily(qu=0.5, co=fill(0.2, 3), theta=h)
        elf_m = ELFFamily(qu=0.5, co=fill(0.2, 3), theta=-h)
        ls_p = GAM.elf_ls(elf_p, y, wt)
        ls_m = GAM.elf_ls(elf_m, y, wt)
        numerical_lsth = (ls_p.ls - ls_m.ls) / (2h)
        @test abs(ls.lsth1 - numerical_lsth) < 1e-4
    end

    @testset "pinball_loss" begin
        y = [1.0, 2.0, 3.0, 4.0]
        mu = [1.5, 1.5, 3.5, 3.5]

        # qu=0.5: symmetric pinball = 0.5 * sum(|y - mu|)
        loss5 = pinball_loss(y, mu, 0.5)
        @test loss5 ≈ 0.5 * sum(abs.(y .- mu))

        # qu=1.0-eps: should heavily penalize under-prediction
        # Per-obs
        losses = pinball_loss(y, mu, 0.9; reduce=false)
        @test length(losses) == 4

        # Errors
        @test_throws ArgumentError pinball_loss(y, mu, 0.0)
        @test_throws ArgumentError pinball_loss(y, mu, 1.0)
    end

    @testset "ELF GAM fitting" begin
        Random.seed!(42)
        n = 200
        x = range(0, 4, length=n) |> collect
        y = sin.(x) .+ 0.5 .* randn(n)
        df = (y=y, x=x)

        # Median fit
        elf = ELFFamily(qu=0.5, co=fill(0.2, n), theta=0.0)
        fit = gam(@gam_formula(y ~ s(x, k=10)), df; family=elf)
        @test fit.converged
        @test cor(fit.fitted_values, sin.(x)) > 0.95

        # Quantile 0.1 — most points should be above
        elf1 = ELFFamily(qu=0.1, co=fill(0.2, n), theta=0.0)
        fit1 = gam(@gam_formula(y ~ s(x, k=10)), df; family=elf1)
        @test fit1.converged
        frac1 = mean(y .< fit1.fitted_values)
        @test 0.0 < frac1 < 0.4  # roughly 10% below, with some slack

        # Quantile 0.9 — most points should be below
        elf9 = ELFFamily(qu=0.9, co=fill(0.2, n), theta=0.0)
        fit9 = gam(@gam_formula(y ~ s(x, k=10)), df; family=elf9)
        @test fit9.converged
        frac9 = mean(y .< fit9.fitted_values)
        @test frac9 > 0.6  # roughly 90% below
    end

    @testset "qgam high-level API" begin
        Random.seed!(123)
        n = 100
        x = range(0, 3, length=n) |> collect
        y = 2.0 .* x .+ randn(n)
        df = (y=y, x=x)

        # Single quantile fit with fixed learning rate
        elf = ELFFamily(qu=0.5, co=fill(0.15, n), theta=-0.5)
        fit = gam(@gam_formula(y ~ s(x, k=8)), df; family=elf)
        @test fit.converged

        # Multiple quantiles
        fits = mqgam(@gam_formula(y ~ s(x, k=8)), df, [0.25, 0.5, 0.75];
                     lsig=-0.5, co=0.15)
        @test length(fits.fits) == 3
        @test haskey(fits.fits, 0.25)
        @test haskey(fits.fits, 0.75)
    end

    @testset "qdo" begin
        Random.seed!(42)
        n = 100
        x = range(0, 3, length=n) |> collect
        y = 2.0 .* x .+ randn(n)
        df = (y=y, x=x)

        fits = mqgam(@gam_formula(y ~ s(x, k=8)), df, [0.25, 0.5, 0.75];
                     lsig=-0.5, co=0.15)

        # Extract model
        m50 = qdo(fits, 0.5)
        @test m50 isa GAM.GamModel
        @test m50.family.qu == 0.5

        # Apply function
        pred = qdo(fits, 0.25, predict)
        @test length(pred) == n

        # Error for missing quantile
        @test_throws ArgumentError qdo(fits, 0.42)
    end

    @testset "cqcheck" begin
        Random.seed!(123)
        n = 200
        x = range(0, 2π, length=n) |> collect
        y = sin.(x) .+ 0.3 .* randn(n)
        df = (y=y, x=x)

        elf = ELFFamily(qu=0.5, co=fill(0.2, n), theta=-0.5)
        fit = gam(@gam_formula(y ~ s(x, k=15)), df; family=elf)

        res = cqcheck(fit, x; nbin=8)
        @test res isa GAM.CQCheckResult
        @test length(res.bin_mid) == 8
        @test length(res.proportions) == 8
        @test length(res.ci_lower) == 8
        @test length(res.ci_upper) == 8
        @test all(res.ci_lower .<= res.ci_upper)
        @test res.target_qu == 0.5
        @test res.lev == 0.05

        # Proportions should be in [0, 1]
        @test all(0.0 .<= res.proportions .<= 1.0)
    end

    @testset "check_qgam" begin
        Random.seed!(456)
        n = 200
        x = range(0, 2π, length=n) |> collect
        y = sin.(x) .+ 0.3 .* randn(n)
        df = (y=y, x=x)

        elf = ELFFamily(qu=0.5, co=fill(0.2, n), theta=-0.5)
        fit = gam(@gam_formula(y ~ s(x, k=15)), df; family=elf)

        chk = check_qgam(fit; nbin=8)
        @test chk isa GAM.QGamCheck
        @test chk.target_qu == 0.5
        @test 0.0 < chk.actual_proportion < 1.0
        @test chk.integrated_abs_bias >= 0.0
        @test length(chk.bias_values) == n
        @test chk.calibration isa GAM.CQCheckResult
    end

    @testset "quantile_residuals" begin
        Random.seed!(789)
        n = 200
        x = range(0, 2π, length=n) |> collect
        y = sin.(x) .+ 0.3 .* randn(n)
        df = (y=y, x=x)

        elf = ELFFamily(qu=0.5, co=fill(0.2, n), theta=-0.5)
        fit = gam(@gam_formula(y ~ s(x, k=15)), df; family=elf)

        qr = quantile_residuals(fit)
        @test length(qr) == n
        @test all(isfinite.(qr))
        # Quantile residuals should be roughly standard normal
        @test abs(mean(qr)) < 1.0
        @test 0.2 < std(qr) < 2.0
    end

    @testset "ELFLSS family" begin
        Random.seed!(101)
        n = 200
        x = range(0, 2π, length=n) |> collect
        # Heteroscedastic data: variance increases with x
        y = sin.(x) .+ (0.1 .+ 0.3 .* x ./ (2π)) .* randn(n)
        df = (y=y, x=x)

        co = 0.3 * sqrt(2π * var(y)) / (2 * log(2))
        fam = ELFLSSFamily(qu=0.5, co=co)

        @test nparams(fam) == 2
        @test param_names(fam) == ["mu", "sigma"]

        # Test NLL computation
        nll = GAM.nll_obs(fam, 1.0, [0.5, log(0.3)])
        @test isfinite(nll)
        @test nll > 0  # NLL should be positive for reasonable inputs

        # Test initial_eta
        η0 = GAM.initial_eta(fam, y)
        @test length(η0) == 2
        @test length(η0[1]) == n
        @test length(η0[2]) == n
    end

    @testset "ELFLSS qgam high-level API and diagnostics" begin
        Random.seed!(202)
        n = 250
        x = collect(range(-2.5, 2.5; length=n))
        sigma = 0.15 .+ 0.25 .* abs.(x)
        y = sin.(x) .+ sigma .* randn(n)
        df = (y=y, x=x)

        formulas = [
            @gam_formula(y ~ s(x, k=12, bs=:cr)),
            @gam_formula(y ~ 0 + s(x, k=10, bs=:cr)),
        ]

        fit = qgam(formulas, df, 0.75)
        @test fit isa GAM.MultiParameterModel
        @test fit.family isa ELFLSSFamily
        @test fit.converged
        @test nparams(fit) == 2

        mu_hat = GAM._elflss_location(fit)
        sig_hat = GAM._apply_link_inv.(Ref(fit.family.links[2]), fit.fitted_eta[2])
        @test length(mu_hat) == n
        @test all(sig_hat .> 0)
        @test cor(mu_hat, sin.(x)) > 0.9
        @test cor(sig_hat, sigma) > 0.5

        fit_fixed = qgam(formulas, df, 0.75; co=0.2)
        fit_direct = gam(formulas, df, ELFLSSFamily(qu=0.75, co=0.2))
        @test fit_fixed.converged
        @test fit_fixed.nll ≈ fit_direct.nll atol=1e-6

        cal = cqcheck(fit, x; nbin=8)
        @test cal isa GAM.CQCheckResult
        @test cal.target_qu == 0.75
        @test all(0.0 .<= cal.proportions .<= 1.0)

        chk = check_qgam(fit; nbin=8)
        @test chk isa GAM.QGamCheck
        @test chk.target_qu == 0.75
        @test 0.0 < chk.actual_proportion < 1.0
        @test chk.integrated_abs_bias >= 0.0

        qr = quantile_residuals(fit)
        @test length(qr) == n
        @test all(isfinite.(qr))
    end

    @testset "ELFLSS prediction with uncertainty" begin
        Random.seed!(303)
        n = 220
        x = collect(range(-2.2, 2.2; length=n))
        sigma = 0.18 .+ 0.2 .* abs.(x)
        y = sin.(x) .+ sigma .* randn(n)
        df = (y=y, x=x)

        formulas = [
            @gam_formula(y ~ s(x, k=12, bs=:cr)),
            @gam_formula(y ~ 0 + s(x, k=10, bs=:cr)),
        ]

        fit = qgam(formulas, df, 0.75; co=0.2)
        @test fit.converged
        @test param_names(fit.family) == ["mu", "sigma"]

        pred_link = predict(fit; type=:link)
        pred_resp = predict(fit; type=:response)
        @test size(pred_link) == (n, 2)
        @test size(pred_resp) == (n, 2)
        @test pred_link[:, 1] ≈ fit.fitted_eta[1] atol=1e-8
        @test pred_link[:, 2] ≈ fit.fitted_eta[2] atol=1e-8
        @test pred_resp[:, 1] ≈ GAM._apply_link_inv.(Ref(fit.family.links[1]), fit.fitted_eta[1]) atol=1e-8
        @test pred_resp[:, 2] ≈ GAM._apply_link_inv.(Ref(fit.family.links[2]), fit.fitted_eta[2]) atol=1e-8

        newx = collect(range(-1.4, 1.4; length=9))
        newdf = (x=newx,)

        pred_link_new, se_link_new = predict(fit, newdf; type=:link, se=true)
        pred_resp_new, se_resp_new = predict(fit, newdf; type=:response, se=true)

        @test size(pred_link_new) == (length(newx), 2)
        @test size(se_link_new) == size(pred_link_new)
        @test size(pred_resp_new) == size(pred_link_new)
        @test size(se_resp_new) == size(pred_link_new)
        @test all(se_link_new .>= 0)
        @test all(se_resp_new .>= 0)
        @test all(pred_resp_new[:, 2] .> 0)

        X_mu = hcat(ones(length(newx)), predict_matrix(fit.smooths[1][1], newdf))
        X_sigma = hcat(ones(length(newx)), predict_matrix(fit.smooths[2][1], newdf))

        β_mu = param_coef(fit, 1)
        β_sigma = param_coef(fit, 2)
        V_mu = @view fit.Vp[(fit.param_offsets[1] + 1):fit.param_offsets[2],
                            (fit.param_offsets[1] + 1):fit.param_offsets[2]]
        V_sigma = @view fit.Vp[(fit.param_offsets[2] + 1):fit.param_offsets[3],
                               (fit.param_offsets[2] + 1):fit.param_offsets[3]]

        manual_link = hcat(X_mu * β_mu, X_sigma * β_sigma)
        manual_se_link = hcat(
            sqrt.(max.(vec(sum((X_mu * V_mu) .* X_mu; dims=2)), 0.0)),
            sqrt.(max.(vec(sum((X_sigma * V_sigma) .* X_sigma; dims=2)), 0.0)),
        )
        manual_resp = hcat(
            GLM.linkinv.(Ref(fit.family.links[1]), manual_link[:, 1]),
            GLM.linkinv.(Ref(fit.family.links[2]), manual_link[:, 2]),
        )
        manual_se_resp = hcat(
            abs.(GLM.mueta.(Ref(fit.family.links[1]), manual_link[:, 1])) .* manual_se_link[:, 1],
            abs.(GLM.mueta.(Ref(fit.family.links[2]), manual_link[:, 2])) .* manual_se_link[:, 2],
        )

        @test pred_link_new ≈ manual_link atol=1e-8
        @test se_link_new ≈ manual_se_link atol=1e-8
        @test pred_resp_new ≈ manual_resp atol=1e-8
        @test se_resp_new ≈ manual_se_resp atol=1e-8
    end
end
