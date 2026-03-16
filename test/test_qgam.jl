using Test, GAM, Random, Statistics

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
end
