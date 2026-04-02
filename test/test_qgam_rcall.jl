using Test, GAM, RCall, Random, Statistics
using StatsAPI: predict

R"library(qgam); library(mgcv)"

@testset "qgam R comparison" begin

    @testset "ELF deviance matches R" begin
        y = [1.0, 2.0, 3.0, -1.0, 0.5]
        mu = [0.5, 1.5, 2.5, 0.0, 0.3]
        wt = ones(5)

        for (qu, co, theta) in [(0.5, 0.2, 0.0), (0.1, 0.1, -1.0), (0.9, 0.3, 0.5)]
            @rput y mu wt qu co theta
            R"""
            fam <- elf(theta=theta, qu=qu, co=co)
            dev_r <- sum(fam$dev.resids(y, mu, wt))
            """
            dev_r = rcopy(R"dev_r")

            elf = ELFFamily(qu=qu, co=fill(co, 5), theta=theta)
            dev_jl = GAM._deviance(elf, y, mu, wt)
            @test abs(dev_jl - dev_r) < 1e-10
        end
    end

    @testset "ELF Dd level 0 matches R" begin
        y = [1.0, 2.0, 3.0, -1.0, 0.5]
        mu = [0.5, 1.5, 2.5, 0.0, 0.3]
        wt = ones(5)
        qu = 0.5; co = 0.2; theta = log(0.5)

        @rput y mu wt qu co theta
        R"""
        fam <- elf(theta=theta, qu=qu, co=co)
        dd <- fam$Dd(y, mu, theta, wt, level=0)
        """

        elf = ELFFamily(qu=qu, co=fill(co, 5), theta=theta)
        dd_jl = GAM.elf_Dd(elf, y, mu, wt; level=0)

        @test maximum(abs.(dd_jl[:Dmu] .- rcopy(R"dd$Dmu"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu2] .- rcopy(R"dd$Dmu2"))) < 1e-12
    end

    @testset "ELF Dd level 1 matches R" begin
        y = [1.0, 2.0, 3.0, -1.0, 0.5]
        mu = [0.5, 1.5, 2.5, 0.0, 0.3]
        wt = ones(5)
        qu = 0.5; co = 0.2; theta = log(0.5)

        @rput y mu wt qu co theta
        R"""
        fam <- elf(theta=theta, qu=qu, co=co)
        dd <- fam$Dd(y, mu, theta, wt, level=1)
        """

        elf = ELFFamily(qu=qu, co=fill(co, 5), theta=theta)
        dd_jl = GAM.elf_Dd(elf, y, mu, wt; level=1)

        @test maximum(abs.(dd_jl[:Dth] .- rcopy(R"dd$Dth"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmuth] .- rcopy(R"dd$Dmuth"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu3] .- rcopy(R"dd$Dmu3"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu2th] .- rcopy(R"dd$Dmu2th"))) < 1e-12
    end

    @testset "ELF Dd level 2 matches R" begin
        y = [1.0, 2.0, 3.0, -1.0, 0.5]
        mu = [0.5, 1.5, 2.5, 0.0, 0.3]
        wt = ones(5)
        qu = 0.8; co = 0.15; theta = -0.5

        @rput y mu wt qu co theta
        R"""
        fam <- elf(theta=theta, qu=qu, co=co)
        dd <- fam$Dd(y, mu, theta, wt, level=2)
        """

        elf = ELFFamily(qu=qu, co=fill(co, 5), theta=theta)
        dd_jl = GAM.elf_Dd(elf, y, mu, wt; level=2)

        @test maximum(abs.(dd_jl[:Dmu4] .- rcopy(R"dd$Dmu4"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dth2] .- rcopy(R"dd$Dth2"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmuth2] .- rcopy(R"dd$Dmuth2"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu2th2] .- rcopy(R"dd$Dmu2th2"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu3th] .- rcopy(R"dd$Dmu3th"))) < 1e-12
    end

    @testset "ELF log-saturated likelihood matches R" begin
        y = [1.0, 2.0, 3.0, -1.0]
        wt = ones(4)

        for (qu, co, theta) in [(0.5, 0.2, 0.0), (0.1, 0.1, -1.0), (0.9, 0.3, 0.5)]
            @rput y wt qu co theta
            R"""
            fam <- elf(theta=theta, qu=qu, co=co)
            ls_r <- fam$ls(y, wt, theta, scale=1)
            """

            elf = ELFFamily(qu=qu, co=fill(co, 4), theta=theta)
            ls_jl = GAM.elf_ls(elf, y, wt)

            @test abs(ls_jl.ls - rcopy(R"ls_r$ls")) < 1e-10
            @test abs(ls_jl.lsth1 - rcopy(R"ls_r$lsth1")) < 1e-10
            @test abs(ls_jl.lsth2 - rcopy(R"ls_r$lsth2")) < 1e-10
        end
    end

    @testset "ELF GAM fit correlated with R" begin
        Random.seed!(42)
        n = 200
        x = collect(range(0, 4, length=n))
        y = sin.(x) .+ 0.5 .* randn(n)
        @rput y x

        R"""
        df_r <- data.frame(y=y, x=x)
        fit_r <- gam(y ~ s(x, k=10, bs="cr"), family=elf(theta=0, qu=0.5, co=0.2), data=df_r)
        fitted_r <- as.numeric(fitted(fit_r))
        """
        fitted_r = rcopy(R"fitted_r")

        elf = ELFFamily(qu=0.5, co=fill(0.2, n), theta=0.0)
        fit_jl = gam(@gam_formula(y ~ s(x, bs=:cr, k=10)), (y=y, x=x); family=elf)

        @test fit_jl.converged
        @test cor(fit_jl.fitted_values, fitted_r) > 0.99
    end

    @testset "pinball loss matches R" begin
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        mu = [1.5, 1.5, 3.5, 3.5, 4.0]

        for qu in [0.1, 0.5, 0.9]
            @rput y mu qu
            R"""
            pin_r <- qgam::pinLoss(y, mu, qu)
            """
            pin_r = rcopy(R"pin_r")
            pin_jl = pinball_loss(y, mu, qu)
            @test abs(pin_jl - pin_r) < 1e-10
        end
    end

    @testset "ELFLSS NLL matches R expression" begin
        for (y_i, η1, η2, qu, co) in [
            (1.2, 0.8, log(0.4), 0.5, 0.2),
            (-0.3, -0.1, log(0.7), 0.8, 0.35),
            (2.0, 1.4, log(0.25), 0.2, 0.15),
        ]
            @rput y_i η1 η2 qu co
            R"""
            mu <- η1
            sig <- exp(η2)
            z <- (y_i - mu) / sig
            nll_r <- -((1 - qu) * z -
                       co * log1p(exp((y_i - mu) / co)) / sig -
                       log(co * beta(co * (1 - qu) / sig, co * qu / sig)))
            """

            fam = ELFLSSFamily(qu=qu, co=co)
            nll_jl = GAM.nll_obs(fam, y_i, [η1, η2])
            @test abs(nll_jl - rcopy(R"nll_r")) < 1e-10
        end
    end

    @testset "ELFLSS predictions match mgcv/qgam family for intercept-only fit" begin
        Random.seed!(404)
        n = 200
        y = 1.5 .+ 0.4 .* randn(n)
        qu = 0.75
        co = 0.2

        formulas = [
            @gam_formula(y ~ 1),
            @gam_formula(y ~ 1),
        ]
        fit_jl = qgam(formulas, (y=y,), qu; co=co)

        newdata_jl = (dummy=ones(5),)
        pred_link_jl = predict(fit_jl, newdata_jl; type=:link)
        pred_resp_jl, se_resp_jl = predict(fit_jl, newdata_jl; type=:response, se=true)

        @rput y qu co
        R"""
        df_r <- data.frame(y=y)
        nd_r <- data.frame(dummy=rep(1, 5))
        fit_r <- mgcv::gam(
            list(y ~ 1, ~ 1),
            family=qgam::elflss(qu=qu, co=co, theta=0, remInter=FALSE),
            data=df_r
        )
        pred_link_r <- predict(fit_r, newdata=nd_r, type="link")
        pred_resp_r <- predict(fit_r, newdata=nd_r, type="response")
        pred_resp_se_r <- predict(fit_r, newdata=nd_r, type="response", se.fit=TRUE)
        """

        pred_link_r = rcopy(R"pred_link_r")
        pred_resp_r = rcopy(R"pred_resp_r")
        se_resp_r = rcopy(R"pred_resp_se_r$se.fit")

        @test size(pred_link_jl) == size(pred_link_r)
        @test size(pred_resp_jl) == size(pred_resp_r)
        @test size(se_resp_jl) == size(se_resp_r)

        @test pred_link_jl ≈ pred_link_r atol=1e-10
        @test pred_resp_jl ≈ pred_resp_r atol=1e-10
        @test se_resp_jl ≈ se_resp_r atol=1e-10
    end
end
