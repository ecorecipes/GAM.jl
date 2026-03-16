using Test, GAM, Random, Statistics, ForwardDiff, LinearAlgebra

@testset "EGPD families" begin

    # ================================================================
    # Basic NLL computation and AD differentiability
    # ================================================================
    @testset "EGPD1 NLL and AD" begin
        fam = EGPD1Family()
        @test GAM.nparams(fam) == 3
        @test GAM.param_names(fam) == ["logscale", "shape", "logkappa"]

        y = 2.0
        η = [log(1.5), 0.2, log(1.5)]
        nll = GAM.nll_obs(fam, y, η)
        @test isfinite(nll)
        @test nll > 0

        # AD gradient
        g = ForwardDiff.gradient(v -> GAM.nll_obs(fam, y, v), η)
        @test length(g) == 3
        @test all(isfinite.(g))

        # AD hessian
        h = ForwardDiff.hessian(v -> GAM.nll_obs(fam, y, v), η)
        @test size(h) == (3, 3)
        @test all(isfinite.(h))
        @test h ≈ h'  # symmetric

        # κ=1 should reduce to standard GPD
        η_gpd = [log(1.5), 0.2, 0.0]  # log(κ)=0 → κ=1
        nll_egpd1 = GAM.nll_obs(fam, y, η_gpd)
        nll_gpd = GAM.nll_obs(GPDFamily(), y, [log(1.5), 0.2])
        @test abs(nll_egpd1 - nll_gpd) < 1e-10

        # Domain checks: out of GPD support should return large penalty
        @test GAM.nll_obs(fam, -3.0, [log(1.0), 0.5, 0.0]) > 1e10

        # Initial eta
        Random.seed!(42)
        y_vec = rand(100) .* 5
        η0 = GAM.initial_eta(fam, y_vec)
        @test length(η0) == 3
        @test all(length.(η0) .== 100)
    end

    @testset "EGPD2 NLL and AD" begin
        fam = EGPD2Family()
        @test GAM.nparams(fam) == 5
        @test GAM.param_names(fam) == ["logscale", "shape", "logkappa1", "logdkappa", "logitp"]

        y = 2.0
        η = [log(1.5), 0.2, log(1.0), log(1.0), 0.0]
        nll = GAM.nll_obs(fam, y, η)
        @test isfinite(nll)
        @test nll > 0

        # AD gradient and hessian
        g = ForwardDiff.gradient(v -> GAM.nll_obs(fam, y, v), η)
        @test length(g) == 5
        @test all(isfinite.(g))

        h = ForwardDiff.hessian(v -> GAM.nll_obs(fam, y, v), η)
        @test size(h) == (5, 5)
        @test all(isfinite.(h))
    end

    @testset "EGPD3 NLL and AD" begin
        fam = EGPD3Family()
        @test GAM.nparams(fam) == 3
        @test GAM.param_names(fam) == ["logscale", "shape", "logdelta"]

        y = 2.0
        η = [log(1.5), 0.2, log(1.0)]
        nll = GAM.nll_obs(fam, y, η)
        @test isfinite(nll)
        @test nll > 0

        g = ForwardDiff.gradient(v -> GAM.nll_obs(fam, y, v), η)
        @test length(g) == 3
        @test all(isfinite.(g))

        h = ForwardDiff.hessian(v -> GAM.nll_obs(fam, y, v), η)
        @test size(h) == (3, 3)
        @test all(isfinite.(h))
    end

    @testset "EGPD4 NLL and AD" begin
        fam = EGPD4Family()
        @test GAM.nparams(fam) == 4
        @test GAM.param_names(fam) == ["logscale", "shape", "logdelta", "logkappa"]

        y = 2.0
        η = [log(1.5), 0.2, log(1.0), log(2.0)]
        nll = GAM.nll_obs(fam, y, η)
        @test isfinite(nll)
        @test nll > 0

        g = ForwardDiff.gradient(v -> GAM.nll_obs(fam, y, v), η)
        @test length(g) == 4
        @test all(isfinite.(g))

        h = ForwardDiff.hessian(v -> GAM.nll_obs(fam, y, v), η)
        @test size(h) == (4, 4)
        @test all(isfinite.(h))
    end

    # ================================================================
    # Default AD-based nll_derivs! via multiparameter framework
    # ================================================================
    @testset "AD nll_derivs! for EGPD families" begin
        Random.seed!(42)
        n = 50
        for (fam, K, η_vals) in [
            (EGPD1Family(), 3, [log(2.0), 0.1, log(1.5)]),
            (EGPD3Family(), 3, [log(2.0), 0.1, log(1.0)]),
            (EGPD4Family(), 4, [log(2.0), 0.1, log(1.0), log(2.0)]),
        ]
            y = rand(n) .* 5
            η_list = [fill(η_vals[k], n) for k in 1:K]
            ncol_out = K + K * (K + 1) ÷ 2
            out = zeros(n, ncol_out)
            GAM.nll_derivs!(fam, out, y, η_list)

            # Check against per-obs ForwardDiff
            for i in [1, n÷2, n]
                η_i = [η_list[k][i] for k in 1:K]
                g_ad = ForwardDiff.gradient(v -> GAM.nll_obs(fam, y[i], v), η_i)
                for k in 1:K
                    @test abs(out[i, k] - g_ad[k]) < 1e-10
                end
            end
        end
    end

    # ================================================================
    # nll_total
    # ================================================================
    @testset "nll_total for EGPD families" begin
        Random.seed!(42)
        n = 100
        y = rand(n) .* 5

        for (fam, η_vals) in [
            (EGPD1Family(), [log(2.0), 0.1, log(1.5)]),
            (EGPD2Family(), [log(2.0), 0.1, log(1.0), log(1.0), 0.0]),
            (EGPD3Family(), [log(2.0), 0.1, log(1.0)]),
            (EGPD4Family(), [log(2.0), 0.1, log(1.0), log(2.0)]),
        ]
            K = length(η_vals)
            η_list = [fill(η_vals[k], n) for k in 1:K]
            nll = GAM.nll_total(fam, y, η_list)
            @test isfinite(nll)

            # Should equal sum of per-obs NLLs
            nll_sum = sum(GAM.nll_obs(fam, y[i], [η_vals[k] for k in 1:K]) for i in 1:n)
            @test abs(nll - nll_sum) < 1e-8
        end
    end

    # ================================================================
    # Constant model fitting via evgam
    # ================================================================
    @testset "evgam EGPD1 constant fit" begin
        Random.seed!(123)
        # Generate EGPD1 data: GPD with power transform G(u) = u^κ
        n = 500
        σ_true, ξ_true, κ_true = 2.0, 0.1, 1.5
        # Sample: u ~ Uniform(0,1), GPD quantile of G_inv(u) = u^(1/κ)
        u = rand(n)
        u_gpd = u .^ (1 / κ_true)
        y = σ_true .* ((1 .- u_gpd) .^ (-ξ_true) .- 1) ./ ξ_true

        df = (; y=y)
        m = evgam(
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df, EGPD1Family()
        )
        @test m.converged
        @test GAM.nparams(m) == 3
        @test length(m.coefficients) == 3
        # Parameters should be in reasonable range
        @test abs(m.coefficients[1] - log(σ_true)) < 1.0
        @test abs(m.coefficients[2] - ξ_true) < 0.3
        @test abs(m.coefficients[3] - log(κ_true)) < 1.0
    end

    @testset "evgam EGPD3 constant fit" begin
        Random.seed!(456)
        n = 500
        σ_true, ξ_true, δ_true = 2.0, 0.1, 1.5
        # Simplified: generate from GPD, parameters should be recoverable
        u = rand(n)
        y = σ_true .* ((1 .- u) .^ (-ξ_true) .- 1) ./ ξ_true

        df = (; y=y)
        m = evgam(
            [@gam_formula(y ~ 1), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df, EGPD3Family()
        )
        @test m.converged
        @test GAM.nparams(m) == 3
        @test length(m.coefficients) == 3
    end

    # ================================================================
    # Smooth model fitting via evgam
    # ================================================================
    @testset "evgam EGPD1 smooth fit" begin
        Random.seed!(789)
        n = 500
        x = range(0, 5, length=n)
        σ = 1.0 .+ 0.5 .* x
        ξ_true = 0.1
        κ_true = 1.5

        u = rand(n)
        u_gpd = u .^ (1 / κ_true)
        y = σ .* ((1 .- u_gpd) .^ (-ξ_true) .- 1) ./ ξ_true

        df = (; y=y, x=collect(x))
        m = evgam(
            [@gam_formula(y ~ s(x, bs=:cr, k=6)), @gam_formula(y ~ 1), @gam_formula(y ~ 1)],
            df, EGPD1Family()
        )
        @test m.converged
        @test length(m.coefficients) > 3  # smooth terms add basis coefficients
    end

end
