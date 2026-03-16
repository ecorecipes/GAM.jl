# Tests for multi-parameter models (evgam: GEV and GPD)
using Test
using GAM
using GAM: nll_total, nll_derivs!, gev_nll_derivs_exact!, gpd_nll_derivs_exact!,
           deriv_ncols, hess_col, grad_col,
           assemble_gradient, assemble_hessian!,
           mp_newton_inner, mp_control, _compute_eta,
           build_penalty_matrices, count_sp,
           param_links, initial_eta
using LinearAlgebra
using Statistics
using Random

@testset "Multi-parameter models" begin

    # ================================================================
    # GEV NLL correctness
    # ================================================================
    @testset "GEV NLL" begin
        fam = GEVFamily()
        # Known computation: y=3, μ=2, ψ=log(1)=0, ξ=0.2
        # σ = 1, t = 0.2*(3-2)/1 = 0.2, s = 1.2
        # NLL = 0 + (5+1)*log(1.2) + 1.2^(-5) = 6*0.1823 + 0.4019 = 1.4959
        y = [3.0]
        η = [Float64[2.0], Float64[0.0], Float64[0.2]]
        nll = nll_total(fam, y, η)
        expected = 0.0 + (1/0.2 + 1) * log(1.2) + 1.2^(-1/0.2)
        @test nll ≈ expected atol=1e-10

        # Gumbel limit (ξ ≈ 0)
        η0 = [Float64[2.0], Float64[0.0], Float64[0.0]]
        nll0 = nll_total(fam, y, η0)
        expected0 = 0.0 + 1.0 + exp(-1.0)  # ψ + z + exp(-z), z = (3-2)/1 = 1
        @test nll0 ≈ expected0 atol=1e-6
    end

    # ================================================================
    # GEV derivatives vs finite differences
    # ================================================================
    @testset "GEV derivatives" begin
        fam = GEVFamily()
        y = [3.5, 2.1, 4.0]
        μ = [2.0, 1.5, 3.0]
        ψ = [0.5, 0.2, -0.1]
        ξ = [0.2, -0.1, 0.05]

        out = zeros(3, 9)
        gev_nll_derivs_exact!(out, y, μ, ψ, ξ)

        # Check gradient via FD for each observation
        h = 1e-5
        for j in 1:3
            η_base = [Float64[μ[j]], Float64[ψ[j]], Float64[ξ[j]]]
            f0 = nll_total(fam, [y[j]], η_base)

            for k in 1:3
                η_p = deepcopy(η_base)
                η_p[k][1] += h
                fp = nll_total(fam, [y[j]], η_p)
                fd = (fp - f0) / h
                @test out[j, k] ≈ fd atol=1e-4
            end
        end
    end

    # ================================================================
    # GPD NLL correctness
    # ================================================================
    @testset "GPD NLL" begin
        fam = GPDFamily()
        # y=1.5, ψ=0, ξ=0.2 → σ=1, t=0.3, s=1.3
        # NLL = 0 + (5+1)*log(1.3) = 6*0.2624 = 1.5745
        y = [1.5]
        η = [Float64[0.0], Float64[0.2]]
        nll = nll_total(fam, y, η)
        expected = 0.0 + (1/0.2 + 1) * log(1 + 0.2*1.5/1.0)
        @test nll ≈ expected atol=1e-10

        # Exponential limit (ξ ≈ 0): NLL = ψ + y/σ
        η0 = [Float64[0.0], Float64[0.0]]
        nll0 = nll_total(fam, y, η0)
        @test nll0 ≈ 0.0 + 1.5 atol=1e-6
    end

    # ================================================================
    # GPD derivatives vs finite differences
    # ================================================================
    @testset "GPD derivatives" begin
        fam = GPDFamily()
        y = [1.5, 0.3, 2.0]
        ψ = [0.3, -0.2, 0.5]
        ξ = [0.15, 0.3, -0.1]

        out = zeros(3, 5)
        gpd_nll_derivs_exact!(out, y, ψ, ξ)

        h = 1e-5
        for j in 1:3
            η_base = [Float64[ψ[j]], Float64[ξ[j]]]
            f0 = nll_total(fam, [y[j]], η_base)

            for k in 1:2
                η_p = deepcopy(η_base)
                η_p[k][1] += h
                fp = nll_total(fam, [y[j]], η_p)
                fd = (fp - f0) / h
                @test out[j, k] ≈ fd atol=1e-4
            end
        end
    end

    # ================================================================
    # Derivative column indexing
    # ================================================================
    @testset "Derivative indexing" begin
        # K=3: 3 grad + 6 Hessian = 9 columns
        @test deriv_ncols(3) == 9
        @test deriv_ncols(2) == 5
        @test deriv_ncols(1) == 2

        # Hessian column indices for K=3
        @test hess_col(3, 1, 1) == 4  # (1,1)
        @test hess_col(3, 1, 2) == 5  # (1,2)
        @test hess_col(3, 2, 2) == 6  # (2,2)
        @test hess_col(3, 1, 3) == 7  # (1,3)
        @test hess_col(3, 2, 3) == 8  # (2,3)
        @test hess_col(3, 3, 3) == 9  # (3,3)
    end

    # ================================================================
    # Gradient/Hessian assembly
    # ================================================================
    @testset "Assembly" begin
        n = 50
        K = 2
        p1, p2 = 3, 2
        p = p1 + p2

        X1 = randn(n, p1)
        X2 = randn(n, p2)
        X_list = Matrix{Float64}[X1, X2]

        # Create synthetic per-obs derivatives
        derivs = randn(n, deriv_ncols(K))

        g = assemble_gradient(derivs, X_list)
        @test length(g) == p
        @test g[1:p1] ≈ X1' * derivs[:, 1]
        @test g[p1+1:p] ≈ X2' * derivs[:, 2]

        H = zeros(p, p)
        assemble_hessian!(H, derivs, X_list)
        @test size(H) == (p, p)
        @test H ≈ H'  # symmetric
        # Check block (1,1): X1' * diag(derivs[:,3]) * X1
        @test H[1:p1, 1:p1] ≈ X1' * Diagonal(derivs[:, hess_col(2,1,1)]) * X1
        # Check block (1,2): X1' * diag(derivs[:,4]) * X2
        @test H[1:p1, p1+1:p] ≈ X1' * Diagonal(derivs[:, hess_col(2,1,2)]) * X2
    end

    # ================================================================
    # Inner Newton convergence
    # ================================================================
    @testset "Inner Newton" begin
        Random.seed!(123)
        fam = GPDFamily()
        n = 200
        # Generate proper GPD data (positive exceedances)
        y = Float64[]
        for _ in 1:n
            u = rand()
            push!(y, 1.0 * ((1-u)^(-0.1) - 1) / 0.1)  # GPD(σ=1, ξ=0.1)
        end

        X1 = hcat(ones(n), randn(n))  # log-scale: intercept + covariate
        X2 = ones(n, 1)                # shape: intercept only
        X_list = Matrix{Float64}[X1, X2]
        p = 3
        β = zeros(p)
        S = zeros(p, p)
        ctrl = mp_control(inner_tol=1e-6, inner_maxit=200)

        β_opt, nll, g, H, conv = mp_newton_inner(fam, y, X_list, β, S, ctrl)
        @test conv
        @test maximum(abs, g) < 1e-2  # gradient near zero (exact depends on Hessian conditioning)
        @test isfinite(nll)
    end

    # ================================================================
    # Full evgam fit — GEV
    # ================================================================
    @testset "evgam GEV fit" begin
        Random.seed!(42)
        n = 200
        x = range(0, 3, length=n)
        μ_true = 2.0 .+ 0.5 .* sin.(2π .* x)

        y = similar(x, Float64)
        for i in 1:n
            u = rand()
            yp = -log(u)
            y[i] = μ_true[i] + 0.5 * (yp^(-0.1) - 1) / 0.1
        end

        df = (; y=collect(y), x=collect(x))

        m = evgam(
            [@gam_formula(y ~ s(x, bs=:cr, k=8)),
             @gam_formula(y ~ 1),
             @gam_formula(y ~ 1)],
            df, GEVFamily()
        )

        @test m.converged
        @test m.nobs == n
        @test nparams(m) == 3
        @test length(m.coefficients) > 3
        # Location intercept should be near 2.0
        @test abs(m.coefficients[1] - 2.0) < 0.5
        # Shape should be near 0.1
        @test abs(param_coef(m, 3)[1] - 0.1) < 0.15
    end

    # ================================================================
    # Full evgam fit — GPD
    # ================================================================
    @testset "evgam GPD fit" begin
        Random.seed!(99)
        n = 300
        y = similar(Vector{Float64}, n)
        for i in 1:n
            u = rand()
            y[i] = 1.0 * ((1-u)^(-0.1) - 1) / 0.1  # GPD(σ=1, ξ=0.1)
        end

        df = (; y=y)

        m = evgam(
            [@gam_formula(y ~ 1),
             @gam_formula(y ~ 1)],
            df, GPDFamily()
        )

        @test m.converged
        @test nparams(m) == 2
        # Log-scale should be near log(1) = 0
        @test abs(param_coef(m, 1)[1]) < 0.5
        # Shape should be near 0.1
        @test abs(param_coef(m, 2)[1] - 0.1) < 0.15
    end

    # ================================================================
    # Modularity: family interface
    # ================================================================
    @testset "Family interface" begin
        gev = GEVFamily()
        gpd = GPDFamily()

        @test nparams(gev) == 3
        @test nparams(gpd) == 2
        @test param_names(gev) == ["location", "logscale", "shape"]
        @test param_names(gpd) == ["logscale", "shape"]
        @test param_links(gev) == [:identity, :log, :identity]
        @test param_links(gpd) == [:log, :identity]

        # initial_eta
        y = [1.0, 2.0, 3.0]
        η_gev = initial_eta(gev, y)
        @test length(η_gev) == 3
        @test all(length(η) == 3 for η in η_gev)

        η_gpd = initial_eta(gpd, y)
        @test length(η_gpd) == 2
        @test all(length(η) == 3 for η in η_gpd)
    end
end
