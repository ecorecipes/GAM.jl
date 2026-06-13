using Test
using GAM
using DataFrames
using Distributions
using Random
using LinearAlgebra
using Statistics
using SpecialFunctions: digamma, trigamma

@testset "Review fixes — extended families / EGPD / gamlss / SCAM / bam" begin

    # ========================================================================
    # Fix 1: NB θ-estimation Newton Hessian (extended_families.jl)
    # g2 must equal dg1/dθ:
    #   trigamma(y+θ) - trigamma(θ) + 1/θ - 2/(μ+θ) + (θ+y)/(μ+θ)²
    # ========================================================================
    @testset "NB theta Newton: Hessian matches FD of gradient" begin
        # Per-observation gradient/Hessian exactly as implemented in
        # estimate_theta!(::NegBinFamily, ...)
        g1_obs(y, mu, th) = digamma(y + th) - digamma(th) + log(th) -
                            log(mu + th) + (mu - y) / (mu + th)
        g2_obs(y, mu, th) = trigamma(y + th) - trigamma(th) + 1.0 / th -
                            2.0 / (mu + th) + (th + y) / (mu + th)^2

        for (y, mu, th) in [(0.0, 1.5, 0.8), (3.0, 2.0, 1.0), (7.0, 4.5, 2.5),
                            (1.0, 0.3, 5.0), (12.0, 9.0, 0.4), (2.0, 2.0, 10.0)]
            h = 1e-6 * max(1.0, th)
            fd = (g1_obs(y, mu, th + h) - g1_obs(y, mu, th - h)) / (2h)
            an = g2_obs(y, mu, th)
            @test isapprox(an, fd; rtol = 1e-5, atol = 1e-8)
        end

        # Behavioral check: estimate_theta! converges to a root of the score
        Random.seed!(20260612)
        n = 2000
        θ_true = 2.5
        mu = fill(3.0, n)
        y = Float64.([rand(NegativeBinomial(θ_true, θ_true / (θ_true + m))) for m in mu])
        fam = NegBinFamily(theta = 1.0, estimate_theta = true)
        GAM.estimate_theta!(fam, y, mu, ones(n), 1.0)
        θ̂ = fam.theta

        # Score at the returned θ̂ should be (near) zero
        score = sum(g1_obs(y[i], mu[i], θ̂) for i in 1:n)
        scale_score = sum(abs(g2_obs(y[i], mu[i], θ̂)) for i in 1:n)
        @test abs(score) / scale_score < 1e-4
        # And θ̂ should be in a sane neighborhood of the truth
        @test 1.5 < θ̂ < 4.0
    end

    # ========================================================================
    # Fix 2: EGPD1/3/4 exact derivatives — small-|ξ| branch + support guards
    # ========================================================================
    @testset "EGPD exact derivatives vs finite differences" begin
        fd_grad = function (fam, yi, η, k, h)
            ηp = copy(η); ηm = copy(η)
            ηp[k] += h; ηm[k] -= h
            return (GAM.nll_obs(fam, yi, ηp) - GAM.nll_obs(fam, yi, ηm)) / (2h)
        end

        yvals = [0.05, 0.3, 1.0, 2.5]
        n = length(yvals)

        configs = [
            (EGPD1Family(), 3, [0.2, 0.3]),          # ψ, lκ
            (EGPD3Family(), 3, [0.2, -0.2]),         # ψ, lδ
            (EGPD4Family(), 4, [0.2, -0.2, 0.3]),    # ψ, lδ, lκ
        ]

        for (fam, K, extra) in configs
            ncols = K + div(K * (K + 1), 2)
            for ξ in (-0.3, -1e-7, 0.0, 1e-7, 0.3)
                η_list = Vector{Vector{Float64}}(undef, K)
                η_list[1] = fill(extra[1], n)
                η_list[2] = fill(ξ, n)
                for k in 3:K
                    η_list[k] = fill(extra[k - 1], n)
                end
                out = zeros(n, ncols)
                GAM.nll_derivs!(fam, out, yvals, η_list)
                @test all(isfinite, out)

                for i in 1:n
                    ηv = [η_list[k][i] for k in 1:K]
                    for k in 1:K
                        # Step: stay inside the small-|ξ| branch when ξ ≈ 0
                        h = (k == 2 && abs(ξ) < 1e-6) ? 2e-7 : 1e-5
                        fd = fd_grad(fam, yvals[i], ηv, k, h)
                        @test isapprox(out[i, k], fd; rtol = 1e-4, atol = 1e-5)
                    end
                end
            end
        end

        # Hessian diagonal spot-check at ξ = ±0.3 (2nd-order central FD)
        for fam in (EGPD1Family(), EGPD3Family())
            for ξ in (-0.3, 0.3)
                η_list = [fill(0.2, n), fill(ξ, n), fill(0.3, n)]
                out = zeros(n, 9)
                GAM.nll_derivs!(fam, out, yvals, η_list)
                hess_diag_cols = (4, 6, 9)  # (1,1), (2,2), (3,3)
                for i in 1:n
                    ηv = [0.2, ξ, 0.3]
                    for (k, col) in zip(1:3, hess_diag_cols)
                        h = 1e-4
                        ηp = copy(ηv); ηm = copy(ηv)
                        ηp[k] += h; ηm[k] -= h
                        f0 = GAM.nll_obs(fam, yvals[i], ηv)
                        fp = GAM.nll_obs(fam, yvals[i], ηp)
                        fm = GAM.nll_obs(fam, yvals[i], ηm)
                        # NB: `2f0` would parse as the Float32 literal 2.0f0
                        fd2 = (fp - 2 * f0 + fm) / h^2
                        @test isapprox(out[i, col], fd2; rtol = 5e-3, atol = 1e-4)
                    end
                end
            end
        end

        # No NaN/Inf crossing ξ = 0 (nll and derivatives), fine grid
        for fam in (EGPD1Family(), EGPD3Family(), EGPD4Family())
            K = GAM.nparams(fam)
            ncols = K + div(K * (K + 1), 2)
            for ξ in vcat(collect(range(-2e-6, 2e-6; length = 41)), [-1e-8, 1e-8, 0.0])
                ηv = vcat([0.2, ξ], fill(0.1, K - 2))
                for yi in yvals
                    @test isfinite(GAM.nll_obs(fam, yi, ηv))
                end
                η_list = [fill(ηv[k], n) for k in 1:K]
                out = zeros(n, ncols)
                GAM.nll_derivs!(fam, out, yvals, η_list)
                @test all(isfinite, out)
            end
        end

        # Out-of-support: 1 + ξy/σ ≤ 0 → huge NLL, zero (finite) derivatives
        for fam in (EGPD1Family(), EGPD3Family(), EGPD4Family())
            K = GAM.nparams(fam)
            ncols = K + div(K * (K + 1), 2)
            ξ = -0.5
            ybad = [3.0]  # σ = 1 (ψ=0): 1 + (-0.5)(3) = -0.5 ≤ 0
            ηv = vcat([0.0, ξ], fill(0.1, K - 2))
            @test GAM.nll_obs(fam, ybad[1], ηv) >= 1e19
            η_list = [fill(ηv[k], 1) for k in 1:K]
            out = fill(NaN, 1, ncols)
            GAM.nll_derivs!(fam, out, ybad, η_list)
            @test all(out .== 0.0)
        end
    end

    # ========================================================================
    # Fix 3: gamlss :local_ml AD working response for generic families
    # ========================================================================
    @testset "gamlss BetaRegression with sp_method=:local_ml" begin
        Random.seed!(20260613)
        n = 200
        x = collect(range(0, 1; length = n))
        μ_true = @. 1 / (1 + exp(-(-1 + 2x)))
        φ = 8.0
        y = [rand(Beta(μ * φ, (1 - μ) * φ)) for μ in μ_true]
        y = clamp.(y, 1e-4, 1 - 1e-4)
        df = DataFrame(x = x, y = y)

        m = gamlss([GAM.@formulak(y ~ s(x, k=8)), GAM.@formulak(y ~ 1)],
                   df, BetaRegression();
                   method = :rs,
                   gamlss_ctrl = gamlss_control(sp_method = :local_ml))
        @test m isa GAM.MultiParameterModel
        μ_fit = @. 1 / (1 + exp(-m.fitted_eta[1]))
        @test cor(μ_fit, μ_true) > 0.9
    end

    # ========================================================================
    # Fix 4: p_ident[1] = false for pure convex/concave SCOP-splines
    # ========================================================================
    @testset "SCAM cv/cx: first coefficient unexponentiated" begin
        Random.seed!(20260614)
        n = 100
        x = collect(range(0, 1; length = n))
        ddf = DataFrame(x = x, y = randn(n))

        for (bs, first_free) in [(:cv, true), (:cx, true), (:mpi, false),
                                 (:mpd, false), (:micx, false), (:micv, false),
                                 (:mdcx, false), (:mdcv, false)]
            gf = GAM.GamFormula(:y, Symbol[], true, [s(:x; bs = bs, k = 10)])
            _, _, _, smooths, _ = GAM.setup_gam(gf, ddf)
            pid = smooths[1].p_ident
            @test pid !== nothing
            if first_free
                @test pid[1] == false
                @test all(pid[2:end])
            else
                @test all(pid)
            end
        end
    end

    @testset "SCAM :cv fits a strictly decreasing concave function" begin
        Random.seed!(20260615)
        n = 400
        x = collect(range(0, 1; length = n))
        f_true = @. -x^2 - 0.5x
        y = f_true .+ 0.05 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(GAM.@formulak(y ~ s(x, bs = :cv, k = 12)), df)
        rmse = sqrt(mean((m.fitted_values .- f_true) .^ 2))
        @test rmse < 0.02   # was ~0.09 before p_ident fix

        # No regression: hump-shaped concave still fits
        f_hump = @. -4 * (x - 0.5)^2
        y2 = f_hump .+ 0.05 .* randn(n)
        df2 = DataFrame(x = x, y = y2)
        m2 = scam(GAM.@formulak(y ~ s(x, bs = :cv, k = 12)), df2)
        rmse2 = sqrt(mean((m2.fitted_values .- f_hump) .^ 2))
        @test rmse2 < 0.02

        # Convex analogue: decreasing convex y = (x-1)²
        f_cx = @. (x - 1)^2
        y3 = f_cx .+ 0.05 .* randn(n)
        df3 = DataFrame(x = x, y = y3)
        m3 = scam(GAM.@formulak(y ~ s(x, bs = :cx, k = 12)), df3)
        rmse3 = sqrt(mean((m3.fitted_values .- f_cx) .^ 2))
        @test rmse3 < 0.02
    end

    # ========================================================================
    # Fix 5: pen_rank for drops_first constraint types
    # ========================================================================
    @testset "SCAM penalty rank matches stored pen_rank" begin
        Random.seed!(20260616)
        n = 120
        x = collect(range(0, 1; length = n))
        ddf = DataFrame(x = x, y = randn(n))

        for bs in (:mpi, :mpd, :cx, :cv, :micx, :micv, :mdcx, :mdcv)
            gf = GAM.GamFormula(:y, Symbol[], true, [s(:x; bs = bs, k = 10)])
            _, _, _, smooths, _ = GAM.setup_gam(gf, ddf)
            sm = smooths[1]
            S = sm.S[1]
            ev = eigvals(Symmetric(S))
            num_rank = count(>(maximum(ev) * 1e-9), ev)
            @test num_rank == sm.rank
        end
    end

    # ========================================================================
    # Fix 6: SCAM EFS uses X_eff = X·diag(Cdiag) — fits still converge
    # ========================================================================
    @testset "SCAM monotone fit converges with corrected EFS" begin
        Random.seed!(20260617)
        n = 300
        x = collect(range(0, 1; length = n))
        f_true = @. 3 / (1 + exp(-8 * (x - 0.5)))
        y = f_true .+ 0.1 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = scam(GAM.@formulak(y ~ s(x, bs = :mpi, k = 10)), df)
        @test m.converged
        rmse = sqrt(mean((m.fitted_values .- f_true) .^ 2))
        @test rmse < 0.08
        # Fitted function should be monotone non-decreasing
        @test all(diff(m.fitted_values) .>= -1e-6)
    end

    # ========================================================================
    # Fixes 7 & 8: bam(gf::GamFormula) honors family; family-aware mustart
    # ========================================================================
    @testset "bam Poisson matches gam (GamFormula path)" begin
        Random.seed!(20260618)
        n = 2000
        x = collect(range(0, 1; length = n))
        μ_true = @. exp(1 + 0.6 * sin(2π * x))
        y = Float64.([rand(Poisson(m)) for m in μ_true])
        df = DataFrame(x = x, y = y)

        # @formulak produces a GamFormula → exercises the previously buggy method
        gf = GAM.@formulak(y ~ s(x, k = 10, bs = :cr))
        @test gf isa GAM.GamFormula

        m_bam = bam(gf, df; family = Poisson())
        @test m_bam.family isa Poisson
        @test m_bam.converged
        @test all(m_bam.fitted_values .> 0)

        m_gam = gam(gf, df; family = Poisson())
        @test cor(m_gam.fitted_values, m_bam.fitted_values) > 0.999
        @test abs(m_gam.deviance_val - m_bam.deviance_val) <
              0.01 * abs(m_gam.deviance_val)

        # Gaussian init no longer clamped to (0.001, 0.999): values far outside
        Random.seed!(20260619)
        yg = 100.0 .+ 5.0 .* sin.(2π .* x) .+ randn(n)
        dfg = DataFrame(x = x, y = yg)
        m_g = bam(GAM.@formulak(y ~ s(x, k = 10, bs = :cr)), dfg)
        @test m_g.converged
        @test sqrt(mean((m_g.fitted_values .- (100.0 .+ 5.0 .* sin.(2π .* x))) .^ 2)) < 0.5
    end
end
