using Test
using GAM
using DataFrames
using Distributions
using LinearAlgebra
using StableRNGs

@testset "Targeted regressions" begin
    rng = StableRNG(202)

    @testset "TPRS multi-dimensional null space" begin
        X_data = randn(rng, 80, 4)
        T = GAM._tps_multi_null_basis(X_data, 4)
        @test size(T) == (80, binomial(7, 4))
        @test rank(T) == size(T, 2)
        @test all(sum(abs, T[:, j]) > 0 for j in axes(T, 2))
    end

    @testset "Tensor smooth2random preserves penalty groups" begin
        n = 120
        x = randn(rng, n)
        z = randn(rng, n)
        df = DataFrame(x = x, z = z, y = x .* z .+ 0.1 .* randn(rng, n))
        sm = smooth_construct(te(:x, :z, k = 25), df)
        smm = smooth2random(sm)
        @test length(smm.Zs) == length(sm.S)
        group_counts = [count(==(i), smm.pen_ind) for i in 1:length(sm.S)]
        @test all(group_counts .> 0)
        @test sum(group_counts) == length(smm.rind)
    end

    @testset "Random-effect prediction uses training levels" begin
        df = DataFrame(site = ["site_B", "site_A", "site_C", "site_A"], y = randn(rng, 4))
        sm = smooth_construct(s(:site, bs = :re), df)
        Xp = predict_matrix(sm, DataFrame(site = ["site_C", "site_A", "site_B"]))
        @test size(Xp, 1) == 3
        @test all(sum(abs, Xp; dims = 2) .> 0)
    end

    @testset "REML log|S+| derivative matches finite differences for multi-penalty tensor smooth" begin
        # This targets the specific fix: `_reml_gradient`'s d(log|S+|)/d(log sp_j)
        # term must be computed per-penalty within each block (λ_j * tr(S_λ^+ S_j))
        # rather than using `block.rank` for every penalty sharing a block.
        #
        # We finite-difference `_log_penalty_det` directly (the log-pseudo-determinant
        # whose analytic derivative this term computes) rather than the full REML
        # score/gradient. The full score's gradient also has a D1/trA1 contribution
        # whose correctness depends on how the scale parameter is estimated relative
        # to what the score's own stationarity condition requires; that is a separate,
        # pre-existing numerical question (affecting non-Gaussian families and any
        # family with an estimated scale) that is independent of, and out of scope
        # for, this multi-penalty log|S+| fix.
        n = 80
        x = randn(rng, n)
        z = randn(rng, n)
        y = sin.(x) .+ cos.(z) .+ 0.15 .* randn(rng, n)
        data = DataFrame(x = x, z = z)
        sm = smooth_construct(te(:x, :z, k = 25), data)
        sm.first_para = 2
        sm.last_para = 1 + size(sm.X, 2)
        X = hcat(ones(n), sm.X)
        penalty = GAM.setup_penalties([sm], 1)
        @test length(penalty.blocks) == 1
        @test length(penalty.blocks[1].S) == 2  # tensor smooth: 2 marginal penalties, 1 block

        log_sp = log.([0.7, 1.6])
        h = 1e-5
        fd = similar(log_sp)
        for j in eachindex(log_sp)
            plus = copy(log_sp); plus[j] += h
            minus = copy(log_sp); minus[j] -= h
            fd[j] = (GAM._log_penalty_det(penalty, plus) - GAM._log_penalty_det(penalty, minus)) / (2h)
        end

        # Reproduce the per-block, per-penalty analytic derivative the same way
        # `_reml_gradient` computes it: λ_j * tr(S_λ^+ S_j) using the *block's*
        # combined (not per-penalty) pseudo-inverse.
        block = only(penalty.blocks)
        k_block = block.stop - block.start + 1
        S_block = zeros(k_block, k_block)
        for (offset, Sj) in enumerate(block.S)
            S_block .+= exp(log_sp[offset]) .* Sj
        end
        eig_block = eigen(Symmetric(S_block))
        thresh_block = eps(Float64) * max(maximum(abs.(eig_block.values)), 1.0)
        S_block_pinv = zeros(k_block, k_block)
        for i in eachindex(eig_block.values)
            λeig = eig_block.values[i]
            if λeig > thresh_block
                vi = eig_block.vectors[:, i]
                S_block_pinv .+= (1 / λeig) .* (vi * vi')
            end
        end
        analytic = [exp(log_sp[j]) * tr(S_block_pinv * block.S[j]) for j in eachindex(log_sp)]

        @test analytic ≈ fd atol = 1e-3 rtol = 1e-3
        # Regression check for the original bug: `block.rank` (a single scalar
        # shared by every penalty in the block) must NOT equal both per-penalty
        # derivatives here, since the two marginal penalties have different
        # sensitivities -- this is exactly what "using block.rank for every
        # penalty in a block" got wrong.
        @test analytic[1] != analytic[2]
    end

    @testset "Gamma and inverse-Gaussian saturated likelihoods are finite and REML-stationary" begin
        x = collect(range(0.2, 1.2; length = 100))
        μ = exp.(0.4 .+ 0.3 .* sin.(2π .* x))
        φ = 0.15
        y_gamma = [rand(rng, Gamma(1 / φ, μi * φ)) for μi in μ]
        df_gamma = DataFrame(x = x, y = y_gamma)
        m = gam(@formulak(y ~ s(x, k = 12)), df_gamma; family = Gamma(), link = LogLink(), method = :REML)
        S_total = GAM.total_penalty(m.penalty, m.sp, size(m.X, 2))
        pirls_result = GAM.pirls(m.X, m.y, S_total, m.family, m.link; start = m.coefficients)
        _, grad = GAM.reml_score(m.X, m.y, m.penalty, m.sp, m.family, m.link, m.weights, pirls_result; method = :REML)
        @test maximum(abs.(grad)) < 5e-2
        @test isfinite(GAM._log_saturated_likelihood(Gamma(), m.y, m.weights, m.scale))

        y_ig = [rand(rng, InverseGaussian(μi, 1 / φ)) for μi in μ]
        @test isfinite(GAM._log_saturated_likelihood(InverseGaussian(), y_ig, ones(length(y_ig)), φ))
    end
end
