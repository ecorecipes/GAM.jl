# Tests for the ti() identifiability fix:
# ti() marginals now have the sum-to-zero constraint absorbed BEFORE the
# tensor product is formed (mgcv's mc=TRUE convention), so the ti() span
# contains no constant and no marginal main effects.

using Test
using GAM
using Random
using LinearAlgebra
using Statistics
using DataFrames
using StatsAPI: fitted, predict

# Relative least-squares residual of v against the column space of A
function rel_resid(A::Matrix{Float64}, v::Vector{Float64})
    return norm(v - A * (A \ v)) / norm(v)
end

@testset "ti() review fixes" begin

    rng = Xoshiro(2026)

    # ------------------------------------------------------------------
    @testset "1. No constant / main effects in ti span (bs=$bs)" for bs in (:cr, :tp)
        n = 300
        x = rand(rng, n) .* 2 .- 1
        z = rand(rng, n) .* 2 .- 1
        data = DataFrame(x=x, z=z)

        sm_ti = smooth_construct(ti(:x, :z, k=25, bs=bs), data)
        Xti = sm_ti.X

        # ti() basis dimension is prod(k_j - 1) = 4*4 = 16
        @test size(Xti) == (n, 16)

        # Constant not in span
        r_const = rel_resid(Xti, ones(n))
        @test r_const > 0.9

        # Marginal main-effect bases (centered, as they enter a fitted model)
        # are not in the ti span — test every column of each marginal smooth
        for v in (:x, :z)
            sm_marg = smooth_construct(s(v, k=5, bs=bs), data)
            for j in 1:size(sm_marg.X, 2)
                col = sm_marg.X[:, j]
                r = rel_resid(Xti, col)
                @test r > 0.9
            end
        end

        # Penalties: one per margin, correct size, symmetric PSD
        @test length(sm_ti.S) == 2
        for S in sm_ti.S
            @test size(S) == (16, 16)
            @test norm(S - S') < 1e-8
            @test all(eigvals(Symmetric(S)) .>= -1e-8)
        end

        # Bookkeeping: null space of ti block = product of marginal nullities.
        # For cubic-type marginals the constrained null space is the centered
        # linear function (dim 1 each) -> 1 for the block.
        @test sm_ti.null_dim == 1
        @test sm_ti.rank == 16 - 1

        # No overall constraint (absorbed in marginals)
        @test sm_ti.constraint === nothing
    end

    # ------------------------------------------------------------------
    @testset "2. Additive truth -> ti EDF small" begin
        rng2 = Xoshiro(42)
        n = 400
        x = rand(rng2, n) .* 2 .- 1
        z = rand(rng2, n) .* 2 .- 1
        y = sin.(2 .* x) .+ z .^ 2 .+ 0.1 .* randn(rng2, n)
        df = DataFrame(x=x, z=z, y=y)

        m = gam(GAM.@formulak(y ~ s(x, k = 8, bs = :cr) + s(z, k = 8, bs = :cr) +
                                  ti(x, z, k = 25, bs = :cr)), df)
        @test m.converged

        i_ti = findfirst(sm -> sm.spec.basis isa TensorInteraction, m.smooths)
        @test i_ti !== nothing
        edf_ti = m.edf[i_ti]
        @info "Additive truth: ti EDF = $edf_ti"
        # mgcv 1.9-4 gives ti EDF = 3.244 on this exact dataset (GAM.jl
        # matches to 3 decimals); the ti block retains one unpenalized
        # bilinear direction, so EDF on additive truth stays moderate
        @test edf_ti < 4.0
    end

    # ------------------------------------------------------------------
    @testset "3. Genuine interaction recovered" begin
        rng3 = Xoshiro(7)
        n = 400
        x = rand(rng3, n) .* 2 .- 1
        z = rand(rng3, n) .* 2 .- 1
        f_true = sin.(2 .* x) .* z
        y = f_true .+ 0.1 .* randn(rng3, n)
        df = DataFrame(x=x, z=z, y=y)

        m_add = gam(GAM.@formulak(y ~ s(x, k = 8, bs = :cr) + s(z, k = 8, bs = :cr)), df)
        m_int = gam(GAM.@formulak(y ~ s(x, k = 8, bs = :cr) + s(z, k = 8, bs = :cr) +
                                      ti(x, z, k = 25, bs = :cr)), df)
        @test m_int.converged

        rmse_add = sqrt(mean((fitted(m_add) .- f_true) .^ 2))
        rmse_int = sqrt(mean((fitted(m_int) .- f_true) .^ 2))
        @info "Interaction truth: RMSE additive = $rmse_add, RMSE with ti = $rmse_int"
        @test rmse_int < 0.5 * rmse_add
    end

    # ------------------------------------------------------------------
    @testset "4. predict(m, traindata) == fitted(m) with ti" begin
        rng4 = Xoshiro(11)
        n = 300
        x = rand(rng4, n) .* 2 .- 1
        z = rand(rng4, n) .* 2 .- 1
        y = sin.(2 .* x) .* z .+ z .^ 2 .+ 0.1 .* randn(rng4, n)
        df = DataFrame(x=x, z=z, y=y)

        m = gam(GAM.@formulak(y ~ s(x, k = 8, bs = :cr) + s(z, k = 8, bs = :cr) +
                                  ti(x, z, k = 25, bs = :cr)), df)
        pred = predict(m, df; type = :response)
        @test maximum(abs.(pred .- fitted(m))) < 1e-6

        # ti-only model too (constraint transforms reused in prediction)
        m2 = gam(GAM.@formulak(y ~ ti(x, z, k = 25, bs = :cr)), df)
        pred2 = predict(m2, df; type = :response)
        @test maximum(abs.(pred2 .- fitted(m2))) < 1e-6

        # New data prediction has the right shape and is finite
        nd = DataFrame(x=rand(rng4, 50) .* 2 .- 1, z=rand(rng4, 50) .* 2 .- 1)
        prednew = predict(m, nd; type = :response)
        @test length(prednew) == 50
        @test all(isfinite, prednew)
    end

    # ------------------------------------------------------------------
    @testset "5a. te() unchanged alongside ti()" begin
        rng5 = Xoshiro(5)
        n = 200
        x1 = randn(rng5, n)
        x2 = randn(rng5, n)
        data = (x1=x1, x2=x2)

        sm_te = smooth_construct(te(:x1, :x2), data)
        sm_ti = smooth_construct(ti(:x1, :x2), data)

        # te() keeps the full product with one overall constraint: 5*5 - 1 = 24
        @test size(sm_te.X, 2) == 24
        @test sm_te.constraint !== nothing
        # ti() has fewer columns than te(): 4*4 = 16
        @test size(sm_ti.X, 2) == 16
        @test size(sm_ti.X, 2) < size(sm_te.X, 2)

        # te() prediction matrix reproduces training matrix
        Xp_te = predict_matrix(sm_te, data)
        @test Xp_te ≈ sm_te.X atol = 1e-8
        Xp_ti = predict_matrix(sm_ti, data)
        @test Xp_ti ≈ sm_ti.X atol = 1e-8
    end
end

# test_t2.jl (the t2 regression suite) runs as part of the main test suite;
# it requires test-only dependencies (StableRNGs), so it is not included here.
