using Test
using GAM
using StableRNGs
using LinearAlgebra
using Statistics
using StatsAPI: coef, fitted, deviance, predict
using DataFrames

@testset "t2() tensor product smooth" begin

    t2_rng = StableRNG(123)

    @testset "SmoothSpec construction" begin
        sp = t2(:x, :z)
        @test sp isa SmoothSpec{T2TensorProduct}
        @test sp.term_vars == [:x, :z]
        @test sp.basis isa T2TensorProduct
        @test sp.k == 25  # 5 * 5

        sp2 = t2(:x, :z, k=4)
        @test sp2.k == 16  # 4 * 4

        sp3 = t2(:x, :z, bs=:ps)
        @test sp3.term_vars == [:x, :z]

        sp4 = t2(:x, :y, :z)
        @test sp4.term_vars == [:x, :y, :z]
        @test sp4.k == 125  # 5^3

        @test_throws ArgumentError t2(:x)  # need at least 2 vars
    end

    @testset "Basis construction — 2 marginals" begin
        n = 200
        x = randn(t2_rng, n)
        z = randn(t2_rng, n)
        data = DataFrame(x=x, z=z)

        spec = t2(:x, :z, k=5, bs=:cr)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{T2TensorProduct}
        @test size(sm.X, 1) == n

        # After constraint absorption, columns = total_k - 1
        k_marginal = round(Int, 25^(1/2))  # 5
        @test size(sm.X, 2) == k_marginal^2 - 1

        # t2 with 2 marginals each having 1 penalty:
        # S1⊗I, I⊗S2, S1⊗S2 → 3 penalties
        @test length(sm.S) == 3

        # Each penalty should be square, matching column count
        k_eff = size(sm.X, 2)
        for (i, S) in enumerate(sm.S)
            @test size(S) == (k_eff, k_eff)
            # Symmetric
            @test S ≈ S' atol=1e-10
            # Positive semi-definite (eigenvalues ≥ -tol)
            eigs = eigvals(Symmetric(S))
            @test all(eigs .>= -1e-8)
        end
    end

    @testset "Basis construction — 3 marginals" begin
        n = 300
        x = randn(t2_rng, n)
        y = randn(t2_rng, n)
        z = randn(t2_rng, n)
        data = DataFrame(x=x, y=y, z=z)

        spec = t2(:x, :y, :z, k=3, bs=:cr)  # 3 per margin
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{T2TensorProduct}
        @test size(sm.X, 1) == n

        # mgcv's t2 (Wood, Scheipl & Faraway 2013) reparameterizes each
        # marginal into null/range parts and gives one penalty per
        # tensor-product block that involves at least one range factor:
        # 2^d - 1 = 7 penalties for d = 3 (verified against
        # mgcv::smoothCon(t2(x,y,z,k=3,bs="cr")), which gives 7 penalties of
        # ranks 1,2,2,2,4,4,4 over 26 columns with null.space.dim 7).
        @test length(sm.S) == 7

        k_eff = size(sm.X, 2)
        for S in sm.S
            @test size(S) == (k_eff, k_eff)
            @test S ≈ S' atol=1e-10
        end
    end

    @testset "Prediction matrix" begin
        n = 200
        x = randn(t2_rng, n)
        z = randn(t2_rng, n)
        data = DataFrame(x=x, z=z)

        spec = t2(:x, :z, k=5, bs=:cr)
        sm = smooth_construct(spec, data)

        # Predict on training data should match X
        Xp = predict_matrix(sm, data)
        @test size(Xp) == size(sm.X)
        @test Xp ≈ sm.X atol=1e-10

        # Predicting on a tiny subset of the training rows must still use the
        # fitted marginal basis, not rebuild a smaller one from newdata.
        subset_idx = [1, 50, 120, 200]
        subset_data = data[subset_idx, :]
        Xp_subset = predict_matrix(sm, subset_data)
        @test size(Xp_subset) == (length(subset_idx), size(sm.X, 2))
        @test Xp_subset ≈ sm.X[subset_idx, :] atol=1e-10

        # Predict on new data
        n_new = 50
        new_data = DataFrame(
            x=randn(t2_rng, n_new),
            z=randn(t2_rng, n_new),
        )
        Xp_new = predict_matrix(sm, new_data)
        @test size(Xp_new, 1) == n_new
        @test size(Xp_new, 2) == size(sm.X, 2)
    end

    @testset "t2 vs te — different penalties, same basis dimension" begin
        n = 200
        x = randn(t2_rng, n)
        z = randn(t2_rng, n)
        data = DataFrame(x=x, z=z)

        spec_te = te(:x, :z, k=5, bs=:cr)
        spec_t2 = t2(:x, :z, k=5, bs=:cr)

        sm_te = smooth_construct(spec_te, data)
        sm_t2 = smooth_construct(spec_t2, data)

        # Same basis matrix dimensions
        @test size(sm_te.X) == size(sm_t2.X)

        # te has 2 penalties (one per marginal), t2 has 3 (2 marginal + 1 interaction)
        @test length(sm_te.S) == 2
        @test length(sm_t2.S) == 3
    end

    @testset "GAM fitting with t2()" begin
        n = 300
        x = randn(t2_rng, n)
        z = randn(t2_rng, n)
        f_true = sin.(x) .+ cos.(z) .+ 0.5 .* x .* z
        y = f_true .+ 0.3 .* randn(t2_rng, n)
        data = DataFrame(x=x, z=z, y=y)

        m = gam(@formulak(y ~ t2(x, z, k=5)), data)

        @test m isa GamModel
        @test m.converged
        @test length(coef(m)) > 1

        # Should explain a good amount of variance
        ss_res = sum((y .- fitted(m)).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r2_val = 1 - ss_res / ss_tot
        @test r2_val > 0.5

        # Prediction should work
        pred = predict(m, data)
        @test length(pred) == n
        @test pred ≈ fitted(m) atol=1e-6
    end

    @testset "GAM fitting t2 vs te — similar fits" begin
        n = 300
        x = randn(t2_rng, n)
        z = randn(t2_rng, n)
        f_true = sin.(x) .+ cos.(z)
        y = f_true .+ 0.3 .* randn(t2_rng, n)
        data = DataFrame(x=x, z=z, y=y)

        m_te = gam(@formulak(y ~ te(x, z, k=5)), data)
        m_t2 = gam(@formulak(y ~ t2(x, z, k=5)), data)

        @test m_te.converged
        @test m_t2.converged

        # Both should give similar (but not identical) fits
        cor_fits = cor(fitted(m_te), fitted(m_t2))
        @test cor_fits > 0.9
    end

    @testset "@formulak parsing" begin
        gf = @formulak(y ~ t2(x, z))
        @test length(gf.smooth_specs) == 1
        @test gf.smooth_specs[1].basis isa T2TensorProduct
        @test gf.smooth_specs[1].term_vars == [:x, :z]
    end

    @testset "penalty block order matches mgcv" begin
        # mgcv orders t2 blocks by descending range-mask (bit i-1 set when
        # marginal i contributes a range factor): d=2 -> rr, nr, rn; d=3 ->
        # rrr, nrr, rnr, nnr, rrn, nrn, rnn. Verified against mgcv 1.9.4,
        # whose `rank` vectors for these smooths are [9,6,6] and
        # [1,2,2,4,2,4,4] respectively.
        rng_o = StableRNG(77)
        n = 200
        d2 = (x = rand(rng_o, n), z = rand(rng_o, n), w = rand(rng_o, n))

        sm2 = smooth_construct(t2(:x, :z, k = 5), d2)
        @test [rank(S; rtol = 1e-8) for S in sm2.S] == [9, 6, 6]

        sm3 = smooth_construct(t2(:x, :z, :w, k = 3), d2)
        @test [rank(S; rtol = 1e-8) for S in sm3.S] == [1, 2, 2, 4, 2, 4, 4]
    end

    @testset "per-marginal k" begin
        # A vector k sets the marginal dimensions directly, matching mgcv's
        # k = c(4, 7); mgcv gives 27 columns, ranks [10, 10, 4], null dim 3.
        rng_k = StableRNG(78)
        n = 200
        dk = (x = rand(rng_k, n), z = rand(rng_k, n))

        smk = smooth_construct(t2(:x, :z, k = [4, 7]), dk)
        @test size(smk.X, 2) == 27
        @test [rank(S; rtol = 1e-8) for S in smk.S] == [10, 10, 4]
        @test smk.null_dim == 3

        # te takes the same convention
        @test size(smooth_construct(te(:x, :z, k = [4, 7]), dk).X, 2) == 27

        # a scalar k remains a TOTAL-dimension hint
        @test size(smooth_construct(t2(:x, :z, k = 5), dk).X, 2) == 24

        @test_throws ArgumentError t2(:x, :z, k = [4, 7, 9])
        @test_throws ArgumentError t2(:x, :z, k = [2, 7])
    end

    @testset "prediction is a fixed linear map of the raw basis" begin
        # The identifiability constraint reproduces mgcv's FITTING constraint
        # (`C` = column sums over the all-null block). Unlike mgcv — which
        # additionally builds a separate `Cp`-based parameterization used only
        # by PredictMat — we keep ONE basis, so predict must be exactly the
        # same linear map of the raw tensor basis that construction used.
        rng_p = StableRNG(79)
        n = 200
        dp = (x = rand(rng_p, n), z = rand(rng_p, n))
        newp = (x = rand(rng_p, 90), z = rand(rng_p, 90))

        spec = t2(:x, :z, k = 5)
        sm = smooth_construct(spec, dp)
        rms = [GAM._build_raw_marginal(m, dp, nothing) for m in GAM._get_marginals(spec)]
        X_raw_tr = GAM._row_kronecker([rm.X for rm in rms])
        X_raw_new = GAM._row_kronecker([GAM._raw_predict_marginal(rm, newp) for rm in rms])

        M = X_raw_tr \ sm.X                      # the construction-time map
        @test maximum(abs, X_raw_tr * M - sm.X) < 1e-8
        @test maximum(abs, X_raw_new * M - predict_matrix(sm, newp)) < 1e-8
    end

end
