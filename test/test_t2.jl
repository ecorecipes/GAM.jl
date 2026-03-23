@testset "t2() tensor product smooth" begin

    t2_rng = StableRNG(123)

    @testset "SmoothSpec construction" begin
        sp = t2(:x, :z)
        @test sp isa SmoothSpec{T2TensorProduct}
        @test sp.term_vars == [:x, :z]
        @test sp.basis isa T2TensorProduct
        @test sp.k == 25  # 5 * 5

        sp2 = t2(:x, :z, k=16)
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

        spec = t2(:x, :z, k=25, bs=:cr)
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

        spec = t2(:x, :y, :z, k=27, bs=:cr)  # 3^3 = 27
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{T2TensorProduct}
        @test size(sm.X, 1) == n

        # 3 marginals, each with 1 penalty:
        # S1⊗I⊗I, I⊗S2⊗I, I⊗I⊗S3, S1⊗S2⊗S3 → 4 penalties
        @test length(sm.S) == 4

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

        spec = t2(:x, :z, k=25, bs=:cr)
        sm = smooth_construct(spec, data)

        # Predict on training data should match X
        Xp = predict_matrix(sm, data)
        @test size(Xp) == size(sm.X)
        @test Xp ≈ sm.X atol=1e-10

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

        spec_te = te(:x, :z, k=25, bs=:cr)
        spec_t2 = t2(:x, :z, k=25, bs=:cr)

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

        m = gam(@gam_formula(y ~ t2(x, z, k=25)), data)

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

        m_te = gam(@gam_formula(y ~ te(x, z, k=25)), data)
        m_t2 = gam(@gam_formula(y ~ t2(x, z, k=25)), data)

        @test m_te.converged
        @test m_t2.converged

        # Both should give similar (but not identical) fits
        cor_fits = cor(fitted(m_te), fitted(m_t2))
        @test cor_fits > 0.9
    end

    @testset "@gam_formula parsing" begin
        gf = @gam_formula(y ~ t2(x, z))
        @test length(gf.smooth_specs) == 1
        @test gf.smooth_specs[1].basis isa T2TensorProduct
        @test gf.smooth_specs[1].term_vars == [:x, :z]
    end

end
