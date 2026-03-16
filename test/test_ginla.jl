@testset "GINLA (Integrated Nested Laplace Approximation)" begin
    @testset "choldrop" begin
        # Build a known positive-definite matrix and its Cholesky
        A = [4.0 2.0 1.0; 2.0 5.0 3.0; 1.0 3.0 6.0]
        R = Matrix(cholesky(Symmetric(A)).U)

        # Drop column/row 2 → should give Cholesky of A[[1,3],[1,3]]
        R_drop = GAM.choldrop(R, 2)
        A_sub = A[[1, 3], [1, 3]]
        @test R_drop' * R_drop ≈ A_sub atol = 1e-10

        # Drop column 1
        R_drop1 = GAM.choldrop(R, 1)
        A_sub1 = A[[2, 3], [2, 3]]
        @test R_drop1' * R_drop1 ≈ A_sub1 atol = 1e-10

        # Drop last column
        R_drop3 = GAM.choldrop(R, 3)
        A_sub3 = A[[1, 2], [1, 2]]
        @test R_drop3' * R_drop3 ≈ A_sub3 atol = 1e-10

        # Larger matrix
        n = 10
        M = randn(rng, n, n)
        A_big = M' * M + 5 * I
        R_big = Matrix(cholesky(Symmetric(A_big)).U)
        for k in [1, 5, n]
            R_dk = GAM.choldrop(R_big, k)
            idx = [1:(k - 1); (k + 1):n]
            A_dk = A_big[idx, idx]
            @test R_dk' * R_dk ≈ A_dk atol = 1e-8
        end
    end

    @testset "_logf joint density" begin
        rng_g = StableRNG(42)
        n = 100
        x = range(0, 2π; length = n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng_g, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, k = 10, bs = :cr)), df)

        # logf at the fitted coefficients should be at the optimum
        nll_opt, grad_opt = GAM._logf(m.coefficients, m, m.X; deriv = true)
        @test isfinite(nll_opt)
        @test nll_opt > 0  # NLL is positive

        # Gradient should be near zero at optimum (up to penalty)
        @test maximum(abs.(grad_opt)) < 1.0

        # logf at a perturbed point should be worse
        beta_bad = m.coefficients .+ 0.5
        nll_bad, _ = GAM._logf(beta_bad, m, m.X; deriv = false)
        @test nll_bad > nll_opt
    end

    @testset "_cubic_interp" begin
        # Interpolate a known function
        x = collect(range(0, 2π; length = 20))
        y = sin.(x)
        xnew = collect(range(0, 2π; length = 100))
        ynew = GAM._cubic_interp(x, y, xnew)

        @test length(ynew) == 100
        @test all(isfinite.(ynew))
        # Should approximate sin reasonably well
        @test maximum(abs.(ynew .- sin.(xnew))) < 0.05
    end

    @testset "_acomp matrix completion" begin
        A = [1.0 0.0 0.0; 0.0 1.0 0.0]
        B, Bi = GAM._acomp(A)
        @test size(B) == (3, 3)
        @test size(Bi) == (3, 3)
        # B * Bi should be identity
        @test B * Bi ≈ I(3) atol = 1e-10

        # First rows of B should be A
        @test B[1:2, :] ≈ A
    end

    @testset "ginla Gaussian basic" begin
        rng_g = StableRNG(123)
        n = 200
        x = range(0, 2π; length = n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng_g, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, k = 10, bs = :cr)), df)

        # Run GINLA
        result = ginla(m; nk = 12, nb = 50)

        @test result isa GinlaResult
        @test size(result.beta) == (size(m.X, 2), 50)
        @test size(result.density) == (size(m.X, 2), 50)
        @test all(result.density .>= 0)

        # Densities should integrate to approximately 1
        for k in 1:size(result.density, 1)
            dx = result.beta[k, 2] - result.beta[k, 1]
            integral = sum(result.density[k, :]) * dx
            @test abs(integral - 1.0) < 0.15
        end

        # Mode of density should be near the coefficient estimate
        for k in 1:min(3, size(result.density, 1))
            mode_idx = argmax(result.density[k, :])
            mode_val = result.beta[k, mode_idx]
            @test abs(mode_val - m.coefficients[k]) < 3 * sqrt(m.Vp[k, k])
        end
    end

    @testset "ginla with coefficient indices" begin
        rng_g = StableRNG(456)
        n = 200
        x = range(0, 2π; length = n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng_g, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, k = 8, bs = :cr)), df)

        # Only compute for first 3 coefficients
        result = ginla(m; A = [1, 2, 3], nk = 10, nb = 40)

        @test size(result.beta) == (3, 40)
        @test size(result.density) == (3, 40)
        @test result.indices == [1, 2, 3]
    end

    @testset "ginla approx levels" begin
        rng_g = StableRNG(789)
        n = 150
        x = range(0, 2π; length = n) |> collect
        y = sin.(x) .+ 0.3 .* randn(rng_g, n)
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, k = 8, bs = :cr)), df)

        # Test all three approximation levels
        r0 = ginla(m; A = [1, 2], nk = 10, nb = 30, approx = 0)
        r1 = ginla(m; A = [1, 2], nk = 10, nb = 30, approx = 1)
        r2 = ginla(m; A = [1, 2], nk = 10, nb = 30, approx = 2)

        # All should produce valid densities
        for r in [r0, r1, r2]
            @test all(r.density .>= 0)
            for k in 1:2
                dx = r.beta[k, 2] - r.beta[k, 1]
                integral = sum(r.density[k, :]) * dx
                @test abs(integral - 1.0) < 0.2
            end
        end

        # approx=0 and approx=1 should give similar results for Gaussian
        @test cor(vec(r0.density), vec(r1.density)) > 0.9
    end

    @testset "ginla Poisson" begin
        rng_g = StableRNG(101)
        n = 200
        x = randn(rng_g, n)
        mu_true = exp.(1.0 .+ 0.5 .* x)
        y = Float64[rand(rng_g, Poisson(m)) for m in mu_true]
        df = DataFrame(x = x, y = y)

        m = gam(@gam_formula(y ~ s(x, k = 8, bs = :cr)), df;
            family = Poisson(), link = LogLink())

        result = ginla(m; A = [1, 2], nk = 10, nb = 40, approx = 1)

        @test all(result.density .>= 0)
        # For Poisson, posteriors may be asymmetric — GINLA should still work
        for k in 1:2
            dx = result.beta[k, 2] - result.beta[k, 1]
            integral = sum(result.density[k, :]) * dx
            @test abs(integral - 1.0) < 0.2
        end
    end
end
