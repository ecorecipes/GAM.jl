using Test
using GAM
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs
using StatsAPI: fitted, coef, predict

const rng_spde = StableRNG(314)

@testset "SPDE Matérn smooth (spde)" begin

    # ------------------------------------------------------------------
    # 1D FEM matrix properties
    # ------------------------------------------------------------------

    @testset "1D FEM mass matrix (C) is SPD" begin
        t = collect(range(0.0, 1.0; length = 20))
        C = GAM._fem_mass_matrix_1d(t)
        @test size(C) == (20, 20)
        @test norm(C - C') < 1e-14  # symmetric
        evals = eigvals(Symmetric(C))
        @test all(evals .> -1e-12)  # PSD
        @test minimum(evals) > 1e-10  # actually SPD for interior mesh
    end

    @testset "1D FEM stiffness matrix (G1) is PSD" begin
        t = collect(range(0.0, 1.0; length = 20))
        G1 = GAM._fem_stiffness_matrix_1d(t)
        @test size(G1) == (20, 20)
        @test norm(G1 - G1') < 1e-14
        evals = eigvals(Symmetric(G1))
        @test all(evals .>= -1e-12)
        # G1 should have a 1D null space (constants)
        @test count(e -> e < 1e-10, evals) == 1
    end

    @testset "1D interpolation matrix rows sum to 1" begin
        t = collect(range(0.0, 10.0; length = 15))
        x = sort(rand(rng_spde, 50) * 10.0)
        A = GAM._fem_interpolation_matrix_1d(x, t)
        @test size(A) == (50, 15)
        row_sums = vec(sum(A; dims = 2))
        @test all(abs.(row_sums .- 1.0) .< 1e-12)
        # Each row has at most 2 nonzero entries
        for i in 1:50
            @test count(A[i, :] .> 0) <= 2
        end
    end

    @testset "Interpolation is exact for linear functions" begin
        t = collect(range(0.0, 5.0; length = 10))
        x = sort(rand(rng_spde, 30) * 5.0)
        A = GAM._fem_interpolation_matrix_1d(x, t)
        # f(x) = 2x + 1 should be interpolated exactly
        f_nodes = 2.0 .* t .+ 1.0
        f_data = 2.0 .* x .+ 1.0
        @test A * f_nodes ≈ f_data atol = 1e-12
    end

    @testset "G2 = G1 * C⁻¹ * G1 relationship" begin
        t = collect(range(0.0, 1.0; length = 15))
        C = GAM._fem_mass_matrix_1d(t)
        G1 = GAM._fem_stiffness_matrix_1d(t)
        C_reg = C + 1e-10 * I
        G2 = G1 * (C_reg \ G1)
        G2 = (G2 + G2') / 2
        # G2 should be PSD
        evals = eigvals(Symmetric(G2))
        @test all(evals .>= -1e-8)
        # G2 should be symmetric
        @test norm(G2 - G2') < 1e-10
    end

    # ------------------------------------------------------------------
    # 1D smooth construction
    # ------------------------------------------------------------------

    @testset "1D SPDE construction: correct dimensions" begin
        n = 100
        x = sort(rand(rng_spde, n) * 2π)
        data = DataFrame(x = x)
        spec = s(:x, bs = :spde, k = 20)

        @test spec.basis isa SPDESmooth
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{SPDESmooth}
        @test size(sm.X, 1) == n
        # k=20 mesh nodes - 1 constraint = 19 columns
        @test size(sm.X, 2) == 19
    end

    @testset "1D SPDE: 3 penalty matrices, all PSD" begin
        n = 100
        x = sort(rand(rng_spde, n) * 2π)
        data = DataFrame(x = x)
        spec = s(:x, bs = :spde, k = 20)
        sm = smooth_construct(spec, data)

        @test length(sm.S) == 3
        ncols = size(sm.X, 2)
        for (i, S) in enumerate(sm.S)
            @test size(S) == (ncols, ncols)
            @test norm(S - S') < 1e-10  # symmetric
            evals = eigvals(Symmetric(S))
            @test all(evals .>= -1e-8)  # PSD
        end
    end

    @testset "1D SPDE: null_dim = 0" begin
        n = 100
        x = sort(rand(rng_spde, n))
        data = DataFrame(x = x)
        spec = s(:x, bs = :spde, k = 15)
        sm = smooth_construct(spec, data)
        @test sm.null_dim == 0
    end

    # ------------------------------------------------------------------
    # GAM fitting with 1D SPDE
    # ------------------------------------------------------------------

    @testset "1D GAM fitting recovers smooth function" begin
        n = 300
        x = sort(rand(rng_spde, n) * 2π)
        f_true = sin.(x)
        y = f_true .+ 0.3 .* randn(rng_spde, n)

        df = DataFrame(x = x, y = y)
        m = gam(@gam_formula(y ~ s(x, bs = :spde, k = 30)), df)

        @test m isa GamModel
        @test m.converged
        @test length(coef(m)) > 0

        # Should recover the true function well
        @test cor(fitted(m), f_true)^2 > 0.7
    end

    @testset "1D prediction works on new data" begin
        n = 200
        x = sort(rand(rng_spde, n) * 2π)
        y = sin.(x) .+ 0.2 .* randn(rng_spde, n)

        df = DataFrame(x = x, y = y)
        m = gam(@gam_formula(y ~ s(x, bs = :spde, k = 25)), df)

        # Predict on new data
        n_new = 50
        x_new = sort(rand(rng_spde, n_new) * 2π)
        df_new = DataFrame(x = x_new)

        pred = predict(m, df_new)
        @test length(pred) == n_new
        @test all(isfinite.(pred))

        # Predictions should be correlated with true function
        f_new = sin.(x_new)
        @test cor(pred, f_new)^2 > 0.5
    end

    # ------------------------------------------------------------------
    # 2D SPDE
    # ------------------------------------------------------------------

    @testset "2D SPDE construction: correct dimensions" begin
        n = 200
        x = rand(rng_spde, n) * 2.0
        y_coord = rand(rng_spde, n) * 2.0
        data = DataFrame(x = x, y = y_coord)

        # k=25 → sqrt(25)=5 → 5×5=25 grid nodes
        spec = s(:x, :y, bs = :spde, k = 25)
        sm = smooth_construct(spec, data)

        @test sm isa ConstructedSmooth{SPDESmooth}
        @test size(sm.X, 1) == n
        # 25 nodes - 1 constraint = 24 columns
        @test size(sm.X, 2) == 24
    end

    @testset "2D SPDE: 3 penalty matrices" begin
        n = 100
        x = rand(rng_spde, n)
        y_coord = rand(rng_spde, n)
        data = DataFrame(x = x, y = y_coord)

        spec = s(:x, :y, bs = :spde, k = 16)
        sm = smooth_construct(spec, data)

        @test length(sm.S) == 3
        ncols = size(sm.X, 2)
        for S in sm.S
            @test size(S) == (ncols, ncols)
            @test norm(S - S') < 1e-8
        end
    end

    @testset "2D GAM fitting" begin
        n = 400
        x1 = rand(rng_spde, n) * 2.0
        x2 = rand(rng_spde, n) * 2.0
        f_true = sin.(x1) .* cos.(x2)
        y = f_true .+ 0.3 .* randn(rng_spde, n)

        df = DataFrame(x1 = x1, x2 = x2, y = y)
        m = gam(@gam_formula(y ~ s(x1, x2, bs = :spde, k = 36)), df)

        @test m isa GamModel
        @test m.converged
        @test cor(fitted(m), f_true)^2 > 0.3
    end

    @testset "2D prediction" begin
        n = 300
        x1 = rand(rng_spde, n) * 2.0
        x2 = rand(rng_spde, n) * 2.0
        y = sin.(x1) .* cos.(x2) .+ 0.3 .* randn(rng_spde, n)

        df = DataFrame(x1 = x1, x2 = x2, y = y)
        m = gam(@gam_formula(y ~ s(x1, x2, bs = :spde, k = 25)), df)

        n_new = 50
        x1n = rand(rng_spde, n_new) * 2.0
        x2n = rand(rng_spde, n_new) * 2.0
        df_new = DataFrame(x1 = x1n, x2 = x2n)

        pred = predict(m, df_new)
        @test length(pred) == n_new
        @test all(isfinite.(pred))
    end

    # ------------------------------------------------------------------
    # Pre-computed FEM matrices via xt
    # ------------------------------------------------------------------

    @testset "Pre-computed FEM matrices via xt" begin
        # Create custom FEM matrices and pass them in
        k = 10
        t = collect(range(0.0, 1.0; length = k))
        C = GAM._fem_mass_matrix_1d(t)
        G1 = GAM._fem_stiffness_matrix_1d(t)
        C_reg = C + 1e-10 * I
        G2 = G1 * (C_reg \ G1)
        G2 = (G2 + G2') / 2

        n = 50
        x = sort(rand(rng_spde, n))
        A = GAM._fem_interpolation_matrix_1d(x, t)

        data = DataFrame(x = x)
        spec = s(:x, bs = :spde, k = k,
                 xt = Dict{Symbol,Any}(:C => C, :G1 => G1, :G2 => G2, :A => A))
        sm = smooth_construct(spec, data)

        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == k - 1  # k - 1 constraint
        @test length(sm.S) == 3
    end

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @testset "Rejects 3D input" begin
        data = DataFrame(x = rand(10), y = rand(10), z = rand(10))
        spec = s(:x, :y, :z, bs = :spde, k = 10)
        @test_throws ArgumentError smooth_construct(spec, data)
    end

    # ------------------------------------------------------------------
    # Comparison with P-spline on same data (sanity check)
    # ------------------------------------------------------------------

    @testset "SPDE vs P-spline: similar recovery of sin(x)" begin
        n = 300
        x = sort(rand(rng_spde, n) * 2π)
        f_true = sin.(x)
        y = f_true .+ 0.3 .* randn(rng_spde, n)
        df = DataFrame(x = x, y = y)

        m_spde = gam(@gam_formula(y ~ s(x, bs = :spde, k = 30)), df)
        m_ps = gam(@gam_formula(y ~ s(x, bs = :ps, k = 30)), df)

        # Both should recover the function well
        r2_spde = cor(fitted(m_spde), f_true)^2
        r2_ps = cor(fitted(m_ps), f_true)^2

        @test r2_spde > 0.6
        @test r2_ps > 0.6
        # They shouldn't differ dramatically
        @test abs(r2_spde - r2_ps) < 0.3
    end
end
