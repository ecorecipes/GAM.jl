using Test
using GAM
using StableRNGs
using LinearAlgebra
using Statistics
using StatsAPI

@testset "Side Constraints (gam.side)" begin
    # Shared test data
    rng = StableRNG(42)
    n = 300
    x = randn(rng, n)
    z = randn(rng, n)
    y = sin.(x) .+ cos.(z) .+ 0.5 .* x .* z .+ randn(rng, n) .* 0.3
    data = (; y, x, z)

    # ----------------------------------------------------------------
    @testset "No overlap — s(x) + s(z)" begin
        m = gam(@gam_formula(y ~ s(x, k = 8) + s(z, k = 8)), data)
        @test m.converged
        @test length(m.smooths) == 2

        # Different variables → no side constraints applied
        for sm in m.smooths
            @test isempty(sm.del_index)
        end
    end

    # ----------------------------------------------------------------
    @testset "1d + 2d overlap — s(x) + s(x, z)" begin
        m = gam(@gam_formula(y ~ s(x, k = 8) + s(x, z, k = 25)), data)
        @test m.converged
        @test length(m.smooths) == 2

        sm1d = m.smooths[1]
        sm2d = m.smooths[2]
        @test length(sm1d.spec.term_vars) == 1
        @test length(sm2d.spec.term_vars) == 2

        # 1d smooth should be untouched
        @test isempty(sm1d.del_index)

        # 2d smooth should have columns removed (linear-in-x overlap)
        @test !isempty(sm2d.del_index)
        @test length(sm2d.del_index) >= 1
    end

    # ----------------------------------------------------------------
    @testset "te() with marginals — s(x) + s(z) + te(x, z)" begin
        m = gam(@gam_formula(y ~ s(x, k = 8) + s(z, k = 8) + te(x, z, k = 25)), data)
        @test m.converged
        @test length(m.smooths) == 3

        sm_x = m.smooths[1]
        sm_z = m.smooths[2]
        sm_te = m.smooths[3]

        # Marginal smooths should be untouched (lower dimensional)
        @test isempty(sm_x.del_index)
        @test isempty(sm_z.del_index)

        # te() includes marginal effects, so side constraints should remove columns
        @test !isempty(sm_te.del_index)
        @test length(sm_te.del_index) >= 1
    end

    # ----------------------------------------------------------------
    @testset "side-constrained tensor keeps linear constraints aligned" begin
        df = DataFrame(y = y, x = x, z = z)
        gf = @gam_formula(y ~ s(x, k = 8, bs = :cr) +
                              s(z, k = 8, bs = :cr) +
                              te(x, z, k = 25, bs = [:sc, :cr], xt = Any[["m+"], nothing]))
        _, _, _, smooths, _ = GAM.setup_gam(gf, df; family = Normal())

        sm_te = smooths[3]
        @test !isempty(sm_te.del_index)
        @test GAM.has_linear_constraints(sm_te)
        @test sm_te.Ain !== nothing
        @test size(sm_te.Ain, 2) == size(sm_te.X, 2)

        Xp = predict_matrix(sm_te, df[1:10, [:x, :z]])
        @test size(Xp) == (10, size(sm_te.X, 2))
    end

    # ----------------------------------------------------------------
    @testset "ti() with marginals — s(x) + s(z) + ti(x, z)" begin
        m = gam(@gam_formula(y ~ s(x, k = 8) + s(z, k = 8) + ti(x, z, k = 25)), data)
        @test m.converged
        @test length(m.smooths) == 3

        sm_x = m.smooths[1]
        sm_z = m.smooths[2]
        sm_ti = m.smooths[3]

        # Marginal smooths untouched
        @test isempty(sm_x.del_index)
        @test isempty(sm_z.del_index)

        # ti() excludes marginal effects by construction — no side constraints needed
        @test isempty(sm_ti.del_index)
    end

    # ----------------------------------------------------------------
    @testset "Predict consistency" begin
        # Use CR basis (exact prediction) and te() for the 2d interaction
        m = gam(@gam_formula(y ~ s(x, k = 8, bs = :cr) + s(z, k = 8, bs = :cr) + te(x, z, k = 25)), data)
        @test m.converged

        # te() has side constraints applied (marginals overlap)
        sm_te = m.smooths[3]
        @test !isempty(sm_te.del_index)

        # predict on training data should match fitted values (CR + te reproduce exactly)
        pred = StatsAPI.predict(m, data; type = :response)
        @test length(pred) == n
        @test pred ≈ m.fitted_values atol = 1e-8

        # predict_matrix should respect del_index for the constrained smooth
        Xp = predict_matrix(sm_te, data)
        @test size(Xp, 2) == size(sm_te.X, 2)
        @test size(Xp, 1) == n

        # predict on new data should return correct length
        rng2 = StableRNG(99)
        newdata = (; x = randn(rng2, 50), z = randn(rng2, 50))
        pred_new = StatsAPI.predict(m, newdata; type = :response)
        @test length(pred_new) == 50
        @test all(isfinite, pred_new)
    end

    # ----------------------------------------------------------------
    @testset "Internal: _fix_dependence" begin
        rng3 = StableRNG(123)
        n_test = 100

        # Case 1: X2 has a column in span of X1 → should return indices
        X1 = randn(rng3, n_test, 3)
        X2_dep = hcat(randn(rng3, n_test, 2), X1[:, 1])  # 3rd col is dependent
        ind = GAM._fix_dependence(X1, X2_dep)
        @test ind isa Vector{Int}
        @test !isempty(ind)

        # Case 2: X2 fully independent of X1 → should return nothing
        X2_indep = randn(rng3, n_test, 3)
        ind2 = GAM._fix_dependence(X1, X2_indep)
        @test ind2 === nothing

        # Case 3: X2 is entirely in span of X1 (all columns dependent)
        X2_full = X1 * randn(rng3, 3, 2)
        ind3 = GAM._fix_dependence(X1, X2_full)
        @test ind3 isa Vector{Int}
        @test length(ind3) == 2  # both columns removed
    end

    # ----------------------------------------------------------------
    @testset "Internal: _augment_smooth_X" begin
        # Construct a real smooth to get realistic matrices
        rng4 = StableRNG(456)
        x_test = randn(rng4, 100)
        spec = s(:x_test, k = 8)
        sm = smooth_construct(spec, (; x_test = x_test))

        nobs = size(sm.X, 1)
        k = size(sm.X, 2)
        np = k  # just use k as total param count for this test

        X_aug = GAM._augment_smooth_X(sm, nobs, np)
        @test size(X_aug) == (nobs + np, k)
        # Top portion should be the original basis matrix
        @test X_aug[1:nobs, :] ≈ sm.X
    end

    # ----------------------------------------------------------------
    @testset "Internal: _should_side_constrain" begin
        rng5 = StableRNG(789)
        x_sc = randn(rng5, 50)

        # TPRS smooth → should be constrained
        spec_tp = s(:x_sc, k = 8)
        sm_tp = smooth_construct(spec_tp, (; x_sc = x_sc))
        @test GAM._should_side_constrain(sm_tp) == true

        # Cubic regression spline → should be constrained
        spec_cr = s(:x_sc, k = 8, bs = :cr)
        sm_cr = smooth_construct(spec_cr, (; x_sc = x_sc))
        @test GAM._should_side_constrain(sm_cr) == true

        # RandomEffect → should NOT be constrained
        spec_re = SmoothSpec(
            [:x_sc], RandomEffect(), 8, nothing, nothing,
            nothing, false, nothing, "s(x_sc,bs=re)"
        )
        sm_re = ConstructedSmooth(
            spec_re,
            Matrix{Float64}(I, 50, 8),       # X
            [Matrix{Float64}(I, 8, 8)],       # S
            Float64[],                          # knots
            0, 8,                               # null_dim, rank
            nothing, nothing,                   # constraint, qrc
            1, 8,                               # first_para, last_para
            nothing, nothing, nothing,          # Sigma, cmX, p_ident
            Int[]                               # del_index
        )
        @test GAM._should_side_constrain(sm_re) == false

        # FactorSmooth → should NOT be constrained
        spec_fs = SmoothSpec(
            [:x_sc], FactorSmooth(), 8, nothing, nothing,
            nothing, false, nothing, "s(x_sc,bs=fs)"
        )
        sm_fs = ConstructedSmooth(
            spec_fs,
            Matrix{Float64}(I, 50, 8),
            [Matrix{Float64}(I, 8, 8)],
            Float64[],
            0, 8,
            nothing, nothing,
            1, 8,
            nothing, nothing, nothing,
            Int[]
        )
        @test GAM._should_side_constrain(sm_fs) == false
    end
end
