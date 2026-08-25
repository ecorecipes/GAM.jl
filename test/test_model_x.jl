# `GamModel.X` is pure duplication: every smooth block of it is bitwise equal
# to the corresponding `ConstructedSmooth.X`, so the only part not recoverable
# from `smooths` is the parametric block. These tests pin that claim, because
# it is the entire justification for `drop_model_matrix!` — if a basis ever
# stopped storing its own block, or the column layout drifted from
# `first_para:last_para`, reassembly would silently return a different matrix
# rather than fail.
#
# Note the reassembly must be BITWISE, not approximate. Re-evaluating the
# bases instead (via `_gam_prediction_matrix`) drifts by ~2.7e-13 on thin-plate
# smooths, which would propagate into `concurvity`'s `qr(X).R`.

@testset "model matrix retention" begin
    _bitwise(A, B) = size(A) == size(B) &&
        all(reinterpret(UInt64, vec(A)) .== reinterpret(UInt64, vec(B)))

    rng_mx = StableRNG(7)
    n = 400
    df = DataFrame(x = rand(rng_mx, n), z = rand(rng_mx, n),
        g = string.(rand(rng_mx, 1:5, n)))
    df.y = sin.(2π .* df.x) .+ df.z .+ 0.3 .* randn(rng_mx, n)

    cases = (
        ("two smooths", GAM.@formula(y ~ s(x) + s(z))),
        ("tensor", GAM.@formula(y ~ te(x, z, k = 5))),
        ("random effect", GAM.@formula(y ~ s(x) + s(g, bs = :re))),
        ("side constraints", GAM.@formula(y ~ s(x) + s(z) + te(x, z, k = 5))),
        ("parametric term", GAM.@formula(y ~ z + s(x))),
    )

    @testset "$lbl" for (lbl, f) in cases
        m = gam(f, df)

        # The premise: each smooth block of X *is* that smooth's stored basis.
        for sm in m.smooths
            @test _bitwise(m.X[:, sm.first_para:sm.last_para], sm.X)
        end

        # Retained by default, and the accessor is then the field itself.
        @test has_model_matrix(m)
        @test model_matrix(m) === m.X

        X_before = copy(m.X)
        GAM.drop_model_matrix!(m)

        @test !has_model_matrix(m)
        @test size(m.X) == (0, 0)
        @test _bitwise(model_matrix(m), X_before)

        # Idempotent, and reassembly is stable across calls.
        GAM.drop_model_matrix!(m)
        @test _bitwise(model_matrix(m), X_before)

        # The retained parametric block is all that is kept, and it is small.
        @test size(m.X_par) == (n, m.n_parametric)
        @test length(m.X_par) < length(X_before)
    end

    @testset "retained bytes" begin
        m = gam(GAM.@formula(y ~ s(x) + s(z)), df)
        before = sizeof(m.X)
        GAM.drop_model_matrix!(m)
        after = sizeof(m.X) + sizeof(m.X_par)
        # Only the parametric columns survive, so the saving is the smooth
        # part of the matrix: p - n_parametric columns of n doubles.
        @test after == n * m.n_parametric * sizeof(Float64)
        @test before > after
    end

    @testset "a hand-built model with no parametric block reports clearly" begin
        # A model claiming parametric columns but carrying none must raise
        # rather than reassemble a matrix with an uninitialised leading block.
        m = gam(GAM.@formula(y ~ s(x)), df)
        m.X = Matrix{Float64}(undef, 0, 0)
        m.X_par = Matrix{Float64}(undef, 0, 0)
        @test_throws ArgumentError model_matrix(m)
    end

    @testset "every consumer works with the matrix dropped" begin
        # The six sites that used to read `m.X` directly now go through
        # `model_matrix` (or `sm.X`, where the block they wanted IS the
        # smooth's own). A dropped model must give them identical results,
        # not a silent 0x0 — `concurvity` is the canary, since it used to
        # throw `BoundsError` here.
        mk() = gam(GAM.@formula(y ~ g + s(x) + s(z)), df)
        mr = mk()
        md = mk()
        GAM.drop_model_matrix!(md)
        @test has_model_matrix(mr)
        @test !has_model_matrix(md)

        # Equality is `==`, not `≈`: reassembly is bitwise, so anything
        # weaker would hide a real divergence.
        @test concurvity(mr; full = true) == concurvity(md; full = true)
        @test concurvity(mr; full = false) == concurvity(md; full = false)
        @test anova_gam(mr).smooth_table.p_value ==
              anova_gam(md).smooth_table.p_value
        @test GAM.partial_residuals(mr).residual ==
              GAM.partial_residuals(md).residual
        @test smooth_estimates(mr; select = 1, n = 50).estimate ==
              smooth_estimates(md; select = 1, n = 50).estimate
        @test GAM.fitted_samples(mr; n = 5, seed = 1) ==
              GAM.fitted_samples(md; n = 5, seed = 1)
        @test predict(mr, df; type = :response) == predict(md, df; type = :response)
        @test lpmatrix(mr, df) == lpmatrix(md, df)
        @test GAM.leverage(mr) == GAM.leverage(md)
        @test GAM.cooksdistance(mr) == GAM.cooksdistance(md)

        # `k_check` runs an unseeded randomization test, so its p-value is not
        # reproducible even between two calls on the SAME model; compare only
        # the deterministic fields.
        kdet(v) = [(r.label, r.k, r.edf, r.k_index) for r in v]
        @test kdet(k_check(mr)) == kdet(k_check(md))
    end

    @testset "per-smooth consumers read sm.X rather than reassembling" begin
        # `_wood_test_statistic`, `partial_residuals` and the plotting recipe
        # want exactly `m.X[:, sm.first_para:sm.last_para]`. That block IS
        # `sm.X`, so they read it directly — reassembling the full matrix to
        # slice it would be strictly worse, and inside a per-smooth loop it
        # would happen once per smooth.
        for f in (GAM.@formula(y ~ s(x) + s(z)),
                  GAM.@formula(y ~ te(x, z, k = 4)),
                  GAM.@formula(y ~ g + s(x)),
                  GAM.@formula(y ~ s(x) + s(z) + te(x, z, k = 4)))
            m = gam(f, df)
            for sm in m.smooths
                @test _bitwise(m.X[:, sm.first_para:sm.last_para], sm.X)
            end
        end
    end
end
