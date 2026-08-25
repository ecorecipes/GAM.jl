# Nested effects (s_nest / gam_nl) — gamFactory-style smooths of estimated
# covariate transformations (Fasiolo et al. 2025, arXiv:2511.19234)

using Test
using GAM
using GAM: TransLinear, TransExpSmooth, TransMGKS, NestedOuterBasis,
    _nested_outer_basis, _nested_design_row, _bspline_row_interior,
    _nested_outer_penalty
using StatsAPI: coef, fitted, predict, deviance, nulldeviance, nobs
using Random, Statistics, LinearAlgebra
using StableRNGs

@testset "Nested effects (s_nest / gam_nl)" begin

    @testset "Spec construction and errors" begin
        sp = s_nest(:l1, :l2, :l3; trans = trans_linear(), k = 12)
        @test sp.basis isa NestedBasis
        @test sp.basis.trans isa TransLinear
        @test sp.k == 12
        @test sp.term_vars == [:l1, :l2, :l3]
        @test has_nested_effects([sp])
        @test !has_nested_effects([s(:x)])

        @test_throws ArgumentError s_nest()
        @test_throws ArgumentError s_nest(:x; trans = trans_mgks())     # needs coords
        @test_throws ArgumentError s_nest(:x; k = 3)
        @test_throws ArgumentError s_nest(:x; by = :g)
        @test_throws ArgumentError trans_linear(penalty = :nope)
        # nested specs cannot go through the standard construction pipeline
        @test_throws ArgumentError smooth_construct(sp, (l1 = [1.0], l2 = [1.0], l3 = [1.0]))
    end

    @testset "Outer basis: s(0)=0 and linear extrapolation" begin
        k = 10
        ob = _nested_outer_basis(k)
        row0 = _bspline_row_interior(0.0, ob.knots, 4, k)
        @test ob.ncols == k - 1
        # design row at u = 0 is exactly zero (s(0) = 0)
        @test maximum(abs, _nested_design_row(ob, 0.0, row0, k)) < 1e-14
        # partition of unity inside the range
        rowu = _bspline_row_interior(1.3, ob.knots, 4, k)
        @test sum(rowu) ≈ 1.0 atol = 1e-10
        # linear extrapolation beyond the boundary: second differences vanish
        us = [3.5, 3.7, 3.9]
        vals = [_nested_design_row(ob, u, row0, k) for u in us]
        second_diff = vals[1] .- 2 .* vals[2] .+ vals[3]
        @test maximum(abs, second_diff) < 1e-10
        # penalty: symmetric PSD
        S = _nested_outer_penalty(k, ob.drop)
        @test issymmetric(S)
        @test minimum(eigvals(Symmetric(S))) > -1e-10
    end

    @testset "Single-index (trans_linear): direction recovery" begin
        rng = StableRNG(7)
        n = 400
        X = randn(rng, n, 3)
        a_true = normalize([0.7, 0.5, 0.2])
        u = X * a_true
        f_true = sin.(1.5 .* u)
        y = f_true .+ 0.2 .* randn(rng, n)
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])

        m = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3, trans = trans_linear(), k = 10)), df)
        @test m isa NestedGamModel
        @test m.converged
        a_hat = inner_coef(m)
        @test length(a_hat) == 3
        @test norm(a_hat) ≈ 1.0 atol = 1e-8
        # the fitter reparameterizes the stored coefficients to unit norm
        @test norm(m.ζ[m.inner_ranges[1]]) ≈ 1.0 atol = 1e-8
        @test abs(cor(X * a_hat, u)) > 0.99
        @test cor(fitted(m), f_true) > 0.98
        @test deviance(m) < nulldeviance(m)
        @test 0.8 < deviance_explained(m) < 1.0
        @test nobs(m) == n

        # prediction reproduces training fit; response/link types work
        p_link = predict(m, df)
        p_resp = predict(m, df; type = :response)
        @test p_resp ≈ fitted(m) atol = 1e-10
        @test p_link ≈ p_resp atol = 1e-10   # identity link
        @test_throws ArgumentError predict(m, df; type = :terms)

        # prediction at new data (including mild extrapolation) is finite
        newdf = (l1 = [0.0, 2.5, -2.5], l2 = [0.0, 2.5, -2.5], l3 = [0.0, 2.5, -2.5])
        @test all(isfinite, predict(m, newdf; type = :response))

        # delta-method standard errors: finite, positive, sane scale, and
        # larger under extrapolation than in the interior
        p_se, se = predict(m, df; se = true)
        @test p_se ≈ predict(m, df) atol = 1e-12
        @test all(isfinite, se) && all(>(0.0), se)
        @test median(se) < 0.2                       # response sd is 0.2
        # ~95% CIs cover the truth at most training points
        @test mean(abs.(p_se .- f_true) .<= 1.96 .* se) > 0.7
        far = (l1 = [8.0], l2 = [8.0], l3 = [8.0])
        near = (l1 = [0.1], l2 = [0.1], l3 = [0.1])
        _, se_far = predict(m, far; se = true)
        _, se_near = predict(m, near; se = true)
        @test se_far[1] > se_near[1]
        # response-scale SEs equal link-scale ones under identity link
        _, se_resp = predict(m, df; type = :response, se = true)
        @test se_resp ≈ se atol = 1e-12
    end

    @testset "gam() auto-routing, mixed smooths, Poisson" begin
        rng = StableRNG(11)
        n = 500
        X = randn(rng, n, 3)
        a_true = normalize([0.6, 0.35, 0.15])
        u = X * a_true
        x0 = rand(rng, n) .* 2
        η = 0.3 .+ 0.5 .* cos.(π .* x0) .+ 0.8 .* tanh.(2 .* u)
        y = Float64.(rand.(rng, Poisson.(exp.(η))))
        df = (y = y, x0 = x0, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])

        m = gam(GAM.@formula(y ~ s(x0, k = 8, bs = :cr) +
                                 s_nest(l1, l2, l3, trans = trans_linear(), k = 10)),
            df, Poisson())
        @test m isa NestedGamModel          # auto-routed
        @test m.converged
        @test length(m.smooths) == 1        # the standard s(x0) smooth
        @test abs(cor(X * inner_coef(m), u)) > 0.98
        @test all(predict(m, df; type = :response) .> 0)
        @test deviance_explained(m) > 0.2

        # extended families are rejected with a clear error
        @test_throws ArgumentError gam(
            GAM.@formula(y ~ s_nest(l1, l2, l3, trans = trans_linear())),
            df, NegBinFamily(theta = 1.0))
    end

    @testset "Adaptive exponential smoothing (trans_nexpsm)" begin
        rng = StableRNG(12)
        n = 600
        x = randn(rng, n)
        ω_true = 0.8
        st = similar(x)
        st[1] = x[1]
        for i in 2:n
            st[i] = ω_true * st[i - 1] + (1 - ω_true) * x[i]
        end
        y = sin.(2 .* st ./ std(st)) .+ 0.15 .* randn(rng, n)
        df = (y = y, x = x)

        m = gam_nl(GAM.@formula(y ~ s_nest(x, trans = trans_nexpsm(), k = 8)), df)
        @test m.converged
        ω_hat = 1.0 / (1.0 + exp(-coef(m)[m.inner_ranges[1]][1]))
        @test abs(ω_hat - ω_true) < 0.1
        @test cor(fitted(m), y) > 0.9
    end

    @testset "Kernel smoothing (trans_mgks)" begin
        rng = StableRNG(13)
        n = 200
        cx, cy = rand(rng, n), rand(rng, n)
        z = sin.(3 .* cx) .+ cos.(3 .* cy) .+ 0.1 .* randn(rng, n)
        y = 2 .* (sin.(3 .* cx) .+ cos.(3 .* cy)) .+ 0.2 .* randn(rng, n)
        df = (y = y, z = z, cx = cx, cy = cy)

        m = gam_nl(GAM.@formula(y ~ s_nest(z, cx, cy, trans = trans_mgks(), k = 8)), df)
        @test cor(fitted(m), y) > 0.9
        # prediction at new coordinates smooths over the stored training data
        newdf = (z = zeros(5), cx = collect(range(0.1, 0.9; length = 5)),
            cy = fill(0.5, 5))
        @test all(isfinite, predict(m, newdf; type = :response))

        # neighborhood-truncated (nn) evaluation agrees with the full O(n²)
        # smoother and stores its fixed training neighborhoods
        m_full = gam_nl(GAM.@formula(y ~ s_nest(z, cx, cy,
            trans = trans_mgks(nn = 0), k = 8)), df)
        @test m.nested_aux[1] isa Matrix{Int}
        @test size(m.nested_aux[1]) == (n, 50)
        @test m_full.nested_aux[1] === nothing
        @test cor(fitted(m), fitted(m_full)) > 0.99
        @test_throws ArgumentError trans_mgks(nn = -1)
    end

    @testset "gam_nl argument validation" begin
        df = (y = randn(10), x = randn(10))
        @test_throws ArgumentError gam_nl(GAM.@formula(y ~ s(x, k = 5)), df)  # no s_nest
        df2 = (y = randn(50), l1 = randn(50), l2 = randn(50))
        @test_throws ArgumentError gam_nl(
            GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear())), df2;
            family = InverseGaussian())
    end

    @testset "Offset support" begin
        rng = StableRNG(31)
        n = 250
        X = randn(rng, n, 3)
        u = X * normalize([0.7, 0.5, 0.2])
        f_true = sin.(1.5 .* u)
        y = f_true .+ 0.2 .* randn(rng, n)
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])

        # identity link + free intercept: a constant offset is absorbed by
        # the intercept — fitted values identical, intercept shifted by −c
        c = 5.0
        m0 = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df)
        m_off = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df; offset = fill(c, n))
        @test maximum(abs, fitted(m_off) .- fitted(m0)) < 1e-3
        @test (coef(m0)[1] - coef(m_off)[1]) ≈ c atol = 1e-3
        # predict must be given the offset to reproduce fitted
        @test predict(m_off, df; offset = fill(c, n), type = :response) ≈
              fitted(m_off) atol = 1e-8
        # without the offset, predictions differ by exactly c on the link scale
        @test predict(m_off, df) .+ c ≈ predict(m_off, df; offset = fill(c, n)) atol = 1e-10
        @test_throws ArgumentError gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df; offset = ones(3))

        # Poisson rate model with log-exposure offset recovers the rate
        expo = rand(rng, n) .* 4 .+ 0.5
        rate = exp.(0.3 .+ 0.8 .* tanh.(2 .* u))
        ycnt = Float64.(rand.(rng, Poisson.(rate .* expo)))
        dfp = (y = ycnt, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        mp = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), dfp;
            family = Poisson(), offset = log.(expo))
        @test cor(predict(mp, dfp; type = :response), rate) > 0.9
    end

    @testset "Weights support" begin
        rng = StableRNG(32)
        n = 250
        X = randn(rng, n, 3)
        u = X * normalize([0.7, 0.5, 0.2])
        f_true = sin.(1.5 .* u)
        y = f_true .+ 0.2 .* randn(rng, n)
        y_bad = copy(y)
        y_bad[1:20] .+= 10.0
        df = (y = y_bad, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        w0 = ones(n); w0[1:20] .= 0.0
        m_w = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df; weights = w0)
        m_u = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df)
        # downweighting the corrupted rows recovers the clean signal
        @test cor(fitted(m_w)[21:end], f_true[21:end]) > 0.98
        @test cor(fitted(m_w)[21:end], f_true[21:end]) >
              cor(fitted(m_u)[21:end], f_true[21:end])
        @test_throws ArgumentError gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df; weights = -ones(n))
        @test_throws ArgumentError gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df; weights = ones(5))
    end

    @testset "Categorical parametric covariates" begin
        rng = StableRNG(33)
        n = 240
        X = randn(rng, n, 3)
        u = X * normalize([0.7, 0.5, 0.2])
        f_true = sin.(1.5 .* u)
        g = repeat(["a", "b", "c"], inner = 80)
        shift = Dict("a" => 0.0, "b" => 1.0, "c" => -0.5)
        y = f_true .+ [shift[gi] for gi in g] .+ 0.2 .* randn(rng, n)
        df = (y = y, g = g, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        m = gam_nl(GAM.@formulak(y ~ g + s_nest(l1, l2, l3,
            trans = trans_linear(), k = 8)), df)
        @test m.converged
        # intercept + 2 dummy columns
        @test m.coef_ranges[:parametric] == 1:3
        @test cor(fitted(m), y) > 0.95
        # level contrasts recovered (b−a = 1.0, c−a = −0.5 up to noise)
        @test coef(m)[2] ≈ 1.0 atol = 0.2
        @test coef(m)[3] ≈ -0.5 atol = 0.2
        # prediction reuses the training schema
        @test predict(m, df; type = :response) ≈ fitted(m) atol = 1e-8
    end

    @testset "te() + nested effect: overlapping-group EFS converges" begin
        rng = StableRNG(42)
        n = 400
        X = randn(rng, n, 3)
        a_true = normalize([0.6, 0.35, 0.15])
        u = X * a_true
        x1 = rand(rng, n); x2 = rand(rng, n)
        y = sin.(2π .* x1) .* cos.(π .* x2) .+ 0.8 .* tanh.(2 .* u) .+
            0.2 .* randn(rng, n)
        df = (y = y, x1 = x1, x2 = x2, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        m = gam_nl(GAM.@formula(y ~ te(x1, x2, k = 5) +
                                    s_nest(l1, l2, l3, trans = trans_linear(), k = 10)), df)
        @test m.converged
        @test abs(cor(X * inner_coef(m), u)) > 0.98
        @test deviance_explained(m) > 0.7
    end

    @testset "Constant offset absorbed by the intercept" begin
        rng = StableRNG(31)
        n = 300
        X = randn(rng, n, 3)
        u = X * normalize([0.7, 0.5, 0.2])
        y = sin.(1.5 .* u) .+ 0.2 .* randn(rng, n)
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2], l3 = X[:, 3])
        m0 = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3, trans = trans_linear(), k = 10)), df)
        mc = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, l3, trans = trans_linear(), k = 10)), df;
            offset = fill(2.5, n))
        @test coef(m0)[1] - coef(mc)[1] ≈ 2.5 atol = 1e-6
        @test maximum(abs.(fitted(m0) .- fitted(mc))) < 1e-6
    end

    @testset "NestedControl" begin
        rng = StableRNG(31)
        n = 200
        X = randn(rng, n, 2)
        u = X * normalize([0.8, 0.6])
        y = sin.(u) .+ 0.2 .* randn(rng, n)
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2])

        ctrl = nested_control(outer_maxit = 2, newton_maxit = 50, tol = 1e-6)
        @test ctrl isa NestedControl
        m2 = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(), k = 8)),
            df; control = ctrl)
        @test m2.iterations <= 2          # outer_maxit honored
        m_full = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(), k = 8)), df)
        @test m_full.converged

        # trace runs without error
        mt = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(), k = 8)),
            df; control = nested_control(trace = true))
        @test mt.converged

        # deprecated loose kwargs still work, with a warning, and match control
        m_dep = @test_logs (:warn, r"deprecated") match_mode = :any gam_nl(
            GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(), k = 8)),
            df; outer_maxit = 2, newton_maxit = 50, tol = 1e-6)
        @test m_dep.iterations <= 2
        @test coef(m_dep) ≈ coef(m2) atol = 1e-10

        @test_throws ArgumentError nested_control(outer_maxit = 0)
        @test_throws ArgumentError nested_control(tol = -1.0)
    end

    @testset "Fixed outer sp is honored" begin
        rng = StableRNG(21)
        n = 300
        X = randn(rng, n, 2)
        u = X * normalize([0.8, 0.6])
        y = sin.(u) .+ 0.2 .* randn(rng, n)
        df = (y = y, l1 = X[:, 1], l2 = X[:, 2])
        m_fix = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(),
            k = 8, sp = 1000.0)), df)
        m_free = gam_nl(GAM.@formula(y ~ s_nest(l1, l2, trans = trans_linear(),
            k = 8)), df)
        # the outer log-sp of the fixed fit stays at log(1000)
        n_std = length(m_fix.sp) - 2
        @test m_fix.sp[n_std + 1] ≈ log(1000.0) atol = 1e-12
        @test m_fix.sp[n_std + 1] != m_free.sp[n_std + 1]
    end

    @testset "s_nest sp= validation" begin
        # A nested effect has exactly ONE outer penalty, so a vector `sp` is a
        # user error rather than an unsupported feature. It used to surface as
        # a bare `MethodError: no method matching Float64(::Vector{Float64})`
        # from an inline `Float64(sp)`; it now routes through the shared
        # `_normalize_sp` validator like every other smooth constructor.
        err = try
            s_nest(:x; trans = trans_linear(), k = 8, sp = [1.0, 2.0])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("single outer penalty", sprint(showerror, err))
        @test occursin("scalar", sprint(showerror, err))

        # Shared validation applies too: non-positive and non-finite rejected.
        @test_throws ArgumentError s_nest(:x; trans = trans_linear(), k = 8, sp = -1.0)
        @test_throws ArgumentError s_nest(:x; trans = trans_linear(), k = 8, sp = 0.0)
        @test_throws ArgumentError s_nest(:x; trans = trans_linear(), k = 8, sp = Inf)

        # Scalar and `nothing` still behave exactly as before.
        @test s_nest(:x; trans = trans_linear(), k = 8, sp = 2.0).sp === 2.0
        @test s_nest(:x; trans = trans_linear(), k = 8).sp === nothing
    end
end
