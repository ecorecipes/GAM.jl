using Test
using GAM
using DataFrames
using Random
using Statistics
using LinearAlgebra
using StatsAPI: coef, fitted, predict

@testset "Review fixes — basis_cr / basis_gp / basis_re" begin

    # ========================================================================
    # Fix 1: cyclic CR wrap-around interval (h_0 ≡ h[end], not h[end-1])
    # ========================================================================
    @testset "cc basis: wrap-around with non-uniform knots" begin
        Random.seed!(20260612)

        # Deliberately irregular knots over one period [0, 2π]
        knots = [0.0, 0.3, 0.9, 1.4, 2.1, 2.7, 3.5, 4.4, 5.1, 2π]
        lo, hi = knots[1], knots[end]

        n = 500
        x = rand(n) .* (hi - lo) .+ lo
        ftrue(t) = sin.(t) .+ 0.5 .* cos.(2 .* t)  # periodic on [0, 2π]
        y = ftrue(x)

        X, S = GAM._cc_basis(x, knots)

        # (a) penalty symmetric PSD
        @test S ≈ S' atol = 1e-10
        ev = eigvals(Symmetric(S))
        @test all(ev .>= -1e-8 * maximum(abs, ev))

        # Penalized LS fit of the periodic function
        β = (X' * X + 1e-8 .* S) \ (X' * y)
        feval(pts) = GAM._cc_basis(collect(pts), knots)[1] * β

        # Fit quality on a grid (smooth periodic target, 9 cyclic basis fns)
        grid = range(lo, hi - 1e-9; length = 400) |> collect
        rmse = sqrt(mean((feval(grid) .- ftrue(grid)) .^ 2))
        @test rmse < 0.1

        # (b) value and first-derivative continuity across the wrap point,
        # compared against the same finite-difference checks at an interior knot
        δ = 1e-5
        f_vals = feval([lo, lo + δ, lo + 2δ, hi - 2δ, hi - δ])
        f_lo, f_lop, f_lopp, f_himm, f_him = f_vals

        # value continuity: f(hi - δ) - f(lo + δ) ≈ 2δ f'(wrap) → O(1e-5)
        @test abs(f_him - f_lop) < 1e-3

        # one-sided derivatives at the wrap point (note f(hi) ≡ f(lo))
        d_right = (f_lop - f_lo) / δ
        d_left = (f_lo - f_him) / δ
        @test abs(d_left - d_right) < 1e-2

        # same check at an interior knot for comparison — wrap should not be worse
        kj = knots[5]
        g_vals = feval([kj - δ, kj, kj + δ])
        d_int_left = (g_vals[2] - g_vals[1]) / δ
        d_int_right = (g_vals[3] - g_vals[2]) / δ
        @test abs(d_left - d_right) < 100 * max(abs(d_int_right - d_int_left), δ)

        # (c) basis evaluation consistent with periodicity
        X_lo = GAM._cc_basis([lo + 1e-9], knots)[1]
        X_hi = GAM._cc_basis([hi - 1e-9], knots)[1]
        @test maximum(abs, X_lo .- X_hi) < 1e-6
        # x beyond the period wraps around
        X_wrap = GAM._cc_basis([hi + 0.37], knots)[1]
        X_base = GAM._cc_basis([lo + 0.37], knots)[1]
        @test X_wrap ≈ X_base atol = 1e-10
    end

    # ========================================================================
    # Fix 2: cyclic penalty rank is k-2 (basis has k-1 cols, constant null)
    # ========================================================================
    @testset "cc penalty rank" begin
        Random.seed!(42)
        n = 200
        x = rand(n) .* 2π
        df = DataFrame(x = x, y = sin.(x))

        for k in [6, 10, 15]
            spec = s(:x, bs = :cc, k = k)
            sm = GAM.smooth_construct(spec, df)

            # cc basis: k-1 columns pre-constraint, k-2 after absorption
            @test size(sm.X, 2) == k - 2
            @test sm.null_dim == 1
            @test sm.rank == k - 2

            # numerical rank of the stored (constrained) penalty
            ev = eigvals(Symmetric(sm.S[1]))
            nrank = count(ev .> maximum(ev) * 1e-9)
            @test nrank == k - 2

            # pre-constraint penalty: (k-1)×(k-1) with 1-dim null space (constant)
            knots = GAM.place_knots(x, k)
            _, S_pre = GAM._cc_basis(x, knots)
            ev_pre = eigvals(Symmetric(S_pre))
            @test count(ev_pre .> maximum(ev_pre) * 1e-9) == k - 2
            # constant is in the null space
            @test norm(S_pre * ones(k - 1)) < maximum(ev_pre) * 1e-8
        end

        # full cc fit still works and predict matches fitted
        Random.seed!(7)
        y = sin.(x) .+ 0.2 .* randn(n)
        df2 = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, bs = :cc, k = 12)), df2)
        @test m.converged
        @test predict(m, df2) ≈ fitted(m) atol = 1e-8
        @test sqrt(mean((fitted(m) .- sin.(x)) .^ 2)) < 0.2
    end

    # ========================================================================
    # Fix 3: GP smooth uses the same length-scale at fit and predict
    # ========================================================================
    @testset "gp smooth: predict matches fitted" begin
        Random.seed!(101)
        n = 150
        x = rand(n) .* 2π  # irregular x → quantile knots exclude extremes
        y = sin.(x) .+ 0.3 .* randn(n)
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, bs = :gp, k = 15)), df)
        @test m.converged
        @test predict(m, df) ≈ fitted(m) atol = 1e-8

        # prediction matrix on training data reproduces the fitted basis
        sm = m.smooths[1]
        @test sm.predict_cache isa GAM.GPPredictCache
        Xp = GAM.predict_matrix(sm, df)
        @test Xp ≈ sm.X atol = 1e-10
    end

    # ========================================================================
    # Fix 4: re smooth prediction with non-integer (string) levels
    # ========================================================================
    @testset "re smooth: string levels at predict" begin
        Random.seed!(2024)
        n_groups = 6
        n_per = 50
        n = n_groups * n_per
        gnames = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        g = repeat(gnames, inner = n_per)
        x = randn(n)
        true_re = randn(n_groups) .* 0.5
        gidx = repeat(1:n_groups, inner = n_per)
        y = cos.(x) .+ true_re[gidx] .+ 0.2 .* randn(n)
        df = DataFrame(x = x, g = g, y = y)

        m = gam(@formulak(y ~ s(x) + s(g, bs = :re)), df)
        @test m.converged

        # before the fix, string levels matched nothing → all-zero RE rows
        pred = @test_logs min_level = Base.CoreLogging.Warn predict(m, df)
        @test pred ≈ fitted(m) atol = 1e-8

        # unseen level: zero row + warning
        sm_re = m.smooths[findfirst(s -> s.spec.basis isa RandomEffect, m.smooths)]
        Xp = @test_logs (:warn,) match_mode = :any GAM.predict_matrix(
            sm_re, (g = ["zulu", "alpha"],))
        @test all(Xp[1, :] .== 0)
        @test sum(Xp[2, :]) == 1.0

        df_new = DataFrame(x = [0.0, 0.0], g = ["zulu", "alpha"])
        pred_new = @test_logs (:warn,) match_mode = :any predict(m, df_new)
        @test all(isfinite, pred_new)
    end

    # ========================================================================
    # Fix 5: no sum-to-zero constraint absorbed for bs=:re
    # ========================================================================
    @testset "re smooth: full-rank identity penalty, no constraint" begin
        Random.seed!(123)
        n_groups = 8
        n_per = 40
        n = n_groups * n_per
        group_id = repeat(1:n_groups, inner = n_per)
        # bs=:re follows mgcv's model.matrix(~ vars - 1) rule: only *factor*
        # variables expand to one column per level. A grouping variable must
        # therefore be categorical — integer codes would enter as a single
        # linear term, exactly as in mgcv.
        group = string.(group_id)
        x = randn(n)
        true_re = randn(n_groups) .* 0.4
        f_true = cos.(x) .+ true_re[group_id]
        y = f_true .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, group = group)

        # construction: k columns, identity penalty, no constraint
        sm = GAM.smooth_construct(s(:group, bs = :re), df)
        @test size(sm.X, 2) == n_groups
        @test sm.constraint === nothing
        @test sm.S[1] ≈ Matrix{Float64}(I, n_groups, n_groups)
        @test sm.null_dim == 0
        @test sm.rank == n_groups

        # full fit
        m = gam(@formulak(y ~ s(x) + s(group, bs = :re)), df)
        @test m.converged
        @test predict(m, df) ≈ fitted(m) atol = 1e-8

        # fit quality: RMSE vs truth no worse than the noise level
        rmse = sqrt(mean((fitted(m) .- f_true) .^ 2))
        @test rmse < 0.2

        # RE coefficients: shrunk toward zero (mean near zero by shrinkage,
        # not exactly zero), and they track the true random effects
        sm_re = m.smooths[findfirst(s -> s.spec.basis isa RandomEffect, m.smooths)]
        @test sm_re.last_para - sm_re.first_para + 1 == n_groups
        b_re = coef(m)[sm_re.first_para:sm_re.last_para]
        @test abs(mean(b_re)) < 0.2
        @test !all(iszero, b_re)
        @test std(b_re) > 0.05
        @test cor(b_re, true_re) > 0.8
    end

    # ========================================================================
    # bs=:re follows mgcv's model.matrix(~ v1:v2:... - 1) semantics:
    # factors expand to indicators, numerics multiply. Verified against
    # mgcv 1.9.4 (see test_rcall.jl for the live comparison).
    # ========================================================================
    @testset "re smooth: factor/numeric semantics match mgcv" begin
        Random.seed!(202)
        n = 200
        gi = repeat(1:5, inner = 40)              # integer codes -> numeric
        gs = repeat(string.('a':'e'), inner = 40) # strings       -> factor
        g2 = repeat(["p", "q"], outer = 100)      # second factor
        xv = rand(n)
        zv = rand(n)
        tbl = (gi = gi, gs = gs, g2 = g2, x = xv, z = zv)

        # Factors expand to one column per level
        @test size(GAM.smooth_construct(s(:gs, bs = :re), tbl).X, 2) == 5
        # Numerics contribute a single multiplier column (mgcv: ncol 1)
        @test size(GAM.smooth_construct(s(:x, bs = :re), tbl).X, 2) == 1
        @test size(GAM.smooth_construct(s(:gi, bs = :re), tbl).X, 2) == 1
        @test size(GAM.smooth_construct(s(:x, :z, bs = :re), tbl).X, 2) == 1
        # Two factors give the product of their level counts
        @test size(GAM.smooth_construct(s(:gs, :g2, bs = :re), tbl).X, 2) == 10
        # Random slopes: the rule is symmetric in the variables
        sm_gx = GAM.smooth_construct(s(:gs, :x, bs = :re), tbl)
        sm_xg = GAM.smooth_construct(s(:x, :gs, bs = :re), tbl)
        @test size(sm_gx.X, 2) == 5
        @test size(sm_xg.X, 2) == 5
        @test sm_gx.X ≈ sm_xg.X

        # A numeric column of repeated integer codes warns that it is being
        # treated as numeric (the user probably meant a grouping factor)
        @test_logs (:warn, r"convert it to a factor") match_mode = :any begin
            GAM.smooth_construct(s(:gi, bs = :re), tbl)
        end

        # Numeric entries really are the multiplier values
        @test vec(GAM.smooth_construct(s(:x, bs = :re), tbl).X) ≈ xv
    end
end

@testset "Count-weighted sum-to-zero constraint" begin
    # `bam(...; discrete=true)` wants to build a smooth at the m distinct
    # covariate values rather than at all n rows. Everything
    # `absorb_constraints!` computes is invariant to row multiplicity EXCEPT
    # C = colSums(X); supplying the per-bin counts reproduces the full-data
    # constraint, so the reduced basis is the full one row-selected.
    rng = StableRNG(7)
    m = 200
    counts = rand(rng, 2:9, m)
    xu = collect(range(0.0, 1.0; length = m))
    xfull = vcat([fill(xu[b], counts[b]) for b in 1:m]...)
    rows = vcat([fill(b, counts[b]) for b in 1:m]...)
    w = Float64.(counts)

    # `cr`/`cc` round-trip their stored knots; `ps`/`bs` store a PADDED
    # B-spline knot vector that re-pads if supplied back (16 -> 22 knots), but
    # they place knots from the covariate range alone, which the grid spans.
    for (bs, supply_knots) in ((:cr, true), (:cc, true), (:ps, false), (:bs, false))
        spec = s(:x; bs = bs, k = 12)
        full = GAM.smooth_construct(spec, (x = xfull,))
        kn = supply_knots && !isempty(full.knots) ? full.knots : nothing
        red = GAM.with_row_weights(w) do
            GAM.smooth_construct(spec, (x = xu,), kn)
        end
        @test size(red.X, 2) == size(full.X, 2)
        @test maximum(abs.(full.X .- red.X[rows, :])) < 1e-11
        @test maximum(abs.(full.S[1] .- red.S[1])) < 1e-9
    end

    # The weighting is load-bearing: without it the constraint is wrong.
    spec = s(:x; bs = :cr, k = 12)
    full = GAM.smooth_construct(spec, (x = xfull,))
    unweighted = GAM.smooth_construct(spec, (x = xu,), full.knots)
    @test maximum(abs.(full.X .- unweighted.X[rows, :])) > 1e-3

    # Outside a `with_row_weights` block the dense path is untouched.
    again = GAM.smooth_construct(spec, (x = xfull,))
    @test all(reinterpret(UInt64, vec(full.X)) .===
              reinterpret(UInt64, vec(again.X)))

    # A mismatched weight vector is rejected rather than silently misapplied.
    X = randn(StableRNG(3), 10, 4)
    S = [Matrix{Float64}(I, 4, 4)]
    @test_throws DimensionMismatch GAM.absorb_constraints!(
        copy(X), deepcopy(S); row_weights = ones(9))
end
