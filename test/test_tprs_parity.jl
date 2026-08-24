# TPRS parity with mgcv — knot rule, translation invariance, partial eigensolver
#
# These pin three properties of the thin-plate basis that were wrong before,
# and that are invisible in a fit-quality check because the fitted values stay
# plausible while the underlying basis differs from mgcv's:
#
#   1. The knot rule. GAM.jl used to drop to a rank-k Nystrom approximation as
#      soon as n > max(3k, 200). mgcv subsamples only above `max.knots` (default
#      2000) and then keeps 2000 knots. The old rule made every n > 200 fit a
#      different model from mgcv's: at n = 500, k = 20 the edf was 11.35 against
#      mgcv's 11.44, and the reported `sp` was off by a factor of 38.
#   2. Translation invariance. The thin-plate kernel depends only on distances,
#      but the polynomial null space and the column-RMS rescaling do not, so
#      without mgcv's mean-centring `shift` the basis (and hence `sp`) depended
#      on the covariate origin.
#   3. The Lanczos eigensolver must agree with the dense one it replaces.
#
# The R-dependent parity checks live in the `_rcall_available` section of
# runtests.jl; everything here is R-free.

@testset "TPRS parity" begin
    tpf(k; bs = :tp, sp = nothing) = GAM.GamFormula(:y, Symbol[], true,
        GAM.SmoothSpec[GAM.s(:x; k = k, bs = bs, sp = sp)])

    # Deterministic data — no RNG, so these are exactly reproducible.
    mkdata(n; shift = 0.0) = begin
        xb = collect((1:n) ./ (n + 1))
        DataFrame(x = xb .+ shift,
            y = sin.(2π .* xb) .+ 0.3 .* sin.(37 .* (1:n)))
    end

    @testset "translation invariance" begin
        # The same data shifted by a constant must give the identical basis,
        # smoothing parameter and fit. Before the `shift` fix, a shift of 100
        # moved sp from 0.010121 to 0.011088.
        m0 = gam(tpf(10), mkdata(200))
        for δ in (100.0, -37.5, 1.0e4)
            mδ = gam(tpf(10), mkdata(200; shift = δ))
            @test exp(mδ.sp[1]) ≈ exp(m0.sp[1]) rtol = 1e-8
            @test sum(GAM.edf(mδ)) ≈ sum(GAM.edf(m0)) rtol = 1e-8
            @test deviance(mδ) ≈ deviance(m0) rtol = 1e-8
        end
    end

    @testset "shift is applied on the prediction path" begin
        # A shifted fit must predict correctly on shifted newdata: if the
        # construction shift were not re-applied at predict time, the kernel
        # distances would be right but the null-space polynomials wrong.
        df = mkdata(150; shift = 250.0)
        m = gam(tpf(10), df)
        @test predict(m, df; type = :response) ≈ fitted(m) atol = 1e-9
    end

    @testset "knot rule follows mgcv's max.knots" begin
        # Below the threshold every unique covariate value is a knot, so the
        # basis is exact rather than a Nystrom approximation.
        for n in (250, 400, 900)
            sm = smooth_construct(GAM.s(:x; k = 10, bs = :tp), mkdata(n))
            @test length(sm.knots) == n
        end
        # Above it, knots are capped at max.knots, not at k.
        sm_big = smooth_construct(GAM.s(:x; k = 10, bs = :tp), mkdata(2600))
        @test length(sm_big.knots) == 2000

        # And the cap is user-settable, mirroring s(..., xt = list(max.knots=)).
        spec = GAM.s(:x; k = 10, bs = :tp)
        spec.xt[:max_knots] = 300
        @test length(smooth_construct(spec, mkdata(900)).knots) == 300
    end

    @testset "knots are reported on the original covariate scale" begin
        # The mean shift is an internal parameterization detail; `knots` should
        # come back in the units the user supplied.
        df = mkdata(300; shift = 500.0)
        sm = smooth_construct(GAM.s(:x; k = 10, bs = :tp), df)
        @test minimum(sm.knots) ≈ minimum(df.x) atol = 1e-9
        @test maximum(sm.knots) ≈ maximum(df.x) atol = 1e-9
    end

    @testset "Lanczos agrees with the dense eigensolver" begin
        # _tprs_top_eigen switches to Lanczos above n = max(400, 4k); the two
        # paths must return the same invariant subspace and eigenvalues.
        x = collect(range(-1, 1; length = 700))
        E = Symmetric(GAM._tps_penalty_matrix(x, 2))
        k = 12
        vl, Ul = GAM._lanczos_eigen(E, k)
        dense = eigen(E)
        idx = sortperm(abs.(dense.values); rev = true)[1:k]
        vd, Ud = dense.values[idx], dense.vectors[:, idx]

        # Tolerances pin what the solver actually achieves at its 1e-12
        # convergence bound (measured: eigenvalues 9.4e-12 relative, projector
        # 1.5e-9). They are deliberately tight: the trailing eigenpair is the
        # least converged, and a regression there would silently perturb the
        # last basis column rather than fail loudly.
        @test vl ≈ vd rtol = 1e-9
        # Eigenvectors are only defined up to sign, so compare the projectors.
        @test Ul * Ul' ≈ Ud * Ud' atol = 1e-7
        @test Ul' * Ul ≈ I atol = 1e-9        # orthonormal
        for j in 1:k                           # genuine eigenpairs
            @test E * Ul[:, j] ≈ vl[j] .* Ul[:, j] atol = 1e-9
        end
    end

    @testset "sp round-trips within GAM.jl" begin
        # Supplying the sp a free fit selected must reproduce that fit exactly.
        # (Cross-package transfer of `sp` is NOT exact for tp — the basis
        # parameterization still differs from mgcv's; see the comment in
        # _construct_tprs.)
        df = mkdata(200)
        mfree = gam(tpf(10), df)
        mfix = gam(tpf(10; sp = exp(mfree.sp[1])), df)
        @test sum(GAM.edf(mfix)) ≈ sum(GAM.edf(mfree)) rtol = 1e-6
        @test deviance(mfix) ≈ deviance(mfree) rtol = 1e-6
    end

    # Not TPRS, but this file is where the basis-parameterization invariants
    # live, and this one guards a defect found while diagnosing bs=:gp against
    # mgcv: the construction and prediction paths each pick the GP correlation
    # function independently, so a default changed in one and not the other
    # yields silently wrong predictions rather than an error.
    @testset "gp correlation function round-trips through predict" begin
        gpf(; xt = Dict{Symbol, Any}()) = begin
            sp = GAM.s(:x; k = 10, bs = :gp)
            merge!(sp.xt, xt)
            GAM.GamFormula(:y, Symbol[], true, GAM.SmoothSpec[sp])
        end
        df = DataFrame(x = collect(range(0, 1; length = 150)),
            y = sin.(2π .* range(0, 1; length = 150)))
        for xt in (Dict{Symbol, Any}(),
                   Dict{Symbol, Any}(:corfun => :mgcv_m32),
                   Dict{Symbol, Any}(:corfun => :matern52),
                   Dict{Symbol, Any}(:corfun => :mgcv_m32, :scale => 2.0))
            m = gam(gpf(; xt = xt), df)
            @test predict(m, df; type = :response) ≈ fitted(m) atol = 1e-8
        end
        # mgcv's default correlation really is the √3-free one, so the two
        # parameterizations must give genuinely different fits.
        m_std = gam(gpf(), df)
        m_mgcv = gam(gpf(; xt = Dict{Symbol, Any}(:corfun => :mgcv_m32)), df)
        @test !isapprox(sum(GAM.edf(m_std)), sum(GAM.edf(m_mgcv)); rtol = 1e-4)
    end

    @testset "bs=:ds warns that it is not a Duchon spline" begin
        # It delegates to tp, which is a different basis from mgcv's ds; the
        # docstring says so but a docstring is not visible at the call site.
        @test_logs (:warn, r"not implemented as a Duchon spline") match_mode = :any begin
            gam(tpf(10; bs = :ds), mkdata(120))
        end
    end

    # ── mgcv parameterization: what makes `sp` transferable ──────────────────
    #
    # The basis below is an exact reparameterization of mgcv's either way — the
    # fitted values agree regardless. What these pin is the *choice* of
    # parameterization, which is observable only through `sp`: mgcv rescales
    # each column of X to RMS 1 and then normalises the penalty by
    # ‖S‖₁/‖X‖∞², and both operations are applied *after* the null-space
    # rotation. Any different-but-valid rotation therefore reports `sp` on a
    # different scale, so an mgcv `sp` fed to GAM.jl (or vice versa) silently
    # fits a different model. Getting these three conventions right took
    # ‖S‖₁/‖X‖∞² from 14.67 to mgcv's 19.017082287 on s(x, k=10), n=200.

    @testset "QT factorization matches mgcv's convention" begin
        # mgcv's QT (matrix.c:394) produces A·Q = [0, T] with T reverse lower
        # triangular, so the null space is the FIRST Ac-Ar columns of Q. This
        # is not LAPACK's `qr(A')` complement.
        for (Ar, Ac) in ((1, 5), (2, 10), (3, 12), (5, 9))
            A = [sin(3.0i + 1.7j) + 0.5cos(i * j) for i in 1:Ar, j in 1:Ac]
            Uhh = GAM._mgcv_qt(A)
            Q = Matrix{Float64}(I, Ac, Ac)
            GAM._hq_mult_right!(Q, Uhh)
            @test maximum(abs.(Q' * Q - I)) < 1e-12          # orthogonal
            AQ = A * Q
            @test maximum(abs.(AQ[:, 1:(Ac - Ar)])) < 1e-11  # null block
            # reverse lower triangular: entries with (i-1)+(j-1) < Ar-1 vanish
            Tb = AQ[:, (Ac - Ar + 1):Ac]
            for i in 1:Ar, j in 1:Ar
                (i - 1) + (j - 1) < Ar - 1 && @test abs(Tb[i, j]) < 1e-11
            end
            # _mgcv_null_space returns exactly that leading block
            Z = GAM._mgcv_null_space(A, Ac, Ar)
            @test size(Z) == (Ac, Ac - Ar)
            @test maximum(abs.(A * Z)) < 1e-11
            @test maximum(abs.(Z' * Z - I)) < 1e-12
        end
    end

    @testset "eigenpairs are emitted in mgcv's descending-signed order" begin
        # Rlanczos selects by |λ| but writes out in descending signed order
        # (mat.c:3852-3862 over a descending tridiagonal spectrum).
        v = [0.5, -3.0, 2.0, -0.25]
        U = Matrix{Float64}(I, 4, 4)
        vo, Uo = GAM._mgcv_eigen_order(v, U)
        @test vo == [2.0, 0.5, -0.25, -3.0]
        @test issorted(vo; rev = true)
        for (j, val) in enumerate(vo)          # columns follow their values
            @test Uo[:, j] == U[:, findfirst(==(val), v)]
        end
        # The real basis path: selection is still by magnitude, order by sign.
        E = Symmetric([exp(-abs(i - j) / 3.0) for i in 1:40, j in 1:40])
        vv, _ = GAM._tprs_top_eigen(E, 6)
        @test issorted(vv; rev = true)
    end

    @testset "null-space monomial order matches gen_tps_poly_powers" begin
        # tprs.c:100-129 is an odometer that increments the LOWEST index first,
        # giving [1, x, z] for d=2,m=2 — not the by-total-degree order, which
        # gives [1, z, x] and permutes the rows of T'U.
        @test GAM._tps_monomial_exponents(2, 2) == [[0, 0], [1, 0], [0, 1]]
        @test GAM._tps_monomial_exponents(1, 2) == [[0], [1]]
        @test GAM._tps_monomial_exponents(1, 3) == [[0], [1], [2]]
        # d=2, m=3: M = 6, degrees 0,1,1,2,2,2 with the lowest index moving first
        e = GAM._tps_monomial_exponents(2, 3)
        @test length(e) == 6
        @test e[1] == [0, 0] && e[2] == [1, 0] && e[3] == [2, 0]
        @test all(sum(v) <= 2 for v in e)
        @test length(unique(e)) == 6
        # d=3, m=2: constant plus the three linear terms, in coordinate order
        @test GAM._tps_monomial_exponents(3, 2) ==
              [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    end

    @testset "penalty normalisation is stable across the basis" begin
        # ‖S‖₁/‖X‖∞² is the constant relating our `sp` to mgcv's; it must not
        # depend on the covariate origin (translation invariance) and must be
        # reproducible run to run.
        function maS(df; k = 10, d1 = true)
            spec = d1 ? GAM.s(:x; k = k) : GAM.s(:x, :z; k = k)
            sm = GAM._construct_tprs(spec, Tables.columntable(df), nothing;
                absorb_cons = false)
            return opnorm(sm.S[1], 1) / opnorm(sm.X, Inf)^2
        end
        base = maS(mkdata(200))
        @test maS(mkdata(200; shift = 100.0)) ≈ base rtol = 1e-10
        @test maS(mkdata(200; shift = -37.5)) ≈ base rtol = 1e-10
        @test base > 0
    end
end
