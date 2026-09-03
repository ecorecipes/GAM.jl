# Analytic derivatives of the NCV criterion (`ncv_score_grad`).
#
# A wrong gradient does not announce itself: the optimizer still "converges",
# just to the wrong place, and every downstream fitted value looks plausible.
# So the gradient is checked DIRECTLY here rather than through fitted results.
#
# Two independent checks, because they fail differently:
#
#   1. CENTRAL FINITE DIFFERENCES of the criterion, refitting P-IRLS at each
#      perturbed rho. This is the only check that exercises the whole chain,
#      including the implicit dependence of beta on the smoothing parameters —
#      which is most of the derivation and the easiest part to get wrong.
#      The step is chosen by taking the best of several rather than assuming
#      one: too large and truncation dominates, too small and cancellation
#      does, and where the sweet spot sits depends on the model.
#
#   2. FORWARDDIFF through an independent, generic reimplementation of the
#      Gaussian leave-one-out case (below). It shares no code with
#      `ncv_score_grad`, so agreement is real corroboration rather than a
#      tautology. It is restricted to Gaussian+identity because there beta has
#      the closed form (X'X + S)^-1 X'y and needs no P-IRLS iteration; the
#      production criterion is `Float64`-annotated throughout and cannot be
#      differentiated directly. (The penalty log-determinant itself is no
#      longer a barrier — it carries analytic first and second derivatives for
#      ForwardDiff, which is what lets `sp_optimizer = :newton` run on
#      multi-penalty blocks — but nothing else on the NCV path is generic.)

@testset "NCV analytic derivatives" begin

    # ── helpers ────────────────────────────────────────────────────────────
    # Build the pieces the criterion needs, the same way the fitting path does.
    function _ncv_setup(df, specs, fam, lnk)
        gf = GAM.GamFormula(:y, Symbol[], true, specs)
        _, X, _, sm, _ = GAM.setup_gam(gf, df)
        y = Float64.(df.y)
        n, p = size(X)
        penalty = GAM.setup_penalties(sm, 1)
        return (X = X, y = y, n = n, p = p, penalty = penalty,
                w = ones(n), off = zeros(n), fam = fam, lnk = lnk)
    end

    function _fit_at(S, ctx, lsp)
        GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
        GAM.pirls(ctx.X, ctx.y, S, ctx.fam, ctx.lnk;
                  weights = ctx.w, offset = ctx.off)
    end

    function _crit_at(S, ctx, nei, lsp)
        r = _fit_at(S, ctx, lsp)
        GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
        return first(GAM.ncv_score(ctx.X, ctx.y, r.coefficients,
            r.linear_predictor, ctx.fam, ctx.lnk, S, ctx.w, ctx.off, nei))
    end

    # Worst relative error between the analytic gradient and central FD, over
    # all smoothing parameters, taking the best step per component.
    function _grad_vs_fd(ctx, nei, lsp; steps = (1e-3, 1e-4, 1e-5, 1e-6))
        S = zeros(ctx.p, ctx.p)
        r = _fit_at(S, ctx, lsp)
        GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
        sc, gr, _ = GAM.ncv_score_grad(ctx.X, ctx.y, r.coefficients,
            r.linear_predictor, ctx.fam, ctx.lnk, ctx.penalty, lsp, S,
            ctx.w, ctx.off, nei)
        worst = 0.0
        for l in eachindex(lsp)
            best = Inf
            for h in steps
                lp = copy(lsp); lm = copy(lsp)
                lp[l] += h; lm[l] -= h
                fd = (_crit_at(S, ctx, nei, lp) - _crit_at(S, ctx, nei, lm)) / (2h)
                best = min(best, abs(fd - gr[l]) / max(abs(fd), 1e-10))
            end
            worst = max(worst, best)
        end
        return (sc, gr, worst)
    end

    _rng = StableRNG(4321)
    _n = 120
    _x = sort(rand(_rng, _n))
    _z = rand(_rng, _n)

    # ── 1. the score agrees with `ncv_score`, so the gradient is a derivative
    #       OF THE CRITERION ACTUALLY USED, not of a parallel formula ────────
    @testset "score matches ncv_score exactly" begin
        df = DataFrame(x = _x, y = sin.(2π .* _x) .+ 0.3 .* randn(_rng, _n))
        ctx = _ncv_setup(df, [GAM.s(:x; k = 10, bs = :cr)], Normal(), IdentityLink())
        nei = GAM.loo_neighbourhoods(ctx.n)
        S = zeros(ctx.p, ctx.p)
        for lsp0 in (-1.0, 1.0, 4.0)
            lsp = [lsp0]
            r = _fit_at(S, ctx, lsp)
            GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
            s_only = first(GAM.ncv_score(ctx.X, ctx.y, r.coefficients,
                r.linear_predictor, ctx.fam, ctx.lnk, S, ctx.w, ctx.off, nei))
            s_grad, _, _ = GAM.ncv_score_grad(ctx.X, ctx.y, r.coefficients,
                r.linear_predictor, ctx.fam, ctx.lnk, ctx.penalty, lsp, S,
                ctx.w, ctx.off, nei)
            @test s_grad ≈ s_only atol = 1e-12
        end
    end

    # ── 2. finite differences, across families, links and neighbourhoods ────
    #
    # Tolerances are loose relative to the measured agreement (1e-9..1e-8 for
    # these models) because central FD accuracy is what is being bounded here,
    # not the analytic derivative — a tight bound would make this a test of the
    # step-size search rather than of the gradient.
    @testset "gradient vs central finite differences" begin
        dfg = DataFrame(x = _x, y = sin.(2π .* _x) .+ 0.3 .* randn(_rng, _n))
        ctx = _ncv_setup(dfg, [GAM.s(:x; k = 10, bs = :cr)], Normal(), IdentityLink())
        nei = GAM.loo_neighbourhoods(ctx.n)
        for lsp0 in (-2.0, 0.0, 2.0, 5.0)
            _, _, w = _grad_vs_fd(ctx, nei, [lsp0])
            @test w < 1e-5
        end

        # Several smoothing parameters: catches a gradient that is right for
        # one component and mis-indexed across the penalty blocks.
        dfm = DataFrame(x = _x, z = _z,
                        y = sin.(2π .* _x) .+ 0.5 .* _z .+ 0.3 .* randn(_rng, _n))
        ctxm = _ncv_setup(dfm, [GAM.s(:x; k = 8, bs = :cr), GAM.s(:z; k = 8, bs = :cr)],
                          Normal(), IdentityLink())
        neim = GAM.loo_neighbourhoods(ctxm.n)
        _, grm, wm = _grad_vs_fd(ctxm, neim, [1.0, 1.0])
        @test length(grm) == 2
        @test wm < 1e-5

        # Poisson/log: non-identity link, so d2mu/deta2 and V'(mu) both enter
        # dw2/deta. A gradient that ignored either would still pass Gaussian.
        mu = exp.(1.0 .+ 0.8 .* sin.(2π .* _x))
        dfp = DataFrame(x = _x, y = Float64.([rand(_rng, Poisson(m)) for m in mu]))
        ctxp = _ncv_setup(dfp, [GAM.s(:x; k = 10, bs = :cr)], Poisson(), LogLink())
        _, _, wp = _grad_vs_fd(ctxp, GAM.loo_neighbourhoods(ctxp.n), [1.0])
        @test wp < 1e-5

        # Interval neighbourhoods: folds drop many points, so the Woodbury
        # block is nd x nd rather than scalar.
        _, _, wi = _grad_vs_fd(ctxp, GAM.interval_neighbourhoods(ctxp.n, 5), [1.0])
        @test wi < 1e-5

        # Binomial/logit.
        pb = 1 ./ (1 .+ exp.(-(0.5 .+ 1.5 .* sin.(2π .* _x))))
        dfb = DataFrame(x = _x, y = Float64.([rand(_rng) < q ? 1.0 : 0.0 for q in pb]))
        ctxb = _ncv_setup(dfb, [GAM.s(:x; k = 8, bs = :cr)], Binomial(), LogitLink())
        _, _, wb = _grad_vs_fd(ctxb, GAM.loo_neighbourhoods(ctxb.n), [1.0])
        @test wb < 1e-5
    end

    # ── 3. ForwardDiff through an independent reimplementation ──────────────
    @testset "gradient vs ForwardDiff (independent formulation)" begin
        ad_ok = try
            @eval using ForwardDiff
            true
        catch
            false
        end
        if !ad_ok
            @info "ForwardDiff unavailable; skipping the AD cross-check"
        else
            df = DataFrame(x = _x, y = sin.(2π .* _x) .+ 0.3 .* randn(_rng, _n))
            ctx = _ncv_setup(df, [GAM.s(:x; k = 10, bs = :cr)], Normal(), IdentityLink())
            nei = GAM.loo_neighbourhoods(ctx.n)
            X, y, p = ctx.X, ctx.y, ctx.p

            # Generic Gaussian-identity LOO NCV, written from the definition and
            # sharing no code with `ncv_score_grad`. For Gaussian identity
            # w2 == 1 and beta is a linear solve, so the fold update is the
            # classic h_ii form and the whole thing is differentiable.
            function ncv_gauss(rho::AbstractVector{T}) where {T}
                S = zeros(T, p, p)
                # Rebuild S(rho) directly from the penalty blocks.
                sp_idx = 1
                for block in ctx.penalty.blocks
                    for (i, Si) in enumerate(block.S)
                        lam = exp(rho[sp_idx])
                        base = block.start + block.offsets[i] - 1
                        m = size(Si, 1)
                        for a in 1:m, b in 1:m
                            S[base + a, base + b] += lam * Si[a, b]
                        end
                        sp_idx += 1
                    end
                end
                H = X' * X + S
                Hi = inv(H)
                beta = Hi * (X' * y)
                eta = X * beta
                acc = zero(T)
                for i in 1:ctx.n
                    xi = @view X[i, :]
                    hii = dot(xi, Hi * xi)
                    r_i = y[i] - eta[i]
                    # eta_cv = eta_i - h_ii * w1_i/(1 - h_ii), w1_i = r_i
                    eta_cv = eta[i] - hii * r_i / (1 - hii)
                    acc += (y[i] - eta_cv)^2
                end
                return acc
            end

            for lsp0 in (0.0, 2.0, 4.0)
                lsp = [lsp0]
                g_ad = ForwardDiff.gradient(ncv_gauss, lsp)
                S = zeros(p, p)
                r = _fit_at(S, ctx, lsp)
                GAM.total_penalty!(S, ctx.penalty, lsp, p)
                sc, gr, _ = GAM.ncv_score_grad(X, y, r.coefficients,
                    r.linear_predictor, ctx.fam, ctx.lnk, ctx.penalty, lsp, S,
                    ctx.w, ctx.off, nei)
                # The reimplementation must agree on the VALUE too, otherwise
                # matching gradients would only mean two consistent mistakes.
                @test ncv_gauss(lsp) ≈ sc rtol = 1e-8
                @test g_ad[1] ≈ gr[1] rtol = 1e-6
            end
        end
    end

    # ── 4. the gradient vanishes where the optimizer stops ─────────────────
    @testset "gradient is ~0 at the selected optimum" begin
        df = DataFrame(x = _x, y = sin.(2π .* _x) .+ 0.3 .* randn(_rng, _n))
        m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), df; method = :NCV)
        @test m.converged
        ctx = _ncv_setup(df, [GAM.s(:x; k = 10, bs = :cr)], Normal(), IdentityLink())
        nei = GAM.loo_neighbourhoods(ctx.n)
        S = zeros(ctx.p, ctx.p)
        lsp = copy(m.sp)
        r = _fit_at(S, ctx, lsp)
        GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
        sc, gr, _ = GAM.ncv_score_grad(ctx.X, ctx.y, r.coefficients,
            r.linear_predictor, ctx.fam, ctx.lnk, ctx.penalty, lsp, S,
            ctx.w, ctx.off, nei)
        @test isfinite(sc)
        # Scaled by the criterion, since NCV is a deviance and its size is
        # data-dependent.
        @test maximum(abs, gr) / max(abs(sc), 1.0) < 1e-4
    end

    # ── 5. structural guards ───────────────────────────────────────────────
    @testset "shape and finiteness" begin
        df = DataFrame(x = _x, z = _z,
                       y = sin.(2π .* _x) .+ 0.5 .* _z .+ 0.3 .* randn(_rng, _n))
        ctx = _ncv_setup(df, [GAM.s(:x; k = 8, bs = :cr), GAM.s(:z; k = 8, bs = :cr)],
                         Normal(), IdentityLink())
        nei = GAM.loo_neighbourhoods(ctx.n)
        S = zeros(ctx.p, ctx.p)
        lsp = [0.5, 1.5]
        r = _fit_at(S, ctx, lsp)
        GAM.total_penalty!(S, ctx.penalty, lsp, ctx.p)
        sc, gr, npd = GAM.ncv_score_grad(ctx.X, ctx.y, r.coefficients,
            r.linear_predictor, ctx.fam, ctx.lnk, ctx.penalty, lsp, S,
            ctx.w, ctx.off, nei)
        @test length(gr) == 2
        @test all(isfinite, gr)
        @test isfinite(sc) && sc > 0
        @test npd == 0

        # `_ncv_apply_sp_component!` must agree with `total_penalty!`'s own
        # block arithmetic — if the two ever drift apart the gradient is
        # differentiating a different penalty from the one being fitted.
        v = randn(StableRNG(9), ctx.p)
        out = zeros(ctx.p)
        tot = zeros(ctx.p)
        for l in 1:2
            GAM._ncv_apply_sp_component!(out, ctx.penalty, lsp, l, v)
            # Same component, obtained by differencing total_penalty! in lambda.
            Sa = zeros(ctx.p, ctx.p); Sb = zeros(ctx.p, ctx.p)
            lp = copy(lsp); lp[l] += 1e-6
            GAM.total_penalty!(Sa, ctx.penalty, lp, ctx.p)
            GAM.total_penalty!(Sb, ctx.penalty, lsp, ctx.p)
            tot .= ((Sa .- Sb) ./ 1e-6) * v
            @test maximum(abs, out .- tot) / max(maximum(abs, tot), 1e-10) < 1e-5
        end
    end
end
