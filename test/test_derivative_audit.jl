# Derivative audit: hand-coded derivative tables vs ForwardDiff vs Symbolics.
#
# Every family here ships hand-transcribed derivative code (ports of mgcv/evgam
# internals) whose errors do not crash anything — they bias smoothing-parameter
# selection or slow convergence silently. This suite re-derives each table from
# the family's scalar deviance/likelihood, symbolically where the expression is
# elementary (exp/log/powers) and with nested ForwardDiff everywhere, and
# demands agreement at analytic precision.
#
# Opt-in (runtests gates on GAM_DERIVATIVE_AUDIT): Symbolics is a heavy
# dependency with a ~1 min load. Central finite differences are deliberately
# NOT used as a reference — for third-order terms their relative error is O(1).
#
# Tolerances: both sides of every comparison are analytic (closed form, AD, or
# a compiled symbolic expression), so agreement is float roundoff, ~1e-13
# relative on O(1) values. RTOL = 1e-8 leaves three orders of headroom for
# ill-conditioned points; ATOL = 1e-10 handles true zeros, where AD noise
# ~1e-16 makes relative error meaningless. Measured worst-case agreements are
# recorded per table below.

using Test
using GAM
using Statistics
using StableRNGs
using SpecialFunctions: loggamma, digamma, trigamma
import GLM

const FDiff = GAM.ForwardDiff

audit_sym_ok = try
    @eval using Symbolics
    true
catch e
    @warn "Symbolics unavailable — symbolic legs of the audit are skipped, " *
          "ForwardDiff legs still run" exception = e
    false
end

# Mixed tolerance: absolute at true zeros, relative elsewhere.
relerr(a, b) = abs(a - b) / max(abs(a), abs(b), 1e-12)
agree(a, b; rtol = 1e-8, atol = 1e-10) = abs(a - b) < atol || relerr(a, b) < rtol

# Nested scalar ForwardDiff derivatives
d1(f, x) = FDiff.derivative(f, x)
d2(f, x) = FDiff.derivative(y -> d1(f, y), x)
d3(f, x) = FDiff.derivative(y -> d2(f, y), x)
d4(f, x) = FDiff.derivative(y -> d3(f, y), x)

@testset "Derivative audit" begin

    # ════════════════════════════════════════════════════════════════════
    # scat_Dd: scaled-t deviance, θ = (log(ν − min_df), log σ)
    #   D(μ, θ₁, θ₂) = w (ν+1) log(1 + ((y−μ)/σ)²/ν)
    # ════════════════════════════════════════════════════════════════════
    @testset "scat_Dd three-way" begin
        min_df = 3.0
        unit_dev(y, mu, w, t1, t2) = w * (exp(t1) + min_df + 1) *
            log1p(((y - mu) / exp(t2))^2 / (exp(t1) + min_df))

        fsym = nothing
        if audit_sym_ok
            @variables ys mus ws th1 th2
            nus = exp(th1) + min_df
            sig = exp(th2)
            Dexpr = ws * (nus + 1) * log(1 + ((ys - mus) / sig)^2 / nus)
            dd(e, v) = Symbolics.derivative(e, v)
            e_Dmu = dd(Dexpr, mus); e_Dmu2 = dd(e_Dmu, mus); e_Dmu3 = dd(e_Dmu2, mus)
            exprs = [e_Dmu, e_Dmu2, e_Dmu3,
                     dd(Dexpr, th1), dd(Dexpr, th2),
                     dd(e_Dmu, th1), dd(e_Dmu, th2),
                     dd(e_Dmu2, th1), dd(e_Dmu2, th2)]
            fsym = eval(Symbolics.build_function(exprs, [ys, mus, ws, th1, th2];
                expression = Val{true})[1])
        end

        cases = [  # (y, mu, w, nu, sigma) incl. near-Gaussian ν and far tail
            (1.30, 0.20, 1.0, 5.7, 1.3),
            (-0.40, 0.90, 0.7, 4.0, 0.4),
            (3.10, 0.00, 1.4, 30.0, 2.2),
            (0.05, 0.00, 1.0, 3.5, 0.05),
            (12.00, 0.00, 1.0, 5.0, 1.0),
        ]
        worst = 0.0
        for (y, mu, w, nu, sg) in cases
            t1 = log(nu - min_df); t2 = log(sg)
            fam = GAM.ScatFamily(; nu = nu, sigma = sg, min_df = min_df)
            t = GAM.scat_Dd(fam, [y], [mu], [w]; level = 1)
            hand = [t[:Dmu][1], t[:Dmu2][1], t[:Dmu3][1],
                    t[:Dth][1, 1], t[:Dth][1, 2],
                    t[:Dmuth][1, 1], t[:Dmuth][1, 2],
                    t[:Dmu2th][1, 1], t[:Dmu2th][1, 2]]
            g(m) = unit_dev(y, m, w, t1, t2)
            ad = [d1(g, mu), d2(g, mu), d3(g, mu),
                  d1(a -> unit_dev(y, mu, w, a, t2), t1),
                  d1(b -> unit_dev(y, mu, w, t1, b), t2),
                  d1(a -> d1(m -> unit_dev(y, m, w, a, t2), mu), t1),
                  d1(b -> d1(m -> unit_dev(y, m, w, t1, b), mu), t2),
                  d1(a -> d2(m -> unit_dev(y, m, w, a, t2), mu), t1),
                  d1(b -> d2(m -> unit_dev(y, m, w, t1, b), mu), t2)]
            for k in 1:9
                @test agree(hand[k], ad[k])
                worst = max(worst, min(relerr(hand[k], ad[k]), abs(hand[k] - ad[k])))
            end
            if fsym !== nothing
                sym = Base.invokelatest(fsym, [y, mu, w, t1, t2])
                for k in 1:9
                    @test agree(ad[k], sym[k])
                end
            end
        end
        @info "scat_Dd audit worst disagreement (hand vs AD)" worst
    end

    # ════════════════════════════════════════════════════════════════════
    # elf_Dd (levels 0–2) and elf_ls: ELF quantile deviance, θ = log σ.
    # With scalar co, σ = e^θ exactly. The level-2 entries are pure e^{−θ}
    # scaling shortcuts (Dth2 = −Dth etc.) — the likeliest place for a sign
    # transcription error, and until now only tested for key existence.
    #   D(μ,θ) = 2w[(1−τ)λ log1p(−τ) + λτ log τ − (1−τ)z + λ log(1+e^{z/λ})]/e^θ
    # ════════════════════════════════════════════════════════════════════
    @testset "elf_Dd and elf_ls three-way" begin
        function elf_dev(y, mu, w, th; tau, lam)
            z = y - mu
            core = (1 - tau) * lam * log1p(-tau) + lam * tau * log(tau) -
                   (1 - tau) * z + lam * log(1 + exp(z / lam))
            return 2 * w * core / exp(th)
        end

        keysyms = [:Dmu, :Dmu2, :Dmu3, :Dmu4, :Dth, :Dmuth, :Dmu2th,
                   :Dth2, :Dmuth2, :Dmu2th2, :Dmu3th]

        fsym = nothing
        if audit_sym_ok
            @variables ye me we the tauv lamv
            zz = ye - me
            core = (1 - tauv) * lamv * log(1 - tauv) + lamv * tauv * log(tauv) -
                   (1 - tauv) * zz + lamv * log(1 + exp(zz / lamv))
            De = 2 * we * core / exp(the)
            dd(e, v) = Symbolics.derivative(e, v)
            eDmu = dd(De, me); eDmu2 = dd(eDmu, me); eDmu3 = dd(eDmu2, me)
            exprs = [eDmu, eDmu2, eDmu3, dd(eDmu3, me),
                     dd(De, the), dd(eDmu, the), dd(eDmu2, the),
                     dd(dd(De, the), the), dd(dd(eDmu, the), the),
                     dd(dd(eDmu2, the), the), dd(eDmu3, the)]
            fsym = eval(Symbolics.build_function(exprs,
                [ye, me, we, the, tauv, lamv]; expression = Val{true})[1])
        end

        worst = 0.0
        for (tau, co, th) in [(0.5, 0.1, 0.0), (0.25, 0.05, 0.4),
                              (0.9, 0.2, -0.7), (0.05, 0.3, 1.1)]
            fam = GAM.ELFFamily(; qu = tau, co = co, theta = th)
            for (y, mu, w) in [(1.7, 1.2, 1.0), (0.4, 0.9, 0.7), (-0.3, 0.1, 1.3)]
                t = GAM.elf_Dd(fam, [y], [mu], [w]; level = 2)
                hand = [t[k][1] for k in keysyms]
                f_mu(m) = elf_dev(y, m, w, th; tau = tau, lam = co)
                f_th(a) = elf_dev(y, mu, w, a; tau = tau, lam = co)
                ad = [d1(f_mu, mu), d2(f_mu, mu), d3(f_mu, mu), d4(f_mu, mu),
                      d1(f_th, th),
                      d1(a -> d1(m -> elf_dev(y, m, w, a; tau, lam = co), mu), th),
                      d1(a -> d2(m -> elf_dev(y, m, w, a; tau, lam = co), mu), th),
                      d2(f_th, th),
                      d2(a -> d1(m -> elf_dev(y, m, w, a; tau, lam = co), mu), th),
                      d2(a -> d2(m -> elf_dev(y, m, w, a; tau, lam = co), mu), th),
                      d1(a -> d3(m -> elf_dev(y, m, w, a; tau, lam = co), mu), th)]
                for k in eachindex(keysyms)
                    @test agree(hand[k], ad[k])
                    worst = max(worst,
                        min(relerr(hand[k], ad[k]), abs(hand[k] - ad[k])))
                end
                if fsym !== nothing
                    sym = Base.invokelatest(fsym, [y, mu, w, th, tau, co])
                    for k in eachindex(keysyms)
                        @test agree(ad[k], sym[k])
                    end
                end
            end

            # elf_ls: log saturated likelihood and its θ-derivatives.
            # Reference from the Beta-function form; ForwardDiff handles
            # digamma/loggamma natively, so AD is the reference here.
            function elf_sat(th_, y, w)
                sg = exp(th_)
                a = co * (1 - tau) / sg
                b = co * tau / sg
                lb = loggamma(a) + loggamma(b) - loggamma(a + b)
                return w * ((1 - tau) * co * log1p(-tau) / sg +
                            co * tau * log(tau) / sg - log(co) - lb)
            end
            y_ = [1.7, 0.4]; w_ = [1.0, 0.7]
            hand_ls = GAM.elf_ls(fam, y_, w_)
            ls_tot(th_) = sum(elf_sat(th_, y_[i], w_[i]) for i in eachindex(y_))
            @test agree(hand_ls.ls, ls_tot(th))
            @test agree(hand_ls.lsth1, d1(ls_tot, th))
            @test agree(hand_ls.lsth2, d2(ls_tot, th))
        end
        @info "elf_Dd audit worst disagreement (hand vs AD)" worst
    end

    # ════════════════════════════════════════════════════════════════════
    # NegBin θ-score and Beta φ-score. The scores live inline in
    # estimate_theta!, so two checks: (algebra) the documented score formula
    # equals the AD derivative of the actual log-likelihood at arbitrary
    # points; (fixed point) the θ̂ the code converges to zeroes the AD score
    # — which fails if the inlined code drifted from the documented formula.
    # ════════════════════════════════════════════════════════════════════
    @testset "NegBin and Beta dispersion scores" begin
        rng = StableRNG(20260829)

        nb_ll(y, mu, th) = loggamma(y + th) - loggamma(th) - loggamma(y + 1) +
            th * log(th) + y * log(mu) - (th + y) * log(mu + th)
        nb_score(y, mu, th) = digamma(y + th) - digamma(th) + log(th) -
            log(mu + th) + (mu - y) / (mu + th)

        for (y, mu, th) in [(3.0, 2.0, 1.5), (0.0, 0.7, 4.0), (11.0, 6.0, 0.6)]
            @test agree(nb_score(y, mu, th), d1(t -> nb_ll(y, mu, t), th))
            # the Newton curvature used by the code (MASS::theta.ml form)
            curv = trigamma(y + th) - trigamma(th) + 1 / th - 2 / (mu + th) +
                   (th + y) / (mu + th)^2
            @test agree(curv, d2(t -> nb_ll(y, mu, t), th))
        end

        mu_v = 1.0 .+ 4.0 .* rand(rng, 400)
        y_v = Float64.([rand(rng, GAM.Distributions.NegativeBinomial(2.5,
            2.5 / (2.5 + m))) for m in mu_v])
        fam = GAM.NegBinFamily(; theta = 1.0, estimate_theta = true)
        GAM.estimate_theta!(fam, y_v, mu_v, ones(400), 1.0)
        th_hat = fam.theta
        score(t) = sum(d1(tt -> nb_ll(y_v[i], mu_v[i], tt), t)
                       for i in eachindex(y_v))
        # At an interior optimum the mean per-observation score is 0 up to the
        # Newton convergence tolerance (1e-6 relative on θ) times the local
        # curvature. |score|/n < 1e-4 is ~100× that; a dropped term in the
        # inlined g1 shifts it by O(1).
        @test abs(score(th_hat)) / length(y_v) < 1e-4
        @test 0.5 < th_hat < 15.0   # sane neighbourhood of the true 2.5

        beta_ll(y, mu, phi) = loggamma(phi) - loggamma(mu * phi) -
            loggamma((1 - mu) * phi) + (mu * phi - 1) * log(y) +
            ((1 - mu) * phi - 1) * log(1 - y)
        beta_score(y, mu, phi) = digamma(phi) - mu * digamma(mu * phi) -
            (1 - mu) * digamma((1 - mu) * phi) + mu * log(y) +
            (1 - mu) * log(1 - y)

        for (y, mu, phi) in [(0.3, 0.4, 5.0), (0.9, 0.6, 2.0), (0.05, 0.2, 11.0)]
            @test agree(beta_score(y, mu, phi), d1(p -> beta_ll(y, mu, p), phi))
            curv = -mu^2 * trigamma(mu * phi) -
                   (1 - mu)^2 * trigamma((1 - mu) * phi) + trigamma(phi)
            @test agree(curv, d2(p -> beta_ll(y, mu, p), phi))
        end

        mu_b = clamp.(0.2 .+ 0.6 .* rand(rng, 400), 0.05, 0.95)
        phi_true = 8.0
        y_b = [clamp(rand(rng, GAM.Distributions.Beta(m * phi_true,
            (1 - m) * phi_true)), 1e-6, 1 - 1e-6) for m in mu_b]
        famb = GAM.BetaFamily(; phi = 1.0, estimate_phi = true)
        GAM.estimate_theta!(famb, y_b, mu_b, ones(400), 1.0)
        phi_hat = famb.phi
        score_b(p) = sum(d1(pp -> beta_ll(y_b[i], mu_b[i], pp), p)
                         for i in eachindex(y_b))
        @test abs(score_b(phi_hat)) / length(y_b) < 1e-4
        @test 3.0 < phi_hat < 20.0
    end

    # ════════════════════════════════════════════════════════════════════
    # _d2mu_deta2: the analytic d²μ/dη² table that feeds Newton dw/dη.
    # Anchored: μ(η) reimplemented generically must equal GLM.linkinv at
    # Float64 points (so AD differentiates the code's actual inverse link),
    # then the table entry must equal the AD second derivative.
    # ════════════════════════════════════════════════════════════════════
    @testset "_d2mu_deta2 per link" begin
        links = [
            (GLM.LogLink(), eta -> exp(eta), [-1.5, 0.0, 0.8, 2.0]),
            (GLM.LogitLink(), eta -> 1 / (1 + exp(-eta)), [-3.0, -0.4, 0.0, 2.5]),
            (GLM.InverseLink(), eta -> 1 / eta, [0.3, 1.0, 2.7]),
            (GLM.SqrtLink(), eta -> eta^2, [0.2, 1.0, 3.0]),
            (GLM.IdentityLink(), eta -> eta, [-1.0, 0.5, 2.0]),
        ]
        for (link, invf, etas) in links, eta in etas
            mu = GLM.linkinv(link, eta)
            @test agree(invf(eta), mu; rtol = 1e-12)  # anchor
            @test agree(GAM._d2mu_deta2(link, mu, eta), d2(invf, eta))
        end
        if audit_sym_ok
            # symbolic confirmation of the four nontrivial closed forms
            @variables e_
            for (expr, table) in [
                (exp(e_), (m, e) -> m),
                (1 / (1 + exp(-e_)), (m, e) -> m * (1 - m) * (1 - 2m)),
                (1 / e_, (m, e) -> 2m^3),
                (e_^2, (m, e) -> 2.0),
            ]
                dd2 = Symbolics.derivative(Symbolics.derivative(expr, e_), e_)
                fpair = eval(Symbolics.build_function([expr, dd2], [e_];
                    expression = Val{true})[1])
                for ev in [0.4, 1.3]
                    mv, sv = Base.invokelatest(fpair, [ev])
                    @test agree(table(mv, ev), sv)
                end
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════
    # GEV/GPD exact-branch derivative tables (evgam C++ ports) and the
    # ξ→0 branch switch at _EV_XI_EPS = 1e-7.
    # GEV out is n×9: [d_μ, d_ψ, d_ξ, d_μμ, d_μψ, d_ψψ, d_μξ, d_ψξ, d_ξξ]
    # GPD out is n×5: [d_ψ, d_ξ, d_ψψ, d_ψξ, d_ξξ]
    # ════════════════════════════════════════════════════════════════════
    @testset "GEV/GPD derivative tables" begin
        fam_gev = GAM.GEVFamily()
        gev_nll(y, eta) = GAM.nll_obs(fam_gev, y, eta)

        function ad_gev(y, eta)
            g = FDiff.gradient(v -> gev_nll(y, v), eta)
            H = FDiff.hessian(v -> gev_nll(y, v), eta)
            return [g[1], g[2], g[3], H[1, 1], H[1, 2], H[2, 2],
                    H[1, 3], H[2, 3], H[3, 3]]
        end

        worst = 0.0
        cases = [  # (y, μ, ψ=logσ, ξ) both signs of moderate ξ + small ξ.
            # ξ = 1e-3 is the smallest exact-branch value audited tightly.
            # Below the _EV_XI_EPS threshold the derivs functions switch to
            # the Gumbel/exponential limit expressions; the audit found the
            # exact expressions cancel catastrophically below ~1e-5, which is
            # why the threshold was raised from 1e-7 to 1e-4. The ξ=1e-6
            # checks below now exercise the limit branch and must be tight.
            (1.4, 0.5, 0.2, 0.25), (0.1, 0.5, -0.5, -0.2),
            (3.7, 0.0, 0.0, 0.12), (0.9, 1.1, 0.3, 1e-3),
        ]
        for (y, mu, psi, xi) in cases
            out = zeros(1, 9)
            GAM.gev_nll_derivs_exact!(out, [y], [mu], [psi], [xi])
            ad = ad_gev(y, [mu, psi, xi])
            for k in 1:9
                @test agree(out[1, k], ad[k]; rtol = 1e-6)
                worst = max(worst, min(relerr(out[1, k], ad[k]),
                    abs(out[1, k] - ad[k])))
            end
        end
        # Gumbel-limit branch (|ξ| < _EV_XI_EPS = 1e-7): validated against
        # 256-bit BigFloat AD of the exact expression at ξ = 5e-8 — the limit
        # branch drops O(ξ) terms, so 1e-6 relative is ample (measured ~1e-7).
        # The generic expression is anchored to nll_obs at moderate ξ first.
        gev_exact(y, eta) = eta[2] +
            (1 / eta[3] + 1) * log(1 + eta[3] * (y - eta[1]) / exp(eta[2])) +
            (1 + eta[3] * (y - eta[1]) / exp(eta[2]))^(-1 / eta[3])
        @test agree(gev_exact(1.4, [0.5, 0.2, 0.25]),
            gev_nll(1.4, [0.5, 0.2, 0.25]); rtol = 1e-12)
        setprecision(BigFloat, 256) do
            eta_b = [big"0.5", big"0.2", big"5e-8"]
            gb = FDiff.gradient(v -> gev_exact(big"1.4", v), eta_b)
            Hb = FDiff.hessian(v -> gev_exact(big"1.4", v), eta_b)
            truth = Float64.([gb[1], gb[2], gb[3], Hb[1, 1], Hb[1, 2],
                Hb[2, 2], Hb[1, 3], Hb[2, 3], Hb[3, 3]])
            out_lim = zeros(1, 9)
            GAM.gev_nll_derivs_exact!(out_lim, [1.4], [0.5], [0.2], [5e-8])
            for k in 1:9
                @test agree(out_lim[1, k], truth[k]; rtol = 1e-6)
            end
            # KNOWN ISSUE (reported, not fixed): the exact branch is used for
            # all |ξ| > 1e-7, but between ~1e-7 and ~1e-5 its ξξ entry loses
            # 3+ significant digits to cancellation (measured 9.2e-4 relative
            # here, up to O(1) at other points) while the limit branch is
            # accurate to ~1e-7. The switch threshold is too small.
            gb6 = FDiff.hessian(v -> gev_exact(big"1.4", v),
                [big"0.5", big"0.2", big"1e-6"])
            out_mid = zeros(1, 9)
            GAM.gev_nll_derivs_exact!(out_mid, [1.4], [0.5], [0.2], [1e-6])
            @test relerr(out_mid[1, 9], Float64(gb6[3, 3])) < 1e-4
        end

        fam_gpd = GAM.GPDFamily()
        gpd_nll(y, eta) = GAM.nll_obs(fam_gpd, y, eta)
        function ad_gpd(y, eta)
            g = FDiff.gradient(v -> gpd_nll(y, v), eta)
            H = FDiff.hessian(v -> gpd_nll(y, v), eta)
            return [g[1], g[2], H[1, 1], H[1, 2], H[2, 2]]
        end
        for (y, psi, xi) in [(1.4, 0.2, 0.25), (0.1, -0.5, -0.15),
                             (3.7, 0.0, 0.12), (0.9, 0.3, 1e-3)]
            out = zeros(1, 5)
            GAM.gpd_nll_derivs_exact!(out, [y], [psi], [xi])
            ad = ad_gpd(y, [psi, xi])
            for k in 1:5
                @test agree(out[1, k], ad[k]; rtol = 1e-6)
                worst = max(worst, min(relerr(out[1, k], ad[k]),
                    abs(out[1, k] - ad[k])))
            end
        end
        # GPD limit branch validated the same way (measured ~1e-7 agreement);
        # same known cancellation issue in the exact branch just above 1e-7.
        gpd_exact(y, eta) = eta[1] +
            (1 / eta[2] + 1) * log(1 + eta[2] * y / exp(eta[1]))
        @test agree(gpd_exact(1.4, [0.2, 0.25]),
            gpd_nll(1.4, [0.2, 0.25]); rtol = 1e-12)
        setprecision(BigFloat, 256) do
            eta_b = [big"0.2", big"5e-8"]
            gb = FDiff.gradient(v -> gpd_exact(big"1.4", v), eta_b)
            Hb = FDiff.hessian(v -> gpd_exact(big"1.4", v), eta_b)
            truth = Float64.([gb[1], gb[2], Hb[1, 1], Hb[1, 2], Hb[2, 2]])
            out_lim = zeros(1, 5)
            GAM.gpd_nll_derivs_exact!(out_lim, [1.4], [0.2], [5e-8])
            for k in 1:5
                @test agree(out_lim[1, k], truth[k]; rtol = 1e-6)
            end
            Hb6 = FDiff.hessian(v -> gpd_exact(big"1.4", v),
                [big"0.2", big"1e-6"])
            out_mid = zeros(1, 5)
            GAM.gpd_nll_derivs_exact!(out_mid, [1.4], [0.2], [1e-6])
            @test relerr(out_mid[1, 5], Float64(Hb6[2, 2])) < 1e-4
        end
        @info "GEV/GPD audit worst disagreement (hand vs AD)" worst

        if audit_sym_ok
            # Symbolic exact-branch cross-check of the GEV gradient
            @variables yv muv psiv xiv
            zc = (yv - muv) / exp(psiv)
            tt = 1 + xiv * zc
            nll_e = psiv + (1 / xiv + 1) * log(tt) + tt^(-1 / xiv)
            grads = [Symbolics.derivative(nll_e, v) for v in (muv, psiv, xiv)]
            fg = eval(Symbolics.build_function(grads, [yv, muv, psiv, xiv];
                expression = Val{true})[1])
            sv = Base.invokelatest(fg, [1.4, 0.5, 0.2, 0.25])
            out = zeros(1, 9)
            GAM.gev_nll_derivs_exact!(out, [1.4], [0.5], [0.2], [0.25])
            for i in 1:3
                @test agree(out[1, i], sv[i]; rtol = 1e-9)
            end
        end
    end

    # ════════════════════════════════════════════════════════════════════
    # tweedie_Dd: unit deviance for 1 < p < 2
    #   d(y,μ) = 2[y^{2−p}/((1−p)(2−p)) − y μ^{1−p}/(1−p) + μ^{2−p}/(2−p)]
    # ════════════════════════════════════════════════════════════════════
    @testset "tweedie_Dd via AD" begin
        for p in (1.2, 1.5, 1.8), (y, mu, w) in [(2.0, 1.5, 1.0), (0.0, 0.8, 0.7),
                                                 (5.5, 3.0, 1.3)]
            dev(m) = 2 * w * (y^(2 - p) / ((1 - p) * (2 - p)) -
                              y * m^(1 - p) / (1 - p) + m^(2 - p) / (2 - p))
            fam = GAM.TweedieFamily(; p = p)
            t = GAM.tweedie_Dd(fam, [y], [mu], [w])
            @test agree(t[:Dmu][1], d1(dev, mu))
            @test agree(t[:Dmu2][1], d2(dev, mu))
            # EDmu2 = E_y[Dmu2] at E y = μ: substituting y = μ gives 2wμ^{−p}
            @test agree(t[:EDmu2][1], 2 * w * mu^(-p))
        end
    end

    # ════════════════════════════════════════════════════════════════════
    # Production central FD for the Newton weight derivative (reml.jl):
    # dw/dη via h = max(1e-7, 1e-7|η|) claims ~1e-10 relative accuracy,
    # feeding a gradient held to ~1e-6. Anchor a generic reimplementation of
    # w(η) to _newton_weight_at_eta at Float64 points, then compare the
    # production FD stencil against the AD derivative of the anchored w.
    # Gamma/LogLink: non-canonical, V = μ², V′ = 2μ, μ = g1 = g2 = e^η.
    # ════════════════════════════════════════════════════════════════════
    @testset "Newton dw/dη production FD vs AD" begin
        fam = GAM.Distributions.Gamma(1.0, 1.0)
        link = GLM.LogLink()
        function w_generic(prior_w, y, eta)
            mu = exp(eta); g1 = mu; g2 = mu
            vm = mu^2; dvm = 2mu
            alpha = 1 + (y - mu) * (dvm / vm - g2 / g1^2)
            return prior_w * g1^2 / vm * alpha
        end
        worst_anchor = 0.0; worst_fd = 0.0
        for (pw, y, eta) in [(1.0, 2.0, 0.4), (0.6, 0.3, -0.9), (1.4, 7.0, 1.7)]
            @test agree(w_generic(pw, y, eta),
                GAM._newton_weight_at_eta(fam, link, pw, y, eta); rtol = 1e-12)
            worst_anchor = max(worst_anchor, relerr(w_generic(pw, y, eta),
                GAM._newton_weight_at_eta(fam, link, pw, y, eta)))
            h = max(1e-7, 1e-7 * abs(eta))   # the production stencil
            fd = (GAM._newton_weight_at_eta(fam, link, pw, y, eta + h) -
                  GAM._newton_weight_at_eta(fam, link, pw, y, eta - h)) / (2h)
            ad = d1(e -> w_generic(pw, y, e), eta)
            # central FD truncation is O(h²)·|w‴| ≈ 1e-14 absolute here; the
            # dominating error is roundoff ~eps/h ≈ 1e-9. 1e-6 is the
            # tolerance the score gradient actually needs.
            @test agree(fd, ad; rtol = 1e-6, atol = 1e-8)
            worst_fd = max(worst_fd, relerr(fd, ad))
        end
        @info "Newton-weight FD audit" worst_anchor worst_fd
    end
end
