# Parameter recovery from simulated data — tests that can actually FAIL.
#
# Every testset here simulates from a known truth, fits, and asserts the
# estimate lies within a tolerance justified by the sampling variability at
# that n (stated per test). This supersedes the placeholder "recovery" checks
# in runtests.jl (NegBin θ / Beta φ assert only `> 0 && isfinite`) and covers
# families/fitters that previously had no recovery test at all:
# NegBinLS and InverseGaussianLS were never fitted anywhere in the suite,
# EGPD2/EGPD4 were never fitted, bam() had zero recovery tests, and the gamm
# σ_RE band (0.1–1.5 for truth 0.5) could not fail.
#
# Tolerances follow the house rule: measured, then justified. Each assertion
# carries the observed estimate from the pinned seed and an approximate
# standard error; bounds sit at roughly 3 SE (or a stated multiple) so a real
# regression fails while seed-to-seed wobble does not.

@testset "Parameter recovery from simulated truth" begin

    # ------------------------------------------------------------------
    # NegBin θ — truth 3.0. The old check (runtests.jl) asserts only θ>0.
    # ------------------------------------------------------------------
    @testset "NegBin theta recovery" begin
        rng = StableRNG(20260829)
        n = 2000
        x = rand(rng, n)
        mu = exp.(1.0 .+ 0.8 .* sin.(2π .* x))
        θ_true = 3.0
        # NB(y; μ, θ): r = θ, p = θ/(θ+μ)
        y = Float64.([rand(rng, NegativeBinomial(θ_true, θ_true / (θ_true + m)))
                      for m in mu])
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, k = 10)), df; family = NegBinFamily())
        θ̂ = m.family.theta
        # Fisher-information SE for θ̂ at n=2000, μ~e¹⁺: ≈ 0.15–0.2; observed
        # θ̂ = 3.07 with this seed. 3 SE ≈ 0.6 → assert within 0.7.
        @test abs(θ̂ - θ_true) < 0.7
        # And the smooth is genuinely recovered, not traded off against θ:
        @test cor(fitted(m), mu) > 0.98
    end

    # ------------------------------------------------------------------
    # Beta φ — truth 10.0. Old check asserts only φ>0.
    # ------------------------------------------------------------------
    @testset "Beta phi recovery" begin
        rng = StableRNG(20260830)
        n = 2000
        x = rand(rng, n)
        mu = @. 1 / (1 + exp(-(0.5 * sin(2π * x))))
        φ_true = 10.0
        y = [rand(rng, Beta(m * φ_true, (1 - m) * φ_true)) for m in mu]
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, k = 10)), df; family = BetaFamily())
        φ̂ = m.family.phi
        # SE(φ̂) ≈ φ·√(2/n) ≈ 0.32 at n=2000 (precision-parameter asymptotics);
        # observed φ̂ = 10.15 with this seed. 3 SE ≈ 1.0 → assert within 1.2.
        @test abs(φ̂ - φ_true) < 1.2
        @test cor(fitted(m), mu) > 0.95
    end

    # ------------------------------------------------------------------
    # Tweedie — p and the dispersion φ (m.scale). p was previously pinned
    # only as "beats the starting value"; φ was never checked anywhere.
    # ------------------------------------------------------------------
    @testset "Tweedie p and dispersion recovery" begin
        rng = StableRNG(20260831)
        n = 2000
        x = rand(rng, n)
        mu = exp.(0.6 .+ 0.5 .* sin.(2π .* x))
        p_true, φ_true = 1.5, 1.2
        # Compound Poisson-Gamma simulation (exact for 1<p<2)
        α = (2 - p_true) / (p_true - 1)
        y = map(mu) do m
            λ = m^(2 - p_true) / (φ_true * (2 - p_true))
            gs = φ_true * (p_true - 1) * m^(p_true - 1)
            N = rand(rng, Poisson(λ))
            N == 0 ? 0.0 : sum(rand(rng, Gamma(α, gs)) for _ in 1:N)
        end
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, k = 10)), df;
            family = TweedieFamily(p = 1.7, estimate_p = true))
        # Profile-likelihood p̂ at n=2000: observed 1.507 with this seed
        # (started from 1.7). Series-based SE(p̂) ≈ 0.02–0.03 → 0.1 is >3 SE.
        @test abs(m.family.p - p_true) < 0.1
        # Pearson dispersion: observed 1.20; SE ≈ φ√(2/n)·(1+CV) ≈ 0.05–0.08.
        @test abs(m.scale - φ_true) < 0.25
    end

    # ------------------------------------------------------------------
    # Gaussian residual scale — gam() and bam() (bam had NO recovery tests).
    # Also bam-vs-gam coefficient parity on identical settings, which the
    # suite previously pinned only as cor(fitted)>0.999.
    # ------------------------------------------------------------------
    @testset "Gaussian scale and bam parity" begin
        rng = StableRNG(20260901)
        n = 1500
        x = rand(rng, n)
        σ_true = 0.4
        f_true = sin.(2π .* x)
        y = f_true .+ σ_true .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        mg = gam(@formulak(y ~ s(x, k = 12, bs = :cr)), df)
        mb = bam(@formulak(y ~ s(x, k = 12, bs = :cr)), df)

        # SE(σ̂²) ≈ σ²√(2/(n−edf)) ≈ 0.006 at n=1500. 3 SE ≈ 0.018 → 0.025.
        @test abs(mg.scale - σ_true^2) < 0.025
        @test abs(mb.scale - σ_true^2) < 0.025

        # Same method, same k, same penalty: coefficient-level parity between
        # the QR/normal-equation paths (a class of bug cor(fitted) cannot see).
        @test maximum(abs.(coef(mg) .- coef(mb))) <
              1e-4 * max(1.0, maximum(abs.(coef(mg))))
        # The two accumulation paths stop at slightly different sp on the flat
        # REML ridge; measured Δedf 0.0034 on this seed. 0.01 catches a real
        # divergence while tolerating ridge wobble.
        @test isapprox(mg.edf_total, mb.edf_total; atol = 0.01)
    end

    @testset "bam Poisson recovery" begin
        rng = StableRNG(20260902)
        n = 3000
        x = rand(rng, n)
        η_true = 0.5 .+ 0.8 .* sin.(2π .* x)
        y = Float64.([rand(rng, Poisson(exp(e))) for e in η_true])
        df = DataFrame(x = x, y = y)
        m = bam(@formulak(y ~ s(x, k = 12, bs = :cr)), df; family = Poisson())
        # RMSE of η̂ against truth, bounded by ~3× the average posterior SE of
        # the linear predictor at this n/k (≈0.05); observed RMSE 0.037.
        rmse = sqrt(mean(abs2, log.(max.(fitted(m), 1e-10)) .- η_true))
        @test rmse < 0.15
    end

    # ------------------------------------------------------------------
    # gamlss location–scale: both linear predictors against truth (previous
    # coverage was correlation-only for μ and nothing for σ).
    # ------------------------------------------------------------------
    @testset "GaussianLS both predictors vs truth" begin
        rng = StableRNG(20260903)
        n = 1200
        x = collect(range(0, 1; length = n))
        μ_true = sin.(2π .* x)
        σ_true = @. 0.4 + 0.2 * cos(2π * x)
        y = μ_true .+ σ_true .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        m = gamlss([@formulak(y ~ s(x, k = 10)), @formulak(y ~ s(x, k = 10))],
            df, GaussianLS())
        @test m.converged
        # RMSE targets: E[RMSE(μ̂)] ≈ mean σ·√(edf/n) ≈ 0.04; allow 3×.
        @test sqrt(mean(abs2, m.fitted_eta[1] .- μ_true)) < 0.12
        # σ̂ = exp(η₂); relative RMSE observed 0.06 with this seed; allow 0.2.
        σ̂ = exp.(m.fitted_eta[2])
        @test sqrt(mean(abs2, (σ̂ .- σ_true) ./ σ_true)) < 0.2
    end

    # NegBinLS and InverseGaussianLS were never fitted anywhere in the suite
    # (only their constructors were touched). Fit each and recover both
    # predictors. If either testset errors, that is a bug in the family, not
    # in this test.
    @testset "NegBinLS fit and recovery" begin
        rng = StableRNG(20260904)
        n = 1500
        x = rand(rng, n)
        μ_true = exp.(1.2 .+ 0.6 .* sin.(2π .* x))
        θ_true = 4.0
        y = Float64.([rand(rng, NegativeBinomial(θ_true, θ_true / (θ_true + m)))
                      for m in μ_true])
        df = DataFrame(x = x, y = y)
        m = gamlss([@formulak(y ~ s(x, k = 10)), @formulak(y ~ 1)],
            df, NegBinLS())
        @test m.converged
        μ̂ = exp.(m.fitted_eta[1])
        @test cor(μ̂, μ_true) > 0.95
        # Intercept-only second predictor: exp(η₂) is the family's dispersion
        # parameterisation; assert it is a single well-determined value in a
        # generous truth-bracketing band (the parameterisation of η₂ is
        # family-internal — the failable content is that it is finite,
        # constant, and not degenerate at a bound).
        d̂ = exp.(m.fitted_eta[2])
        @test std(d̂) / mean(d̂) < 0.01
        @test 0.05 < mean(d̂) < 100.0
    end

    @testset "InverseGaussianLS fit and recovery" begin
        rng = StableRNG(20260905)
        n = 1500
        x = rand(rng, n)
        μ_true = exp.(0.5 .+ 0.4 .* sin.(2π .* x))
        λ_true = 6.0
        y = [rand(rng, InverseGaussian(m, λ_true)) for m in μ_true]
        df = DataFrame(x = x, y = y)
        # NB: `NegBinLS()` is exported but the matching `InverseGaussianLS()`
        # alias is not — reported as an exposition gap; the qualified
        # constructor is the public long-form name.
        m = gamlss([@formulak(y ~ s(x, k = 10)), @formulak(y ~ 1)],
            df, GAM.InverseGaussianLocationScale())
        @test m.converged
        μ̂ = exp.(m.fitted_eta[1])
        @test cor(μ̂, μ_true) > 0.95
        d̂ = exp.(m.fitted_eta[2])
        @test std(d̂) / mean(d̂) < 0.01
        @test all(isfinite, d̂)
    end

    # ------------------------------------------------------------------
    # Extreme-value families. GEV/GPD ξ at n=1200 (the existing tests sit at
    # n=200–300 with atol 0.15 ≈ 1.5–2 SE); EGPD1 with failable bands;
    # EGPD3 with EXACT inverse-CDF simulation; EGPD2/EGPD4 fitted for the
    # first time anywhere in the suite.
    # ------------------------------------------------------------------
    @testset "GEV xi recovery (tightened)" begin
        rng = StableRNG(20260906)
        n = 1200
        μt, σt, ξt = 1.0, 2.0, 0.15
        u = rand(rng, n)
        # inverse CDF: y = μ + σ((-log u)^-ξ − 1)/ξ
        y = μt .+ σt .* ((-log.(u)) .^ (-ξt) .- 1) ./ ξt
        df = (; y = y)
        m = evgam([@formulak(y ~ 1), @formulak(y ~ 1), @formulak(y ~ 1)],
            df, GEVFamily())
        @test m.converged
        # Asymptotic SE(ξ̂) ≈ 0.86/√n ≈ 0.025 at n=1200; observed ξ̂ = 0.155.
        # 3 SE ≈ 0.075 → assert within 0.08 (halves the old 0.15 band).
        @test abs(m.coefficients[3] - ξt) < 0.08
        @test abs(m.coefficients[1] - μt) < 0.2
        @test abs(exp(m.coefficients[2]) - σt) < 0.3
    end

    @testset "GPD xi recovery (tightened)" begin
        rng = StableRNG(20260907)
        n = 1200
        σt, ξt = 1.5, 0.2
        u = rand(rng, n)
        y = σt .* ((1 .- u) .^ (-ξt) .- 1) ./ ξt
        df = (; y = y)
        m = evgam([@formulak(y ~ 1), @formulak(y ~ 1)], df, GPDFamily())
        @test m.converged
        # SE(ξ̂) ≈ (1+ξ)/√n ≈ 0.035 at n=1200; observed ξ̂ = 0.185.
        @test abs(m.coefficients[2] - ξt) < 0.1
        @test abs(exp(m.coefficients[1]) - σt) < 0.25
    end

    @testset "EGPD1 recovery (failable bands)" begin
        rng = StableRNG(20260908)
        n = 2000
        σt, ξt, κt = 2.0, 0.1, 1.5
        u = rand(rng, n)
        v = u .^ (1 / κt)                              # G⁻¹ for G(v)=v^κ
        y = σt .* ((1 .- v) .^ (-ξt) .- 1) ./ ξt       # GPD quantile
        df = (; y = y)
        m = evgam([@formulak(y ~ 1), @formulak(y ~ 1), @formulak(y ~ 1)],
            df, EGPD1Family())
        @test m.converged
        # Old bands were ±1.0 on the log scale (≈ e×, could not fail short of
        # divergence). At n=2000 the log-scale SEs are ≈0.08 (σ), 0.10 (κ);
        # observed: logσ̂−logσ = 0.04, ξ̂−ξ = 0.02, logκ̂−logκ = −0.05.
        @test abs(m.coefficients[1] - log(σt)) < 0.3
        @test abs(m.coefficients[2] - ξt) < 0.12
        @test abs(m.coefficients[3] - log(κt)) < 0.3
    end

    @testset "EGPD3 recovery (exact simulation)" begin
        # F(y) = 1 − ((1+δ)S − S^{1+δ})/δ with S the GPD survivor, derived
        # from the implemented density f = h·(1+δ)/δ·(1−S^δ). g(S) =
        # (1+δ)S − S^{1+δ} is monotone increasing on [0,1] with g(1)=δ, so
        # S = g⁻¹(δ(1−u)) by bisection is an EXACT simulator — EGPD3
        # previously only ever had `converged` asserted.
        rng = StableRNG(20260909)
        n = 2000
        σt, ξt, δt = 2.0, 0.1, 1.5
        g(S) = (1 + δt) * S - S^(1 + δt)
        y = map(rand(rng, n)) do u
            target = δt * (1 - u)
            lo, hi = 0.0, 1.0
            for _ in 1:60
                mid = (lo + hi) / 2
                g(mid) < target ? (lo = mid) : (hi = mid)
            end
            S = (lo + hi) / 2
            σt * (S^(-ξt) - 1) / ξt                     # invert the survivor
        end
        df = (; y = y)
        m = evgam([@formulak(y ~ 1), @formulak(y ~ 1), @formulak(y ~ 1)],
            df, EGPD3Family())
        @test m.converged

        # Measured on this seed: the fitter lands at (σ̂,ξ̂,δ̂) = (1.31, 0.145,
        # 0.069) with NLL 0.57 BELOW the truth's — (σ, ξ, δ) trade along a
        # near-flat likelihood ridge, so per-parameter recovery is not an
        # assertable property of this family at n=2000. What IS assertable:
        # (a) optimizer competence — the fit must be at least as good as the
        #     generating parameters (fails if the fitter lands materially
        #     worse than a point it should dominate);
        # (b) distributional recovery — the fitted CDF must match the sample
        #     (probability-integral transform ≈ Uniform), which a broken
        #     fitter or a wrong density fails regardless of the ridge.
        fam3 = EGPD3Family()
        nll_at(c) = sum(GAM.nll_obs(fam3, yi, c) for yi in y)
        @test nll_at(m.coefficients) <= nll_at([log(σt), ξt, log(δt)]) + 2.0

        σ̂, ξ̂, δ̂ = exp(m.coefficients[1]), m.coefficients[2],
                    exp(m.coefficients[3])
        pit = map(y) do yi
            S = (1 + ξ̂ * yi / σ̂)^(-1 / ξ̂)
            1 - ((1 + δ̂) * S - S^(1 + δ̂)) / δ̂
        end
        ks = maximum(abs.(sort(pit) .- (1:n) ./ n))
        # KS 95% band ≈ 1.36/√n ≈ 0.030 at n=2000; observed 0.0084. Estimated
        # parameters tighten the PIT, so 0.04 is a conservative failure bar.
        @test ks < 0.04
        # ξ̂ is the one identified parameter on this ridge (observed 0.145).
        @test abs(ξ̂ - ξt) < 0.15
    end

    @testset "EGPD2 and EGPD4 fit at all" begin
        # Never previously fitted anywhere in the suite. Data from EGPD1
        # truth; the richer families must converge to a finite fit whose
        # scale/shape sit near the generating values (their extra transform
        # parameters absorb the rest — asserted finite, not pinned).
        rng = StableRNG(20260910)
        n = 1500
        σt, ξt, κt = 2.0, 0.1, 1.5
        u = rand(rng, n)
        v = u .^ (1 / κt)
        y = σt .* ((1 .- v) .^ (-ξt) .- 1) ./ ξt
        df = (; y = y)

        m2 = evgam([@formulak(y ~ 1) for _ in 1:5], df, EGPD2Family())
        @test m2.converged
        @test all(isfinite, m2.coefficients)
        @test abs(m2.coefficients[2] - ξt) < 0.3

        m4 = evgam([@formulak(y ~ 1) for _ in 1:4], df, EGPD4Family())
        @test m4.converged
        @test all(isfinite, m4.coefficients)
        @test abs(m4.coefficients[2] - ξt) < 0.3
    end

    # ------------------------------------------------------------------
    # gamm: σ_RE within ~3 SE of truth (the existing band 0.1–1.5 for truth
    # 0.5 could not fail), plus the model-assembly identity η = Xβ.
    # ------------------------------------------------------------------
    @testset "gamm sigma_RE recovery and assembly" begin
        rng = StableRNG(20260911)
        n_groups, n_per = 40, 25
        n = n_groups * n_per
        group = repeat(1:n_groups, inner = n_per)
        x = rand(rng, n)
        σ_re, σ_eps = 0.5, 0.3
        b_true = σ_re .* randn(rng, n_groups)
        y = sin.(2π .* x) .+ b_true[group] .+ σ_eps .* randn(rng, n)
        df = DataFrame(x = x, y = y, group = string.(group))

        m = gamm(@formula(y ~ s(x) + (1 | group)), df)
        @test m.gam_model.converged

        vc = VarCorr(m)
        # SE(σ̂_RE) ≈ σ_RE/√(2(g−1)) ≈ 0.057 at g=40; observed σ̂ = 0.52.
        @test abs(vc[1].std - σ_re) < 0.17
        # Residual σ likewise (SE ≈ 0.007): observed 0.298.
        @test abs(vc[end].std - σ_eps) < 0.05

        # Assembly identity: the linear predictor IS the model matrix times
        # the full coefficient vector (fixed + smooth + RE-as-smooth blocks) —
        # the Xβ + Zb identity in this parameterisation.
        gm = m.gam_model
        @test maximum(abs.(GAM.model_matrix(gm) * coef(gm) .-
                           gm.linear_predictor)) < 1e-8
    end
end
