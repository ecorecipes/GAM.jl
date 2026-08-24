# Integration tests: compare GAM.jl results against R mgcv via RCall
#
# These tests verify that GAM.jl produces statistically equivalent results
# to R's mgcv package. Exact numerical equality is not expected for all
# basis types (different basis construction algorithms), but fitted values,
# EDF, deviance, and predictions should agree closely.
#
# Requirements: R with mgcv installed, RCall.jl

using Test
using GAM
using RCall
using DataFrames
using Distributions
using LinearAlgebra
using Statistics
using StatsAPI: loglikelihood, dof, aic, bic, nobs, predict, fitted
using StatsBase: coef
using StableRNGs

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

"""
Pull a fitted mgcv model's summary statistics into a NamedTuple.
Assumes `m` is already defined in the R global environment.
"""
function r_gam_summary(model_name::String)
    reval("""
    .m <- $model_name
    .out <- list(
        edf       = sum(.m[["edf"]]),
        deviance  = deviance(.m),
        scale     = .m[["scale"]],
        sp        = .m[["sp"]],
        coef      = coef(.m),
        fitted    = fitted(.m),
        resid_dev = residuals(.m, type="deviance"),
        r2        = summary(.m)[["r.sq"]],
        n         = nobs(.m),
        edf_per   = .m[["edf"]]
    )
    """)
    out = rcopy(reval(".out"))
    return out
end

function simulate_tweedie_rcall(rng, mu::AbstractVector{<:Real}, p::Real, phi::Real)
    1.0 < p < 2.0 || throw(ArgumentError("simulate_tweedie_rcall requires 1 < p < 2"))
    phi > 0 || throw(ArgumentError("simulate_tweedie_rcall requires phi > 0"))

    alpha = (2.0 - p) / (p - 1.0)
    y = Vector{Float64}(undef, length(mu))
    @inbounds for i in eachindex(mu)
        mui = Float64(mu[i])
        lambda = mui^(2.0 - p) / (phi * (2.0 - p))
        gamma_scale = phi * (p - 1.0) * mui^(p - 1.0)
        n_terms = rand(rng, Poisson(lambda))
        total = 0.0
        if n_terms > 0
            dist = Gamma(alpha, gamma_scale)
            for _ in 1:n_terms
                total += rand(rng, dist)
            end
        end
        y[i] = total
    end
    return y
end

# ──────────────────────────────────────────────────────────────────────────────

@testset "R Integration Tests (mgcv)" begin

    # ──────────────────────────────────────────────────────────────────────
    # 1. Gaussian GAM — cubic regression spline (exact basis match)
    # ──────────────────────────────────────────────────────────────────────
    @testset "Gaussian CR — sine curve" begin
        R"""
        set.seed(123)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_cr <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                     method="REML")
        """
        rs = r_gam_summary("r_cr")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

        # CR basis should give near-exact matches
        @test m.edf_total ≈ rs[:edf] atol = 0.05
        @test m.deviance_val ≈ rs[:deviance] atol = 0.01
        @test m.scale ≈ rs[:scale] atol = 0.001
        @test cor(m.fitted_values, rs[:fitted]) > 0.9999
        @test maximum(abs.(m.fitted_values .- rs[:fitted])) < 0.01
    end

    # ──────────────────────────────────────────────────────────────────────
    # 2. Gaussian GAM — TPRS smooth
    # ──────────────────────────────────────────────────────────────────────
    @testset "Gaussian TPRS — sine curve" begin
        R"""
        set.seed(42)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_tp <- gam(y ~ s(x, k=15, bs="tp"), data=data.frame(x=x, y=y),
                     method="REML")
        """
        rs = r_gam_summary("r_tp")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :tp)), df; method = :REML)

        # Since the `max_knots` fix (mgcv's rule: subsample only above 2000 knots,
        # not the old rank-k Nyström fallback above max(3k, 200)) the TPRS basis
        # matches mgcv's to near machine precision on this model.
        @test abs(m.edf_total - rs[:edf]) < 1e-4
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 1e-6
        @test maximum(abs.(m.fitted_values .- rs[:fitted])) < 1e-5
    end

    # ──────────────────────────────────────────────────────────────────────
    # 2b. Gaussian GAM — 2-D TPRS
    # ──────────────────────────────────────────────────────────────────────
    # This is the case the `max_knots` fix actually repairs. At n = 300 the old
    # rule (n > max(3k, 200)) dropped to a rank-k Nyström basis, giving edf error
    # 5.4e-2 and max fitted error 2.5e-2 against mgcv. The 1-D case above was
    # already close before the fix, so its tight tolerances are necessary but not
    # sufficient — this testset is what would catch a revert of the knot rule.
    @testset "Gaussian 2-D TPRS — interacting surface" begin
        R"""
        set.seed(11)
        n <- 300
        u <- runif(n); v <- runif(n)
        y <- sin(2*pi*u) * cos(pi*v) + rnorm(n, sd=0.2)
        r_tp2 <- gam(y ~ s(u, v, k=30, bs="tp"), data=data.frame(u=u, v=v, y=y),
                     method="REML")
        """
        rs2 = r_gam_summary("r_tp2")
        df2 = DataFrame(u = rcopy(R"u"), v = rcopy(R"v"), y = rcopy(R"y"))
        m2 = gam(@formulak(y ~ s(u, v, k = 30, bs = :tp)), df2; method = :REML)

        @test abs(m2.edf_total - rs2[:edf]) < 1e-4
        @test abs(m2.deviance_val - rs2[:deviance]) / rs2[:deviance] < 1e-6
        @test maximum(abs.(m2.fitted_values .- rs2[:fitted])) < 1e-5
    end

    # ──────────────────────────────────────────────────────────────────────
    # 3. Gaussian GAM — P-spline
    # ──────────────────────────────────────────────────────────────────────
    @testset "Gaussian P-spline — sine curve" begin
        R"""
        set.seed(42)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_ps <- gam(y ~ s(x, k=15, bs="ps"), data=data.frame(x=x, y=y),
                     method="REML")
        """
        rs = r_gam_summary("r_ps")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :ps)), df; method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 1.5
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.05
        @test cor(m.fitted_values, rs[:fitted]) > 0.999
    end

    # ──────────────────────────────────────────────────────────────────────
    # 4. Gaussian GAM — multiple smooth terms
    # ──────────────────────────────────────────────────────────────────────
    @testset "Gaussian CR — two smooths" begin
        R"""
        set.seed(77)
        n <- 300
        x1 <- seq(0, 2*pi, length.out=n)
        x2 <- rnorm(n)
        y <- sin(x1) + 0.5*x2^2 + rnorm(n, sd=0.3)
        r_multi <- gam(y ~ s(x1, k=12, bs="cr") + s(x2, k=10, bs="cr"),
                       data=data.frame(x1=x1, x2=x2, y=y), method="REML")
        """
        rs = r_gam_summary("r_multi")
        r_x1 = rcopy(R"x1")
        r_x2 = rcopy(R"x2")
        r_y = rcopy(R"y")

        df = DataFrame(x1 = r_x1, x2 = r_x2, y = r_y)
        m = gam(@formulak(y ~ s(x1, k = 12, bs = :cr) + s(x2, k = 10, bs = :cr)),
            df; method = :REML)

        @test m.edf_total ≈ rs[:edf] atol = 0.1
        @test m.deviance_val ≈ rs[:deviance] atol = 0.1
        @test m.scale ≈ rs[:scale] atol = 0.005
        @test cor(m.fitted_values, rs[:fitted]) > 0.9999
    end

    # ──────────────────────────────────────────────────────────────────────
    # 5. Poisson GAM — CR smooth
    # ──────────────────────────────────────────────────────────────────────
    @testset "Poisson CR — count data" begin
        R"""
        set.seed(99)
        n <- 300
        x <- seq(0, 2*pi, length.out=n)
        mu <- exp(0.5 * sin(x) + 0.5)
        y <- rpois(n, mu)
        r_pois <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                      family=poisson(), method="REML")
        """
        rs = r_gam_summary("r_pois")
        r_x = rcopy(R"x")
        r_y = rcopy(R"as.numeric(y)")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            family = Poisson(), link = LogLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 2.0
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.05
        @test cor(m.fitted_values, rs[:fitted]) > 0.99
    end

    # ──────────────────────────────────────────────────────────────────────
    # 6. Binomial GAM — CR smooth
    # ──────────────────────────────────────────────────────────────────────
    @testset "Binomial CR — binary response" begin
        R"""
        set.seed(55)
        n <- 400
        x <- seq(-3, 3, length.out=n)
        p <- plogis(2*sin(x))
        y <- rbinom(n, 1, p)
        r_binom <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                       family=binomial(), method="REML")
        """
        rs = r_gam_summary("r_binom")
        r_x = rcopy(R"x")
        r_y = rcopy(R"as.numeric(y)")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            family = Binomial(), link = LogitLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 2.0
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.1
        @test cor(m.fitted_values, rs[:fitted]) > 0.99
    end

    # ──────────────────────────────────────────────────────────────────────
    # 7. Quasi-Poisson GAM — overdispersed count data
    # ──────────────────────────────────────────────────────────────────────
    @testset "QuasiPoisson CR — overdispersed count data" begin
        R"""
        set.seed(199)
        n <- 350
        x <- seq(0, 2*pi, length.out=n)
        mu <- exp(0.6 + 0.5*sin(x))
        theta <- 1.7
        y <- rnbinom(n, size=theta, mu=mu)
        r_qpois <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                      family=quasipoisson(), method="REML")
        """
        rs = r_gam_summary("r_qpois")
        r_x = rcopy(R"x")
        r_y = rcopy(R"as.numeric(y)")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            family = QuasiPoissonFamily(), link = LogLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 4.0
        @test abs(m.deviance_val - rs[:deviance]) / max(rs[:deviance], 1.0) < 0.1
        @test abs(m.scale - rs[:scale]) / rs[:scale] < 0.25
        @test cor(m.fitted_values, rs[:fitted]) > 0.94
        @test m.scale > 1.0
    end

    # ──────────────────────────────────────────────────────────────────────
    # 8. Quasi-Binomial GAM — overdispersed grouped proportions
    # ──────────────────────────────────────────────────────────────────────
    @testset "QuasiBinomial CR — grouped proportions" begin
        R"""
        set.seed(299)
        n <- 320
        x <- seq(-3, 3, length.out=n)
        eta <- -0.3 + 1.2*sin(x)
        mu <- plogis(eta)
        w <- rep(20, n)
        phi <- 12
        p_latent <- rbeta(n, mu*phi, (1-mu)*phi)
        success <- rbinom(n, size=w, prob=p_latent)
        y <- success / w
        r_qbin <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y, w=w),
                     weights=w, family=quasibinomial(), method="REML")
        """
        rs = r_gam_summary("r_qbin")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")
        r_w = rcopy(R"as.numeric(w)")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            family = QuasiBinomialFamily(), link = LogitLink(),
            weights = r_w, method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 3.0
        @test abs(m.deviance_val - rs[:deviance]) / max(rs[:deviance], 1.0) < 0.1
        @test abs(m.scale - rs[:scale]) / rs[:scale] < 0.25
        @test cor(m.fitted_values, rs[:fitted]) > 0.995
        @test m.scale > 1.0
    end

    # ──────────────────────────────────────────────────────────────────────
    # 9. Gamma GAM — CR smooth with log link
    # ──────────────────────────────────────────────────────────────────────
    @testset "Gamma CR — positive response" begin
        R"""
        set.seed(88)
        n <- 300
        x <- seq(0.1, 3, length.out=n)
        mu <- exp(0.5 + 0.3*sin(2*x))
        y <- rgamma(n, shape=5, rate=5/mu)
        r_gamma <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                       family=Gamma(link="log"), method="REML")
        """
        rs = r_gam_summary("r_gamma")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            family = Gamma(), link = LogLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 2.0
        @test abs(m.deviance_val - rs[:deviance]) / max(rs[:deviance], 1.0) < 0.1
        @test cor(m.fitted_values, rs[:fitted]) > 0.99
    end

    @testset "Tweedie log density vs mgcv" begin
        R"""
        y_tw <- c(0, 0.35, 1.1, 2.4, 4.2)
        mu_tw <- c(0.4, 0.8, 1.3, 2.0, 3.1)
        p_tw <- 1.4
        phi_tw <- 0.7
        ld_tw <- mgcv:::ldTweedie(y_tw, mu=mu_tw, p=p_tw, phi=phi_tw)[,1]
        """
        y_tw = rcopy(R"y_tw")
        mu_tw = rcopy(R"mu_tw")
        p_tw = Float64(rcopy(R"p_tw"))
        phi_tw = Float64(rcopy(R"phi_tw"))
        ld_tw = rcopy(R"as.numeric(ld_tw)")

        jl_ld = [GAM._tweedie_logdensity(yi, mui, p_tw, phi_tw) for (yi, mui) in zip(y_tw, mu_tw)]

        @test maximum(abs.(jl_ld .- ld_tw)) < 1e-8
    end

    @testset "Tweedie Dd vs mgcv tw()" begin
        y_tw = [0.0, 0.35, 1.1, 2.4, 4.2]
        mu_tw = [0.4, 0.8, 1.3, 2.0, 3.1]
        wt_tw = [1.0, 0.7, 1.2, 0.9, 1.5]
        p_tw = 1.4

        dd_jl = GAM.tweedie_Dd(TweedieFamily(p = p_tw), y_tw, mu_tw, wt_tw; level=0)

        R"""
        y_dd <- $y_tw
        mu_dd <- $mu_tw
        wt_dd <- $wt_tw
        p_dd <- $p_tw
        fam_dd <- tw(theta = p_dd)
        dd_tw <- fam_dd$Dd(y_dd, mu_dd, fam_dd$getTheta(), wt_dd, level = 0)
        """

        @test maximum(abs.(dd_jl[:Dmu] .- rcopy(R"dd_tw$Dmu"))) < 1e-12
        @test maximum(abs.(dd_jl[:Dmu2] .- rcopy(R"dd_tw$Dmu2"))) < 1e-12
        @test maximum(abs.(dd_jl[:EDmu2] .- rcopy(R"dd_tw$EDmu2"))) < 1e-12
    end

    @testset "Tweedie model loglikelihood vs mgcv density" begin
        rng_tw = StableRNG(778)
        n = 240
        x = range(0, 1; length = n) |> collect
        mu_true = exp.(0.3 .+ 0.5 .* cos.(2π .* x))
        true_p = 1.35
        true_phi = 0.7
        y = simulate_tweedie_rcall(rng_tw, mu_true, true_p, true_phi)

        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, k = 12, bs = :cr)), df;
            family = TweedieFamily(p = true_p), method = :REML)

        R"""
        y_ll <- $y
        mu_ll <- $(m.fitted_values)
        wt_ll <- $(m.weights)
        p_ll <- $(m.family.p)
        phi_ll <- $(m.scale)
        ll_tw_model <- sum(mgcv:::ldTweedie(y_ll, mu=mu_ll, p=p_ll, phi=phi_ll)[,1] * wt_ll)
        """
        r_ll = Float64(rcopy(R"ll_tw_model"))

        @test loglikelihood(m) ≈ r_ll atol = 1e-8
        @test aic(m) ≈ -2 * r_ll + 2 * dof(m) atol = 1e-8
    end

    @testset "Tweedie CR — estimated power" begin
        rng_tw = StableRNG(777)
        n = 320
        x = range(0, 1; length=n) |> collect
        mu_true = exp.(0.35 .+ 0.6 .* sin.(2π .* x))
        true_p = 1.45
        true_phi = 0.8
        y = simulate_tweedie_rcall(rng_tw, mu_true, true_p, true_phi)

        R"""
        df_tw <- data.frame(x = $x, y = $y)
        r_tw <- gam(y ~ s(x, k=12, bs="cr"), data=df_tw,
                    family=tw(theta=-1.8), method="REML")
        r_tw_p <- r_tw$family$getTheta(TRUE)
        r_tw_fit <- fitted(r_tw)
        """
        r_tw_p = Float64(rcopy(R"r_tw_p"))
        r_tw_fit = rcopy(R"as.numeric(r_tw_fit)")

        df = DataFrame(x=x, y=y)
        m = gam(@formulak(y ~ s(x, k = 12, bs = :cr)), df;
            family = TweedieFamily(p = 1.8, estimate_p = true), method = :REML)

        @test m.converged
        @test abs(m.family.p - r_tw_p) < 0.15
        @test cor(m.fitted_values, r_tw_fit) > 0.995
    end

    # ──────────────────────────────────────────────────────────────────────
    # 8. Fixed smoothing parameter — bypass outer iteration
    # ──────────────────────────────────────────────────────────────────────
    @testset "Fixed sp — CR sine curve" begin
        R"""
        set.seed(123)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_fix <- gam(y ~ s(x, k=15, bs="cr", sp=1.0), data=data.frame(x=x, y=y))
        """
        rs = r_gam_summary("r_fix")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        # Fit Julia with fixed sp (fx=true means unpenalised; use manual approach)
        df = DataFrame(x = r_x, y = r_y)
        spec = s(:x, bs = :cr, k = 15)
        data_t = Tables.columntable(df)
        sm = smooth_construct(spec, data_t)
        n_j = length(r_y)
        X = hcat(ones(n_j), sm.X)
        sm.first_para = 2
        sm.last_para = 1 + size(sm.X, 2)

        penalty = GAM.setup_penalties([sm], 1)
        # Set sp = log(1.0) = 0.0 to match R's sp=1.0
        log_sp = [0.0]
        p = size(X, 2)
        S_total = GAM.total_penalty(penalty, log_sp, p)
        result = GAM.pirls(X, r_y, S_total, Normal(), IdentityLink())

        @test result.converged
        fitted_j = result.fitted_values
        @test cor(fitted_j, rs[:fitted]) > 0.995
        @test abs(result.deviance - rs[:deviance]) / rs[:deviance] < 0.05
    end

    # ──────────────────────────────────────────────────────────────────────
    # 9. Basis matrix comparison — CR
    # ──────────────────────────────────────────────────────────────────────
    @testset "CR basis matrix vs mgcv" begin
        R"""
        n <- 50
        x <- seq(0, 1, length.out=n)
        sm_r <- smoothCon(s(x, k=10, bs="cr"), data=data.frame(x=x),
                          absorb.cons=TRUE)[[1]]
        """
        r_X = rcopy(reval("sm_r[['X']]"))
        r_S = rcopy(reval("sm_r[['S']][[1]]"))
        r_rank = Int(rcopy(reval("sm_r[['rank']]")))
        r_x = rcopy(R"x")

        spec = s(:x, bs = :cr, k = 10)
        sm = smooth_construct(spec, (x = r_x,))

        # Dimensions must match exactly
        @test size(sm.X) == size(r_X)
        @test size(sm.S[1]) == size(r_S)
        @test sm.rank == r_rank

        # Basis columns span the same space (check via fitted values)
        # Fit OLS with both bases and compare predictions
        beta_r = r_X \ sin.(2π .* r_x)
        beta_j = sm.X \ sin.(2π .* r_x)
        pred_r = r_X * beta_r
        pred_j = sm.X * beta_j
        @test cor(pred_r, pred_j) > 0.9999
        @test maximum(abs.(pred_r .- pred_j)) < 1e-6
    end

    # ──────────────────────────────────────────────────────────────────────
    # 10. Scale estimation comparison
    # ──────────────────────────────────────────────────────────────────────
    @testset "Scale estimation — Gaussian" begin
        R"""
        set.seed(200)
        n <- 500
        x <- seq(0, 2*pi, length.out=n)
        true_sd <- 0.5
        y <- sin(x) + rnorm(n, sd=true_sd)
        r_scale <- gam(y ~ s(x, k=20, bs="cr"), data=data.frame(x=x, y=y),
                       method="REML")
        """
        rs = r_gam_summary("r_scale")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 20, bs = :cr)), df; method = :REML)

        # Scale should be close to true variance 0.25
        @test m.scale ≈ rs[:scale] atol = 0.005
        @test abs(m.scale - 0.25) < 0.05  # close to true σ² = 0.25
    end

    # ──────────────────────────────────────────────────────────────────────
    # 11. Prediction at new data points
    # ──────────────────────────────────────────────────────────────────────
    @testset "Prediction at new data — CR" begin
        R"""
        set.seed(111)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        m_pred <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                      method="REML")
        x_new <- seq(0.5, 5.5, length.out=50)
        pred_r <- predict(m_pred, newdata=data.frame(x=x_new))
        """
        r_pred = rcopy(R"as.numeric(pred_r)")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")
        r_xnew = rcopy(R"x_new")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

        # Build prediction matrix at new points
        sm = m.smooths[1]
        X_new_sm = predict_matrix(sm, (x = r_xnew,))
        X_new = hcat(ones(length(r_xnew)), X_new_sm)
        pred_j = X_new * coef(m)

        @test cor(pred_j, r_pred) > 0.9999
        @test maximum(abs.(pred_j .- r_pred)) < 0.01
    end

    # ──────────────────────────────────────────────────────────────────────
    # 12. Wigglier function — multiple frequencies
    # ──────────────────────────────────────────────────────────────────────
    @testset "Wiggly function — CR" begin
        R"""
        set.seed(333)
        n <- 400
        x <- seq(0, 1, length.out=n)
        f <- sin(2*pi*x) + 0.5*sin(4*pi*x) + 0.2*cos(8*pi*x)
        y <- f + rnorm(n, sd=0.2)
        r_wiggly <- gam(y ~ s(x, k=30, bs="cr"), data=data.frame(x=x, y=y),
                        method="REML")
        """
        rs = r_gam_summary("r_wiggly")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")
        r_f = rcopy(R"f")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 30, bs = :cr)), df; method = :REML)

        # Both should recover the true function well
        rmse_r = sqrt(mean((rs[:fitted] .- r_f) .^ 2))
        rmse_j = sqrt(mean((m.fitted_values .- r_f) .^ 2))

        @test m.edf_total ≈ rs[:edf] atol = 0.5
        @test m.deviance_val ≈ rs[:deviance] atol = 0.5
        @test cor(m.fitted_values, rs[:fitted]) > 0.9999
        # Both recover truth similarly well
        @test abs(rmse_j - rmse_r) < 0.02
    end

    # ──────────────────────────────────────────────────────────────────────
    # 13. Large n — verify scaling
    # ──────────────────────────────────────────────────────────────────────
    @testset "Large n (n=2000) — CR" begin
        R"""
        set.seed(444)
        n <- 2000
        x <- runif(n, 0, 2*pi)
        y <- sin(x) + rnorm(n, sd=0.5)
        r_large <- gam(y ~ s(x, k=20, bs="cr"), data=data.frame(x=x, y=y),
                       method="REML")
        """
        rs = r_gam_summary("r_large")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 20, bs = :cr)), df; method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 0.5
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.01
        @test m.scale ≈ rs[:scale] atol = 0.01
        @test cor(m.fitted_values, rs[:fitted]) > 0.9999
    end

    # ──────────────────────────────────────────────────────────────────────
    # 14. EDF per-smooth comparison — two smooths
    # ──────────────────────────────────────────────────────────────────────
    @testset "Per-smooth EDF — two CR smooths" begin
        R"""
        set.seed(555)
        n <- 300
        x1 <- seq(0, 2*pi, length.out=n)
        x2 <- rnorm(n)
        y <- sin(x1) + 0.3*x2 + rnorm(n, sd=0.3)
        r_edf2 <- gam(y ~ s(x1, k=12, bs="cr") + s(x2, k=8, bs="cr"),
                      data=data.frame(x1=x1, x2=x2, y=y), method="REML")
        """
        rs = r_gam_summary("r_edf2")
        r_x1 = rcopy(R"x1")
        r_x2 = rcopy(R"x2")
        r_y = rcopy(R"y")

        df = DataFrame(x1 = r_x1, x2 = r_x2, y = r_y)
        m = gam(@formulak(y ~ s(x1, k = 12, bs = :cr) + s(x2, k = 8, bs = :cr)),
            df; method = :REML)

        r_edf_per = rs[:edf_per]
        # Per-smooth EDFs should roughly agree
        # (indexing: R edf includes intercept as first element in some versions)
        j_edf_per = m.edf
        @test length(j_edf_per) == 2
        @test abs(j_edf_per[1] - sum(r_edf_per[1:11])) < 1.0  # s(x1)
        @test abs(j_edf_per[2] - sum(r_edf_per[12:end])) < 1.0  # s(x2)
    end

    # ──────────────────────────────────────────────────────────────────────
    # 15. Residual deviance comparison
    # ──────────────────────────────────────────────────────────────────────
    @testset "Deviance residuals — Gaussian CR" begin
        R"""
        set.seed(123)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_resid <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                       method="REML")
        """
        rs = r_gam_summary("r_resid")
        r_resid_dev = rs[:resid_dev]
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

        # Gaussian deviance residuals = y - mu
        j_resid = r_y .- m.fitted_values
        @test cor(j_resid, r_resid_dev) > 0.9999
        @test maximum(abs.(j_resid .- r_resid_dev)) < 0.01
    end

    # ──────────────────────────────────────────────────────────────────────
    # 16. BAM vs R bam — Gaussian CR
    # ──────────────────────────────────────────────────────────────────────
    @testset "BAM vs R bam — Gaussian CR" begin
        R"""
        set.seed(42)
        n <- 5000
        x <- rnorm(n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_bam1 <- bam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                      method="fREML")
        """
        rs = r_gam_summary("r_bam1")
        r_x = rcopy(R"x")
        r_y = rcopy(R"y")

        df = DataFrame(x = r_x, y = r_y)
        m = bam(@formulak(y ~ s(x, k = 15, bs = :cr)), df;
            bam_ctrl = bam_control(chunk_size = 1000))

        # Fitted values should correlate well
        @test cor(m.fitted_values, rs[:fitted]) > 0.999
        # EDF should be in reasonable range
        @test abs(sum(edf(m)) - rs[:edf]) < 2.0
    end

    # ──────────────────────────────────────────────────────────────────────
    # 17. BAM vs R bam — Poisson
    # ──────────────────────────────────────────────────────────────────────
    @testset "BAM vs R bam — Poisson" begin
        R"""
        set.seed(99)
        n <- 3000
        x <- rnorm(n)
        y <- rpois(n, exp(1 + 0.5*sin(x)))
        r_bam_p <- bam(y ~ s(x, k=12, bs="cr"), data=data.frame(x=x, y=as.numeric(y)),
                       family=poisson(), method="fREML")
        """
        rs = r_gam_summary("r_bam_p")
        r_x = rcopy(R"x")
        r_y = rcopy(reval("as.numeric(y)"))

        df = DataFrame(x = r_x, y = r_y)
        m = bam(@formulak(y ~ s(x, k = 12, bs = :cr)), df;
            family = Poisson(), link = LogLink(),
            bam_ctrl = bam_control(chunk_size = 500))

        @test cor(m.fitted_values, rs[:fitted]) > 0.99
        @test abs(sum(edf(m)) - rs[:edf]) < 2.0
    end

    # ──────────────────────────────────────────────────────────────────────────
    # 18. GINLA — Gaussian, compare posterior modes vs R
    # ──────────────────────────────────────────────────────────────────────────
    @testset "18. GINLA Gaussian vs R" begin
        reval("""
        set.seed(42)
        n <- 200
        x <- runif(n)
        y <- sin(2*pi*x) + rnorm(n, 0, 0.3)
        G <- gam(y ~ s(x, k=10, bs="cr"), fit=FALSE)
        gi <- ginla(G, nk=16, nb=100)
        r_beta <- gi[["beta"]]
        r_dens <- gi[["density"]]
        """)
        r_beta = rcopy(reval("r_beta"))   # p × nb
        r_dens = rcopy(reval("r_dens"))   # p × nb

        # Same data in Julia (use exact R data)
        r_x = rcopy(reval("x"))
        r_y = rcopy(reval("y"))
        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), df)
        gi = ginla(m; nk = 16, nb = 100)

        p = size(gi.beta, 1)
        @test p == size(r_beta, 1)

        # Posterior modes should be close
        for k in 1:p
            jl_mode = gi.beta[k, argmax(gi.density[k, :])]
            r_mode = r_beta[k, argmax(r_dens[k, :])]
            # Modes within 0.1 of each other
            @test abs(jl_mode - r_mode) < 0.15
        end

        # Posterior means (density-weighted) should be close
        for k in 1:min(3, p)
            db_jl = gi.beta[k, 2] - gi.beta[k, 1]
            jl_mean = sum(gi.beta[k, :] .* gi.density[k, :]) * db_jl
            db_r = r_beta[k, 2] - r_beta[k, 1]
            r_mean = sum(r_beta[k, :] .* r_dens[k, :]) * db_r
            @test abs(jl_mean - r_mean) < 0.15
        end
    end

    # ──────────────────────────────────────────────────────────────────────────
    # 19. GINLA — Poisson, compare posterior modes vs R
    # ──────────────────────────────────────────────────────────────────────────
    @testset "19. GINLA Poisson vs R" begin
        reval("""
        set.seed(99)
        n <- 300
        x <- runif(n)
        eta <- 1.5 * sin(2*pi*x)
        y <- rpois(n, exp(eta))
        G <- gam(y ~ s(x, k=8, bs="cr"), family=poisson(), fit=FALSE)
        gip <- ginla(G, nk=16, nb=100)
        r_beta_p <- gip[["beta"]]
        r_dens_p <- gip[["density"]]
        r_coef_p <- coef(gam(G=G))
        """)
        r_beta_p = rcopy(reval("r_beta_p"))
        r_dens_p = rcopy(reval("r_dens_p"))
        r_coef = rcopy(reval("r_coef_p"))

        # Same data in Julia
        r_x = rcopy(reval("x"))
        r_y = rcopy(reval("as.numeric(y)"))
        df = DataFrame(x = r_x, y = r_y)
        m = gam(@formulak(y ~ s(x, k = 8, bs = :cr)), df;
            family = Poisson(), link = LogLink())
        gi = ginla(m; nk = 16, nb = 100)

        p = size(gi.beta, 1)
        @test p == size(r_beta_p, 1)

        # For Poisson, posterior modes should agree within 0.2
        for k in 1:p
            jl_mode = gi.beta[k, argmax(gi.density[k, :])]
            r_mode = r_beta_p[k, argmax(r_dens_p[k, :])]
            @test abs(jl_mode - r_mode) < 0.25
        end

        # All densities should be non-negative and finite
        @test all(gi.density .>= 0)
        @test all(isfinite.(gi.density))
        @test all(isfinite.(gi.beta))
    end

    # ──────────────────────────────────────────────────────────────────────
    # Inference parity: sp, coefficients, prediction SEs, AIC, p-values
    # (elementwise, tolerances ~5-10x empirically observed agreement)
    # ──────────────────────────────────────────────────────────────────────
    @testset "Inference parity — Gaussian CR (sp, coef, SE, AIC, p)" begin
        R"""
        set.seed(123)
        n <- 200
        x <- seq(0, 2*pi, length.out=n)
        y <- sin(x) + rnorm(n, sd=0.3)
        r_inf <- gam(y ~ s(x, k=15, bs="cr"), data=data.frame(x=x, y=y),
                     method="REML")
        pr_inf <- predict(r_inf, se.fit=TRUE)
        sm_inf <- summary(r_inf)
        """
        rs = r_gam_summary("r_inf")
        df = DataFrame(x = rcopy(R"x"), y = rcopy(R"y"))
        m = gam(@formulak(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

        # Smoothing parameter (log scale; observed agreement ~1e-5)
        @test m.sp[1] ≈ log(rs[:sp][1]) atol = 0.05
        # Coefficient vector, elementwise (observed max abs diff ~9e-8)
        @test length(coef(m)) == length(rs[:coef])
        @test maximum(abs.(coef(m) .- rs[:coef])) < 1e-5
        # Prediction standard errors vs predict(se.fit=TRUE), elementwise
        # (observed max rel diff ~5e-7)
        se_r = rcopy(Vector{Float64}, R"pr_inf$se.fit")
        _, se_jl = predict(m, df; se = true)
        @test maximum(abs.(se_jl .- se_r) ./ se_r) < 1e-4
        # AIC. Both mgcv conventions are now reproduced directly:
        #     AIC(m)  = family$aic(...) + 2*sum(edf2)  [stats::AIC -> logLik.gam]
        #     m$aic   = family$aic(...) + 2*sum(edf)   [gam.outer]
        # `aic(m)` targets the first (what an mgcv user gets from AIC());
        # `conditional_aic(m)` targets the second.
        @test aic(m) ≈ rcopy(R"AIC(r_inf)") atol = 2e-3
        @test conditional_aic(m) ≈ rcopy(R"r_inf$aic") atol = 1e-3
        @test loglikelihood(m) ≈ rcopy(R"as.numeric(logLik(r_inf))") atol = 1e-3
        # the edf2 correction itself, isolated from any fit difference
        edf2_gap = rcopy(R"2*(sum(r_inf$edf2) - sum(r_inf$edf))")
        @test aic(m) - conditional_aic(m) ≈ edf2_gap atol = 2e-3
        # logLik's df attribute is sum(edf2) + 1 when the scale is estimated
        @test dof(m) ≈ rcopy(R"""attr(logLik(r_inf), "df")""") atol = 1e-3
        # BIC follows the same decomposition (df enters as log(n)*df)
        @test bic(m) - (-2loglikelihood(m) + log(nobs(m)) * conditional_dof(m)) ≈
              log(nobs(m)) * rcopy(R"(sum(r_inf$edf2) - sum(r_inf$edf))") atol = 6e-3
        # edf1 / edf2 / Ref.df totals. Basis-invariant only: elementwise
        # comparison is meaningless because the bases are ordered differently
        # and mgcv rescales each S by its own S.scale.
        @test sum(m.edf1) ≈ rcopy(R"sum(r_inf$edf1)") rtol = 1e-3
        @test sum(edf2(m)) ≈ rcopy(R"sum(r_inf$edf2)") rtol = 1e-3
        @test ref_df(m)[1] ≈ rcopy(R"""sm_inf$s.table[1, "Ref.df"]""") rtol = 1e-3
        # Vc, through the quantity it exists to change: mgcv's
        # predict(unconditional = TRUE) standard errors, and the widening
        # factor they imply relative to the conditional ones.
        R"""pr_unc <- predict(r_inf, se.fit = TRUE, unconditional = TRUE)"""
        se_unc_r = rcopy(Vector{Float64}, R"pr_unc$se.fit")
        _, se_unc = predict(m, df; se = true, unconditional = true)
        @test maximum(abs.(se_unc .- se_unc_r) ./ se_unc_r) < 1e-3
        @test mean(se_unc ./ se_jl) ≈ mean(se_unc_r ./ se_r) rtol = 1e-3
        # Smooth-term test: edf and ref_df match; the test STATISTIC differs
        # (GAM.jl uses a documented simplification of Wood (2013) testStat,
        # observed F 160.4 vs mgcv 132.6), so only edf and the (here
        # decisive) p-value magnitude are asserted.
        st = anova_gam(m).smooth_table
        @test st.edf[1] ≈ rcopy(R"sm_inf$s.table[1,\"edf\"]") atol = 0.05
        @test st.p_value[1] < 1e-10 && rcopy(R"sm_inf$s.table[1,\"p-value\"]") < 1e-10
    end

    @testset "Inference parity — Poisson CR (sp, SE, AIC)" begin
        R"""
        set.seed(77)
        n <- 300
        xp <- runif(n, 0, 2)
        yp <- rpois(n, exp(0.5 + 0.8*sin(2*pi*xp)))
        r_pinf <- gam(yp ~ s(xp, k=12, bs="cr"), family=poisson(),
                      method="REML", data=data.frame(xp=xp, yp=yp))
        pr_pinf <- predict(r_pinf, se.fit=TRUE)
        """
        df = DataFrame(xp = rcopy(R"xp"), yp = Float64.(rcopy(R"yp")))
        m = gam(@formulak(yp ~ s(xp, k = 12, bs = :cr)), df;
            family = Poisson(), method = :REML)
        # observed: sp log-diff 0.012, SE max rel 0.0021
        @test m.sp[1] ≈ rcopy(R"log(r_pinf$sp[1])") atol = 0.1
        se_r = rcopy(Vector{Float64}, R"pr_pinf$se.fit")
        _, se_jl = predict(m, df; se = true)
        @test maximum(abs.(se_jl .- se_r) ./ se_r) < 0.02
        # Same AIC conventions as the Gaussian case above. The residual
        # (~0.004) is fit difference, not convention:
        # EFS and mgcv's outer Newton stop at slightly different sp.
        @test aic(m) ≈ rcopy(R"AIC(r_pinf)") atol = 0.05
        @test conditional_aic(m) ≈ rcopy(R"r_pinf$aic") atol = 0.05
        @test aic(m) - conditional_aic(m) ≈
              rcopy(R"2*(sum(r_pinf$edf2) - sum(r_pinf$edf))") atol = 0.05
        # Poisson's scale is known, so mgcv's REML Hessian carries no log-φ
        # coordinate and Vc is built on the M×M block directly.
        @test sum(edf2(m)) ≈ rcopy(R"sum(r_pinf$edf2)") rtol = 0.03
        @test sum(m.edf1) ≈ rcopy(R"sum(r_pinf$edf1)") rtol = 0.03
    end

    @testset "Cyclic cubic (bs=:cc) vs mgcv" begin
        R"""
        set.seed(5)
        n <- 250
        tt <- runif(n, 0, 24)
        yc <- sin(2*pi*tt/24) + 0.5*cos(4*pi*tt/24) + rnorm(n, sd=0.25)
        r_cc2 <- gam(yc ~ s(tt, k=12, bs="cc"), method="REML",
                     data=data.frame(tt=tt, yc=yc),
                     knots=list(tt=seq(0, 24, length.out=12)))
        """
        df = DataFrame(tt = rcopy(R"tt"), yc = rcopy(R"yc"))
        m = gam(@formulak(yc ~ s(tt, k = 12, bs = :cc)), df; method = :REML)
        f_r = rcopy(Vector{Float64}, R"fitted(r_cc2)")
        # observed: edf diff 0.005, fitted max abs 0.033, cor 0.99996
        @test m.edf_total ≈ rcopy(R"sum(r_cc2$edf)") atol = 0.1
        @test cor(m.fitted_values, f_r) > 0.999
        @test maximum(abs.(m.fitted_values .- f_r)) < 0.1
    end

    @testset "Factor smooth (bs=:fs) vs mgcv" begin
        R"""
        set.seed(9)
        ng <- 5; npg <- 60
        gf <- factor(rep(1:ng, each=npg))
        xf <- runif(ng*npg)
        bf <- rnorm(ng, sd=0.5)
        yf <- sin(2*pi*xf) + bf[as.integer(gf)] +
              0.3*as.integer(gf)/ng*cos(2*pi*xf) + rnorm(ng*npg, sd=0.2)
        r_fs2 <- gam(yf ~ s(xf, gf, bs="fs", k=8), method="REML",
                     data=data.frame(xf=xf, gf=gf, yf=yf))
        """
        df = DataFrame(xf = rcopy(R"xf"), g = string.(rcopy(R"as.integer(gf)")),
            yf = rcopy(R"yf"))
        m = gam(@formulak(yf ~ s(xf, g, bs = :fs, k = 8)), df; method = :REML)
        f_r = rcopy(Vector{Float64}, R"fitted(r_fs2)")
        # post-rewrite fs matches mgcv basis dimension and fit
        # (observed: 41 = 41 coefficients, fitted cor 0.9999, max abs 0.037)
        @test length(coef(m)) == rcopy(R"length(coef(r_fs2))")
        @test cor(m.fitted_values, f_r) > 0.999
        @test maximum(abs.(m.fitted_values .- f_r)) < 0.1
    end


    @testset "Fletcher scale — quasipoisson vs mgcv" begin
        # Both sides default to Fletcher (2012); observed rel diff 0.017.
        rng_fs = StableRNG(77)
        n_fs = 400
        x_fs = rand(rng_fs, n_fs) .* 3
        mu_fs = exp.(0.5 .+ 0.6 .* sin.(x_fs .* 2))
        y_fs = Float64[rand(rng_fs, Distributions.NegativeBinomial(1.4, 1.4 / (1.4 + m))) for m in mu_fs]
        m_fs = gam(GAM.@formulak(y ~ s(x, k = 10, bs = :cr)),
            DataFrame(x = x_fs, y = y_fs); family = QuasiPoissonFamily())
        @rput y_fs x_fs
        RCall.reval("""
        dat_fs <- data.frame(x = x_fs, y = y_fs)
        m_fsr <- mgcv::gam(y ~ s(x, k = 10, bs = "cr"), data = dat_fs,
                           family = quasipoisson(), method = "REML")
        sc_fsr <- summary(m_fsr)[["scale"]]
        """)
        sc_r = rcopy(R"sc_fsr")
        @test m_fs.scale ≈ sc_r rtol = 0.05
    end
    # ── Round-4 parity: shrinkage bases ts/cs vs mgcv ──────────────────────
    # Measured agreement (StableRNG(501), n=300, null smooth on x2):
    #   ts: fitted max-abs 0.015, cor 0.999967, per-smooth edf diff ≤ 0.15
    #   cs: fitted max-abs 0.007, cor 0.999990
    # Tolerances at ~5x measured.
    @testset "Shrinkage bases ts/cs vs mgcv" begin
        rng = StableRNG(501)
        n = 300
        x1 = rand(rng, n); x2 = rand(rng, n)
        y = sin.(2π .* x1) .+ 0.3 .* randn(rng, n)
        df = DataFrame(y=y, x1=x1, x2=x2)
        @rput y x1 x2
        RCall.reval("dr_sh <- data.frame(y=y, x1=x1, x2=x2)")
        for (mj, bstr, tol) in (
            (gam(GAM.@formula(y ~ s(x1, k=10, bs=:ts) + s(x2, k=10, bs=:ts)), df), "ts", 0.08),
            (gam(GAM.@formula(y ~ s(x1, k=10, bs=:cs) + s(x2, k=10, bs=:cs)), df), "cs", 0.05),
        )
            RCall.reval("""
            mr_sh <- gam(y ~ s(x1, k=10, bs="$bstr") + s(x2, k=10, bs="$bstr"),
                         data=dr_sh, method="REML")
            fit_sh <- as.vector(fitted(mr_sh)); edf_sh <- summary(mr_sh)\$edf
            """)
            fit_r = rcopy(Vector{Float64}, R"fit_sh")
            edf_r = rcopy(Vector{Float64}, R"edf_sh")
            @test maximum(abs.(fitted(mj) .- fit_r)) < tol
            @test cor(fitted(mj), fit_r) > 0.999
            @test all(abs.(mj.edf .- edf_r) .< 0.5)
            # both implementations shrink the null smooth on x2
            @test mj.edf[2] < 1.5 && edf_r[2] < 1.5
        end
    end

    # ── Round-4 parity: NB and Beta fits vs mgcv nb()/negbin()/betar() ─────
    # Measured: NB fixed θ — fitted max-rel 0.037, cor 0.9996, smooth-edf
    # diff 0.15; NB free θ — θ̂ 1.875 vs 1.778; Beta — fitted max-abs 0.011,
    # φ̂ 7.98 vs 7.87. (mgcv embeds θ/φ in REML; GAM.jl alternates ML
    # updates — small estimate differences are expected.)
    @testset "NegBin and Beta families vs mgcv" begin
        rng = StableRNG(502)
        n = 300
        x = rand(rng, n) .* 2
        mu = exp.(0.5 .+ sin.(π .* x))
        ynb = Float64.([rand(rng, NegativeBinomial(2.0, 2.0 / (2.0 + m))) for m in mu])
        dfn = DataFrame(y=ynb, x=x)
        @rput ynb x
        RCall.reval("""
        drn <- data.frame(y=ynb, x=x)
        mr_nbf <- gam(y ~ s(x, k=10, bs="cr"), data=drn, family=negbin(theta=2), method="REML")
        fit_nbf <- as.vector(fitted(mr_nbf)); edf_nbf <- sum(summary(mr_nbf)\$edf)
        mr_nbe <- gam(y ~ s(x, k=10, bs="cr"), data=drn, family=nb(), method="REML")
        fit_nbe <- as.vector(fitted(mr_nbe)); th_nbe <- mr_nbe\$family\$getTheta(TRUE)
        """)
        mjf = gam(GAM.@formula(y ~ s(x, k=10, bs=:cr)), dfn,
            NegBinFamily(theta=2.0, estimate_theta=false))
        fit_rf = rcopy(Vector{Float64}, R"fit_nbf")
        edf_rf = rcopy(Float64, R"edf_nbf")
        @test cor(fitted(mjf), fit_rf) > 0.999
        @test maximum(abs.(fitted(mjf) .- fit_rf) ./ fit_rf) < 0.15
        @test abs(mjf.edf[1] - edf_rf) < 0.75    # mgcv summary edf excludes intercept
        mje = gam(GAM.@formula(y ~ s(x, k=10, bs=:cr)), dfn, NegBinFamily())
        fit_re = rcopy(Vector{Float64}, R"fit_nbe")
        th_re = rcopy(Float64, R"th_nbe")
        @test abs(mje.family.theta - th_re) < 0.5
        @test cor(fitted(mje), fit_re) > 0.995

        ybeta = clamp.([rand(rng, Beta(pm * 8, (1 - pm) * 8))
                        for pm in (0.2 .+ 0.6 .* sin.(π .* x ./ 2) .^ 2)], 1e-4, 1 - 1e-4)
        dfb = DataFrame(y=ybeta, x=x)
        @rput ybeta x
        RCall.reval("""
        drb <- data.frame(y=ybeta, x=x)
        mr_bt <- gam(y ~ s(x, k=10, bs="cr"), data=drb, family=betar(), method="REML")
        fit_bt <- as.vector(fitted(mr_bt)); phi_bt <- mr_bt\$family\$getTheta(TRUE)
        """)
        mjb = gam(GAM.@formula(y ~ s(x, k=10, bs=:cr)), dfb, BetaFamily())
        fit_rb = rcopy(Vector{Float64}, R"fit_bt")
        phi_rb = rcopy(Float64, R"phi_bt")
        @test maximum(abs.(fitted(mjb) .- fit_rb)) < 0.05
        @test cor(fitted(mjb), fit_rb) > 0.999
        @test abs(mjb.family.phi - phi_rb) < 1.0
    end

    # ── Round-4 parity: smooth-term CIs vs mgcv terms/iterms SEs ───────────
    # Semantics pairing (measured to machine level):
    #   smooth_estimates default (overall_uncertainty=true)  ≙ type="iterms"
    #     — both include intercept uncertainty (max-rel 3.6e-7)
    #   smooth_estimates(overall_uncertainty=false)          ≙ type="terms"
    #     — centered smooth only (max-rel 4.1e-7)
    @testset "Smooth CI semantics vs mgcv terms/iterms" begin
        rng = StableRNG(505)
        n = 300
        xg = rand(rng, n) .* 2π
        yg = sin.(xg) .+ 0.3 .* randn(rng, n)
        dfg = DataFrame(y=yg, x=xg)
        mg = gam(GAM.@formula(y ~ s(x, k=12, bs=:cr)), dfg)
        se_def = smooth_estimates(mg; n=200)
        se_no = smooth_estimates(mg; n=200, overall_uncertainty=false)
        @rput yg xg
        RCall.reval("""
        drg <- data.frame(y=yg, x=xg)
        mr_ci <- gam(y ~ s(x, k=12, bs="cr"), data=drg, method="REML")
        grid_ci <- data.frame(x = seq(min(xg), max(xg), length=200))
        pt <- predict(mr_ci, newdata=grid_ci, type="terms", se.fit=TRUE)
        est_tm <- as.vector(pt\$fit[,1]); se_tm <- as.vector(pt\$se.fit[,1])
        se_it <- as.vector(predict(mr_ci, newdata=grid_ci, type="iterms", se.fit=TRUE)\$se.fit[,1])
        """)
        est_tm = rcopy(Vector{Float64}, R"est_tm")
        se_tm = rcopy(Vector{Float64}, R"se_tm")
        se_it = rcopy(Vector{Float64}, R"se_it")
        @test maximum(abs.(se_def.estimate .- est_tm)) < 5e-6
        @test maximum(abs.(se_def.se .- se_it) ./ se_it) < 5e-6
        @test maximum(abs.(se_no.se .- se_tm) ./ se_tm) < 5e-6
    end

    # ── Round-4 parity: tensor smooths te/ti elementwise ───────────────────
    # Measured: te fitted max-abs 1.4e-5 with smooth-edf (total−intercept)
    # matching to 0.001; ti (with cr mains) max-abs 1.3e-4, edf to 0.01.
    @testset "Tensor te/ti elementwise vs mgcv" begin
        rng = StableRNG(506)
        n = 300
        xt = rand(rng, n); zt = rand(rng, n)
        yt = sin.(2π .* xt) .* cos.(π .* zt) .+ 0.2 .* randn(rng, n)
        dft = DataFrame(y=yt, x=xt, z=zt)
        @rput yt xt zt
        RCall.reval("""
        drt <- data.frame(y=yt, x=xt, z=zt)
        mr_te <- gam(y ~ te(x, z), data=drt, method="REML")
        fit_te <- as.vector(fitted(mr_te)); edf_te <- sum(summary(mr_te)\$edf)
        mr_ti <- gam(y ~ s(x, k=8, bs="cr") + s(z, k=8, bs="cr") + ti(x, z),
                     data=drt, method="REML")
        fit_ti <- as.vector(fitted(mr_ti)); edf_ti <- sum(summary(mr_ti)\$edf)
        """)
        mte = gam(GAM.@formula(y ~ te(x, z)), dft)
        fit_te = rcopy(Vector{Float64}, R"fit_te")
        edf_te = rcopy(Float64, R"edf_te")
        @test maximum(abs.(fitted(mte) .- fit_te)) < 1e-3
        @test abs((mte.edf_total - 1) - edf_te) < 0.2
        mti = gam(GAM.@formula(y ~ s(x, k=8, bs=:cr) + s(z, k=8, bs=:cr) + ti(x, z)), dft)
        fit_ti = rcopy(Vector{Float64}, R"fit_ti")
        edf_ti = rcopy(Float64, R"edf_ti")
        @test maximum(abs.(fitted(mti) .- fit_ti)) < 2e-3
        @test abs((mti.edf_total - 1) - edf_ti) < 0.2
    end

    # ========================================================================
    # bs="re" follows mgcv's smooth.construct.re.smooth.spec, which builds
    # model.matrix(~ v1:v2:...:vk - 1): factor variables expand to one column
    # per level (several factors give the product of their level counts),
    # while numeric variables contribute no columns of their own and instead
    # multiply the indicators. The rule is symmetric in the variables.
    # ========================================================================
    @testset "bs=:re basis dimension matches mgcv" begin
        n = 200
        gi = repeat(1:5, inner = 40)
        gs = repeat(string.('a':'e'), inner = 40)
        g2 = repeat(["p", "q"], outer = 100)
        xv = collect(range(0, 1; length = n))
        zv = collect(range(1, 2; length = n))
        tbl = (gi = gi, gs = gs, g2 = g2, x = xv, z = zv)

        @rput gi gs g2 xv zv
        RCall.reval("""
        library(mgcv)
        dre <- data.frame(gi=gi, gs=factor(gs), g2=factor(g2), x=xv, z=zv)
        recol <- function(f) ncol(smoothCon(eval(parse(text=f)), data=dre,
                                            absorb.cons=FALSE)[[1]]\$X)
        nc_gi  <- recol('s(gi,bs="re")');    nc_gs <- recol('s(gs,bs="re")')
        nc_x   <- recol('s(x,bs="re")');     nc_xz <- recol('s(x,z,bs="re")')
        nc_gsg2<- recol('s(gs,g2,bs="re")'); nc_gsx<- recol('s(gs,x,bs="re")')
        nc_xgs <- recol('s(x,gs,bs="re")')
        """)

        jl(sp) = size(GAM.smooth_construct(sp, tbl).X, 2)
        # Integer codes are numeric in both: a single multiplier column
        @test jl(s(:gi, bs = :re)) == rcopy(Int, R"nc_gi") == 1
        # Factors expand to one column per level
        @test jl(s(:gs, bs = :re)) == rcopy(Int, R"nc_gs") == 5
        # Continuous covariates: one column, not one per observation
        @test jl(s(:x, bs = :re)) == rcopy(Int, R"nc_x") == 1
        @test jl(s(:x, :z, bs = :re)) == rcopy(Int, R"nc_xz") == 1
        # Two factors: product of level counts
        @test jl(s(:gs, :g2, bs = :re)) == rcopy(Int, R"nc_gsg2") == 10
        # Random slopes, and symmetry in the variable order
        @test jl(s(:gs, :x, bs = :re)) == rcopy(Int, R"nc_gsx") == 5
        @test jl(s(:x, :gs, bs = :re)) == rcopy(Int, R"nc_xgs") == 5
    end
end
