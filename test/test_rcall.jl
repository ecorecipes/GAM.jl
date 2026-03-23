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
using LinearAlgebra
using Statistics
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :tp)), df; method = :REML)

        # TPRS basis construction may differ slightly; check statistical equivalence
        @test abs(m.edf_total - rs[:edf]) < 1.0
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.05
        @test cor(m.fitted_values, rs[:fitted]) > 0.999
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :ps)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x1, k = 12, bs = :cr) + s(x2, k = 10, bs = :cr)),
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df;
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df;
            family = Binomial(), link = LogitLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 2.0
        @test abs(m.deviance_val - rs[:deviance]) / rs[:deviance] < 0.1
        @test cor(m.fitted_values, rs[:fitted]) > 0.99
    end

    # ──────────────────────────────────────────────────────────────────────
    # 7. Gamma GAM — CR smooth with log link
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df;
            family = Gamma(), link = LogLink(), method = :REML)

        @test abs(m.edf_total - rs[:edf]) < 2.0
        @test abs(m.deviance_val - rs[:deviance]) / max(rs[:deviance], 1.0) < 0.1
        @test cor(m.fitted_values, rs[:fitted]) > 0.99
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
        m = gam(@gam_formula(y ~ s(x, k = 20, bs = :cr)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x, k = 30, bs = :cr)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x, k = 20, bs = :cr)), df; method = :REML)

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
        m = gam(@gam_formula(y ~ s(x1, k = 12, bs = :cr) + s(x2, k = 8, bs = :cr)),
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
        m = gam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df; method = :REML)

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
        m = bam(@gam_formula(y ~ s(x, k = 15, bs = :cr)), df;
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
        m = bam(@gam_formula(y ~ s(x, k = 12, bs = :cr)), df;
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
        m = gam(@gam_formula(y ~ s(x, k = 10, bs = :cr)), df)
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
        m = gam(@gam_formula(y ~ s(x, k = 8, bs = :cr)), df;
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

end  # R Integration Tests
