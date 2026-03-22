#!/usr/bin/env Rscript
# Generate reference data for GAMLSS vs mgcv comparison tests.
#
# Fits gaulss (Gaussian location-scale) and gammals (Gamma location-scale)
# models in mgcv, saving fitted values and summary statistics as CSV for
# consumption by Julia tests.
#
# Parameterization notes:
#   gaulss:  models mu (mean) and tau = 1/sigma (precision).
#            fitted[,2] is tau; we save sigma = 1/tau.
#   gammals: models log(mu) and log(phi) where Var = phi * mu^2.
#            fitted[,1] is mu (post-exp'd), fitted[,2] is log(phi).
#            We save sigma = sqrt(phi) = exp(fitted[,2]/2) so that the
#            scale parameter matches GAM.jl's GammaLocationScale(sigma = CV).

library(mgcv)

outdir <- file.path("test", "r_comparison")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)
n <- 500
x <- runif(n, 0, 2 * pi)

# ====================================================================
# Test 1: Gaussian location-scale (gaulss)
# ====================================================================
mu_true  <- sin(x)
sigma_true <- exp(0.5 * cos(x))
y_gaulss <- rnorm(n, mu_true, sigma_true)

dat_gaulss <- data.frame(y = y_gaulss, x = x)
write.csv(dat_gaulss, file.path(outdir, "gamlss_gaulss_data.csv"), row.names = FALSE)

# --- CR splines ---
m_gaulss_cr <- gam(
    list(y ~ s(x, k = 20, bs = "cr"), ~ s(x, k = 10, bs = "cr")),
    family = gaulss(), data = dat_gaulss, method = "REML")

fv_cr <- fitted(m_gaulss_cr)
mu_fit_cr     <- fv_cr[, 1]
sigma_fit_cr  <- 1 / fv_cr[, 2]  # convert precision to sigma

sm_cr   <- summary(m_gaulss_cr)
edf_cr  <- sm_cr$s.table[, "edf"]
sp_cr   <- m_gaulss_cr$sp

write.csv(
    data.frame(mu_fitted = mu_fit_cr, sigma_fitted = sigma_fit_cr),
    file.path(outdir, "gamlss_gaulss_cr_fitted.csv"), row.names = FALSE)

write.csv(data.frame(
    deviance  = deviance(m_gaulss_cr),
    aic       = AIC(m_gaulss_cr),
    edf_mu    = edf_cr[1],
    edf_sigma = edf_cr[2],
    sp_mu     = unname(sp_cr[1]),
    sp_sigma  = unname(sp_cr[2]),
    nll       = -as.numeric(logLik(m_gaulss_cr))
), file.path(outdir, "gamlss_gaulss_cr_summary.csv"), row.names = FALSE)

# --- TP splines ---
m_gaulss_tp <- gam(
    list(y ~ s(x, k = 20, bs = "tp"), ~ s(x, k = 10, bs = "tp")),
    family = gaulss(), data = dat_gaulss, method = "REML")

fv_tp <- fitted(m_gaulss_tp)
sm_tp <- summary(m_gaulss_tp)
edf_tp <- sm_tp$s.table[, "edf"]
sp_tp  <- m_gaulss_tp$sp

write.csv(
    data.frame(mu_fitted = fv_tp[, 1], sigma_fitted = 1 / fv_tp[, 2]),
    file.path(outdir, "gamlss_gaulss_tp_fitted.csv"), row.names = FALSE)

write.csv(data.frame(
    deviance  = deviance(m_gaulss_tp),
    aic       = AIC(m_gaulss_tp),
    edf_mu    = edf_tp[1],
    edf_sigma = edf_tp[2],
    sp_mu     = unname(sp_tp[1]),
    sp_sigma  = unname(sp_tp[2]),
    nll       = -as.numeric(logLik(m_gaulss_tp))
), file.path(outdir, "gamlss_gaulss_tp_summary.csv"), row.names = FALSE)

# ====================================================================
# Test 2: Gamma location-scale (gammals)
# ====================================================================
# DGP:  mu = exp(sin(x)),  phi = exp(cos(x))  (Var = phi * mu^2)
#        shape = 1/phi,     scale = mu * phi
# In GAM.jl: sigma = CV = sqrt(phi) = exp(0.5*cos(x))
mu_gamma_true  <- exp(sin(x))
phi_true       <- exp(cos(x))
shape_true     <- 1 / phi_true
scale_true     <- mu_gamma_true * phi_true  # mean = shape*scale = mu

y_gamma <- rgamma(n, shape = shape_true, scale = scale_true)

dat_gamma <- data.frame(y = y_gamma, x = x)
write.csv(dat_gamma, file.path(outdir, "gamlss_gammals_data.csv"), row.names = FALSE)

# --- CR splines ---
m_gammals_cr <- gam(
    list(y ~ s(x, k = 20, bs = "cr"), ~ s(x, k = 10, bs = "cr")),
    family = gammals(), data = dat_gamma, method = "REML")

fv_g_cr <- fitted(m_gammals_cr)
mu_g_cr    <- fv_g_cr[, 1]
# fitted[,2] = log(phi); sigma = sqrt(phi) = exp(log(phi)/2)
sigma_g_cr <- exp(fv_g_cr[, 2] / 2)

sm_g_cr  <- summary(m_gammals_cr)
edf_g_cr <- sm_g_cr$s.table[, "edf"]
sp_g_cr  <- m_gammals_cr$sp

write.csv(
    data.frame(mu_fitted = mu_g_cr, sigma_fitted = sigma_g_cr),
    file.path(outdir, "gamlss_gammals_cr_fitted.csv"), row.names = FALSE)

write.csv(data.frame(
    deviance  = deviance(m_gammals_cr),
    aic       = AIC(m_gammals_cr),
    edf_mu    = edf_g_cr[1],
    edf_sigma = edf_g_cr[2],
    sp_mu     = unname(sp_g_cr[1]),
    sp_sigma  = unname(sp_g_cr[2]),
    nll       = -as.numeric(logLik(m_gammals_cr))
), file.path(outdir, "gamlss_gammals_cr_summary.csv"), row.names = FALSE)

# --- TP splines ---
m_gammals_tp <- gam(
    list(y ~ s(x, k = 20, bs = "tp"), ~ s(x, k = 10, bs = "tp")),
    family = gammals(), data = dat_gamma, method = "REML")

fv_g_tp <- fitted(m_gammals_tp)
sm_g_tp  <- summary(m_gammals_tp)
edf_g_tp <- sm_g_tp$s.table[, "edf"]
sp_g_tp  <- m_gammals_tp$sp

write.csv(
    data.frame(mu_fitted = fv_g_tp[, 1], sigma_fitted = exp(fv_g_tp[, 2] / 2)),
    file.path(outdir, "gamlss_gammals_tp_fitted.csv"), row.names = FALSE)

write.csv(data.frame(
    deviance  = deviance(m_gammals_tp),
    aic       = AIC(m_gammals_tp),
    edf_mu    = edf_g_tp[1],
    edf_sigma = edf_g_tp[2],
    sp_mu     = unname(sp_g_tp[1]),
    sp_sigma  = unname(sp_g_tp[2]),
    nll       = -as.numeric(logLik(m_gammals_tp))
), file.path(outdir, "gamlss_gammals_tp_summary.csv"), row.names = FALSE)

# ====================================================================
# Also save as combined RDS for convenience
# ====================================================================
results <- list(
    gaulss_cr = list(
        mu_fitted    = mu_fit_cr,
        sigma_fitted = sigma_fit_cr,
        edf          = edf_cr,
        sp           = sp_cr,
        deviance     = deviance(m_gaulss_cr),
        aic          = AIC(m_gaulss_cr),
        nll          = -as.numeric(logLik(m_gaulss_cr)),
        coef         = coef(m_gaulss_cr)
    ),
    gaulss_tp = list(
        mu_fitted    = fv_tp[, 1],
        sigma_fitted = 1 / fv_tp[, 2],
        edf          = edf_tp,
        sp           = sp_tp,
        deviance     = deviance(m_gaulss_tp),
        aic          = AIC(m_gaulss_tp),
        nll          = -as.numeric(logLik(m_gaulss_tp)),
        coef         = coef(m_gaulss_tp)
    ),
    gammals_cr = list(
        mu_fitted    = mu_g_cr,
        sigma_fitted = sigma_g_cr,
        edf          = edf_g_cr,
        sp           = sp_g_cr,
        deviance     = deviance(m_gammals_cr),
        aic          = AIC(m_gammals_cr),
        nll          = -as.numeric(logLik(m_gammals_cr)),
        coef         = coef(m_gammals_cr)
    ),
    gammals_tp = list(
        mu_fitted    = fv_g_tp[, 1],
        sigma_fitted = exp(fv_g_tp[, 2] / 2),
        edf          = edf_g_tp,
        sp           = sp_g_tp,
        deviance     = deviance(m_gammals_tp),
        aic          = AIC(m_gammals_tp),
        nll          = -as.numeric(logLik(m_gammals_tp)),
        coef         = coef(m_gammals_tp)
    )
)
saveRDS(results, file.path(outdir, "gamlss_mgcv_ref.rds"))

cat("All reference data generated successfully.\n")
cat(sprintf("  gaulss CR deviance: %.4f, AIC: %.4f\n", deviance(m_gaulss_cr), AIC(m_gaulss_cr)))
cat(sprintf("  gaulss TP deviance: %.4f, AIC: %.4f\n", deviance(m_gaulss_tp), AIC(m_gaulss_tp)))
cat(sprintf("  gammals CR deviance: %.4f, AIC: %.4f\n", deviance(m_gammals_cr), AIC(m_gammals_cr)))
cat(sprintf("  gammals TP deviance: %.4f, AIC: %.4f\n", deviance(m_gammals_tp), AIC(m_gammals_tp)))
