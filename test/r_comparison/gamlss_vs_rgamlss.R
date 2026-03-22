#!/usr/bin/env Rscript
# Generate reference data for GAMLSS (GAM.jl) vs R gamlss package comparison.
#
# Fits Normal (NO) and Gamma (GA) location-scale models using R's gamlss
# package with pb() P-spline smooths and local ML smoothing parameter
# selection.  Saves fitted values and summary statistics as CSV for
# consumption by Julia tests.
#
# Parameterization notes:
#   NO:  mu = mean, sigma = sd.  Links: identity(mu), log(sigma).
#        Same as GAM.jl GaussianLS.
#   GA:  mu = mean, sigma = CV (coefficient of variation).
#        shape = 1/sigma^2, rate = 1/(mu * sigma^2).
#        Links: log(mu), log(sigma).
#        Same as GAM.jl GammaLocationScale.
#
#   pb(x, inter=N, degree=d) creates N + d basis functions.
#   To match in Julia, use s(x, k = N + d, bs = :ps).

if (!requireNamespace("gamlss", quietly = TRUE)) {
    install.packages("gamlss", repos = "https://cloud.r-project.org")
}
library(gamlss)

outdir <- file.path("test", "r_comparison")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)
n <- 500
x <- runif(n, 0, 2 * pi)

# ====================================================================
# Test 1: Normal location-scale (NO)
# ====================================================================
mu_true    <- sin(x)
sigma_true <- exp(0.5 * cos(x))
y_no       <- rnorm(n, mu_true, sigma_true)

dat_no <- data.frame(y = y_no, x = x)
write.csv(dat_no, file.path(outdir, "gamlss_rgamlss_no_data.csv"), row.names = FALSE)

# pb(x, inter=20, degree=3) → 23 basis functions (match Julia k=23)
# pb(x, inter=8,  degree=3) → 11 basis functions (match Julia k=11)
m_no <- gamlss(
    y ~ pb(x, inter = 20, degree = 3, order = 2, method = "ML"),
    sigma.formula = ~ pb(x, inter = 8, degree = 3, order = 2, method = "ML"),
    family = NO, data = dat_no,
    control = gamlss.control(n.cyc = 50, trace = FALSE))

mu_fit_no    <- fitted(m_no, "mu")
sigma_fit_no <- fitted(m_no, "sigma")

write.csv(
    data.frame(mu_fitted = mu_fit_no, sigma_fitted = sigma_fit_no),
    file.path(outdir, "gamlss_rgamlss_no_fitted.csv"), row.names = FALSE)

nll_no <- -as.numeric(logLik(m_no))
write.csv(data.frame(
    deviance  = m_no$G.deviance,
    aic       = m_no$aic,
    nll       = nll_no,
    edf_mu    = m_no$mu.df,
    edf_sigma = m_no$sigma.df
), file.path(outdir, "gamlss_rgamlss_no_summary.csv"), row.names = FALSE)

# Also fit with GAIC method for sp selection
m_no_gaic <- gamlss(
    y ~ pb(x, inter = 20, degree = 3, order = 2, method = "GAIC", k = 2),
    sigma.formula = ~ pb(x, inter = 8, degree = 3, order = 2, method = "GAIC", k = 2),
    family = NO, data = dat_no,
    control = gamlss.control(n.cyc = 50, trace = FALSE))

mu_fit_no_gaic    <- fitted(m_no_gaic, "mu")
sigma_fit_no_gaic <- fitted(m_no_gaic, "sigma")

write.csv(
    data.frame(mu_fitted = mu_fit_no_gaic, sigma_fitted = sigma_fit_no_gaic),
    file.path(outdir, "gamlss_rgamlss_no_gaic_fitted.csv"), row.names = FALSE)

nll_no_gaic <- -as.numeric(logLik(m_no_gaic))
write.csv(data.frame(
    deviance  = m_no_gaic$G.deviance,
    aic       = m_no_gaic$aic,
    nll       = nll_no_gaic,
    edf_mu    = m_no_gaic$mu.df,
    edf_sigma = m_no_gaic$sigma.df
), file.path(outdir, "gamlss_rgamlss_no_gaic_summary.csv"), row.names = FALSE)

# ====================================================================
# Test 2: Gamma location-scale (GA)
# ====================================================================
# DGP: mu = exp(sin(x)), sigma = CV = exp(0.5*cos(x))
# shape = 1/sigma^2, rate = 1/(mu * sigma^2)
mu_gamma_true    <- exp(sin(x))
sigma_gamma_true <- exp(0.5 * cos(x))
shape_true       <- 1 / sigma_gamma_true^2
rate_true        <- 1 / (mu_gamma_true * sigma_gamma_true^2)

y_gamma <- rgamma(n, shape = shape_true, rate = rate_true)

dat_gamma <- data.frame(y = y_gamma, x = x)
write.csv(dat_gamma, file.path(outdir, "gamlss_rgamlss_ga_data.csv"), row.names = FALSE)

m_ga <- gamlss(
    y ~ pb(x, inter = 20, degree = 3, order = 2, method = "ML"),
    sigma.formula = ~ pb(x, inter = 8, degree = 3, order = 2, method = "ML"),
    family = GA, data = dat_gamma,
    control = gamlss.control(n.cyc = 50, trace = FALSE))

mu_fit_ga    <- fitted(m_ga, "mu")
sigma_fit_ga <- fitted(m_ga, "sigma")

write.csv(
    data.frame(mu_fitted = mu_fit_ga, sigma_fitted = sigma_fit_ga),
    file.path(outdir, "gamlss_rgamlss_ga_fitted.csv"), row.names = FALSE)

nll_ga <- -as.numeric(logLik(m_ga))
write.csv(data.frame(
    deviance  = m_ga$G.deviance,
    aic       = m_ga$aic,
    nll       = nll_ga,
    edf_mu    = m_ga$mu.df,
    edf_sigma = m_ga$sigma.df
), file.path(outdir, "gamlss_rgamlss_ga_summary.csv"), row.names = FALSE)

# Also fit with GAIC method
m_ga_gaic <- gamlss(
    y ~ pb(x, inter = 20, degree = 3, order = 2, method = "GAIC", k = 2),
    sigma.formula = ~ pb(x, inter = 8, degree = 3, order = 2, method = "GAIC", k = 2),
    family = GA, data = dat_gamma,
    control = gamlss.control(n.cyc = 50, trace = FALSE))

mu_fit_ga_gaic    <- fitted(m_ga_gaic, "mu")
sigma_fit_ga_gaic <- fitted(m_ga_gaic, "sigma")

write.csv(
    data.frame(mu_fitted = mu_fit_ga_gaic, sigma_fitted = sigma_fit_ga_gaic),
    file.path(outdir, "gamlss_rgamlss_ga_gaic_fitted.csv"), row.names = FALSE)

nll_ga_gaic <- -as.numeric(logLik(m_ga_gaic))
write.csv(data.frame(
    deviance  = m_ga_gaic$G.deviance,
    aic       = m_ga_gaic$aic,
    nll       = nll_ga_gaic,
    edf_mu    = m_ga_gaic$mu.df,
    edf_sigma = m_ga_gaic$sigma.df
), file.path(outdir, "gamlss_rgamlss_ga_gaic_summary.csv"), row.names = FALSE)

# ====================================================================
# Summary
# ====================================================================
cat("All R gamlss reference data generated successfully.\n")
cat(sprintf("  NO  (ML)   deviance: %.4f, AIC: %.4f, NLL: %.4f\n",
            m_no$G.deviance, m_no$aic, nll_no))
cat(sprintf("  NO  (GAIC) deviance: %.4f, AIC: %.4f, NLL: %.4f\n",
            m_no_gaic$G.deviance, m_no_gaic$aic, nll_no_gaic))
cat(sprintf("  GA  (ML)   deviance: %.4f, AIC: %.4f, NLL: %.4f\n",
            m_ga$G.deviance, m_ga$aic, nll_ga))
cat(sprintf("  GA  (GAIC) deviance: %.4f, AIC: %.4f, NLL: %.4f\n",
            m_ga_gaic$G.deviance, m_ga_gaic$aic, nll_ga_gaic))
