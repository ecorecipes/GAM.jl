#!/usr/bin/env Rscript
# Compare SPDE smooth against standard mgcv smooths
# The SPDE smooth from Miller, Glennie & Seaton (2020) requires INLA for
# mesh construction. This script generates reference data using standard
# mgcv smooths for baseline comparison, and attempts the SPDE smooth if
# INLA is available.

library(mgcv)

set.seed(42)
n <- 300
x <- sort(runif(n, 0, 2 * pi))
y <- sin(x) + rnorm(n) * 0.3
dat <- data.frame(y = y, x = x)
write.csv(dat, "test/r_comparison/spde_data.csv", row.names = FALSE)

# Fit thin-plate spline as baseline (both SPDE and TP should recover sin(x))
m_tp <- gam(y ~ s(x, k = 30, bs = "tp"), data = dat, method = "REML")
write.csv(data.frame(fitted = fitted(m_tp)),
          "test/r_comparison/spde_tp_fitted.csv", row.names = FALSE)
write.csv(data.frame(deviance = deviance(m_tp), edf = sum(m_tp$edf)),
          "test/r_comparison/spde_tp_summary.csv", row.names = FALSE)

cat("TP baseline: deviance =", round(deviance(m_tp), 2),
    ", edf =", round(sum(m_tp$edf), 2), "\n")

# Try SPDE smooth if INLA is available
spde_available <- FALSE
tryCatch({
  require(INLA)

  # Define SPDE smooth (from Miller, Glennie & Seaton 2020)
  smooth.construct.spde.smooth.spec <- function(object, data, knots) {
    dim <- length(object$term)
    if (dim != 1) stop("Only 1D SPDE supported here")
    x <- data[[object$term]]
    t <- seq(min(x), max(x), len = object$bs.dim)
    mesh <- inla.mesh.1d(loc = t, degree = 2, boundary = "free")
    object$X <- as.matrix(inla.spde.make.A(mesh, x))
    inlamats <- inla.mesh.fem(mesh)
    object$S <- list()
    object$S[[1]] <- as.matrix(inlamats$c1)
    object$S[[2]] <- 2 * as.matrix(inlamats$g1)
    object$S[[3]] <- as.matrix(inlamats$g2)
    object$L <- matrix(c(2, 2, 2, 4, 2, 0), ncol = 2)
    object$rank <- rep(object$bs.dim, 3)
    object$null.space.dim <- 0
    object$mesh <- mesh
    object$df <- ncol(object$X)
    class(object) <- "spde.smooth"
    return(object)
  }

  Predict.matrix.spde.smooth <- function(object, data) {
    dim <- length(object$term)
    x <- data[[object$term]]
    Xp <- inla.spde.make.A(object$mesh, x)
    return(as.matrix(Xp))
  }

  m_spde <- gam(y ~ s(x, bs = "spde", k = 30), data = dat, method = "REML")
  write.csv(data.frame(fitted = fitted(m_spde)),
            "test/r_comparison/spde_spde_fitted.csv", row.names = FALSE)
  write.csv(data.frame(deviance = deviance(m_spde), edf = sum(m_spde$edf)),
            "test/r_comparison/spde_spde_summary.csv", row.names = FALSE)
  cat("SPDE: deviance =", round(deviance(m_spde), 2),
      ", edf =", round(sum(m_spde$edf), 2), "\n")
  spde_available <- TRUE
}, error = function(e) {
  cat("INLA not available, skipping R SPDE comparison.\n")
  cat("Only comparing against TP baseline.\n")
})

cat("Reference files written.\n")
if (spde_available) {
  cat("SPDE reference available for comparison.\n")
} else {
  cat("Only TP baseline available (INLA not installed).\n")
}
