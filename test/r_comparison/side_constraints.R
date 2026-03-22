#!/usr/bin/env Rscript
# Generate reference data for side constraint (gam.side) comparison tests
# Run: Rscript test/r_comparison/side_constraints.R

library(mgcv)

set.seed(42)
n <- 300
x <- rnorm(n)
z <- rnorm(n)
y <- sin(x) + cos(z) + 0.5 * x * z + rnorm(n) * 0.3
dat <- data.frame(y = y, x = x, z = z)

# Save data
write.csv(dat, "test/r_comparison/side_data.csv", row.names = FALSE)

# Model 1: s(x) + s(x,z) — 1d + 2d overlap
m1 <- gam(y ~ s(x, k = 8) + s(x, z, k = 25), data = dat, method = "REML")
cat("Model 1: s(x) + s(x,z)\n")
cat("  s(x) ncol:", length(m1$smooth[[1]]$first.para:m1$smooth[[1]]$last.para), "\n")
cat("  s(x,z) ncol:", length(m1$smooth[[2]]$first.para:m1$smooth[[2]]$last.para), "\n")

# Model 2: s(x) + s(z) + te(x,z) — tensor with marginals
# Julia te(x,z,k=25) gives 5x5=25-1=24 cols (matching R's k=c(5,5))
m2 <- gam(y ~ s(x, k = 8) + s(z, k = 8) + te(x, z, k = c(5, 5)), data = dat, method = "REML")
cat("\nModel 2: s(x) + s(z) + te(x,z)\n")
cat("  s(x) ncol:", length(m2$smooth[[1]]$first.para:m2$smooth[[1]]$last.para), "\n")
cat("  s(z) ncol:", length(m2$smooth[[2]]$first.para:m2$smooth[[2]]$last.para), "\n")
cat("  te(x,z) ncol:", length(m2$smooth[[3]]$first.para:m2$smooth[[3]]$last.para), "\n")

# Model 3: s(x) + s(z) + ti(x,z) — tensor interaction (no removal expected)
# Julia ti(x,z,k=25) gives (5-1)*(5-1)=16? No, our ti gives 8 cols for k=25
# R ti(x,z,k=c(5,5)) gives (5-1)*(5-1)=16
# Use k=c(4,3) to get (4-1)*(3-1)=6? Try c(4,4) → (4-1)*(4-1)=9
# Actually Julia ti(k=25) → sqrt split of 5 per dim → (5-1)*(5-1)=16... no it gives 8
# Just use k=c(5,5) and verify no-removal behavior separately
m3 <- gam(y ~ s(x, k = 8) + s(z, k = 8) + ti(x, z, k = c(5, 5)), data = dat, method = "REML")
cat("\nModel 3: s(x) + s(z) + ti(x,z)\n")
cat("  s(x) ncol:", length(m3$smooth[[1]]$first.para:m3$smooth[[1]]$last.para), "\n")
cat("  s(z) ncol:", length(m3$smooth[[2]]$first.para:m3$smooth[[2]]$last.para), "\n")
cat("  ti(x,z) ncol:", length(m3$smooth[[3]]$first.para:m3$smooth[[3]]$last.para), "\n")

# Save summary for each model
save_summary <- function(model, prefix) {
  sm_info <- data.frame(
    smooth_label = sapply(model$smooth, function(s) s$label),
    ncol = sapply(model$smooth, function(s) length(s$first.para:s$last.para)),
    edf = model$edf[unlist(lapply(model$smooth, function(s) s$first.para:s$last.para))],
    stringsAsFactors = FALSE
  )
  # Per-smooth EDF and ncol
  smooth_summary <- data.frame(
    label = sapply(model$smooth, function(s) s$label),
    ncol = sapply(model$smooth, function(s) length(s$first.para:s$last.para)),
    edf = sapply(1:length(model$smooth), function(i) sum(model$edf[model$smooth[[i]]$first.para:model$smooth[[i]]$last.para])),
    stringsAsFactors = FALSE
  )
  write.csv(smooth_summary, paste0("test/r_comparison/", prefix, "_smooths.csv"), row.names = FALSE)

  # Model-level summary
  model_summary <- data.frame(
    total_ncoef = length(coef(model)),
    deviance = deviance(model),
    aic = AIC(model),
    reml = model$gcv.ubre,
    scale = model$scale
  )
  write.csv(model_summary, paste0("test/r_comparison/", prefix, "_summary.csv"), row.names = FALSE)

  # Fitted values
  write.csv(data.frame(fitted = fitted(model)), paste0("test/r_comparison/", prefix, "_fitted.csv"), row.names = FALSE)
}

save_summary(m1, "side_m1")
save_summary(m2, "side_m2")
save_summary(m3, "side_m3")

cat("\nReference files written successfully.\n")
