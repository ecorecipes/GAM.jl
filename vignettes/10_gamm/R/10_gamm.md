# GAMM: R Comparison
Simon Frost

- [Overview](#overview)
- [Setup](#setup)
- [Example 1: Gaussian GAMM with Random
  Intercepts](#example-1-gaussian-gamm-with-random-intercepts)
  - [Fitting with `gamm()`](#fitting-with-gamm)
  - [Random effects](#random-effects)
  - [Variance components](#variance-components)
  - [Comparison with true values](#comparison-with-true-values)
  - [Equivalence with
    `s(subject, bs="re")`](#equivalence-with-ssubject-bsre)
  - [Plot: Data by Subject with Population
    Smooth](#plot-data-by-subject-with-population-smooth)
  - [Plot: Smooth Effect](#plot-smooth-effect)
- [Example 2: Poisson GAMM for Count
  Data](#example-2-poisson-gamm-for-count-data)
  - [Fitting](#fitting)
  - [Random effects](#random-effects-1)
  - [Plot: Data by Site with Population
    Smooth](#plot-data-by-site-with-population-smooth)
  - [Plot: Smooth Effect](#plot-smooth-effect-1)
- [Example 3: Factor-Smooth Interactions
  (`bs="fs"`)](#example-3-factor-smooth-interactions-bsfs)
  - [Fitting](#fitting-1)
  - [What a random intercept cannot
    do](#what-a-random-intercept-cannot-do)
  - [Plot](#plot)
  - [Agreement with GAM.jl, and the one
    caveat](#agreement-with-gamjl-and-the-one-caveat)
- [Syntax Comparison](#syntax-comparison)
  - [Key differences](#key-differences)

## Overview

This vignette shows the R equivalent of the GAMM models fit in
`10_gamm.qmd`, using `mgcv::gamm()` which internally delegates to
`nlme::lme()` for Gaussian models and penalized quasi-likelihood (PQL)
for non-Gaussian models.

## Setup

``` r
library(mgcv)
```

    Loading required package: nlme

    This is mgcv 1.9-4. For overview type '?mgcv'.

``` r
library(nlme)
```

## Example 1: Gaussian GAMM with Random Intercepts

``` r
dat <- read.csv("../data_gaussian_gamm.csv")
cat(sprintf("n = %d, subjects = %d\n", nrow(dat), length(unique(dat$subject))))
```

    n = 480, subjects = 12

``` r
cat(sprintf("y range: [%.2f, %.2f]\n", min(dat$y), max(dat$y)))
```

    y range: [-2.47, 3.06]

### Fitting with `gamm()`

In mgcv, random effects are specified via the `random=` argument, not in
the formula:

``` r
dat$subject <- factor(dat$subject)
m <- gamm(y ~ s(x, k = 15), random = list(subject = ~1), data = dat)
summary(m$gam)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    y ~ s(x, k = 15)

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept)   0.1847     0.1297   1.424    0.155

    Approximate significance of smooth terms:
           edf Ref.df     F p-value    
    s(x) 11.67  11.67 328.8  <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    R-sq.(adj) =  0.769   
      Scale est. = 0.14724   n = 480

### Random effects

``` r
re <- ranef(m$lme)$subject
cat("Random intercept estimates:\n")
```

    Random intercept estimates:

``` r
for (lev in rownames(re)) {
  cat(sprintf("  Subject %2s: b = %+.3f\n", lev, re[lev, 1]))
}
```

      Subject 1/1: b = +0.364
      Subject 1/2: b = -0.503
      Subject 1/3: b = -0.608
      Subject 1/4: b = +0.404
      Subject 1/5: b = +0.418
      Subject 1/6: b = -0.362
      Subject 1/7: b = +0.338
      Subject 1/8: b = -0.177
      Subject 1/9: b = -0.199
      Subject 1/10: b = -0.441
      Subject 1/11: b = -0.081
      Subject 1/12: b = +0.847

### Variance components

``` r
vc <- VarCorr(m$lme)
print(vc)
```

                Variance        StdDev   
    g =         pdIdnot(Xr - 1)          
    Xr1         52.9494478      7.2766371
    Xr2         52.9494478      7.2766371
    Xr3         52.9494478      7.2766371
    Xr4         52.9494478      7.2766371
    Xr5         52.9494478      7.2766371
    Xr6         52.9494478      7.2766371
    Xr7         52.9494478      7.2766371
    Xr8         52.9494478      7.2766371
    Xr9         52.9494478      7.2766371
    Xr10        52.9494478      7.2766371
    Xr11        52.9494478      7.2766371
    Xr12        52.9494478      7.2766371
    Xr13        52.9494478      7.2766371
    subject =   pdLogChol(1)             
    (Intercept)  0.1976426      0.4445702
    Residual     0.1472440      0.3837238

``` r
sigma_re <- as.numeric(vc[rownames(vc) == "(Intercept)", "StdDev"])
sigma_res <- as.numeric(vc[rownames(vc) == "Residual", "StdDev"])
cat(sprintf("RE σ: %.4f (true: 0.6)\n", sigma_re))
```

    RE σ: 0.4446 (true: 0.6)

``` r
cat(sprintf("Residual σ: %.4f (true: 0.4)\n", sigma_res))
```

    Residual σ: 0.3837 (true: 0.4)

### Comparison with true values

``` r
true_re <- tapply(dat$re_true, dat$subject, function(x) x[1])
cat(sprintf("Correlation of estimated vs true RE: %.4f\n",
            cor(re[, 1], true_re)))
```

    Correlation of estimated vs true RE: 0.9874

### Equivalence with `s(subject, bs="re")`

In mgcv, `s(subject, bs="re")` is the smooth-term equivalent of a random
intercept:

``` r
m_gam <- gam(y ~ s(x, k = 15) + s(subject, bs = "re"),
             data = dat, method = "REML")
cat(sprintf("Fitted values correlation (gamm vs gam+re): %.6f\n",
            cor(fitted(m$lme), fitted(m_gam))))
```

    Fitted values correlation (gamm vs gam+re): 1.000000

``` r
cat(sprintf("Scale (gamm): %.6f\n", m$gam$scale))
```

    Scale (gamm): 1.000000

``` r
cat(sprintf("Scale (gam):  %.6f\n", m_gam$scale))
```

    Scale (gam):  0.147477

### Plot: Data by Subject with Population Smooth

``` r
par(mfrow = c(1, 2))

# Scatter colored by subject with population smooth
subject_ids <- sort(unique(dat$subject))
n_subj <- length(subject_ids)
cols <- rainbow(n_subj, alpha = 0.5)
plot(dat$x, dat$y, col = cols[match(dat$subject, subject_ids)],
     pch = 16, cex = 0.6, xlab = "x", ylab = "y",
     main = "Gaussian GAMM: data by subject")
x_grid <- seq(min(dat$x), max(dat$x), length.out = 200)
pop_pred <- predict(m$gam, newdata = data.frame(x = x_grid))
lines(x_grid, pop_pred, col = "black", lwd = 3)

# Grouped bar: estimated vs true random intercepts
re_mat <- rbind(Estimated = re[, 1], True = true_re[rownames(re)])
barplot(re_mat, beside = TRUE, names.arg = rownames(re),
        col = c("steelblue", "darkorange"), las = 1,
        main = "Random intercepts: estimated vs true",
        xlab = "Subject", ylab = "Random intercept")
legend("topright", c("Estimated", "True"),
       fill = c("steelblue", "darkorange"), cex = 0.8)
```

![](10_gamm_files/figure-commonmark/unnamed-chunk-9-1.png)

### Plot: Smooth Effect

``` r
par(mfrow = c(1, 2))

# Smooth effect
plot(m$gam, shade = TRUE, main = "Population smooth f(x)",
     xlab = "x", ylab = "f(x)")

# Random effects
re_est <- re[, 1]
names(re_est) <- rownames(re)
barplot(sort(re_est), horiz = TRUE, las = 1,
        main = "Random intercepts", xlab = "Estimated b_j",
        col = "steelblue")
abline(v = 0, lty = 2)
```

![](10_gamm_files/figure-commonmark/unnamed-chunk-10-1.png)

## Example 2: Poisson GAMM for Count Data

``` r
dat2 <- read.csv("../data_poisson_gamm.csv")
dat2$site <- factor(dat2$site)
cat(sprintf("n = %d, sites = %d\n", nrow(dat2), length(unique(dat2$site))))
```

    n = 480, sites = 8

### Fitting

For non-Gaussian families, `gamm()` uses PQL (penalized
quasi-likelihood):

``` r
m2 <- gamm(y ~ s(x, k = 15), random = list(site = ~1),
           family = poisson(), data = dat2)
```


     Maximum number of PQL iterations:  20 

    iteration 1

    iteration 2

    iteration 3

    iteration 4

``` r
summary(m2$gam)
```


    Family: poisson 
    Link function: log 

    Formula:
    y ~ s(x, k = 15)

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)   0.9809     0.1259   7.792 4.22e-14 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Approximate significance of smooth terms:
           edf Ref.df     F p-value    
    s(x) 6.662  6.662 54.34  <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    R-sq.(adj) =  0.349   
      Scale est. = 1         n = 480

### Random effects

``` r
re2 <- ranef(m2$lme)$site
true_re2 <- tapply(dat2$re_true, dat2$site, function(x) x[1])
cat(sprintf("RE correlation with truth: %.4f\n",
            cor(re2[, 1], true_re2)))
```

    RE correlation with truth: 0.9743

### Plot: Data by Site with Population Smooth

``` r
par(mfrow = c(1, 2))

# Scatter colored by site with population smooth
site_ids <- sort(unique(dat2$site))
n_sites <- length(site_ids)
cols2 <- rainbow(n_sites, alpha = 0.5)
plot(dat2$x, dat2$y, col = cols2[match(dat2$site, site_ids)],
     pch = 16, cex = 0.6, xlab = "x", ylab = "y (count)",
     main = "Poisson GAMM: data by site")
x_grid2 <- seq(min(dat2$x), max(dat2$x), length.out = 200)
pop_pred2 <- predict(m2$gam, newdata = data.frame(x = x_grid2), type = "response")
lines(x_grid2, pop_pred2, col = "black", lwd = 3)

# Grouped bar: estimated vs true random effects
re2_mat <- rbind(Estimated = re2[, 1], True = true_re2[rownames(re2)])
barplot(re2_mat, beside = TRUE, names.arg = rownames(re2),
        col = c("steelblue", "darkorange"), las = 1,
        main = "Site random effects: estimated vs true",
        xlab = "Site", ylab = "Random intercept (log-scale)")
legend("topright", c("Estimated", "True"),
       fill = c("steelblue", "darkorange"), cex = 0.8)
```

![](10_gamm_files/figure-commonmark/unnamed-chunk-14-1.png)

### Plot: Smooth Effect

``` r
par(mfrow = c(1, 2))
plot(m2$gam, shade = TRUE, main = "Poisson GAMM: f(x)",
     xlab = "x", ylab = "f(x)")

re2_est <- re2[, 1]
names(re2_est) <- rownames(re2)
barplot(sort(re2_est), horiz = TRUE, las = 1,
        main = "Site random intercepts", xlab = "Estimated b_j",
        col = "darkorange")
abline(v = 0, lty = 2)
```

![](10_gamm_files/figure-commonmark/unnamed-chunk-15-1.png)

## Example 3: Factor-Smooth Interactions (`bs="fs"`)

mgcv’s `s(x, fac, bs="fs")` fits the same factor-smooth interaction as
GAM.jl’s `s(x, group, bs=:fs)`: one smooth curve per level, with a small
fixed set of smoothing parameters shared across levels rather than one
per level.

``` r
traj <- read.csv("../data_fs_trajectories.csv")
traj$subject <- factor(as.integer(traj$subject))
cat(sprintf("n = %d, subjects = %d\n", nrow(traj), nlevels(traj$subject)))
```

    n = 375, subjects = 15

### Fitting

``` r
m_fs <- gam(y ~ s(t, subject, bs = "fs", k = 8), data = traj, method = "REML")
cat(sprintf("edf = %.2f\n", sum(m_fs$edf)))
```

    edf = 89.64

``` r
cat(sprintf("deviance explained = %.4f\n", summary(m_fs)$dev.expl))
```

    deviance explained = 0.9546

``` r
cat(sprintf("smoothing parameters: %d\n", length(m_fs$sp)))
```

    smoothing parameters: 3

``` r
cat(sprintf("sp = %s\n", paste(sprintf("%.5g", m_fs$sp), collapse = ", ")))
```

    sp = 0.010982, 0.0092252, 0.0044993

Three smoothing parameters for fifteen subjects, exactly as in GAM.jl:
one for the per-subject intercepts, one for the slopes, one for the
wiggliness.

### What a random intercept cannot do

``` r
truth <- traj$f_pop + traj$dev_true
m_ri <- gam(y ~ s(t, k = 10) + s(subject, bs = "re"), data = traj, method = "REML")
cat(sprintf("fs        : edf %6.2f   RMSE vs true curves %.4f\n",
            sum(m_fs$edf), sqrt(mean((fitted(m_fs) - truth)^2))))
```

    fs        : edf  89.64   RMSE vs true curves 0.1675

``` r
cat(sprintf("pop + re  : edf %6.2f   RMSE vs true curves %.4f\n",
            sum(m_ri$edf), sqrt(mean((fitted(m_ri) - truth)^2))))
```

    pop + re  : edf  20.73   RMSE vs true curves 0.4250

### Plot

``` r
par(mfrow = c(1, 2))
tg <- seq(0.02, 0.98, length.out = 150)
levs <- levels(traj$subject)
cols <- rainbow(length(levs))

plot(traj$t, traj$y, col = adjustcolor(cols[traj$subject], alpha.f = 0.35),
     pch = 16, cex = 0.5, xlab = "t (scaled follow-up)", ylab = "y",
     main = 'bs="fs" — one curve per subject')
for (i in seq_along(levs)) {
  pr <- predict(m_fs, newdata = data.frame(t = tg, subject = factor(levs[i], levels = levs)))
  lines(tg, pr, col = cols[i], lwd = 1.4)
}
lines(tg, 3 + 4 * sin(pi * tg) - 1.2 * tg, col = "black", lwd = 3, lty = 2)

plot(range(tg), range(traj$y), type = "n", xlab = "t (scaled follow-up)",
     ylab = "y", main = "fs (solid) vs random intercept (dashed)")
for (i in 1:4) {
  sub <- traj[traj$subject == levs[i], ]
  lines(sub$t, sub$f_pop + sub$dev_true, col = adjustcolor(cols[i], alpha.f = 0.4), lwd = 3)
  nd <- data.frame(t = tg, subject = factor(levs[i], levels = levs))
  lines(tg, predict(m_fs, newdata = nd), col = cols[i], lwd = 2)
  lines(tg, predict(m_ri, newdata = nd), col = cols[i], lwd = 1.6, lty = 2)
}
```

![](10_gamm_files/figure-commonmark/unnamed-chunk-19-1.png)

### Agreement with GAM.jl, and the one caveat

On this dataset the two packages agree closely: `edf` 89.64 (mgcv)
against 89.69 (GAM.jl), identical deviance explained (0.9546), and RMSE
against the true subject curves of 0.1675 against 0.1672. Fitted values
correlate at 0.9999971, a maximum difference of 0.16% of the fitted
range. The random-intercept comparison agrees to four decimal places in
both packages (edf 20.73, RMSE 0.4250).

The **smoothing parameters do not transfer**, though. Both packages
report three, and the wiggliness parameter agrees closely (mgcv
0.010982, GAM.jl 0.010959), but mgcv applies the `nat.param(type=1)`
reparameterization to the null-space components, so the remaining two
are expressed on a different footing and do not correspond elementwise.
Feeding mgcv’s `sp` into GAM.jl makes the fits agree *less* well (0.52%
of range, versus 0.16% when each package selects its own). Compare `fs`
fits at freely selected smoothing parameters, and compare fitted values
and EDF rather than raw `sp`.

## Syntax Comparison

| Feature | Julia (GAM.jl) | R (mgcv) |
|----|----|----|
| Random intercept | `gamm(@formula(y ~ s(x) + (1\|g)), df)` | `gamm(y ~ s(x), random=list(g=~1), data=df)` |
| `@formula` path | `gamm(@formula(y ~ cr(x, 10) + (1\|g)), df)` | — |
| `re()` shorthand | `gamm(@formula(y ~ cr(x, 10) + re(g)), df)` | — |
| Equivalent GAM | `gam(@formula(y ~ s(x) + s(g, bs=:re)), df)` | `gam(y ~ s(x) + s(g, bs="re"), data=df)` |
| Factor-smooth | `s(x, g, bs=:fs)` | `s(x, g, bs="fs")` |
| Non-Gaussian | `gamm(..., Poisson())` | `gamm(..., family=poisson())` |
| Extract RE | `ranef(m)` | `ranef(m$lme)` |
| Variance components | `VarCorr(m)` | `VarCorr(m$lme)` |
| Prediction | `predict(m, newdata)` | `predict(m$gam, newdata)` |

### Key differences

1.  **Formula syntax**: Julia puts random effects *in the formula* with
    `(1|group)`, matching lme4/MixedModels.jl conventions. R mgcv uses a
    separate `random=` argument.

2.  **Fitting algorithm**: GAM.jl treats random effects as smooth terms
    with identity penalty (via PIRLS+REML). R’s `gamm()` converts
    smooths to mixed-model form and delegates to `nlme::lme()`
    (Gaussian) or uses PQL (non-Gaussian).

3.  **Result structure**: Julia returns a `GammModel` with direct
    access. R returns a list with `$gam` (the GAM component) and `$lme`
    (the mixed model component).

4.  **Interface**: Julia `gamm()` returns a single model object. R
    requires navigating `m$gam` vs `m$lme` for different aspects of the
    fit.
