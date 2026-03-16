# Shape-Constrained Additive Models
GAM.jl Contributors

## Introduction

This vignette demonstrates shape-constrained additive models using the R
**scam** package, fitting the same simulated data as the Julia vignette
for comparison.

## Setup

``` r
library(scam)
```

    This is scam 1.2-19.

``` r
library(mgcv)
```

    Loading required package: nlme

    This is mgcv 1.9-3. For overview type 'help("mgcv-package")'.

## Example 1: Monotone increasing (dose-response)

### Load data

True function: *f*(*x*) = 3(1 − *e*<sup>−5*x*</sup>)

``` r
dat <- read.csv("../data.csv")
x <- dat$x
y <- dat$y
n <- nrow(dat)
f_true <- 3.0 * (1.0 - exp(-5.0 * x))
```

### Fit unconstrained GAM vs SCAM

``` r
m_gam <- gam(y ~ s(x, k = 15, bs = "cr"), data = dat)
m_scam <- scam(y ~ s(x, k = 15, bs = "mpi"), data = dat)
```

### Compare fitted values

``` r
yhat_gam <- predict(m_gam)
yhat_scam <- predict(m_scam)

rmse_gam <- sqrt(mean((yhat_gam - f_true)^2))
rmse_scam <- sqrt(mean((yhat_scam - f_true)^2))

cat("RMSE (unconstrained GAM):", round(rmse_gam, 4), "\n")
```

    RMSE (unconstrained GAM): 0.0596 

``` r
cat("RMSE (SCAM, monotone increasing):", round(rmse_scam, 4), "\n")
```

    RMSE (SCAM, monotone increasing): 0.0501 

### Verify monotonicity

``` r
diffs_scam <- diff(yhat_scam)
diffs_gam <- diff(yhat_gam)

cat("Min successive difference (SCAM):", round(min(diffs_scam), 6), "\n")
```

    Min successive difference (SCAM): 1e-06 

``` r
cat("All non-decreasing (SCAM):", all(diffs_scam >= -1e-10), "\n")
```

    All non-decreasing (SCAM): TRUE 

``` r
cat("Min successive difference (GAM):", round(min(diffs_gam), 6), "\n")
```

    Min successive difference (GAM): 1e-06 

``` r
cat("All non-decreasing (GAM):", all(diffs_gam >= -1e-10), "\n")
```

    All non-decreasing (GAM): TRUE 

## Example 2: Convex function

### Simulate data

True function: *f*(*x*) = 2*x*<sup>2</sup>

``` r
dat_cx <- read.csv("../data_cx.csv")
x_cx <- dat_cx$x
y_cx <- dat_cx$y

dat2 <- data.frame(y = y_cx, x = x_cx)
f_true2 <- 2 * x_cx^2
```

### Fit with convexity constraint

``` r
m_cx <- scam(y ~ s(x, k = 15, bs = "cx"), data = dat2)

yhat_cx <- predict(m_cx)
rmse_cx <- sqrt(mean((yhat_cx - f_true2)^2))
cat("RMSE (convex SCAM):", round(rmse_cx, 4), "\n")
```

    RMSE (convex SCAM): 6.7673 

### Verify convexity

``` r
first_diffs <- diff(yhat_cx)
second_diffs <- diff(first_diffs)
cat("Min second difference:", round(min(second_diffs), 6), "\n")
```

    Min second difference: -0.061069 

``` r
cat("All convex:", all(second_diffs >= -1e-10), "\n")
```

    All convex: FALSE 

## Example 3: Monotone increasing and concave

### Simulate data

True function: $f(x) = 3\sqrt{x}$

``` r
dat_micv <- read.csv("../data_micv.csv")
x_micv <- dat_micv$x
y_micv <- dat_micv$y

dat3 <- data.frame(y = y_micv, x = x_micv)
f_true3 <- 3 * sqrt(x_micv)
```

### Fit with monotone increasing + concave constraint

``` r
m_micv <- scam(y ~ s(x, k = 15, bs = "micv"), data = dat3)

yhat_micv <- predict(m_micv)
rmse_micv <- sqrt(mean((yhat_micv - f_true3)^2))
cat("RMSE (monotone increasing + concave):", round(rmse_micv, 4), "\n")
```

    RMSE (monotone increasing + concave): 1.5225 

### Verify constraints

``` r
first_diffs_micv <- diff(yhat_micv)
second_diffs_micv <- diff(first_diffs_micv)
cat("Min first difference (monotonicity):", round(min(first_diffs_micv), 6), "\n")
```

    Min first difference (monotonicity): 0 

``` r
cat("Max second difference (concavity):", round(max(second_diffs_micv), 6), "\n")
```

    Max second difference (concavity): 0.032084 

``` r
cat("Monotone increasing:", all(first_diffs_micv >= -1e-10), "\n")
```

    Monotone increasing: TRUE 

``` r
cat("Concave:", all(second_diffs_micv <= 1e-10), "\n")
```

    Concave: FALSE 

## SCAM model summaries

``` r
summary(m_scam)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    y ~ s(x, k = 15, bs = "mpi")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  2.41623    0.02038   118.6   <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Approximate significance of smooth terms:
           edf Ref.df     F p-value    
    s(x) 3.408  4.201 337.9  <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    R-sq.(adj) =  0.8768   Deviance explained = 87.9%
    GCV score = 0.084902  Scale est. = 0.083031  n = 200

``` r
summary(m_cx)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    y ~ s(x, k = 15, bs = "cx")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  1.74232    0.02041   85.37   <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Approximate significance of smooth terms:
           edf Ref.df     F p-value    
    s(x) 1.857  2.189 289.6  <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    Rank: 14/15

    R-sq.(adj) =  0.7606   Deviance explained = 76.3%
    GCV score = 0.08451  Scale est. = 0.083303  n = 200

``` r
summary(m_micv)
```


    Family: gaussian 
    Link function: identity 

    Formula:
    y ~ s(x, k = 15, bs = "micv")

    Parametric coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 0.591117   0.007384   80.06   <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Approximate significance of smooth terms:
           edf Ref.df     F p-value    
    s(x) 3.393  3.714 160.9  <2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    Rank: 14/15

    R-sq.(adj) =  0.7491   Deviance explained = 75.3%
    GCV score = 0.011149  Scale est. = 0.010904  n = 200

    BFGS termination condition:
    1.99322e-05

## Comparison table

<table>
<thead>
<tr>
<th>Feature</th>
<th>R <code>scam</code></th>
<th>Julia GAM.jl <code>scam</code></th>
</tr>
</thead>
<tbody>
<tr>
<td>Monotone increasing</td>
<td><code>bs="mpi"</code></td>
<td><code>bs=:mpi</code></td>
</tr>
<tr>
<td>Monotone decreasing</td>
<td><code>bs="mpd"</code></td>
<td><code>bs=:mpd</code></td>
</tr>
<tr>
<td>Convex</td>
<td><code>bs="cx"</code></td>
<td><code>bs=:cx</code></td>
</tr>
<tr>
<td>Concave</td>
<td><code>bs="cv"</code></td>
<td><code>bs=:cv</code></td>
</tr>
<tr>
<td>Mono. inc. + convex</td>
<td><code>bs="micx"</code></td>
<td><code>bs=:micx</code></td>
</tr>
<tr>
<td>Mono. inc. + concave</td>
<td><code>bs="micv"</code></td>
<td><code>bs=:micv</code></td>
</tr>
<tr>
<td>Mono. dec. + convex</td>
<td><code>bs="mdcx"</code></td>
<td><code>bs=:mdcx</code></td>
</tr>
<tr>
<td>Mono. dec. + concave</td>
<td><code>bs="mdcv"</code></td>
<td><code>bs=:mdcv</code></td>
</tr>
</tbody>
</table>
