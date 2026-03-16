# GAM.jl vs mgcv benchmark suite
#
# Compares fitting time and accuracy between GAM.jl and R's mgcv.
# Usage: julia --project benchmark/benchmarks.jl

using GAM
using DataFrames
using Statistics
using Printf
using RCall

R"library(mgcv)"

function benchmark_one(label, julia_fn, r_code; n_reps=5)
    # Julia timing
    julia_fn()  # warmup
    j_times = Float64[]
    j_result = nothing
    for _ in 1:n_reps
        t = @elapsed j_result = julia_fn()
        push!(j_times, t)
    end

    # R timing
    reval(r_code)  # warmup
    r_times = Float64[]
    for _ in 1:n_reps
        t_r = rcopy(reval("system.time({ $r_code })[['elapsed']]"))
        push!(r_times, t_r)
    end

    j_med = median(j_times)
    r_med = median(r_times)
    speedup = r_med / j_med

    @printf("%-40s  Julia: %7.3fs  R: %7.3fs  Speedup: %5.1fx\n",
        label, j_med, r_med, speedup)

    return (label=label, julia_time=j_med, r_time=r_med, speedup=speedup,
        julia_result=j_result)
end

function run_benchmarks()
    println("=" ^ 80)
    println("GAM.jl vs mgcv Benchmark Suite")
    println("=" ^ 80)
    println()

    results = []

    # ── 1. Small Gaussian CR ─────────────────────────────────────────────
    R"""
    set.seed(42)
    bm_x <- rnorm(500)
    bm_y <- sin(bm_x) + rnorm(500, sd=0.3)
    bm_df <- data.frame(x=bm_x, y=bm_y)
    """
    x = rcopy(R"bm_x"); y = rcopy(R"bm_y")
    df = DataFrame(x=x, y=y)

    push!(results, benchmark_one("Gaussian CR n=500 k=15",
        () -> gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df),
        "gam(y ~ s(x, k=15, bs='cr'), data=bm_df, method='REML')"))

    # ── 2. Medium Gaussian CR ────────────────────────────────────────────
    R"""
    set.seed(42)
    bm_x2 <- rnorm(5000)
    bm_y2 <- sin(bm_x2) + rnorm(5000, sd=0.3)
    bm_df2 <- data.frame(x=bm_x2, y=bm_y2)
    """
    x2 = rcopy(R"bm_x2"); y2 = rcopy(R"bm_y2")
    df2 = DataFrame(x=x2, y=y2)

    push!(results, benchmark_one("Gaussian CR n=5000 k=20",
        () -> gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df2),
        "gam(y ~ s(x, k=20, bs='cr'), data=bm_df2, method='REML')"))

    # ── 3. Large Gaussian CR ─────────────────────────────────────────────
    R"""
    set.seed(42)
    bm_x3 <- rnorm(50000)
    bm_y3 <- sin(bm_x3) + rnorm(50000, sd=0.5)
    bm_df3 <- data.frame(x=bm_x3, y=bm_y3)
    """
    x3 = rcopy(R"bm_x3"); y3 = rcopy(R"bm_y3")
    df3 = DataFrame(x=x3, y=y3)

    push!(results, benchmark_one("Gaussian CR n=50000 k=20",
        () -> gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df3),
        "gam(y ~ s(x, k=20, bs='cr'), data=bm_df3, method='REML')";
        n_reps=3))

    # ── 4. Two smooths ──────────────────────────────────────────────────
    R"""
    set.seed(42)
    bm_x1m <- rnorm(2000)
    bm_x2m <- rnorm(2000)
    bm_ym <- sin(bm_x1m) + 0.5*bm_x2m^2 + rnorm(2000, sd=0.3)
    bm_dfm <- data.frame(x1=bm_x1m, x2=bm_x2m, y=bm_ym)
    """
    x1m = rcopy(R"bm_x1m"); x2m = rcopy(R"bm_x2m"); ym = rcopy(R"bm_ym")
    dfm = DataFrame(x1=x1m, x2=x2m, y=ym)

    push!(results, benchmark_one("Gaussian 2 CR smooths n=2000",
        () -> gam(@gam_formula(y ~ s(x1, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), dfm),
        "gam(y ~ s(x1, k=15, bs='cr') + s(x2, k=10, bs='cr'), data=bm_dfm, method='REML')"))

    # ── 5. Poisson CR ────────────────────────────────────────────────────
    R"""
    set.seed(42)
    bm_xp <- rnorm(2000)
    bm_yp <- rpois(2000, exp(1 + 0.5*sin(bm_xp)))
    bm_dfp <- data.frame(x=bm_xp, y=bm_yp)
    """
    xp = rcopy(R"bm_xp"); yp = rcopy(R"as.numeric(bm_yp)")
    dfp = DataFrame(x=xp, y=yp)

    push!(results, benchmark_one("Poisson CR n=2000 k=15",
        () -> gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), dfp;
            family=Poisson(), link=LogLink()),
        "gam(y ~ s(x, k=15, bs='cr'), data=bm_dfp, family=poisson(), method='REML')"))

    # ── 6. BAM vs bam — large Gaussian ───────────────────────────────────
    R"""
    set.seed(42)
    bm_x6 <- rnorm(100000)
    bm_y6 <- sin(bm_x6) + rnorm(100000, sd=0.5)
    bm_df6 <- data.frame(x=bm_x6, y=bm_y6)
    """
    x6 = rcopy(R"bm_x6"); y6 = rcopy(R"bm_y6")
    df6 = DataFrame(x=x6, y=y6)

    push!(results, benchmark_one("BAM Gaussian n=100000 k=20",
        () -> bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df6;
            bam_ctrl=bam_control(chunk_size=10000)),
        "bam(y ~ s(x, k=20, bs='cr'), data=bm_df6, method='fREML')";
        n_reps=3))

    # ── Summary ──────────────────────────────────────────────────────────
    println()
    println("=" ^ 80)
    geo_mean = exp(mean(log.(getfield.(results, :speedup))))
    @printf("Geometric mean speedup: %.1fx\n", geo_mean)
    println("=" ^ 80)

    return results
end

run_benchmarks()
