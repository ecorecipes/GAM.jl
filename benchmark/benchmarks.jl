# GAM.jl vs mgcv benchmark suite
#
# Compares fitting time and accuracy between GAM.jl and R equivalents.
# Usage: julia --project benchmark/benchmarks.jl

using GAM
using DataFrames
using Statistics
using Printf
using RCall
using StatsAPI: predict

R"library(mgcv)"

const BYTES_PER_MIB = 1024.0^2

function benchmark_one(label, julia_fn, r_code; n_reps=5)
    # Julia timing — two warmups to ensure full JIT compilation
    julia_fn()
    julia_fn()
    j_times = Float64[]
    j_result = nothing
    for _ in 1:n_reps
        GC.gc(false)
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
    r_med = max(median(r_times), 1e-4)  # floor to avoid Inf
    speedup = r_med / max(j_med, 1e-6)

    @printf("  %-44s  Julia: %7.4fs  R: %7.4fs  Speedup: %6.2fx\n",
        label, j_med, r_med, speedup)

    return (label=label, julia_time=j_med, r_time=r_med, speedup=speedup,
        julia_result=j_result)
end

_extract_iterations(result) = hasproperty(result, :iterations) ? getproperty(result, :iterations) : missing

function _median_iter(values)
    vals = collect(skipmissing(values))
    return isempty(vals) ? missing : Int(round(median(vals)))
end

_fmt_iter(iter) = ismissing(iter) ? "NA" : string(iter)

function _r_bench_once(r_code::AbstractString, r_iter_expr::AbstractString)
    reval("""
    local({
        gc(reset = TRUE)
        tmp <- tempfile()
        fit <- NULL
        Rprofmem(tmp)
        t <- system.time({
            fit <- $r_code
        })[['elapsed']]
        Rprofmem(NULL)
        mem_lines <- if (file.exists(tmp)) readLines(tmp, warn = FALSE) else character()
        unlink(tmp)
        mem_bytes <- suppressWarnings(sum(as.numeric(sub(" .*", "", mem_lines)), na.rm = TRUE))
        iter <- tryCatch({ $r_iter_expr }, error = function(e) NA_integer_)
        if (is.null(iter) || length(iter) == 0 || is.na(iter[[1]])) {
            iter <- NA_integer_
        } else {
            iter <- as.integer(iter[[1]])
        }
        assign(".bench_time", as.numeric(t), envir = .GlobalEnv)
        assign(".bench_mem_bytes", as.numeric(mem_bytes), envir = .GlobalEnv)
        assign(".bench_iter", iter, envir = .GlobalEnv)
        invisible(NULL)
    })
    """)

    iter_val = rcopy(R".bench_iter")
    iter = ismissing(iter_val) ? missing : Int(iter_val)
    return (
        time = Float64(rcopy(R".bench_time")),
        mem_mib = Float64(rcopy(R".bench_mem_bytes")) / BYTES_PER_MIB,
        iter = iter,
    )
end

function benchmark_one_detailed(label, julia_fn, r_code, r_iter_expr;
                                n_reps=3, r_label="R")
    julia_fn()
    julia_fn()

    j_times = Float64[]
    j_mems = Float64[]
    j_iters = Union{Missing, Int}[]
    j_result = nothing
    for _ in 1:n_reps
        GC.gc(false)
        elapsed = Ref(0.0)
        bytes = @allocated begin
            elapsed[] = @elapsed j_result = julia_fn()
        end
        push!(j_times, elapsed[])
        push!(j_mems, bytes / BYTES_PER_MIB)
        push!(j_iters, _extract_iterations(j_result))
    end

    _r_bench_once(r_code, r_iter_expr)  # warmup
    r_times = Float64[]
    r_mems = Float64[]
    r_iters = Union{Missing, Int}[]
    for _ in 1:n_reps
        bench = _r_bench_once(r_code, r_iter_expr)
        push!(r_times, bench.time)
        push!(r_mems, bench.mem_mib)
        push!(r_iters, bench.iter)
    end

    j_med = median(j_times)
    r_med = max(median(r_times), 1e-4)
    speedup = r_med / max(j_med, 1e-6)
    j_mem_med = median(j_mems)
    r_mem_med = median(r_mems)
    j_iter_med = _median_iter(j_iters)
    r_iter_med = _median_iter(r_iters)

    @printf("  %-32s  Julia: %7.4fs %7.1f MiB %4s it  %-9s %7.4fs %7.1f MiB %4s it  Speedup: %6.2fx\n",
        label, j_med, j_mem_med, _fmt_iter(j_iter_med),
        r_label * ":", r_med, r_mem_med, _fmt_iter(r_iter_med), speedup)

    return (
        label = label,
        julia_time = j_med,
        r_time = r_med,
        speedup = speedup,
        julia_mem_mib = j_mem_med,
        r_mem_mib = r_mem_med,
        julia_iter = j_iter_med,
        r_iter = r_iter_med,
        julia_result = j_result,
    )
end

function _run_gamlss_section!(all_results::Vector{NamedTuple})
    println("\n── GAMLSS (Multi-Parameter) ────────────────────────────────────")

    R"""
    suppressPackageStartupMessages(library(gamlss))
    """

    f_mu_cr = @gam_formula(y ~ s(x, k = 20, bs = :cr))
    f_sigma_cr = @gam_formula(y ~ s(x, k = 10, bs = :cr))
    f_mu_ps = @gam_formula(y ~ s(x, k = 23, bs = :ps))
    f_sigma_ps = @gam_formula(y ~ s(x, k = 11, bs = :ps))
    ctrl_local_ml = GAM.gamlss_control(sp_method = :local_ml, n_cyc = 50, trace = false)

    for n in (500, 2000, 10000)
        R"""
        set.seed(42)
        bm_xl <- runif($n, 0, 2 * pi)
        bm_mu_l <- sin(bm_xl)
        bm_sig_l <- exp(0.5 * cos(bm_xl))
        bm_yl <- rnorm($n, bm_mu_l, bm_sig_l)
        bm_dfl <- data.frame(x = bm_xl, y = bm_yl)
        """
        xl = rcopy(R"bm_xl")
        yl = rcopy(R"bm_yl")
        dfl = DataFrame(x = xl, y = yl)

        push!(all_results, benchmark_one_detailed("GAMLSS Normal LS n=$n  (EFS)",
            () -> gamlss([f_mu_cr, f_sigma_cr], dfl, GaussianLS()),
            """gam(list(y ~ s(x, k = 20, bs = "cr"), ~ s(x, k = 10, bs = "cr")),
                    family = gaulss(), data = bm_dfl, method = "REML")""",
            "if (!is.null(fit\$outer.info\$iter)) fit\$outer.info\$iter else NA_integer_";
            n_reps = 3, r_label = "R(mgcv)"))

        push!(all_results, benchmark_one_detailed("GAMLSS Normal LS n=$n (RS+ML)",
            () -> gamlss([f_mu_ps, f_sigma_ps], dfl, GaussianLS();
                method = :rs, gamlss_ctrl = ctrl_local_ml),
            """gamlss(y ~ pb(x, inter = 20, degree = 3, order = 2, method = "ML"),
                      sigma.formula = ~ pb(x, inter = 8, degree = 3, order = 2, method = "ML"),
                      family = NO(), data = bm_dfl,
                      control = gamlss.control(n.cyc = 50, trace = FALSE))""",
            "if (!is.null(fit\$iter)) fit\$iter else NA_integer_";
            n_reps = 3, r_label = "R(gamlss)"))
    end
end

function run_gamlss_benchmarks()
    println("=" ^ 80)
    println("GAM.jl vs R GAMLSS Benchmarks")
    println("=" ^ 80)
    all_results = NamedTuple[]
    _run_gamlss_section!(all_results)
    println()
    @printf("  %-25s  Geometric mean speedup: %6.2fx\n",
        "GAMLSS", exp(mean(log.(getfield.(all_results, :speedup)))))
    println("=" ^ 80)
    return all_results
end

function run_benchmarks()
    println("=" ^ 80)
    println("GAM.jl vs R Benchmark Suite")
    println("=" ^ 80)

    all_results = NamedTuple[]

    # ═══════════════════════════════════════════════════════════════════════
    # Section 1: GAM fitting (gam vs mgcv::gam)
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── GAM Fitting ─────────────────────────────────────────────────")

    # 1a. Small Gaussian CR
    R"""
    set.seed(42)
    bm_x <- rnorm(500)
    bm_y <- sin(bm_x) + rnorm(500, sd=0.3)
    bm_df <- data.frame(x=bm_x, y=bm_y)
    """
    x = rcopy(R"bm_x"); y = rcopy(R"bm_y")
    df = DataFrame(x=x, y=y)

    push!(all_results, benchmark_one("Gaussian CR n=500 k=15",
        () -> gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df),
        "gam(y ~ s(x, k=15, bs='cr'), data=bm_df, method='REML')"))

    # 1b. Medium Gaussian CR
    R"""
    set.seed(42)
    bm_x2 <- rnorm(5000)
    bm_y2 <- sin(bm_x2) + rnorm(5000, sd=0.3)
    bm_df2 <- data.frame(x=bm_x2, y=bm_y2)
    """
    x2 = rcopy(R"bm_x2"); y2 = rcopy(R"bm_y2")
    df2 = DataFrame(x=x2, y=y2)

    push!(all_results, benchmark_one("Gaussian CR n=5000 k=20",
        () -> gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df2),
        "gam(y ~ s(x, k=20, bs='cr'), data=bm_df2, method='REML')"))

    # 1c. Large Gaussian CR
    R"""
    set.seed(42)
    bm_x3 <- rnorm(50000)
    bm_y3 <- sin(bm_x3) + rnorm(50000, sd=0.5)
    bm_df3 <- data.frame(x=bm_x3, y=bm_y3)
    """
    x3 = rcopy(R"bm_x3"); y3 = rcopy(R"bm_y3")
    df3 = DataFrame(x=x3, y=y3)

    push!(all_results, benchmark_one("Gaussian CR n=50000 k=20",
        () -> gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df3),
        "gam(y ~ s(x, k=20, bs='cr'), data=bm_df3, method='REML')";
        n_reps=3))

    # 1d. TPRS (default basis)
    push!(all_results, benchmark_one("Gaussian TP n=5000 k=20",
        () -> gam(@gam_formula(y ~ s(x, k=20)), df2),
        "gam(y ~ s(x, k=20), data=bm_df2, method='REML')"))

    # 1e. Two smooths
    R"""
    set.seed(42)
    bm_x1m <- rnorm(2000)
    bm_x2m <- rnorm(2000)
    bm_ym <- sin(bm_x1m) + 0.5*bm_x2m^2 + rnorm(2000, sd=0.3)
    bm_dfm <- data.frame(x1=bm_x1m, x2=bm_x2m, y=bm_ym)
    """
    x1m = rcopy(R"bm_x1m"); x2m = rcopy(R"bm_x2m"); ym = rcopy(R"bm_ym")
    dfm = DataFrame(x1=x1m, x2=x2m, y=ym)

    push!(all_results, benchmark_one("Gaussian 2 smooths n=2000",
        () -> gam(@gam_formula(y ~ s(x1, k=15, bs=:cr) + s(x2, k=10, bs=:cr)), dfm),
        "gam(y ~ s(x1, k=15, bs='cr') + s(x2, k=10, bs='cr'), data=bm_dfm, method='REML')"))

    # 1f. Three smooths
    R"""
    set.seed(42)
    bm_x3m <- rnorm(3000)
    bm_dfm3 <- data.frame(x1=bm_x1m[1:3000], x2=bm_x2m[1:3000], x3=bm_x3m, y=NA)
    bm_dfm3$x1 <- rnorm(3000); bm_dfm3$x2 <- rnorm(3000)
    bm_dfm3$y <- sin(bm_dfm3$x1) + 0.5*bm_dfm3$x2^2 + cos(bm_x3m) + rnorm(3000, sd=0.3)
    """
    x1_3 = rcopy(R"bm_dfm3$x1"); x2_3 = rcopy(R"bm_dfm3$x2")
    x3_3 = rcopy(R"bm_x3m"); y_3 = rcopy(R"bm_dfm3$y")
    dfm3 = DataFrame(x1=x1_3, x2=x2_3, x3=x3_3, y=y_3)

    push!(all_results, benchmark_one("Gaussian 3 smooths n=3000",
        () -> gam(@gam_formula(y ~ s(x1, k=10, bs=:cr) + s(x2, k=10, bs=:cr) + s(x3, k=10, bs=:cr)), dfm3),
        "gam(y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr'), data=bm_dfm3, method='REML')"))

    # 1g. Poisson GLM-GAM
    R"""
    set.seed(42)
    bm_xp <- rnorm(2000)
    bm_yp <- rpois(2000, exp(1 + 0.5*sin(bm_xp)))
    bm_dfp <- data.frame(x=bm_xp, y=bm_yp)
    """
    xp = rcopy(R"bm_xp"); yp = rcopy(R"as.numeric(bm_yp)")
    dfp = DataFrame(x=xp, y=yp)

    push!(all_results, benchmark_one("Poisson CR n=2000 k=15",
        () -> gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), dfp;
            family=Poisson(), link=LogLink()),
        "gam(y ~ s(x, k=15, bs='cr'), data=bm_dfp, family=poisson(), method='REML')"))

    # 1h. Gamma GLM-GAM
    R"""
    set.seed(42)
    bm_xg <- rnorm(2000)
    bm_mu <- exp(1 + 0.5*sin(bm_xg))
    bm_yg <- rgamma(2000, shape=5, rate=5/bm_mu)
    bm_dfg <- data.frame(x=bm_xg, y=bm_yg)
    """
    xg = rcopy(R"bm_xg"); yg = rcopy(R"bm_yg")
    dfg = DataFrame(x=xg, y=yg)

    push!(all_results, benchmark_one("Gamma CR n=2000 k=15",
        () -> gam(@gam_formula(y ~ s(x, k=15, bs=:cr)), dfg;
            family=Gamma(), link=LogLink()),
        "gam(y ~ s(x, k=15, bs='cr'), data=bm_dfg, family=Gamma(link=log), method='REML')"))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 2: BAM (large-scale)
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── BAM (Large-Scale) ───────────────────────────────────────────")

    R"""
    set.seed(42)
    bm_x6 <- rnorm(100000)
    bm_y6 <- sin(bm_x6) + rnorm(100000, sd=0.5)
    bm_df6 <- data.frame(x=bm_x6, y=bm_y6)
    """
    x6 = rcopy(R"bm_x6"); y6 = rcopy(R"bm_y6")
    df6 = DataFrame(x=x6, y=y6)

    push!(all_results, benchmark_one("BAM Gaussian n=100000 k=20",
        () -> bam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df6;
            bam_ctrl=bam_control(chunk_size=10000)),
        "bam(y ~ s(x, k=20, bs='cr'), data=bm_df6, method='fREML')";
        n_reps=3))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 3: Prediction
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── Prediction ──────────────────────────────────────────────────")

    m_julia = gam(@gam_formula(y ~ s(x, k=20, bs=:cr)), df2)
    R"bm_m_r <- gam(y ~ s(x, k=20, bs='cr'), data=bm_df2, method='REML')"

    # predict at new data
    R"""
    set.seed(99)
    bm_newx <- rnorm(10000)
    bm_newdf <- data.frame(x=bm_newx)
    """
    newx = rcopy(R"bm_newx")
    newdf = DataFrame(x=newx)

    push!(all_results, benchmark_one("predict(m, new) n=10000",
        () -> predict(m_julia, newdf),
        "predict(bm_m_r, newdata=bm_newdf)"))

    # predict at new data with SE
    push!(all_results, benchmark_one("predict(m, new, se) n=10000",
        () -> predict(m_julia, newdf; se=true),
        "predict(bm_m_r, newdata=bm_newdf, se.fit=TRUE)"))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 4: Basis construction
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── Basis Construction ──────────────────────────────────────────")

    # CR basis
    spec_cr = GAM.s(:x; bs=:cr, k=20)
    R"bm_sm_cr <- smoothCon(s(x, k=20, bs='cr'), data=bm_df2, absorb.cons=TRUE)[[1]]"
    push!(all_results, benchmark_one("CR basis n=5000 k=20",
        () -> GAM.smooth_construct(spec_cr, df2),
        "smoothCon(s(x, k=20, bs='cr'), data=bm_df2, absorb.cons=TRUE)"))

    # TPRS basis
    spec_tp = GAM.s(:x; bs=:tp, k=20)
    R"bm_sm_tp <- smoothCon(s(x, k=20, bs='tp'), data=bm_df2, absorb.cons=TRUE)[[1]]"
    push!(all_results, benchmark_one("TPRS basis n=5000 k=20",
        () -> GAM.smooth_construct(spec_tp, df2),
        "smoothCon(s(x, k=20, bs='tp'), data=bm_df2, absorb.cons=TRUE)"))

    # Large TPRS basis
    spec_tp_lg = GAM.s(:x; bs=:tp, k=30)
    push!(all_results, benchmark_one("TPRS basis n=50000 k=30",
        () -> GAM.smooth_construct(spec_tp_lg, df3),
        "smoothCon(s(x, k=30, bs='tp'), data=bm_df3, absorb.cons=TRUE)";
        n_reps=3))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 5: SCAM (shape-constrained)
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── SCAM (Shape-Constrained) ────────────────────────────────────")

    R"library(scam)"
    R"""
    set.seed(42)
    bm_xs <- sort(runif(1000))
    bm_ys <- 2*bm_xs + 0.5*sin(4*bm_xs) + rnorm(1000, sd=0.2)
    bm_dfs <- data.frame(x=bm_xs, y=bm_ys)
    """
    xs = rcopy(R"bm_xs"); ys = rcopy(R"bm_ys")
    dfs = DataFrame(x=xs, y=ys)

    push!(all_results, benchmark_one("SCAM monotone incr n=1000 k=15",
        () -> scam(@gam_formula(y ~ s(x, k=15, bs=:mpi)), dfs),
        "scam(y ~ s(x, k=15, bs='mpi'), data=bm_dfs)"))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 6: QGAM (quantile regression)
    # ═══════════════════════════════════════════════════════════════════════
    println("\n── QGAM (Quantile Regression) ──────────────────────────────────")

    R"library(qgam)"

    # Pre-calibrate sigma in R to get a comparable lsig, then benchmark the ELF fit only
    R"""
    set.seed(42)
    bm_qfit_r <- qgam(y ~ s(x, k=15, bs='cr'), data=bm_df2, qu=0.5)
    bm_lsig_r <- bm_qfit_r$family$getTheta()
    """
    lsig_r = rcopy(R"bm_lsig_r")

    # QGAM with pre-calibrated sigma (ELF fit only, no bootstrap)
    push!(all_results, benchmark_one("QGAM ELF fit n=5000 k=15",
        () -> qgam(@gam_formula(y ~ s(x, k=15, bs=:cr)), df2, 0.5; lsig=lsig_r),
        "qgam(y ~ s(x, k=15, bs='cr'), data=bm_df2, qu=0.5, lsig=$lsig_r)";
        n_reps=3))

    # Full QGAM with calibration (smaller dataset, fewer reps due to cost)
    dfq_small = DataFrame(x=x2[1:1000], y=y2[1:1000])
    R"bm_dfq_small <- bm_df2[1:1000,]"
    push!(all_results, benchmark_one("QGAM full calib n=1000 k=10",
        () -> qgam(@gam_formula(y ~ s(x, k=10, bs=:cr)), dfq_small, 0.5),
        "qgam(y ~ s(x, k=10, bs='cr'), data=bm_dfq_small, qu=0.5)";
        n_reps=1))

    # ═══════════════════════════════════════════════════════════════════════
    # Section 7: GAMLSS (multi-parameter)
    # ═══════════════════════════════════════════════════════════════════════
    _run_gamlss_section!(all_results)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    println()
    println("=" ^ 80)
    println("Summary")
    println("=" ^ 80)

    # Group by section
    sections = [
        ("GAM Fitting", 1:8),
        ("BAM", 9:9),
        ("Prediction", 10:11),
        ("Basis Construction", 12:14),
        ("SCAM", 15:15),
        ("QGAM", 16:17),
        ("GAMLSS", 18:23),
    ]

    for (name, range) in sections
        subset = all_results[intersect(range, 1:length(all_results))]
        isempty(subset) && continue
        geo = exp(mean(log.(getfield.(subset, :speedup))))
        @printf("  %-25s  Geometric mean speedup: %6.2fx\n", name, geo)
    end

    overall_geo = exp(mean(log.(getfield.(all_results, :speedup))))
    println()
    @printf("  %-25s  Geometric mean speedup: %6.2fx\n", "OVERALL", overall_geo)
    println("=" ^ 80)

    return all_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
