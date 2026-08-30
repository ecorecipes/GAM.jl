#!/usr/bin/env julia
#
# Data-generation script for the vignette datasets whose data-generating
# processes are stated in the vignette text. Running this script regenerates
# the CSVs from the documented DGPs with fixed seeds, so data and narrative
# cannot drift apart.
#
# Usage (from the repository root or vignettes/):
#   julia vignettes/generate_data.jl            # regenerate all files below
#
# Files NOT regenerated here (their existing data already matches the vignette
# narrative, and regenerating would churn rendered output for no benefit):
#   01, 03, 08, 09, 11 data files; the pre-existing 05 CSVs; the pre-existing 02 and 04 CSVs
#   (data_shrink/data_adaptive and data_incidence/data_incidence_od ARE
#   generated below); 07 data.csv;
#   10 data_repeated_measures.csv (unused by the current vignette).

using Random
using Distributions
using Printf
using DataFrames
using CSV

const VIGNETTES = @__DIR__

function write_csv(path::String, cols::AbstractVector{<:Pair{String, <:AbstractVector}})
    open(path, "w") do io
        println(io, join(("\"$(first(c))\"" for c in cols), ","))
        n = length(last(first(cols)))
        for i in 1:n
            println(io, join((string(last(c)[i]) for c in cols), ","))
        end
    end
    println("wrote $(path)")
end

# ── 05_diagnostics: data_2d.csv ─────────────────────────────────────────────
# Two-dimensional surface for the vis_gam / gamcontour and data_slice
# demonstrations:
#   f(x, z) = sin(2πx) cos(πz),  y = f + ε,  ε ~ N(0, 0.3²),  x, z ~ U(0, 1)
# Written with CSV.write (unquoted header) to match the checked-in file.
function gen_2d()
    rng = Xoshiro(4242)
    n = 400
    x = rand(rng, n)          # one vectorized pass per variable
    z = rand(rng, n)
    e = randn(rng, n)
    f = sin.(2π .* x) .* cos.(π .* z)
    y = f .+ 0.3 .* e
    path = joinpath(VIGNETTES, "05_diagnostics", "data_2d.csv")
    CSV.write(path, DataFrame(x = x, z = z, y = y))
    println("wrote $(path)")
end

# ── 06_extreme_values: data_gev.csv ─────────────────────────────────────────
# GEV block maxima with covariate-dependent location and scale:
#   μ(x) = 5 + 2 sin(2πx),  log σ(x) = -0.5 + 0.5x,  ξ = 0.1 (constant)
# Sampled by inverse CDF: y = μ + σ/ξ ((-log u)^(-ξ) - 1), u ~ U(0,1).
function gen_gev()
    rng = MersenneTwister(20260313)
    n = 500
    x = rand(rng, n)          # one vectorized pass per variable
    u = rand(rng, n)
    ξ = 0.1
    μ = 5.0 .+ 2.0 .* sin.(2π .* x)
    σ = exp.(-0.5 .+ 0.5 .* x)
    y = μ .+ σ ./ ξ .* ((-log.(u)) .^ (-ξ) .- 1.0)
    write_csv(joinpath(VIGNETTES, "06_extreme_values", "data_gev.csv"),
        ["y" => y, "x" => x])
end

# ── 06_extreme_values: data_gpd.csv ─────────────────────────────────────────
# GPD threshold-exceedance data with covariate-dependent scale:
#   log σ(x) = 0.5 sin(2πx),  ξ = 0.15 (constant)
function gen_gpd()
    rng = MersenneTwister(20260406)
    n = 500
    x = rand(rng, n)
    ξ = 0.15
    σ = exp.(0.5 .* sin.(2π .* x))
    u = rand(rng, n)
    # Inverse-CDF sampling: y = σ/ξ * ((1-u)^(-ξ) - 1)
    y = σ ./ ξ .* ((1 .- u) .^ (-ξ) .- 1.0)
    write_csv(joinpath(VIGNETTES, "06_extreme_values", "data_gpd.csv"),
        ["y" => y, "x" => x])
end

# ── 07_shape_constraints: data_cx.csv ───────────────────────────────────────
# Convex example: f(x) = 2x² on [0, 3], Gaussian noise (sd = 1.0)
function gen_cx()
    rng = MersenneTwister(20260407)
    n = 200
    x = sort(3.0 .* rand(rng, n))
    y = 2.0 .* x .^ 2 .+ 1.0 .* randn(rng, n)
    write_csv(joinpath(VIGNETTES, "07_shape_constraints", "data_cx.csv"),
        ["x" => x, "y" => y])
end

# ── 07_shape_constraints: data_micv.csv ─────────────────────────────────────
# Monotone-increasing + concave example: f(x) = 3√x on [0, 1], noise sd = 0.3
function gen_micv()
    rng = MersenneTwister(20260408)
    n = 200
    x = sort(rand(rng, n))
    y = 3.0 .* sqrt.(x) .+ 0.3 .* randn(rng, n)
    write_csv(joinpath(VIGNETTES, "07_shape_constraints", "data_micv.csv"),
        ["x" => x, "y" => y])
end

# ── 10_gamm: data_gaussian_gamm.csv ─────────────────────────────────────────
# 12 subjects × 40 observations; population mean μ(x) = 1.5 sin(1.5x);
# random intercepts b_j ~ N(0, σ_b²) with σ_b = 0.6; residual sd σ_ε = 0.4.
# Columns: x, y, subject, mu_true (population mean), re_true (subject effect).
function gen_gaussian_gamm()
    rng = MersenneTwister(20260410)
    n_subj, n_per = 12, 40
    σ_b, σ_ε = 0.6, 0.4
    b = σ_b .* randn(rng, n_subj)
    x = Float64[]; y = Float64[]; subject = Int[]; mu_true = Float64[]; re_true = Float64[]
    for j in 1:n_subj, _ in 1:n_per
        xi = rand(rng, Uniform(-π, π))
        μ = 1.5 * sin(1.5 * xi)
        push!(x, xi); push!(subject, j); push!(mu_true, μ); push!(re_true, b[j])
        push!(y, μ + b[j] + σ_ε * randn(rng))
    end
    write_csv(joinpath(VIGNETTES, "10_gamm", "data_gaussian_gamm.csv"),
        ["x" => x, "y" => y, "subject" => subject, "mu_true" => mu_true, "re_true" => re_true])
end

# ── 10_gamm: data_poisson_gamm.csv ──────────────────────────────────────────
# 8 sites × 60 observations; log λ = 1 + 0.8 sin(x) + b_j, b_j ~ N(0, σ_b²)
# with σ_b = 0.4. Columns: x, y, site, eta_true (INCLUDING the site effect),
# re_true (site effect).
function gen_poisson_gamm()
    rng = MersenneTwister(20260411)
    n_site, n_per = 8, 60
    σ_b = 0.4
    b = σ_b .* randn(rng, n_site)
    # Draw each variable in its own pass. Sampling a Poisson consumes a
    # *version-dependent* number of draws from the stream, so interleaving it
    # with the covariate draws (the previous shape of this loop) made `x` depend
    # on Distributions.jl's sampler internals: a Distributions upgrade would
    # silently stop this CSV regenerating from its seed. `x` now comes straight
    # from Base `rand`, so only `y` sits downstream of the Poisson sampler.
    n = n_site * n_per
    site = repeat(1:n_site; inner = n_per)
    x = 2π .* rand(rng, n) .- π
    re_true = b[site]
    eta_true = 1.0 .+ 0.8 .* sin.(x) .+ re_true
    y = [rand(rng, Poisson(exp(η))) for η in eta_true]
    write_csv(joinpath(VIGNETTES, "10_gamm", "data_poisson_gamm.csv"),
        ["x" => x, "y" => y, "site" => site, "eta_true" => eta_true, "re_true" => re_true])
end

# ── 10_gamm: data_fs_trajectories.csv ───────────────────────────────────────
# Factor-smooth (bs=:fs) example: 15 subjects x 25 visits on a common grid of
# scaled follow-up times t in [0.02, 0.98]. Subjects differ in the SHAPE of
# their trajectory, not merely its level:
#   population    f(t)   = 3 + 4 sin(pi t) - 1.2 t
#   subject dev   g_j(t) = a_j sin(pi t) + b_j sin(2 pi t)
#   a_j ~ N(0, 0.7^2),  b_j ~ N(0, 0.5^2),  eps ~ N(0, 0.35^2)
#   y_ij = f(t_i) + g_j(t_i) + eps_ij
# Columns: t, y, subject, f_pop (population mean), dev_true (subject deviation).
# NOTE: `subject` is stored as 1.0, 2.0, ... because the array literal below
# promotes the Int vector to Float64 to match its siblings. That matches
# data_gaussian_gamm.csv; "fixing" it to integers changes the file's bytes and
# drifts vignette 10's numbers. Full precision, no r6 rounding, for the same
# reason (r6 is local to the 15-series generators, not a file-wide convention).
function gen_fs_trajectories()
    rng = MersenneTwister(20260501)
    n_subj, n_per = 15, 25
    tgrid = collect(range(0.02, 0.98; length = n_per))
    subj = repeat(1:n_subj, inner = n_per)
    t = repeat(tgrid, outer = n_subj)
    # one vectorized pass per variable (see README on reproducibility)
    a = 0.7 .* randn(rng, n_subj)
    b = 0.5 .* randn(rng, n_subj)
    eps = 0.35 .* randn(rng, n_subj * n_per)
    f_pop = 3.0 .+ 4.0 .* sin.(pi .* t) .- 1.2 .* t
    dev = a[subj] .* sin.(pi .* t) .+ b[subj] .* sin.(2pi .* t)
    y = f_pop .+ dev .+ eps
    write_csv(joinpath(VIGNETTES, "10_gamm", "data_fs_trajectories.csv"),
        ["t" => t, "y" => y, "subject" => subj,
         "f_pop" => f_pop, "dev_true" => dev])
end

# ── 12_nested_effects: data_si.csv, data_expsm.csv ──────────────────────────
# Single-index data: u = X·a with a ∝ (0.7, 0.5, 0.2) (unit norm),
#   y = sin(1.5·u) + ε,  ε ~ N(0, 0.2²)
# Exponential-smoothing data: s̃ᵢ = ω s̃ᵢ₋₁ + (1−ω) xᵢ with ω = 0.8,
#   y = sin(2·s̃/sd(s̃)) + ε,  ε ~ N(0, 0.15²)
function gen_nested()
    rng = MersenneTwister(20260823)
    n = 400
    X = randn(rng, n, 3)
    a = [0.7, 0.5, 0.2]
    a ./= sqrt(sum(abs2, a))
    u = X * a
    f = sin.(1.5 .* u)
    y = f .+ 0.2 .* randn(rng, n)
    write_csv(joinpath(VIGNETTES, "12_nested_effects", "data_si.csv"),
        ["y" => y, "l1" => X[:, 1], "l2" => X[:, 2], "l3" => X[:, 3],
         "u_true" => u, "f_true" => f])

    n2 = 600
    x = randn(rng, n2)
    ω = 0.8
    st = similar(x)
    st[1] = x[1]
    for i in 2:n2
        st[i] = ω * st[i - 1] + (1 - ω) * x[i]
    end
    sdst = sqrt(sum(abs2, st .- sum(st) / n2) / (n2 - 1))
    y2 = sin.(2 .* st ./ sdst) .+ 0.15 .* randn(rng, n2)
    write_csv(joinpath(VIGNETTES, "12_nested_effects", "data_expsm.csv"),
        ["y" => y2, "x" => x, "st_true" => st])
end

# ── 13_migrating_from_mgcv: data.csv ────────────────────────────────────────
# Gaussian smooth used for the side-by-side mgcv comparison:
#   y = sin(x) + ε,  x on a regular grid over [0, 2π],  ε ~ N(0, 0.3²)
# Matches the reference model used by the package's R-parity tests.
function gen_migration()
    rng = MersenneTwister(20260824)
    n = 200
    x = collect(range(0, 2π; length = n))
    y = sin.(x) .+ 0.3 .* randn(rng, n)
    write_csv(joinpath(VIGNETTES, "13_migrating_from_mgcv", "data.csv"),
        ["y" => y, "x" => x])
end

# ── 14_model_selection: data.csv ────────────────────────────────────────────
# Gu & Wahba four-term example with one null smooth, plus a deliberately
# concurve covariate and a gross outlier for the diagnostics workflow:
#   f0(x) = 2 sin(πx), f1(x) = exp(2x),
#   f2(x) = 0.2 x^11 (10(1−x))^6 + 10 (10x)^3 (1−x)^10, f3(x) = 0
#   x4 = x1 + small noise  (near-duplicate of x1, for concurvity)
#   y = f0 + f1 + f2 + ε,  ε ~ N(0, 2²);  observation 100 shifted by +15
function gen_model_selection()
    rng = MersenneTwister(2024)
    n = 400
    x0 = rand(rng, n); x1 = rand(rng, n); x2 = rand(rng, n); x3 = rand(rng, n)
    x4 = x1 .+ 0.03 .* randn(rng, n)
    f0(x) = 2 * sin(π * x)
    f1(x) = exp(2 * x)
    f2(x) = 0.2 * x^11 * (10 * (1 - x))^6 + 10 * (10 * x)^3 * (1 - x)^10
    y = f0.(x0) .+ f1.(x1) .+ f2.(x2) .+ 2.0 .* randn(rng, n)
    y[100] += 15.0   # gross outlier for the influence section
    write_csv(joinpath(VIGNETTES, "14_model_selection", "data.csv"),
        ["y" => y, "x0" => x0, "x1" => x1, "x2" => x2, "x3" => x3, "x4" => x4])
end

# ── 14_model_selection: data_ar1.csv ────────────────────────────────────────
# Smooth mean plus strongly autocorrelated errors, for the NCV section:
#   y_i = 2 sin(2πx_i) + u_i,  u_i = 0.9 u_{i-1} + 0.6 ε_i,  ε ~ N(0,1)
# u is started from its stationary distribution so the series is stationary
# throughout rather than burning in. The true mean has ~3 effective df, which
# GCV badly overestimates on correlated data and NCV recovers.
function gen_ar1()
    rng = MersenneTwister(20260901)
    n = 300
    rho, sigma = 0.9, 0.6
    x = collect(range(0, 1; length = n))
    f = 2 .* sin.(2π .* x)
    e = randn(rng, n)                    # one vectorized pass
    u = similar(e)
    u[1] = e[1] * sigma / sqrt(1 - rho^2)
    for i in 2:n
        u[i] = rho * u[i - 1] + sigma * e[i]
    end
    write_csv(joinpath(VIGNETTES, "14_model_selection", "data_ar1.csv"),
        ["x" => x, "y" => f .+ u, "f_true" => f])
end

# ── 15_large_and_spatial: data_large.csv, data_mrf.csv, nb.csv, data_sphere.csv
#
# (a) data_large.csv — a plain additive Gaussian model, large enough to show
#     the `gam` → `bam` crossover:
#       f1(x) = sin(2πx), f2(x) = x², f3(x) = exp(-8(x-0.4)²)
#       y = f1(x1) + f2(x2) + f3(x3) + ε,  ε ~ N(0, 0.5²),  n = 20000
#
# (b) data_mrf.csv + nb.csv — areal data on a 6×6 rook-contiguity lattice.
#     Region r(i,j) has spatial effect g(i,j) = sin(0.9i) + cos(0.7j), and
#       y = g(region) + f(z) + ε,  f(z) = 1.5 sin(πz),  ε ~ N(0, 0.4²)
#     with 20 observations per region (n = 720). `nb.csv` is the 36×36
#     rook adjacency matrix, rows/columns ordered r01 … r36, read by both
#     the Julia vignette and its R companion.
#
# (c) data_sphere.csv — n = 800 points uniform on the sphere (lon uniform,
#     sin(lat) uniform), latitude/longitude stored in DEGREES:
#       y = f(lat, lon) + ε,  f = 2 sin(2 lat_rad) cos(lon_rad) + cos(lat_rad),
#       ε ~ N(0, 0.3²)
function gen_large_spatial()
    dir = joinpath(VIGNETTES, "15_large_and_spatial")
    isdir(dir) || mkpath(dir)
    r6(v) = round.(v; digits = 6)

    # (a) large additive data
    rng = MersenneTwister(20260815)
    n = 20_000
    x1 = rand(rng, n); x2 = rand(rng, n); x3 = rand(rng, n)
    f1(x) = sin(2π * x)
    f2(x) = x^2
    f3(x) = exp(-8 * (x - 0.4)^2)
    y = f1.(x1) .+ f2.(x2) .+ f3.(x3) .+ 0.5 .* randn(rng, n)
    write_csv(joinpath(dir, "data_large.csv"),
        ["y" => r6(y), "x1" => r6(x1), "x2" => r6(x2), "x3" => r6(x3)])

    # (b) MRF lattice
    rng = MersenneTwister(20260816)
    nrow_grid = 6; ncol_grid = 6
    labels = String[]
    gi = Int[]; gj = Int[]
    for i in 1:nrow_grid, j in 1:ncol_grid
        push!(labels, "r" * lpad((i - 1) * ncol_grid + j, 2, '0'))
        push!(gi, i); push!(gj, j)
    end
    nreg = length(labels)
    A = zeros(Int, nreg, nreg)
    for a in 1:nreg, b in 1:nreg
        if a != b && abs(gi[a] - gi[b]) + abs(gj[a] - gj[b]) == 1
            A[a, b] = 1
        end
    end
    write_csv(joinpath(dir, "nb.csv"),
        [labels[c] => A[:, c] for c in 1:nreg])

    g = [sin(0.9 * gi[r]) + cos(0.7 * gj[r]) for r in 1:nreg]
    per = 20
    region = repeat(labels; inner = per)
    reg_idx = repeat(1:nreg; inner = per)
    nobs = nreg * per
    z = rand(rng, nobs)
    ymrf = g[reg_idx] .+ 1.5 .* sin.(π .* z) .+ 0.4 .* randn(rng, nobs)
    write_csv(joinpath(dir, "data_mrf.csv"),
        ["y" => r6(ymrf), "region" => region, "z" => r6(z),
         "g_true" => r6(g[reg_idx])])

    # (c) spherical data (degrees)
    rng = MersenneTwister(20260817)
    ns = 800
    lon_rad = 2π .* rand(rng, ns) .- π
    lat_rad = asin.(2 .* rand(rng, ns) .- 1)
    fsph = 2 .* sin.(2 .* lat_rad) .* cos.(lon_rad) .+ cos.(lat_rad)
    ysph = fsph .+ 0.3 .* randn(rng, ns)
    write_csv(joinpath(dir, "data_sphere.csv"),
        ["y" => r6(ysph), "lat" => r6(rad2deg.(lat_rad)),
         "lon" => r6(rad2deg.(lon_rad)), "f_true" => r6(fsph)])
end

# ── 02_basis_types: data_shrink.csv ─────────────────────────────────────────
# Shrinkage-basis demo: x drives the response, z is irrelevant.
#   y = sin(2πx) + ε,  ε ~ N(0, 0.4²);  z ~ U(0,1) enters no model.
function gen_shrink()
    rng = MersenneTwister(20260830)
    n = 400
    x = rand(rng, n)
    z = rand(rng, n)
    e = randn(rng, n)
    y = sin.(2π .* x) .+ 0.4 .* e
    write_csv(joinpath(VIGNETTES, "02_basis_types", "data_shrink.csv"),
        ["x" => x, "z" => z, "y" => y])
end

# ── 02_basis_types: data_adaptive.csv ───────────────────────────────────────
# Adaptive-smooth demo: flat on the left, oscillating on the right.
#   f(x) = sin(10πx) / (1 + exp(-40(x - 0.5))),  y = f(x) + ε,  ε ~ N(0, 0.15²)
function gen_adaptive()
    rng = MersenneTwister(20260831)
    n = 400
    x = sort(rand(rng, n))
    e = randn(rng, n)
    gate = 1 ./ (1 .+ exp.(-40 .* (x .- 0.5)))
    f = gate .* sin.(10π .* x)
    y = f .+ 0.15 .* e
    write_csv(joinpath(VIGNETTES, "02_basis_types", "data_adaptive.csv"),
        ["x" => x, "y" => y, "f_true" => f])
end

# ── 04_families: data_incidence.csv ─────────────────────────────────────────
# District-week case counts with a population offset:
#   y ~ Poisson(μ),  log μ = log(pop) + β₀ + f(x),  β₀ = -6.0,
#   f(x) = 0.9 sin(2πx),  log pop ~ N(9, 0.7²)
function gen_incidence()
    rng = MersenneTwister(20260830)
    n = 400
    x = rand(rng, n)
    logpop = 9.0 .+ 0.7 .* randn(rng, n)
    pop = round.(Int, exp.(logpop))
    μ = exp.(-6.0 .+ 0.9 .* sin.(2π .* x)) .* pop
    y = [rand(rng, Poisson(m)) for m in μ]
    write_csv(joinpath(VIGNETTES, "04_families", "data_incidence.csv"),
        ["x" => x, "pop" => pop, "y" => y])
end

# ── 04_families: data_incidence_od.csv ──────────────────────────────────────
# Weekly counts at a single surveillance site (constant population, so no
# offset), with genuine extra-Poisson variation:
#   y ~ NegBin(μ, θ),  log μ = 2.5 + 0.9 sin(2πx),  θ = 2.0
function gen_incidence_od()
    rng = MersenneTwister(20260831)
    n = 400
    θ = 2.0
    x = rand(rng, n)
    μ = exp.(2.5 .+ 0.9 .* sin.(2π .* x))
    y = [rand(rng, NegativeBinomial(θ, θ / (θ + m))) for m in μ]
    write_csv(joinpath(VIGNETTES, "04_families", "data_incidence_od.csv"),
        ["x" => x, "y" => y])
end

# ── 16_seasonality ──────────────────────────────────────────────────────────
# Seasonal shape, period 52 weeks. Two harmonics so the curve is not a plain
# sinusoid (a cyclic basis should have something to do). Periodic by
# construction: season(0) == season(52).
season(w) = sin(2π * w / 52) + 0.35 * sin(4π * w / 52)

# 16_seasonality: data_season.csv
# Single-site weekly vector-abundance surveillance, 8 years x 53 weeks.
# Weeks are indexed 0..52, where week 0 and week 52 both mark the turn of the
# year — the same point in the seasonal cycle. That makes the observed range
# of `week` equal to the period, which is what a cyclic basis assumes (see the
# `knots=` limitation noted in the vignette).
#   log abundance y = 3.0 + A(year)*season(week) + trend(t) + e,  e ~ N(0, 0.25^2)
#   A(year) = 0.8 + 0.10*(year - 1)     (seasonal amplitude grows over time)
#   trend(t) = 0.6 * (t/t_max)^1.5      (accelerating multi-year increase)
function gen_season()
    rng = MersenneTwister(20240601)
    nyear = 8
    weeks = repeat(0:52, outer = nyear)
    years = repeat(1:nyear, inner = 53)
    t = (years .- 1) .* 52 .+ weeks
    t_max = maximum(t)

    amp = 0.8 .+ 0.10 .* (years .- 1)
    trend = 0.6 .* (t ./ t_max) .^ 1.5
    mu = 3.0 .+ amp .* season.(weeks) .+ trend
    eps = 0.25 .* randn(rng, length(weeks))      # one vectorized pass
    y = mu .+ eps

    df = DataFrame(week = weeks, year = years, t = t,
        y = y, mu_true = mu, amp_true = amp)
    CSV.write(joinpath(VIGNETTES, "16_seasonality", "data_season.csv"), df)
    println("wrote $(joinpath(VIGNETTES, "16_seasonality", "data_season.csv"))")
end

# 16_seasonality: data_region.csv
# Three-region weekly surveillance, 6 years x 53 weeks per region.
# Regions differ in BOTH mean level and seasonal amplitude:
#   coastal  level 3.4, amplitude 1.40   (strong seasonality)
#   inland   level 3.0, amplitude 0.90   (moderate)
#   highland level 2.6, amplitude 0.35   (weak)
# Rainfall (standardized) acts with a coefficient that itself varies through
# the season, beta(week) = 0.45*sin(2*pi*(week - 8)/52), so its effect is a
# varying-coefficient term rather than a constant slope.
#   y = level_r + amp_r*season(week) + beta(week)*rainfall + e, e ~ N(0, 0.25^2)
function gen_region()
    rng = MersenneTwister(20240602)
    regions = ["coastal", "inland", "highland"]
    levels = Dict("coastal" => 3.4, "inland" => 3.0, "highland" => 2.6)
    amps = Dict("coastal" => 1.40, "inland" => 0.90, "highland" => 0.35)
    nyear = 6

    nper = 53 * nyear
    region = repeat(regions, inner = nper)
    weeks = repeat(repeat(0:52, outer = nyear), outer = length(regions))
    years = repeat(repeat(1:nyear, inner = 53), outer = length(regions))

    n = length(weeks)
    rainfall = randn(rng, n)                      # one vectorized pass
    eps = 0.25 .* randn(rng, n)                   # one vectorized pass

    beta = 0.45 .* sin.(2π .* (weeks .- 8) ./ 52)
    lvl = [levels[r] for r in region]
    amp = [amps[r] for r in region]
    mu = lvl .+ amp .* season.(weeks) .+ beta .* rainfall
    y = mu .+ eps

    df = DataFrame(week = weeks, year = years, region = region,
        rainfall = rainfall, y = y, mu_true = mu,
        amp_true = amp, beta_true = beta)
    CSV.write(joinpath(VIGNETTES, "16_seasonality", "data_region.csv"), df)
    println("wrote $(joinpath(VIGNETTES, "16_seasonality", "data_region.csv"))")
end


gen_2d()
gen_gev()
gen_gpd()
gen_cx()
gen_micv()
gen_gaussian_gamm()
gen_poisson_gamm()
gen_fs_trajectories()
gen_nested()
gen_migration()
gen_model_selection()
gen_ar1()
gen_large_spatial()
gen_shrink()
gen_adaptive()
gen_incidence()
gen_incidence_od()
gen_season()
gen_region()
println("done")
