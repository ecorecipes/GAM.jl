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
#   01-05, 08, 09, 11 data files; 06 data_gev.csv; 07 data.csv;
#   10 data_repeated_measures.csv (unused by the current vignette).

using Random
using Distributions
using Printf

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

gen_gpd()
gen_cx()
gen_micv()
gen_gaussian_gamm()
gen_poisson_gamm()
gen_nested()
gen_migration()
gen_model_selection()
gen_large_spatial()
println("done")
