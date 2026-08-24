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
    x = Float64[]; y = Int[]; site = Int[]; eta_true = Float64[]; re_true = Float64[]
    for j in 1:n_site, _ in 1:n_per
        xi = rand(rng, Uniform(-π, π))
        η = 1.0 + 0.8 * sin(xi) + b[j]
        push!(x, xi); push!(site, j); push!(eta_true, η); push!(re_true, b[j])
        push!(y, rand(rng, Poisson(exp(η))))
    end
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

gen_gpd()
gen_cx()
gen_micv()
gen_gaussian_gamm()
gen_poisson_gamm()
gen_nested()
gen_migration()
gen_model_selection()
println("done")
