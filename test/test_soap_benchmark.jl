# Soap film (`bs=:so`) accuracy on the standard benchmark.
#
# GAM.jl's soap film is a DIFFERENT construction from mgcv's, not a degraded
# one. Both solve the soap PDE on a square-celled grid with a sparse LU and a
# cyclic boundary basis; they differ in specific ways (see the notes at the
# `SoapFilm` constructor). Because they differ, `bs=:so` is the one basis whose
# fits are not expected to match mgcv elementwise, so ordinary parity testing
# does not apply to it.
#
# What CAN be tested — and is what a user actually cares about — is whether the
# construction recovers a known function well. This pins that against the
# canonical benchmark for the problem: Ramsay's (2002) horseshoe domain, which
# mgcv ships as `fs.test`/`fs.boundary` and which exists precisely because
# ordinary bivariate smoothers leak across the gap between the horseshoe arms.
#
# Measured against mgcv 1.9-4 on the same simulated data across five seeds,
# with mgcv given the interior knots it requires:
#
#     seed   mgcv RMSE   GAM.jl RMSE   mgcv edf   GAM.jl edf
#        1     0.10161       0.07604      40.90        17.29
#        2     0.11151       0.09010      37.85        17.45
#        3     0.09434       0.07306      48.15        20.69
#        4     0.11064       0.08280      42.54        18.33
#        5     0.10053       0.07862      38.70        14.50
#
# GAM.jl was more accurate on every seed (mean RMSE 0.0801 against 0.1037, a
# 23% reduction) using roughly 43% of the effective degrees of freedom. The
# tolerance below is set at mgcv's own level rather than loosely above GAM.jl's,
# so the test fails if this basis ever regresses to merely matching mgcv.
#
# The RNG here is StableRNG, so these numbers are not the R-seeded ones above;
# measured values are RMSE 0.0577-0.0756 and edf 15.4-17.7 across the seeds used.

@testset "Soap film accuracy on the horseshoe benchmark" begin

    # mgcv's `fs.boundary` (R/soap.r:853-869), a closed form.
    function _fs_boundary(; r0 = 0.1, r = 0.5, l = 3.0, n_theta = 20)
        rr = r + (r - r0)
        th = range(π, π / 2; length = n_theta)
        x = rr .* cos.(th);  y = rr .* sin.(th)
        th = range(π / 2, -π / 2; length = 2n_theta)
        x = vcat(x, (r - r0) .* cos.(th) .+ l)
        y = vcat(y, (r - r0) .* sin.(th) .+ r)
        th = range(π / 2, π; length = n_theta)
        x = vcat(x, r0 .* cos.(th));  y = vcat(y, r0 .* sin.(th))
        return vcat(x, reverse(x)), vcat(y, -reverse(y))
    end

    # mgcv's `fs.test` (R/soap.r:818-850). Returns the field and the exclusion
    # mask; points outside the horseshoe are excluded rather than evaluated.
    function _fs_test(x, y; r0 = 0.1, r = 0.5, l = 3.0, b = 1.0)
        q = π * r / 2
        a = zero(x);  d = zero(x);  ex = falses(length(x))
        for i in eachindex(x)
            if x[i] >= 0 && y[i] > 0
                a[i] = q + x[i];         d[i] = y[i] - r
            elseif x[i] >= 0 && y[i] <= 0
                a[i] = -q - x[i];        d[i] = -r - y[i]
            else
                a[i] = -atan(y[i] / x[i]) * r
                d[i] = sqrt(x[i]^2 + y[i]^2) - r
            end
            ex[i] = abs(d[i]) > r - r0 ||
                    (x[i] > l && (x[i] - l)^2 + d[i]^2 > (r - r0)^2)
        end
        return a .* b .+ d .^ 2, ex
    end

    bx, by = _fs_boundary()
    bnd = [hcat(bx, by)]

    # The boundary must be a closed horseshoe, not a degenerate blob — if this
    # were wrong every assertion below would still "pass" on an easy domain.
    @test length(bx) == length(by)
    @test length(bx) > 100
    @test maximum(bx) > 3.0            # the arms extend to x ~ 3.4
    @test minimum(by) < -0.8 && maximum(by) > 0.8

    for seed in (11, 12, 13)
        rng = StableRNG(seed)
        N = 900
        xs = rand(rng, N) .* 5 .- 1
        ys = rand(rng, N) .* 2 .- 1
        truth, excl = _fs_test(xs, ys)
        keep = .!excl .& isfinite.(truth)
        xs, ys, truth = xs[keep], ys[keep], truth[keep]
        z = truth .+ 0.3 .* randn(rng, length(truth))
        df_fs = DataFrame(x = xs, y = ys, z = z)

        @test nrow(df_fs) > 400        # enough interior points to be a real fit

        spec = GAM.s(:x, :y; bs = :so, k = 41,
                     xt = Dict{Symbol, Any}(:bnd => bnd))
        gf = GAM.GamFormula(:z, Symbol[], true, GAM.SmoothSpec[spec])
        m = gam(gf, df_fs; method = :REML)

        @test m.converged
        rmse = sqrt(mean((fitted(m) .- truth) .^ 2))

        # mgcv attains ~0.10 here. Failing above that means this basis has
        # regressed to no better than the implementation it deliberately
        # differs from, which is exactly what we want to hear about.
        @test rmse < 0.10

        # Two smoothing parameters: the cyclic boundary penalty and the soap
        # interior penalty. mgcv reports two as well.
        @test length(m.sp) == 2

        # Genuinely penalised, not interpolating: measured 15.4-17.7 of 40
        # smooth columns. A fit that saturated would still hit the RMSE bound
        # on noise-free-ish data, so pin the flexibility too.
        e = sum(edf(m))
        @test 8.0 < e < 30.0
    end
end
