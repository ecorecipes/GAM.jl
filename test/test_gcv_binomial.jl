# Binomial GCV/UBRE smoothing-parameter selection — R parity regression.
#
# Background. A source audit reported that GAM.jl's binomial GCV/UBRE path
# failed, returning a near-null fit (edf ≈ 2 where mgcv gets ≈ 4.5), blamed on
# an unweighted initial smoothing parameter plus a one-sided Nelder–Mead
# simplex that could not search downward. That claim was tested across 30
# binomial configurations and did NOT reproduce: GAM.jl's `:UBRE` matches
# mgcv's `method="GCV.Cp"` to 4–5 significant figures throughout, including
# cases whose optimum lies more than three decades below the starting value.
#
# The underlying mechanism is nevertheless real but benign:
#
#   * `_initial_sp` (src/penalty.jl) uses an UNweighted diag(X'X), whereas
#     mgcv's `initial.spg` (R/mgcv.r:4602-4605) passes sqrt(w)·X with the
#     IRLS weights at the start value. On the k=10 case below that is
#     log sp = 0.5032 (ours) against −1.1708 (mgcv), a gap of 1.67, with the
#     converged optimum at −1.0748. mgcv therefore starts essentially at the
#     answer while GAM.jl starts ~1.6 above it.
#   * Nelder–Mead's reflection step searches downward perfectly well despite
#     the one-sided initial simplex, so this costs iterations, not accuracy.
#
# These tests pin the agreement so that a genuine regression in either the
# initial value or the simplex would be caught. Reference values are from
# mgcv 1.9-4. Tolerances are loose enough to absorb P-IRLS convergence noise
# but far tighter than the reported failure mode (edf 2.0 vs 4.5) would be.

@testset "Binomial GCV/UBRE selection matches mgcv" begin

    # ---- Case 1: 0/1 response, single smooth -------------------------------
    # mgcv 1.9-4: gam(y ~ s(x, k), family=binomial(), method="GCV.Cp")
    #   k=10  edf 3.7846  dev 368.4873  sp 0.34137
    #   k=15  edf 3.7969  dev 368.4686  sp 0.39720
    let rng = StableRNG(11), n = 300
        x = sort(rand(rng, n))
        f = @. 2 * sin(pi * x) - 1.0
        p = @. 1 / (1 + exp(-f))
        y = Float64.(rand(rng, n) .< p)
        df = DataFrame(x = x, y = y)

        m10 = gam(@formula(y ~ s(x, k = 10)), df;
            family = Binomial(), method = :UBRE)
        @test m10.converged
        @test isapprox(m10.edf_total, 3.7846; atol = 5e-3)
        @test isapprox(deviance(m10), 368.4873; rtol = 1e-4)
        @test isapprox(exp(m10.sp[1]), 0.34137; rtol = 2e-2)

        m15 = gam(@formula(y ~ s(x, k = 15)), df;
            family = Binomial(), method = :UBRE)
        @test isapprox(m15.edf_total, 3.7969; atol = 5e-3)
        @test isapprox(deviance(m15), 368.4686; rtol = 1e-4)

        # The reported failure mode was a collapse to the null space. Guard
        # it directly: an edf near 2 would mean the smooth was flattened out.
        @test m10.edf_total > 3.0
        @test m15.edf_total > 3.0

        # :GCV uses a different criterion from mgcv's known-scale UBRE, so it
        # is not expected to match mgcv numerically — but it must not collapse.
        mg = gam(@formula(y ~ s(x, k = 10)), df;
            family = Binomial(), method = :GCV)
        @test mg.converged
        @test mg.edf_total > 3.0
    end

    # ---- Case 2: optimum far BELOW any plausible starting value -----------
    # This is the configuration that would strand an optimizer unable to
    # search downward: mgcv selects sp = 2.8256e-05, i.e. log sp ≈ −10.5.
    # mgcv 1.9-4: k=10  edf 9.8820  dev 132.617  sp 2.8256e-05
    let rng = StableRNG(4), n = 150
        x = sort(rand(rng, n))
        f = @. 2.5 * sin(8.0 * pi * x)
        p = @. 1 / (1 + exp(-f))
        y = Float64.(rand(rng, n) .< p)
        df = DataFrame(x = x, y = y)

        m = gam(@formula(y ~ s(x, k = 10)), df;
            family = Binomial(), method = :UBRE)
        @test m.converged
        @test isapprox(m.edf_total, 9.8820; atol = 1e-2)
        @test isapprox(deviance(m), 132.617; rtol = 1e-3)
        # The decisive assertion: the optimizer reached a sp three decades
        # below its starting point.
        @test exp(m.sp[1]) < 1e-4
    end

    # ---- Case 3: highly variable prior weights ----------------------------
    # Where an unweighted initial sp would be most misleading: binomial trial
    # counts spanning 2–199 rather than the near-constant 0.19 of 0/1 data.
    # mgcv 1.9-4: k=10  edf 9.2571  dev 251.429  sp 0.066207
    let rng = StableRNG(21), n = 250
        x = sort(rand(rng, n))
        f = @. 2.0 * sin(3 * pi * x)
        p = @. 1 / (1 + exp(-f))
        nt = [rand(rng, 1:200) for _ in 1:n]
        succ = [Float64(sum(rand(rng, nt[i]) .< p[i])) for i in 1:n]
        df = DataFrame(x = x, prop = succ ./ nt)
        w = Float64.(nt)

        m = gam(@formula(prop ~ s(x, k = 10)), df;
            family = Binomial(), method = :UBRE, weights = w)
        @test m.converged
        @test isapprox(m.edf_total, 9.2571; atol = 1e-2)
        @test isapprox(deviance(m), 251.429; rtol = 1e-3)
        @test isapprox(exp(m.sp[1]), 0.066207; rtol = 5e-2)
    end
end
