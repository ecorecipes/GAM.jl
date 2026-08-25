@testset "P-IRLS false convergence on heavily halved steps" begin
    # Regression for a fit-level defect found against mgcv 1.9-4: at high
    # dispersion, InverseGaussian + LogLink converged to a point 195.5 away
    # from stationarity while reporting `converged = true`.
    #
    # Mechanism: the Fisher search direction is poor enough there that step
    # halving crushed the step to 2^-23 of its length. The accepted step then
    # moved the penalized deviance by 3.8e-5 (inside gam.fit3's roundoff
    # acceptance threshold of 10(0.1+|pdev|)sqrt(eps) = 6.4e-5), so the
    # RELATIVE deviance criterion read 8.8e-8 < epsilon on iteration 1 and
    # declared success. Deviance 427.73 against mgcv's 299.64.
    #
    # The fix requires a near-full step before the deviance criterion may
    # declare convergence: a heavily halved step is evidence the direction was
    # bad, not that we are at an optimum.
    #
    # mgcv needs no such guard because for non-canonical links it steps with
    # full Newton weights (R/gam.fit3.r:504-513), reaching this fit in 2
    # iterations. Fisher and Newton share the fixed point -- w*(z-eta) is
    # algebraically identical, only the Hessian differs -- so Fisher gets
    # there too, just in ~10 iterations instead of 2.

    rng = StableRNG(11)
    n = 300
    x = collect(range(0.05, 1.0; length = n))
    mu_true = exp.(0.5 .+ 0.8 .* sin.(2π .* x))
    # lambda = 1 is high dispersion for InverseGaussian: var = mu^3/lambda
    y = [rand(rng, InverseGaussian(m, 1.0)) for m in mu_true]
    df = DataFrame(x = x, y = y)

    # sp fixed at mgcv's selected value, so the smoothing-parameter optimizer
    # is out of the comparison and any gap is genuinely in the fit.
    sp_mgcv = 394.2079948166541
    gf = GAM.GamFormula(:y, Symbol[], true,
        [GAM.s(:x; bs = :cr, k = 15, sp = sp_mgcv)])
    m = gam(gf, df; family = InverseGaussian(), link = LogLink())

    @test m.converged

    # Stationarity of the penalized score. For InverseGaussian (V = mu^3) with
    # a log link (g' = 1/mu), the working residual collapses to (y-mu)/mu^2.
    S = GAM.total_penalty(m.penalty, [log(sp_mgcv)], size(m.X, 2))
    b = coef(m)
    mu_hat = exp.(m.X * b)
    score = m.X' * ((y .- mu_hat) ./ (mu_hat .^ 2)) .- S * b
    # Was 195.5 before the fix.
    @test maximum(abs, score) < 1e-2

    # mgcv 1.9-4 reference at the same fixed sp. Was 427.73 / 5.99 before.
    @test m.deviance_val ≈ 299.6395385772956 rtol = 1e-4
    @test isapprox(sum(GAM.edf(m)) + 1, 5.574768396712776; atol = 1e-3)
end

@testset "InverseGaussian link coverage" begin
    # The defect was specific to NON-CANONICAL links at high dispersion.
    # InverseGaussian's canonical link is 1/mu^2, where alpha == 1 identically
    # and Fisher scoring is already full Newton, so it was never affected.
    rng = StableRNG(7)
    n = 200
    x = collect(range(0.05, 1.0; length = n))
    mu_true = exp.(0.5 .+ 0.8 .* sin.(2π .* x))

    for lambda in (0.5, 6.0)
        y = [rand(rng, InverseGaussian(m, lambda)) for m in mu_true]
        df = DataFrame(x = x, y = y)
        for lnk in (LogLink(), IdentityLink())
            gf = GAM.GamFormula(:y, Symbol[], true,
                [GAM.s(:x; bs = :cr, k = 10, sp = 0.01)])
            m = gam(gf, df; family = InverseGaussian(), link = lnk)
            @test m.converged
            @test all(isfinite, fitted(m))
            @test all(>(0), fitted(m))
        end
    end
end
