@testset "Discrete design vs mgcv::bam(discrete=TRUE)" begin
    # Tolerances here are calibrated from measurement, not chosen a priori.
    # On this configuration (n = 50,000, two `cr` smooths at k = 20):
    #
    #   GAM.jl discrete vs GAM.jl dense   1.39e-3   (of the fitted range)
    #   GAM.jl discrete vs mgcv discrete  4.25e-4
    #   GAM.jl dense    vs mgcv dense     7.33e-4
    #   mgcv   discrete vs mgcv dense     2.32e-3
    #
    # i.e. the two packages' discrete fits agree with each other more closely
    # than either agrees with its own dense fit — which is the right shape,
    # since both are approximating by rounding covariates onto the same kind
    # of grid. Note mgcv's discrete path also switches optimiser to `fREML`
    # (`bam.r:430-895`), so this is not a like-for-like optimiser comparison
    # and smoothing parameters are deliberately not compared elementwise.
    rng_r = StableRNG(21)
    n = 50_000
    x1 = rand(rng_r, n)
    x2 = rand(rng_r, n)
    y = sin.(2π .* x1) .+ x2 .^ 2 .+ 0.3 .* randn(rng_r, n)
    df = DataFrame(x1 = x1, x2 = x2, y = y)
    gf = GAM.GamFormula(:y, Symbol[], true,
        [GAM.s(:x1; k = 20, bs = :cr), GAM.s(:x2; k = 20, bs = :cr)])

    jd = bam(gf, df)
    jk = bam(gf, df; discrete = true)

    R"""
    suppressMessages(library(mgcv))
    d <- data.frame(x1 = $x1, x2 = $x2, y = $y)
    rd <- bam(y ~ s(x1, k = 20, bs = 'cr') + s(x2, k = 20, bs = 'cr'),
              data = d, method = 'REML')
    rk <- bam(y ~ s(x1, k = 20, bs = 'cr') + s(x2, k = 20, bs = 'cr'),
              data = d, method = 'fREML', discrete = TRUE)
    """
    rdf = rcopy(R"fitted(rd)")
    rkf = rcopy(R"fitted(rk)")
    rke = rcopy(R"sum(rk$edf)")
    rng_f = maximum(rdf) - minimum(rdf)

    # Our discrete fit tracks mgcv's discrete fit.
    @test maximum(abs.(fitted(jk) .- rkf)) / rng_f < 2e-3
    @test abs(jk.edf_total - rke) / rke < 0.02

    # ...and does so at least as well as it tracks its own dense fit, which
    # is the property that says the approximation is mgcv's and not ours.
    @test maximum(abs.(fitted(jk) .- rkf)) <=
          2 * maximum(abs.(fitted(jk) .- fitted(jd)))

    # mgcv's own dense-vs-discrete gap bounds what any faithful port can do.
    @test maximum(abs.(fitted(jk) .- fitted(jd))) / rng_f <
          3 * maximum(abs.(rkf .- rdf)) / rng_f
end
