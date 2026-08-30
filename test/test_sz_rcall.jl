@testset "sz penalty structure matches mgcv" begin
    using DataFrames, CSV, Statistics
    using StatsAPI: fitted, deviance

    # mgcv gives an `sz` term one penalty (hence one smoothing parameter) per
    # factor level by default — `R/smooth.r:2281-2286`. GAM.jl emitted a single
    # summed penalty until this was fixed, which forced every level to share
    # one lambda and inflated the deviation term's edf by 43% (14.63 against
    # mgcv's 10.24) on the model below.
    #
    # What is compared, and what deliberately is not: the two packages use
    # DIFFERENT sum-to-zero contrast bases for the level index — GAM.jl absorbs
    # an orthonormal `Q_L` at construction, mgcv applies its `XZKr` contrast
    # afterwards (`R/smooth.r:4139-4147`). The bases are rotation-equivalent,
    # so fitted values, edf and deviance agree, but the smoothing parameters
    # live on different scales and are NOT comparable elementwise (measured
    # ratios 10280 / 5193 / 830 across the three levels — not even a constant
    # rescale). So this asserts on the penalty COUNT, the edf, the deviance and
    # the fitted values, never on sp itself.

    df = CSV.read(joinpath(@__DIR__, "..", "vignettes", "16_seasonality",
                           "data_region.csv"), DataFrame)
    levs = sort(unique(df.region))
    L = length(levs)

    m = gam(@formulak(y ~ s(week, k = 12, bs = :cc) +
                          s(week, region, k = 12, bs = :sz)), df)

    RCall.reval("suppressMessages(library(mgcv))")
    wk = df.week; rg = String.(df.region); yy = df.y
    @rput wk rg yy
    RCall.reval("""
        dsz <- data.frame(week = wk, region = factor(rg), y = yy)
        msz <- gam(y ~ s(week, k = 12, bs = "cc") +
                       s(week, region, k = 12, bs = "sz"),
                   data = dsz, method = "REML")
        r_nsp  <- length(msz\$sp)
        r_edf  <- as.numeric(summary(msz)\$s.table[, "edf"])
        r_dev  <- deviance(msz)
        r_fit  <- as.numeric(fitted(msz))
        r_ncf  <- length(coef(msz))
    """)
    r_nsp = Int(first(rcopy(RCall.reval("r_nsp"))))
    r_edf = rcopy(RCall.reval("r_edf"))
    r_dev = Float64(first(rcopy(RCall.reval("r_dev"))))
    r_fit = rcopy(RCall.reval("r_fit"))
    r_ncf = Int(first(rcopy(RCall.reval("r_ncf"))))

    # One sp for the cyclic smooth plus one per level, in both packages.
    @test length(m.sp) == r_nsp
    @test length(m.sp) == 1 + L

    # Same basis: identical coefficient count.
    @test length(coef(m)) == r_ncf

    # The cyclic term was already correct and stays so; the deviation term is
    # what the fix moved, from 14.63 to mgcv's value.
    e = edf(m)
    @test isapprox(e[1], r_edf[1]; atol = 5e-3)
    @test isapprox(e[2], r_edf[2]; atol = 5e-2)
    @test e[2] < 11.0        # fails loudly on the old single-penalty code

    @test isapprox(deviance(m), r_dev; rtol = 1e-4)

    # Fitted values agree to ~1e-5 of the fitted range.
    f = fitted(m)
    rng = maximum(f) - minimum(f)
    @test maximum(abs.(f .- r_fit)) / rng < 1e-3
    @test cor(f, r_fit) > 0.999999
end
