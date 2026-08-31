@testset "sz configurable base basis matches mgcv" begin
    using DataFrames, CSV, Statistics
    using StatsAPI: fitted, deviance

    # mgcv lets an `sz` smooth pick the basis for its continuous marginal
    # (`smooth.construct.sz.smooth.spec`, mgcv 1.9-4):
    #
    #   if (is.null(object$xt)) object$base.bs <- "tp"
    #   else if (is.list(object$xt)) {
    #     if (is.null(object$xt$bs)) object$base.bs <- "tp"
    #     else object$base.bs <- object$xt$bs
    #   } else { object$base.bs <- object$xt; object$xt <- NULL }
    #
    # GAM.jl's `xt` is a Dict, so the list form maps: `xt = Dict(:bs => :cr)`.
    #
    # Compared at FREE sp, deliberately. As `test_sz_rcall.jl` records, the two
    # packages use different sum-to-zero contrast bases for the level index, so
    # `sz` smoothing parameters are on different scales and do not transfer —
    # a fixed-sp comparison would measure that mismatch rather than the base
    # basis. Confirmed while writing this: at sp = 0.5 the `:ps` marginal gives
    # edf 15.79 here against mgcv's 20.78, while the FREE fits below agree to
    # 2e-4. The standalone `s(x, bs=:ps)` smooth matches mgcv exactly at fixed
    # sp (edf 10.135189, deviance 371.506004 in both), which is what pins the
    # difference to sz's contrast rather than to the P-spline basis.

    df_sz = CSV.read(joinpath(@__DIR__, "..", "vignettes", "16_seasonality",
        "data_region.csv"), DataFrame)

    @rput df_sz
    R"""
    suppressMessages(library(mgcv))
    df_sz$region <- factor(df_sz$region)
    """

    for b in (:cr, :ps, :cc)
        bs_str = String(b)
        @rput bs_str
        R"""
        m_r <- gam(y ~ s(week, region, bs = "sz", xt = list(bs = bs_str), k = 12),
                   data = df_sz, method = "REML")
        r_ncol <- m_r$smooth[[1]]$last.para - m_r$smooth[[1]]$first.para + 1
        r_npen <- length(m_r$smooth[[1]]$S)
        # NOTE: sum(pen.edf(m)) returns one value PER PENALTY, so it triple
        # counts a three-penalty sz term (47.26 on a 24-column basis). The
        # s.table row is the per-term edf.
        r_edf  <- summary(m_r)$s.table[1, "edf"]
        r_dev  <- deviance(m_r)
        """
        r_ncol = Int(@rget r_ncol)
        r_npen = Int(@rget r_npen)
        r_edf = Float64(@rget r_edf)
        r_dev = Float64(@rget r_dev)

        spec = GAM.s(:week, :region; k = 12, bs = :sz,
            xt = Dict{Symbol, Any}(:bs => b))
        m_jl = gam(GAM.GamFormula(:y, Symbol[], true, [spec]), df_sz;
            method = :REML)
        sm = m_jl.smooths[1]

        # Structure must match exactly — these are integers, not estimates.
        @test size(sm.X, 2) == r_ncol
        @test length(GAM.penalty_matrices(sm)) == r_npen

        # edf and deviance agree to the level of two optimizers stopping at
        # slightly different points on a flat criterion.
        @test isapprox(sum(edf(m_jl)), r_edf; atol = 5e-3)
        @test isapprox(deviance(m_jl), r_dev; rtol = 1e-5)
    end

    # The default must equal an explicit "tp" in BOTH packages — that is the
    # claim that this is an added option, not a behaviour change.
    R"""
    m_def <- gam(y ~ s(week, region, bs = "sz", k = 12), data = df_sz, method = "REML")
    m_tp  <- gam(y ~ s(week, region, bs = "sz", xt = list(bs = "tp"), k = 12),
                 data = df_sz, method = "REML")
    d_def <- deviance(m_def); d_tp <- deviance(m_tp)
    """
    @test isapprox(Float64(@rget d_def), Float64(@rget d_tp); rtol = 1e-12)

    spec_def = GAM.s(:week, :region; k = 12, bs = :sz)
    spec_tp = GAM.s(:week, :region; k = 12, bs = :sz,
        xt = Dict{Symbol, Any}(:bs => :tp))
    m_def = gam(GAM.GamFormula(:y, Symbol[], true, [spec_def]), df_sz; method = :REML)
    m_tp = gam(GAM.GamFormula(:y, Symbol[], true, [spec_tp]), df_sz; method = :REML)
    @test deviance(m_def) == deviance(m_tp)
end
