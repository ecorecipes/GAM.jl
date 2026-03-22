using Test
using GAM
using CSV
using DataFrames
using StableRNGs
using Statistics
using StatsAPI

@testset "Side Constraints vs mgcv" begin
    refdir = joinpath(@__DIR__, "r_comparison")
    data_file = joinpath(refdir, "side_data.csv")

    if !isfile(data_file)
        @warn "R reference data not found. Run: Rscript test/r_comparison/side_constraints.R"
        return
    end

    df = CSV.read(data_file, DataFrame)
    data = (; y = df.y, x = df.x, z = df.z)

    @testset "Model 1: s(x) + s(x,z) — 1d+2d overlap" begin
        ref_sm = CSV.read(joinpath(refdir, "side_m1_smooths.csv"), DataFrame)
        ref_sum = CSV.read(joinpath(refdir, "side_m1_summary.csv"), DataFrame)
        ref_fit = CSV.read(joinpath(refdir, "side_m1_fitted.csv"), DataFrame)

        f = GAM.@gam_formula(y ~ s(x, k = 8) + s(x, z, k = 25))
        m = gam(f, data)

        # Column counts should match R
        @test size(m.smooths[1].X, 2) == ref_sm.ncol[1]  # s(x): 7
        @test size(m.smooths[2].X, 2) == ref_sm.ncol[2]  # s(x,z): 23

        # del_index: s(x) untouched, s(x,z) has 1 removed
        @test isempty(m.smooths[1].del_index)
        @test length(m.smooths[2].del_index) == 1

        # Total coefficients match
        @test length(StatsAPI.coef(m)) == ref_sum.total_ncoef[1]

        # Fitted values correlate well with R
        @test cor(m.fitted_values, ref_fit.fitted) > 0.99
    end

    @testset "Model 2: s(x) + s(z) + te(x,z) — tensor with side constraints" begin
        ref_sm = CSV.read(joinpath(refdir, "side_m2_smooths.csv"), DataFrame)
        ref_fit = CSV.read(joinpath(refdir, "side_m2_fitted.csv"), DataFrame)

        # Julia te(x,z,k=25) matches R te(x,z,k=c(5,5)) in total basis size
        f = GAM.@gam_formula(y ~ s(x, k = 8) + s(z, k = 8) + te(x, z, k = 25))
        m = gam(f, data)

        # s(x) and s(z) untouched
        @test isempty(m.smooths[1].del_index)
        @test isempty(m.smooths[2].del_index)

        # te(x,z) should have 2 columns removed (same as R)
        @test length(m.smooths[3].del_index) == 2
        @test size(m.smooths[3].X, 2) == ref_sm.ncol[3]  # 22

        # Fitted values correlate well
        @test cor(m.fitted_values, ref_fit.fitted) > 0.99
    end

    @testset "Model 3: s(x) + s(z) + ti(x,z) — no removal" begin
        ref_fit = CSV.read(joinpath(refdir, "side_m3_fitted.csv"), DataFrame)

        # Julia ti() has different k split from R ti(k=c(5,5)),
        # but the key behavior is the same: NO columns are removed
        f = GAM.@gam_formula(y ~ s(x, k = 8) + s(z, k = 8) + ti(x, z, k = 25))
        m = gam(f, data)

        # No columns removed from any smooth
        for sm in m.smooths
            @test isempty(sm.del_index)
        end

        # Fitted values correlate reasonably with R despite different ti() dimensions
        # (Julia ti(k=25) → 8 cols vs R ti(k=c(5,5)) → 16 cols)
        @test cor(m.fitted_values, ref_fit.fitted) > 0.85
    end
end
