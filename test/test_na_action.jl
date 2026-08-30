# na.action, weight/offset validation, and construction-time argument checks
#
# mgcv defaults to `na.omit`; GAM.jl defaults to `:fail` (erroring, with a
# message that names the variable) but now offers `:omit` for parity. These
# tests pin both policies, the row bookkeeping that lets a caller realign
# results with the original table, and the argument checks that used to
# surface as `DomainError`/`MethodError` from deep inside the fitters.

using Test
using GAM
using DataFrames
using Random
using Statistics
using StableRNGs
using Distributions: Poisson, Bernoulli
using StatsAPI: fitted, coef, deviance, residuals, predict
using Tables
using StatsModels

# helper: attach lag columns for the gam_nl weight check
function df_with_lags(df, rng)
    d = copy(df)
    d.l1 = rand(rng, nrow(d)); d.l2 = rand(rng, nrow(d)); d.l3 = rand(rng, nrow(d))
    return d
end

@testset "na.action and input validation" begin
    rng = StableRNG(20260824)
    n = 200
    x = rand(rng, n)
    z = rand(rng, n)
    grp = string.(mod.(1:n, 10))
    y = 2 .* sin.(2π .* x) .+ 0.4 .* randn(rng, n)
    df = DataFrame(x = x, z = z, grp = grp, y = y)

    # ------------------------------------------------------------------
    # _apply_na_action / na_omit_rows
    # ------------------------------------------------------------------
    @testset "row selection" begin
        d = DataFrame(y = Union{Float64, Missing}[1.0, 2.0, missing, 4.0, 5.0],
                      x = Union{Float64, Missing}[1.0, NaN, 3.0, missing, 5.0],
                      w = [1.0, 2.0, 3.0, 4.0, Inf],
                      unused = Union{Float64, Missing}[missing, 1, 2, 3, 4])

        # :fail leaves the table alone — the specific validators report it
        t, kept = GAM._apply_na_action(d, :y, [:x], :fail)
        @test kept == collect(1:5)

        # :omit drops rows 2 (NaN x), 3 (missing y), 4 (missing x).
        # Row 1 survives despite `unused` being missing — it is not a model
        # variable. Row 5 survives because `w` was not passed in.
        _, kept = GAM._apply_na_action(d, :y, [:x], :omit)
        @test kept == [1, 5]

        # passing the weights makes the Inf in row 5 disqualifying too
        _, kept = GAM._apply_na_action(d, :y, [:x], :omit; weights = d.w)
        @test kept == [1]

        # na_omit_rows agrees with what the fit would keep
        @test GAM.na_omit_rows(d, :y, [:x]) == [1, 5]
        @test GAM.na_omit_rows(d, :y, [:x]; weights = d.w) == [1]

        # the filtered table keeps every column, subset consistently
        filt, kept = GAM._apply_na_action(d, :y, [:x], :omit)
        @test Tables.columnnames(filt) == Tables.columnnames(Tables.columntable(d))
        @test collect(filt.y) == [1.0, 5.0]

        @test_throws ArgumentError GAM._apply_na_action(d, :y, [:x], :drop)

        # every row incomplete is an error, not an empty fit
        allbad = DataFrame(y = [missing, missing], x = [1.0, 2.0])
        @test_throws ArgumentError GAM._apply_na_action(allbad, :y, [:x], :omit)
    end

    # ------------------------------------------------------------------
    # bam
    # ------------------------------------------------------------------
    @testset "bam na_action" begin
        dfm = copy(df)
        allowmissing!(dfm, :y)
        allowmissing!(dfm, :x)
        dfm.y[3] = missing
        dfm.x[10] = missing
        dfm.z[15] = NaN

        f = @formulak(y ~ s(x, k = 8) + s(z, k = 8))
        @test_throws ArgumentError bam(f, dfm)                    # :fail default
        @test_throws ArgumentError bam(f, dfm; na_action = :fail)

        m = bam(f, dfm; na_action = :omit)
        keep = GAM.na_omit_rows(dfm, :y, [:x, :z])
        @test keep == sort(setdiff(1:n, [3, 10, 15]))
        @test length(fitted(m)) == n - 3

        # the :omit fit equals the fit on the hand-filtered table
        m_ref = bam(f, dfm[keep, :])
        @test coef(m) ≈ coef(m_ref) atol = 1e-12
        @test deviance(m) ≈ deviance(m_ref) atol = 1e-12

        # a missing in a column the model never uses is not a reason to drop
        dfu = copy(df)
        dfu.spare = Vector{Union{Missing, Float64}}(rand(rng, n))
        dfu.spare[4] = missing
        @test length(fitted(bam(f, dfu))) == n

        @test_throws ArgumentError bam(f, dfm; na_action = :sometimes)
    end

    @testset "bam weights and offset validation" begin
        f = @formulak(y ~ s(x, k = 8))

        badw = ones(n); badw[7] = -1.0
        err = try bam(f, df; weights = badw) catch e; e end
        @test err isa ArgumentError
        @test occursin("non-negative", sprint(showerror, err))

        nanw = ones(n); nanw[7] = NaN
        @test_throws ArgumentError bam(f, df; weights = nanw)
        @test_throws ArgumentError bam(f, df; weights = ones(n - 1))

        @test_throws ArgumentError bam(f, df; offset = fill(Inf, n))
        @test_throws ArgumentError bam(f, df; offset = zeros(n - 1))

        # zero weights remain legal (they exclude an observation)
        w0 = ones(n); w0[1:5] .= 0.0
        @test length(fitted(bam(f, df; weights = w0))) == n

        # a `missing` weight is an error under :fail, a dropped row under :omit
        wm = Vector{Union{Missing, Float64}}(ones(n))
        wm[20] = missing
        @test_throws ArgumentError bam(f, df; weights = wm)
        @test length(fitted(bam(f, df; weights = wm, na_action = :omit))) == n - 1
    end

    # ------------------------------------------------------------------
    # gamm
    # ------------------------------------------------------------------
    @testset "gamm na_action" begin
        dfg = copy(df)
        allowmissing!(dfg, :y)
        dfg.y[5] = missing

        @test_throws ArgumentError gamm(@formula(y ~ s(x, k = 8) + (1 | grp)), dfg)
        m = gamm(@formula(y ~ s(x, k = 8) + (1 | grp)), dfg; na_action = :omit)
        @test length(fitted(m)) == n - 1

        # a missing in the GROUPING variable also disqualifies the row
        dfg2 = copy(df)
        dfg2.grp = Vector{Union{Missing, String}}(dfg2.grp)
        dfg2.grp[8] = missing
        m2 = gamm(@formula(y ~ s(x, k = 8) + (1 | grp)), dfg2; na_action = :omit)
        @test length(fitted(m2)) == n - 1

        badw = ones(n); badw[7] = -1.0
        @test_throws ArgumentError gamm(@formula(y ~ s(x, k = 8) + (1 | grp)), df;
                                        weights = badw)
    end

    # ------------------------------------------------------------------
    # gam_nl
    # ------------------------------------------------------------------
    @testset "gam_nl na_action" begin
        dfn = copy(df)
        dfn.l1 = rand(rng, n); dfn.l2 = rand(rng, n); dfn.l3 = rand(rng, n)
        allowmissing!(dfn, :l2)
        dfn.l2[9] = missing

        fnl = @formulak(y ~ s_nest(l1, l2, l3))
        @test_throws ArgumentError gam_nl(fnl, dfn)
        m = gam_nl(fnl, dfn; na_action = :omit)
        @test length(fitted(m)) == n - 1

        badw = ones(n); badw[7] = -1.0
        err = try gam_nl(fnl, df_with_lags(df, rng); weights = badw) catch e; e end
        @test err isa ArgumentError
    end

    # ------------------------------------------------------------------
    # sp= with fx=true, rejected at construction (item 3)
    # ------------------------------------------------------------------
    @testset "sp with fx rejected at construction" begin
        for ctor in (() -> s(:x, sp = 1.0, fx = true),
                     () -> te(:x, :z, sp = 1.0, fx = true),
                     () -> ti(:x, :z, sp = 1.0, fx = true),
                     () -> t2(:x, :z, sp = 1.0, fx = true))
            err = try ctor() catch e; e end
            @test err isa ArgumentError
            @test occursin("incompatible", sprint(showerror, err))
        end

        # each alone is still fine
        @test s(:x, sp = 1.0) isa GAM.SmoothSpec
        @test s(:x, fx = true, k = 5) isa GAM.SmoothSpec
        @test te(:x, :z, sp = 1.0) isa GAM.SmoothSpec
        @test te(:x, :z, fx = true) isa GAM.SmoothSpec

        # and it fires from inside a formula, before any fitting happens
        @test_throws ArgumentError @formulak(y ~ s(x, sp = 1.0, fx = true))
    end

    # ------------------------------------------------------------------
    # the StatsModels @formula (FormulaTerm) front-ends
    # ------------------------------------------------------------------
    @testset "FormulaTerm front-ends" begin
        dff = copy(df)
        allowmissing!(dff, :x)
        dff.x[11] = missing

        # _model_covariates drops the response and keeps everything the model
        # reads, including a random-effect grouping factor
        f = StatsModels.@formula(y ~ s(x, 8) + z)
        @test sort(GAM._model_covariates(f)) == [:x, :z]
        fg = StatsModels.@formula(y ~ s(x, 8) + (1 | grp))
        @test sort(GAM._model_covariates(fg)) == [:grp, :x]

        # bam(::FormulaTerm)
        @test_throws ArgumentError bam(f, dff)
        @test length(fitted(bam(f, dff; na_action = :omit))) == n - 1

        # gamm(::FormulaTerm) — neither front-end may leak a `missing` into
        # smooth_construct as a bare MethodError
        @test_throws ArgumentError gamm(fg, dff)
        @test length(fitted(gamm(fg, dff; na_action = :omit).gam_model)) == n - 1

        # gamm with no random effect falls back to gam(); the rows have
        # already been filtered by then
        fn = StatsModels.@formula(y ~ s(x, 8))
        @test_throws ArgumentError gamm(fn, dff)
        @test length(fitted(gamm(fn, dff; na_action = :omit))) == n - 1
    end

    # ------------------------------------------------------------------
    # Row REALIGNMENT — the docstring's actual promise. This file's header
    # claims to pin "the row bookkeeping that lets a caller realign results
    # with the original table"; until now only result LENGTHS were asserted.
    # A silent off-by-one in the kept-row set would misalign every downstream
    # join a user does — exactly the failure class the feature exists to
    # prevent — and would have passed the old tests.
    # ------------------------------------------------------------------
    @testset "na_omit_rows realignment round-trip" begin
        rng2 = StableRNG(20260912)
        dfm = DataFrame(
            x = Vector{Union{Missing, Float64}}(rand(rng2, n)),
            z = rand(rng2, n),
            w = Vector{Union{Missing, Float64}}(ones(n)),
            off = rand(rng2, n),
        )
        dfm.y = Vector{Union{Missing, Float64}}(
            2 .* sin.(2π .* coalesce.(dfm.x, 0.0)) .+ 0.3 .* randn(rng2, n))
        # incomplete rows scattered through the table, one per column kind
        dfm.y[7] = missing
        dfm.x[42] = missing
        dfm.z[91] = NaN
        dfm.w[120] = missing
        dfm.off[155] = Inf

        # Direct unit test of na_omit_rows, including the weights/offset
        # keywords (previously never exercised anywhere).
        keep = GAM.na_omit_rows(dfm, :y, [:x, :z];
            weights = dfm.w, offset = dfm.off)
        expected = setdiff(1:n, [7, 42, 91, 120, 155])
        @test keep == expected
        @test issorted(keep)                    # original-table order preserved
        # Without weights/offset those rows are complete again:
        @test GAM.na_omit_rows(dfm, :y, [:x, :z]) ==
              setdiff(1:n, [7, 42, 91])

        f = @formulak(y ~ s(x, k = 8) + s(z, k = 8))
        m = bam(f, dfm; weights = dfm.w, offset = dfm.off, na_action = :omit)
        clean = dfm[keep, :]
        m_ref = bam(f, DataFrame(clean);
            weights = Float64.(clean.w), offset = Float64.(clean.off))

        # The realignment identity: scattering the :omit fit's results back
        # through `keep` places each value at ITS original row. Verified by
        # checking the scattered vector agrees, row by original row, with the
        # hand-filtered fit indexed the same way — for fitted values and
        # residuals both.
        aligned_fit = fill(NaN, n)
        aligned_fit[keep] .= fitted(m)
        aligned_res = fill(NaN, n)
        aligned_res[keep] .= residuals(m)
        for (pos, orig_row) in enumerate(keep)
            @test aligned_fit[orig_row] ≈ fitted(m_ref)[pos] atol = 1e-10
            @test aligned_res[orig_row] ≈ residuals(m_ref)[pos] atol = 1e-10
        end
        # Dropped rows stay unfilled — nothing leaked into them.
        @test all(isnan, aligned_fit[[7, 42, 91, 120, 155]])

        # And prediction on the kept subtable reproduces the fit (predict on
        # newdata excludes the training offset, so add it back), so a user
        # can regenerate aligned predictions from the original data at will.
        pred = predict(m, DataFrame(clean))
        @test maximum(abs.(pred .+ Float64.(clean.off) .- fitted(m))) < 1e-8
    end
end
