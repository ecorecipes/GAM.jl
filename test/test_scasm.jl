using Test, GAM, DataFrames, Random, Statistics, Distributions

@testset "SCASM linear-constraint support" begin
    rng = MersenneTwister(123)
    n = 160
    x = sort(rand(rng, n))
    z = rand(rng, n)
    data = (x = x, z = z)

    function capture_stderr_text(f)
        path, io = mktemp()
        try
            result = redirect_stderr(io) do
                f()
            end
            flush(io)
            close(io)
            return result, read(path, String)
        finally
            isopen(io) && close(io)
            rm(path; force = true)
        end
    end

    @testset "sc basis exposes linear constraints" begin
        sm = smooth_construct(s(:x, bs = :sc, k = 12, xt = ["m+"]), data)
        @test GAM.has_linear_constraints(sm)
        @test sm.Ain !== nothing
        @test size(sm.Ain, 2) == size(sm.X, 2)
        @test !has_shape_constraints([sm])
    end

    @testset "positive sc smooth skips centering constraint" begin
        sm = smooth_construct(s(:x, bs = :sc, k = 12, xt = ["+"]), data)
        @test GAM.has_linear_constraints(sm)
        @test sm.constraint === nothing
        @test sm.Ain !== nothing
        @test size(sm.Ain, 1) == size(sm.X, 2)
    end

    @testset "scad basis builds adaptive penalties" begin
        xt = Dict{Symbol,Any}(:constraints => ["m+", "c+"], :n_penalties => 4)
        sm = smooth_construct(s(:x, bs = :scad, k = 14, xt = xt), data)
        @test GAM.has_linear_constraints(sm)
        @test length(sm.S) == 4
        @test sm.Ain !== nothing
        @test size(sm.Ain, 2) == size(sm.X, 2)
    end

    @testset "pc constraints append to smooth" begin
        pc = Dict(
            :ineq => Dict(
                :x => reshape(x, 1, n),
                :weights => fill(1 / n, 1, n),
                :rhs => [0.4],
            ),
        )
        sm = smooth_construct(s(:x, bs = :cr, k = 10, pc = pc), data)
        @test GAM.has_linear_constraints(sm)
        @test sm.Ain !== nothing
        @test size(sm.Ain, 1) == 1
    end

    @testset "tensor constraints merge from constrained marginals" begin
        spec = te(:x, :z, bs = [:sc, :cr], k = 25, xt = Any[["m+"], nothing])
        sm = smooth_construct(spec, data)
        @test GAM.has_linear_constraints(sm)
        @test sm.Ain !== nothing
        @test size(sm.Ain, 2) == size(sm.X, 2)
        Xp = predict_matrix(sm, (x = x[1:25], z = z[1:25]))
        @test size(Xp) == (25, size(sm.X, 2))
    end

    @testset "constrained gam fit converges" begin
        y = exp.(1.5 .* x) .+ 0.05 .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12)), df)
        @test m isa GamModel
        @test m.converged
        @test cor(m.fitted_values, y) > 0.95
    end

    @testset "gam auto-dispatches to linear-constraint backend" begin
        y = exp.(1.2 .* x) .+ 0.05 .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12)), df)
        @test m isa GamModel
        @test m.converged
        @test cor(m.fitted_values, y) > 0.95
    end

    @testset "constrained gam validates control, method, and start" begin
        y = exp.(0.9 .* x) .+ 0.05 .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        f = @formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12))

        @test_throws DimensionMismatch gam(f, df; start = zeros(3))
        @test_throws ArgumentError gam(f, df; control = gam_control(sp_optimizer = :newton))
        @test_throws ArgumentError gam(f, df; method = :UBRE)

        m_ml = gam(f, df; method = :ML)
        m_gcv = gam(f, df; method = :GCV)
        @test m_ml.converged
        @test m_gcv.converged
        @test m_ml.method == :ML
        @test m_gcv.method == :GCV
    end

    @testset "positive smooth fit remains nonnegative" begin
        y = 0.5 .+ x .^ 2 .+ 0.03 .* randn(rng, n)
        df = DataFrame(x = x, y = y)
        m = gam(@formulak(y ~ 0 + s(x, bs = :sc, xt = ["+"], k = 12)), df)
        @test m isa GamModel
        @test m.converged
        @test minimum(m.fitted_values) > -1e-6
    end

    @testset "constrained ExtendedFamily fit converges" begin
        μ = exp.(0.8 .+ 0.9 .* x)
        y = Float64[rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + m))) for m in μ]
        df = DataFrame(x = x, y = y)
        m, stderr_text = capture_stderr_text() do
            gam(@formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12)), df;
                family = NegBinFamily(theta = 3.0, estimate_theta = false))
        end
        @test m isa GamModel
        @test m.family isa NegBinFamily
        @test m.converged
        @test all(m.fitted_values .> 0)
        @test !occursin("non-convex", lowercase(stderr_text))
    end

    @testset "constrained GAMLSS fits across solvers" begin
        gamlss_rng = MersenneTwister(123)
        xg = sort(rand(gamlss_rng, n))
        μ = 0.3 .+ 1.1 .* xg
        σ = fill(0.15, n)
        y = μ .+ σ .* randn(gamlss_rng, n)
        df = DataFrame(x = xg, y = y)
        formulas = [
            @formulak(y ~ s(x, bs = :sc, xt = ["m+"], k = 12)),
            @formulak(y ~ 1),
        ]
        ctrl = gamlss_control(n_cyc = 50, i_cyc = 100, c_crit = 1e-4)

        @testset "efs" begin
            m, stderr_text = capture_stderr_text() do
                gamlss(formulas, df, GaussianLS(); method = :efs, gamlss_ctrl = ctrl)
            end
            @test m.converged
            @test all(isfinite, m.fitted_eta[1])
            @test all(diff(m.fitted_eta[1]) .>= -1e-8)
            @test !occursin("non-convex", lowercase(stderr_text))
        end

        for method in (:rs, :cg)
            @testset "$method" begin
                m, stderr_text = capture_stderr_text() do
                    gamlss(formulas, df, GaussianLS(); method = method, gamlss_ctrl = ctrl)
                end
                @test m.converged
                @test cor(m.fitted_eta[1], μ) > 0.9
                @test !occursin("non-convex", lowercase(stderr_text))
            end
        end
    end

    @testset "constrained GAMM fit converges" begin
        n_groups = 8
        n_per = 20
        group = repeat(1:n_groups, inner = n_per)
        xg = sort(rand(rng, n_groups * n_per))
        re = 0.4 .* randn(rng, n_groups)
        y = 0.5 .+ xg .+ re[group] .+ 0.05 .* randn(rng, length(xg))
        df = DataFrame(x = xg, y = y, group = group)
        m = gamm(@formula(y ~ s(x, bs = :sc, xt = ["m+"], k = 10) + (1 | group)), df)
        @test m isa GammModel
        @test m.gam_model.converged
        @test cor(m.gam_model.fitted_values, y) > 0.9
    end

    @testset "constrained Poisson GAMM PQL fit converges" begin
        n_groups = 6
        n_per = 25
        group = repeat(1:n_groups, inner = n_per)
        xg = sort(rand(rng, n_groups * n_per))
        re = 0.25 .* randn(rng, n_groups)
        η = 0.3 .+ 0.8 .* xg .+ re[group]
        y = Float64[rand(rng, Poisson(exp(ηi))) for ηi in η]
        df = DataFrame(x = xg, y = y, group = group)
        m = gamm(@formula(y ~ s(x, bs = :sc, xt = ["m+"], k = 10) + (1 | group)),
            df, Poisson())
        @test m isa GammModel
        @test m.gam_model.converged
        @test all(m.gam_model.fitted_values .> 0)
    end
end
