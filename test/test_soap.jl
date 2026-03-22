@testset "Soap Film Smooths (bs=:so)" begin
    using Random, DataFrames, Statistics, LinearAlgebra, StatsAPI

    import StatsAPI: coef, predict

    # ── Helpers ──────────────────────────────────────────────────────────
    square_bnd() = [hcat([0, 1, 1, 0, 0.0], [0, 0, 1, 1, 0.0])]

    function make_soap_data(n; bnd = square_bnd(), seed = 42)
        Random.seed!(seed)
        x = 0.05 .+ 0.9 .* rand(n)
        y = 0.05 .+ 0.9 .* rand(n)
        z = sin.(2π .* x) .* cos.(2π .* y) .+ 0.1 .* randn(n)
        DataFrame(x = x, y = y, z = z), bnd
    end

    # ── Construction tests ───────────────────────────────────────────────
    @testset "Basic construction" begin
        df, bnd = make_soap_data(200)
        spec = s(:x, :y, bs = :so, k = 10,
                 xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 30))
        sm = smooth_construct(spec, df)

        @test size(sm.X, 1) == 200
        @test size(sm.X, 2) > 0
        @test length(sm.S) >= 1
        @test all(issymmetric.(sm.S))
        @test sm.null_dim >= 0
    end

    @testset "Penalty matrices are positive semidefinite" begin
        df, bnd = make_soap_data(150)
        spec = s(:x, :y, bs = :so, k = 8,
                 xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 25))
        sm = smooth_construct(spec, df)

        for (i, Si) in enumerate(sm.S)
            evals = eigvals(Symmetric(Si))
            @test all(evals .>= -1e-10)
        end
    end

    @testset "Different k values" begin
        df, bnd = make_soap_data(200)
        for k in [6, 10, 15]
            spec = s(:x, :y, bs = :so, k = k,
                     xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 25))
            sm = smooth_construct(spec, df)
            @test size(sm.X, 2) > 0
            @test size(sm.X, 2) <= k + 10  # some margin for boundary + interior
        end
    end

    @testset "Points at boundary get valid values" begin
        df, bnd = make_soap_data(100)
        # Add points near boundary edges
        near_edge = DataFrame(
            x = [0.05, 0.95, 0.5, 0.5],
            y = [0.5, 0.5, 0.05, 0.95],
            z = [0.0, 0.0, 0.0, 0.0],
        )
        df2 = vcat(df, near_edge)
        spec = s(:x, :y, bs = :so, k = 8,
                 xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 25))
        sm = smooth_construct(spec, df2)
        @test !any(isnan, sm.X)
        @test !any(isinf, sm.X)
    end

    # ── GAM fitting tests ────────────────────────────────────────────────
    @testset "GAM fit on square domain" begin
        df, bnd = make_soap_data(300; seed = 123)
        m = gam(@gam_formula(z ~ s(x, y, bs = :so, k = 15,
                    xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 40))),
                df; control = GAM.gam_control(trace = false))

        @test length(coef(m)) > 0
        @test sum(m.edf) > 1.0
        @test m.converged
    end

    @testset "GAM fit recovers signal" begin
        Random.seed!(77)
        n = 400
        x = 0.1 .+ 0.8 .* rand(n)
        y = 0.1 .+ 0.8 .* rand(n)
        true_f = sin.(2π .* x) .* cos.(2π .* y)
        z = true_f .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, z = z)
        bnd = square_bnd()

        m = gam(@gam_formula(z ~ s(x, y, bs = :so, k = 20,
                    xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 50))),
                df; control = GAM.gam_control(trace = false))

        fitted = m.fitted_values
        @test cor(fitted, true_f) > 0.8
    end

    # ── L-shaped domain ──────────────────────────────────────────────────
    @testset "L-shaped domain" begin
        # L-shape boundary: unit square minus upper-right quadrant
        bnd_L = [hcat(
            [0, 1, 1, 0.5, 0.5, 0, 0.0],
            [0, 0, 0.5, 0.5, 1, 1, 0.0],
        )]

        Random.seed!(55)
        n = 300
        # Sample inside L
        x = Float64[]; y = Float64[]
        while length(x) < n
            px, py = rand(), rand()
            if px <= 0.5 || py <= 0.5  # inside L
                push!(x, 0.05 + 0.9 * px)
                push!(y, 0.05 + 0.9 * py)
            end
        end
        # Clip to be safely inside
        x = clamp.(x, 0.05, 0.95)
        y = clamp.(y, 0.05, 0.95)
        z = sin.(3 .* x) .+ cos.(3 .* y) .+ 0.2 .* randn(n)
        df = DataFrame(x = x, y = y, z = z)

        spec = s(:x, :y, bs = :so, k = 12,
                 xt = Dict{Symbol,Any}(:bnd => bnd_L, :nmax => 30))
        sm = smooth_construct(spec, df)
        @test size(sm.X, 1) == n
        @test !any(isnan, sm.X)
    end

    # ── Prediction ───────────────────────────────────────────────────────
    @testset "Prediction at new data" begin
        df, bnd = make_soap_data(300; seed = 99)
        m = gam(@gam_formula(z ~ s(x, y, bs = :so, k = 12,
                    xt = Dict{Symbol,Any}(:bnd => bnd, :nmax => 35))),
                df; control = GAM.gam_control(trace = false))

        newdf = DataFrame(
            x = 0.2 .+ 0.6 .* rand(50),
            y = 0.2 .+ 0.6 .* rand(50),
        )
        preds = predict(m, newdf; type = :response)
        @test length(preds) == 50
        @test !any(isnan, preds)
        @test !any(isinf, preds)
    end
end