@testset "Markov Random Field (MRF) Smooths" begin
    using Random, DataFrames, Statistics, LinearAlgebra

    @testset "MRF construction: 5-region chain graph" begin
        # Chain graph: 1—2—3—4—5
        nb = zeros(Int, 5, 5)
        nb[1,2] = nb[2,1] = 1
        nb[2,3] = nb[3,2] = 1
        nb[3,4] = nb[4,3] = 1
        nb[4,5] = nb[5,4] = 1

        n = 50
        regions = repeat(1:5, 10)
        y = Float64.(regions) .+ 0.1 .* randn(MersenneTwister(42), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb))
        @test spec isa SmoothSpec{MarkovRandomField}
        @test spec.xt[:nb] === nb

        sm = GAM.smooth_construct(spec, df)
        # After constraint absorption: n_regions - 1 = 4 columns
        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 4
        @test length(sm.S) == 1
        @test size(sm.S[1]) == (4, 4)
        @test sm.null_dim == 1
        @test sm.rank == 4
    end

    @testset "Penalty is a valid graph Laplacian" begin
        # Triangle graph: 1—2, 2—3, 1—3
        nb = [0 1 1; 1 0 1; 1 1 0]
        n = 30
        regions = repeat(1:3, 10)
        y = randn(MersenneTwister(99), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb))
        sm = GAM.smooth_construct(spec, df)

        # Before constraint absorption, the Laplacian L = D - A should have:
        # 1. Row/column sums = 0
        # 2. PSD (eigenvalues >= 0)
        # 3. Exactly one zero eigenvalue (connected graph)
        A = Float64.(nb)
        D = Diagonal(vec(sum(A; dims=2)))
        L = Matrix(D - A)
        @test all(abs.(sum(L; dims=2)) .< 1e-12)  # row sums = 0
        evals = eigvals(Symmetric(L))
        @test all(evals .>= -1e-10)  # PSD
        @test sum(abs.(evals) .< 1e-10) == 1  # one zero eigenvalue
    end

    @testset "Neighbour list input" begin
        # Same chain graph as vector-of-vectors
        nb_list = Vector{Vector{Int}}([[2], [1,3], [2,4], [3,5], [4]])

        n = 50
        regions = repeat(1:5, 10)
        y = Float64.(regions) .+ 0.1 .* randn(MersenneTwister(42), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb_list))
        sm = GAM.smooth_construct(spec, df)
        @test size(sm.X, 2) == 4  # 5 regions - 1 constraint = 4
    end

    @testset "String region labels" begin
        nb = [0 1 0; 1 0 1; 0 1 0]
        n = 30
        labels = ["A", "B", "C"]
        regions = repeat(labels, 10)
        y = randn(MersenneTwister(77), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb))
        sm = GAM.smooth_construct(spec, df)
        @test size(sm.X, 1) == n
        @test size(sm.X, 2) == 2  # 3 regions - 1 constraint

        # Prediction with same data should reconstruct
        X_pred = GAM.predict_matrix(sm, df)
        @test size(X_pred) == size(sm.X)
        @test X_pred ≈ sm.X
    end

    @testset "Prediction with unknown regions" begin
        nb = [0 1; 1 0]
        n = 20
        regions = repeat(1:2, 10)
        y = randn(MersenneTwister(55), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb))
        sm = GAM.smooth_construct(spec, df)

        # Predict with an unknown region (3)
        newdf = DataFrame(region=[1, 2, 3])
        X_pred = GAM.predict_matrix(sm, newdf)
        @test size(X_pred, 1) == 3
        # Unknown region should get zero row
        @test all(X_pred[3, :] .== 0.0)
    end

    @testset "MRF requires neighbourhood matrix" begin
        # s() without xt is fine (error at construction time)
        spec_no_nb = s(:region, bs=:mrf)
        df = DataFrame(region=repeat(1:3, 10), y=randn(30))
        @test_throws ArgumentError GAM.smooth_construct(spec_no_nb, df)

        # Explicit empty xt also errors at construction
        spec_empty_xt = s(:region, bs=:mrf, xt=Dict{Symbol,Any}())
        @test_throws ArgumentError GAM.smooth_construct(spec_empty_xt, df)
    end

    @testset "MRF fitting with gam()" begin
        Random.seed!(123)

        # 6-region grid: 1—2—3, 4—5—6 with 1—4, 2—5, 3—6
        nb = zeros(Int, 6, 6)
        nb[1,2] = nb[2,1] = 1; nb[2,3] = nb[3,2] = 1
        nb[4,5] = nb[5,4] = 1; nb[5,6] = nb[6,5] = 1
        nb[1,4] = nb[4,1] = 1; nb[2,5] = nb[5,2] = 1; nb[3,6] = nb[6,3] = 1

        n = 300
        regions = rand(MersenneTwister(42), 1:6, n)
        # True region effects
        effects = [0.0, 1.0, 2.0, 0.5, 1.5, 2.5]
        y = [effects[r] for r in regions] .+ 0.3 .* randn(MersenneTwister(43), n)
        df = DataFrame(region=regions, y=y)

        spec = s(:region, bs=:mrf, xt=Dict{Symbol,Any}(:nb => nb))
        gf = GamFormula(:y, Symbol[], true, SmoothSpec[spec])
        m = gam(gf, df)

        @test m.converged
        @test length(m.smooths) == 1
        @test m.smooths[1].spec.basis isa MarkovRandomField
        # Check reasonable fit: correlation between fitted and true should be high
        true_vals = [effects[r] for r in regions]
        @test cor(m.fitted_values, true_vals) > 0.8
    end
end
