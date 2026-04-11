using Test
using GAM
using DataFrames
using Random
using LinearAlgebra: diag

@testset "vis_gam" begin

    # ── Shared 2D test data ──────────────────────────────────────────────
    rng = MersenneTwister(42)
    n = 200
    x1 = rand(rng, n)
    x2 = rand(rng, n)
    y = sin.(2π .* x1) .* cos.(2π .* x2) .+ 0.3 .* randn(rng, n)
    df = DataFrame(x1 = x1, x2 = x2, y = y)

    m_te = gam(@formulak(y ~ te(x1, x2, k = 5)), df)

    # Also build a 1D model for error-path testing
    y1d = sin.(2π .* x1) .+ 0.3 .* randn(rng, n)
    df1d = DataFrame(x = x1, y = y1d)
    m_1d = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), df1d)

    # ── Basic construction ───────────────────────────────────────────────
    @testset "basic construction" begin
        v = vis_gam(m_te; select = 1, n_grid = 20)
        @test v isa VisGamData
        @test length(v.x1) == 20
        @test length(v.x2) == 20
        @test size(v.z) == (20, 20)
        @test v.se === nothing
        @test v.x1_label == "x1"
        @test v.x2_label == "x2"
        @test v.z_label == "Effect"
        @test occursin("te(", v.smooth_label)
        @test all(isfinite, v.z)
    end

    # ── Standard errors ──────────────────────────────────────────────────
    @testset "standard errors" begin
        v = vis_gam(m_te; select = 1, n_grid = 15, se = true)
        @test v.se !== nothing
        @test size(v.se) == (15, 15)
        @test all(x -> isfinite(x) && x >= 0, v.se)
    end

    # ── Response scale ───────────────────────────────────────────────────
    @testset "response scale (Gaussian identity)" begin
        v_link = vis_gam(m_te; select = 1, n_grid = 10, type = :link)
        v_resp = vis_gam(m_te; select = 1, n_grid = 10, type = :response)
        # For Gaussian with identity link, link == response
        @test v_link.z ≈ v_resp.z
        @test v_resp.z_label == "Response"
    end

    @testset "response scale (Poisson log-link)" begin
        rng3 = MersenneTwister(99)
        n3 = 300
        a = rand(rng3, n3)
        b = rand(rng3, n3)
        λ = exp.(0.5 .* sin.(2π .* a) .+ 0.3 .* cos.(2π .* b))
        y_pois = Float64.([rand(rng3, Poisson(l)) for l in λ])
        df_pois = DataFrame(a = a, b = b, y = y_pois)
        m_pois = gam(@formulak(y ~ te(a, b, k = 5)), df_pois;
                     family = Poisson(), link = LogLink())

        v_link = vis_gam(m_pois; n_grid = 10, type = :link)
        v_resp = vis_gam(m_pois; n_grid = 10, type = :response)
        # Response scale must be positive (exp of link)
        @test all(x -> isfinite(x) && x > 0, filter(!isnan, v_resp.z))
        # They should differ for non-identity link
        @test !isapprox(v_link.z, v_resp.z)
    end

    # ── too_far masking ──────────────────────────────────────────────────
    @testset "too_far masking" begin
        v_all = vis_gam(m_te; select = 1, n_grid = 20, too_far = 0.0)
        @test !any(isnan, v_all.z)

        v_masked = vis_gam(m_te; select = 1, n_grid = 20, too_far = 0.01)
        # With a very tight threshold, many grid points should be NaN
        @test any(isnan, v_masked.z)
        # But not all of them
        @test any(!isnan, v_masked.z)
    end

    @testset "too_far with SE" begin
        v = vis_gam(m_te; n_grid = 15, se = true, too_far = 0.01)
        nan_z = isnan.(v.z)
        nan_se = isnan.(v.se)
        # SE should be NaN exactly where z is NaN
        @test nan_z == nan_se
    end

    # ── _exclude_too_far helper ──────────────────────────────────────────
    @testset "_exclude_too_far" begin
        # Simple case: data at corners of unit square
        x1d = [0.0, 0.0, 1.0, 1.0]
        x2d = [0.0, 1.0, 0.0, 1.0]
        grid1 = [0.0, 0.5, 1.0]
        grid2 = [0.0, 0.5, 1.0]

        # Very generous threshold — nothing excluded
        mask_loose = GAM._exclude_too_far(grid1, grid2, x1d, x2d, 10.0)
        @test !any(mask_loose)

        # Very tight threshold — center excluded, corners kept
        mask_tight = GAM._exclude_too_far(grid1, grid2, x1d, x2d, 0.1)
        @test !mask_tight[1, 1]  # near (0,0)
        @test !mask_tight[3, 3]  # near (1,1)
        @test mask_tight[2, 2]   # center (0.5, 0.5) is far from all corners
    end

    # ── Error on 1D smooth ───────────────────────────────────────────────
    @testset "error on 1D smooth" begin
        @test_throws ArgumentError vis_gam(m_1d; select = 1)
        try
            vis_gam(m_1d; select = 1)
        catch e
            @test occursin("2D", e.msg)
            @test occursin("gamplot", e.msg)
        end
    end

    # ── Argument validation ──────────────────────────────────────────────
    @testset "argument validation" begin
        @test_throws ArgumentError vis_gam(m_te; select = 0)
        @test_throws ArgumentError vis_gam(m_te; select = 100)
        @test_throws ArgumentError vis_gam(m_te; type = :banana)
    end
end
