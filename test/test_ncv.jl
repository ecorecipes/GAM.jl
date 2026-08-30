# Neighbourhood Cross Validation (`method = :NCV`).
#
# Reference values are from mgcv 1.9-4 and pinned here so the suite needs no R.
# The data are DETERMINISTIC (no RNG) for the same reason as
# test_gp_parity.jl: a pinned reference must not depend on an RNG stream that
# a Julia upgrade could change.
#
# The strongest test here is not the mgcv comparison but
# "one-step Newton == brute-force refit": for a Gaussian identity model the
# single Newton step NCV uses is EXACT, so the criterion must reproduce an
# actual leave-one-out refit to machine precision. That pins the Woodbury
# algebra, the sign conventions, and the β − d update independently of mgcv.

@testset "NCV — neighbourhood cross validation" begin

    # ------------------------------------------------------------------
    # 1. The approximation is exact where it can be checked exactly.
    # ------------------------------------------------------------------
    @testset "one-step Newton == brute-force LOO refit (Gaussian)" begin
        rng = StableRNG(7)
        n = 60
        x = sort(rand(rng, n))
        y = sin.(2π .* x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        # Fixed sp: the comparison is of the criterion, not of an optimizer.
        gf = GAM.GamFormula(:y, Symbol[], true,
            [GAM.s(:x; k = 8, bs = :cr, sp = 0.05)])
        m = gam(gf, df; method = :NCV)

        X = model_matrix(m)
        p = size(X, 2)
        S = GAM.total_penalty(m.penalty, m.penalty.sp, p)
        beta = coef(m)
        eta = X * beta

        # Gaussian identity: dℓ/dη = y − η and −d²ℓ/dη² = 1.
        w1 = y .- eta
        w2 = ones(n)
        H = X' * X + S

        nei = GAM.loo_neighbourhoods(n)
        eta_cv, npd = GAM.ncv_eta(X, beta, w1, w2, H, nei)
        @test npd == 0

        eta_bf = similar(eta_cv)
        for i in 1:n
            idx = setdiff(1:n, i)
            Xi = X[idx, :]
            bi = (Xi' * Xi + S) \ (Xi' * y[idx])
            eta_bf[i] = dot(view(X, i, :), bi)
        end

        # Exact, not approximate: measured 7.8e-16 on this fit.
        @test maximum(abs.(eta_cv .- eta_bf)) < 1e-11

        # And the assembled score equals the brute-force CV deviance.
        score, _ = GAM.ncv_score(X, y, beta, eta, Normal(), IdentityLink(), S,
            ones(n), zeros(n), nei)
        @test score ≈ sum((y .- eta_bf) .^ 2) rtol = 1e-10
    end

    # ------------------------------------------------------------------
    # 2. mgcv parity, at fixed sp so no optimizer enters the comparison.
    # ------------------------------------------------------------------
    @testset "mgcv 1.9-4 parity at fixed sp" begin
        n = 120
        x = collect((1:n) ./ (n + 1))
        y = sin.(2π .* x) .+ 0.3 .* sin.(37 .* (1:n))
        df = DataFrame(x = x, y = y)

        # mgcv: gam(y ~ s(x,k=10,bs="cr"), method="NCV", sp=<sp>)$gcv.ubre
        mgcv_ncv = [
            (0.001, 6.310240995),
            (0.01, 6.309612130),
            (0.1, 6.303473331),
            (1.0, 6.253864557),
        ]
        for (spv, ref) in mgcv_ncv
            gf = GAM.GamFormula(:y, Symbol[], true,
                [GAM.s(:x; k = 10, bs = :cr, sp = spv)])
            m = gam(gf, df; method = :NCV)
            # Agreement is to every digit mgcv prints; 1e-8 is slack.
            @test sp_criterion(m) ≈ ref rtol = 1e-8
        end

        # Free fit: mgcv selects sp = 18.860492 with NCV = 6.072108505.
        # The criterion agrees to 4e-9; the selected sp differs by ~0.03%
        # because GAM.jl minimizes without analytic NCV derivatives (see
        # src/ncv.jl) — a flat optimum, not a different criterion.
        gf2 = GAM.GamFormula(:y, Symbol[], true, [GAM.s(:x; k = 10, bs = :cr)])
        mf = gam(gf2, df; method = :NCV)
        @test sp_criterion(mf) ≈ 6.072108505 rtol = 1e-6
        @test exp(mf.sp[1]) ≈ 18.860492 rtol = 5e-3
    end

    # ------------------------------------------------------------------
    # 3. The property NCV exists for: correlated data.
    # ------------------------------------------------------------------
    @testset "wide neighbourhoods resist autocorrelation" begin
        # Several seeds, because the point is a PROPERTY of the criterion, not
        # a number from one realization. An earlier draft pinned
        # `edf < 10` from a single prototype run; that threshold held for the
        # RNG it was measured on and failed on the next one, while the
        # relative ordering below held for every seed tried.
        for seed in (11, 12, 13)
            rng = StableRNG(seed)
            n = 200
            x = collect(range(0, 1; length = n))
            ftrue = sin.(2π .* x)
            rho = 0.9
            e = zeros(n)
            e[1] = randn(rng)
            for i in 2:n
                e[i] = rho * e[i-1] + sqrt(1 - rho^2) * randn(rng)
            end
            y = ftrue .+ 0.5 .* e
            df = DataFrame(x = x, y = y)

            fitm(; kw...) = gam(@formulak(y ~ s(x, k = 30, bs = :cr)), df; kw...)
            rmse(m) = sqrt(mean((predict(m, df) .- ftrue) .^ 2))

            m_gcv = fitm(method = :GCV)
            m_loo = fitm(method = :NCV)
            m_wide = fitm(method = :NCV,
                nei = GAM.interval_neighbourhoods(n, 15))

            e_gcv = sum(edf(m_gcv))
            e_loo = sum(edf(m_loo))
            e_wide = sum(edf(m_wide))

            # The true function has edf ≈ 3, but with rho = 0.9 both GCV and
            # leave-ONE-out CV chase the correlated noise and run up against
            # the k = 30 basis. Measured 27.2-27.7 across these seeds.
            @test e_gcv > 15
            @test e_loo > 15

            # Widening the neighbourhood past the correlation range is what
            # fixes it. Measured margins e_gcv - e_wide: 8.5, 16.8, 14.8.
            @test e_wide < e_gcv - 5

            # Smoother AND closer to the truth — smoothing alone would not be
            # a win. Measured RMSE 0.531→0.505, 0.359→0.295, 0.427→0.356.
            @test rmse(m_wide) < rmse(m_gcv)

            # Leave-one-out NCV IS ordinary cross validation, so it must land
            # near GCV rather than near the wide fit. This is the guard that
            # the default `nei` really is leave-one-out.
            @test abs(e_loo - e_gcv) < 5
        end
    end

    # ------------------------------------------------------------------
    # 4. Neighbourhood construction and validation.
    # ------------------------------------------------------------------
    @testset "neighbourhood structures" begin
        loo = GAM.loo_neighbourhoods(5)
        @test loo.k == 1:5
        @test loo.m == 1:5
        @test loo.ind == 1:5
        @test loo.mi == 1:5

        # half_width = 0 is leave-one-out.
        @test GAM.interval_neighbourhoods(5, 0).k == GAM.loo_neighbourhoods(5).k

        iv = GAM.interval_neighbourhoods(5, 1)
        # Fold 1 drops {1,2}; fold 3 drops {2,3,4}; fold 5 drops {4,5}.
        @test iv.k[1:iv.m[1]] == [1, 2]
        @test iv.k[(iv.m[2]+1):iv.m[3]] == [2, 3, 4]
        @test iv.k[(iv.m[4]+1):iv.m[5]] == [4, 5]
        @test iv.ind == 1:5

        @test_throws ArgumentError GAM.loo_neighbourhoods(0)
        @test_throws ArgumentError GAM.interval_neighbourhoods(5, -1)

        # Validation catches malformed structures with a usable message
        # rather than a BoundsError inside the criterion.
        @test GAM.validate_neighbourhoods(loo, 5) === nothing
        @test_throws ArgumentError GAM.validate_neighbourhoods(loo, 4)  # index 5 > n
        bad_m = GAM.NeighbourhoodStructure([1, 2], [9], [1], [1])
        @test_throws ArgumentError GAM.validate_neighbourhoods(bad_m, 5)
        mismatched = GAM.NeighbourhoodStructure([1], [1], [1], [1, 2])
        @test_throws ArgumentError GAM.validate_neighbourhoods(mismatched, 5)
    end

    # ------------------------------------------------------------------
    # 5. API surface.
    # ------------------------------------------------------------------
    @testset "dispatch and errors" begin
        rng = StableRNG(21)
        n = 80
        x = sort(rand(rng, n))
        y = sin.(2π .* x) .+ 0.3 .* randn(rng, n)
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, k = 8, bs = :cr)), df; method = :NCV)
        @test m.method == :NCV
        # The achieved score lands in `criterion` (mgcv's `b$gcv.ubre`),
        # not in `reml`, so `show` labels it correctly.
        @test isfinite(sp_criterion(m))
        @test isnan(m.reml)
        @test sp_criterion(m) == m.criterion

        # An explicit LOO structure must give the same answer as the default.
        m_loo = gam(@formulak(y ~ s(x, k = 8, bs = :cr)), df;
            method = :NCV, nei = GAM.loo_neighbourhoods(n))
        @test sp_criterion(m_loo) ≈ sp_criterion(m) rtol = 1e-8

        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df; method = :NOPE)
        # `nei` without :NCV is a user error, not silently ignored.
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df;
            method = :REML, nei = GAM.loo_neighbourhoods(n))
        # A malformed `nei` is caught before the criterion runs.
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df;
            method = :NCV, nei = GAM.loo_neighbourhoods(n + 10))
        # Extended families are not supported (they take :REML/:ML).
        @test_throws ArgumentError gam(@formulak(y ~ s(x)), df;
            family = NegBinFamily(), link = LogLink(), method = :NCV)
    end

    # ------------------------------------------------------------------
    # 6. Non-Gaussian NCV runs and is sane. The Newton step is an
    #    approximation here (unlike the Gaussian case above), so this checks
    #    behaviour rather than exactness.
    # ------------------------------------------------------------------
    @testset "Poisson NCV" begin
        rng = StableRNG(31)
        n = 150
        x = sort(rand(rng, n))
        mu = exp.(1.0 .+ sin.(2π .* x))
        y = Float64.([rand(rng, Poisson(m)) for m in mu])
        df = DataFrame(x = x, y = y)

        m = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), df;
            family = Poisson(), link = LogLink(), method = :NCV)
        @test m.converged
        @test isfinite(sp_criterion(m))
        @test sp_criterion(m) > 0
        # Recovers the mean function about as well as REML does.
        mr = gam(@formulak(y ~ s(x, k = 10, bs = :cr)), df;
            family = Poisson(), link = LogLink(), method = :REML)
        rmse(mm) = sqrt(mean((predict(mm, df; type = :response) .- mu) .^ 2))
        @test rmse(m) < 2 * rmse(mr)
    end
end
