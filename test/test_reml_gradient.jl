# Finite-difference validation of the analytic REML/ML gradient
# (`_reml_gradient`, reached via `reml_score(...; compute_gradient=true)`).
#
# The analytic gradient is the total derivative of the PROFILED score — the
# score evaluated at the penalized MLE β̂(ρ) — so the reference must refit
# P-IRLS to convergence at each perturbed ρ rather than holding β fixed.
#
# Two regimes are pinned deliberately:
#   * scale known or supplied  → exact (validated to ~1e-7 here, ~1e-9 in
#     the development harness with tighter P-IRLS tolerances)
#   * scale estimated internally → approximate; the `dσ̂²/dρ` chain term is
#     omitted. The bound below documents that gap rather than hiding it, and
#     will tighten to the exact regime if a future change profiles the scale.

@testset "REML gradient vs finite differences" begin
    _tight = gam_control(epsilon = 1e-13, maxit = 500)

    # Build (X, y, penalty) with an unpenalized intercept plus one smooth.
    function _grad_case(family, link; tensor = false, n = 120, seed = 11)
        rng = StableRNG(seed)
        x = collect(range(-2.0, 2.0; length = n))
        z = collect(range(-1.5, 1.5; length = n))
        shuffle!(rng, z)
        ft = 0.8 .* sin.(1.3 .* x) .+ 0.4 .* cos.(1.1 .* z)
        y = if family isa Normal
            ft .+ 0.25 .* randn(rng, n)
        elseif family isa Poisson
            Float64[rand(rng, Poisson(exp(0.6 * e + 0.5))) for e in ft]
        elseif family isa Bernoulli
            Float64[rand(rng, Bernoulli(1 / (1 + exp(-e)))) for e in ft]
        else  # Gamma
            [rand(rng, Gamma(4.0, exp(0.5 * e + 0.3) / 4.0)) for e in ft]
        end
        data = DataFrame(x = x, z = z)
        sm = tensor ? smooth_construct(te(:x, :z, k = 16), data) :
             smooth_construct(s(:x, k = 8, bs = :cr), data)
        sm.first_para = 2
        sm.last_para = 1 + size(sm.X, 2)
        X = hcat(ones(n), sm.X)
        pen = GAM.setup_penalties([sm], 1)
        return X, y, pen
    end

    # Profiled score / analytic gradient at a given log_sp.
    function _score_grad(X, y, pen, lsp, family, link, w, off; method, scale, want_grad)
        S = GAM.total_penalty(pen, lsp, size(X, 2))
        res = GAM.pirls(X, y, S, family, link;
            weights = w, offset = off, control = _tight)
        return GAM.reml_score(X, y, pen, lsp, family, link, w, res;
            method = method, scale = scale, compute_gradient = want_grad)
    end

    function _check(family, link; tensor = false, method = :REML, scale = -1.0,
                    use_weights = false, use_offset = false, lsp = nothing,
                    seed = 11, tol)
        X, y, pen = _grad_case(family, link; tensor = tensor, seed = seed)
        n = length(y)
        w = use_weights ? (0.5 .+ collect(range(0.0, 1.0; length = n))) : ones(n)
        off = use_offset ? fill(0.13, n) : zeros(n)
        ρ = lsp === nothing ? (tensor ? [0.4, -0.7] : [0.4]) : lsp

        _, analytic = _score_grad(X, y, pen, ρ, family, link, w, off;
            method = method, scale = scale, want_grad = true)

        h = 1e-5
        fd = similar(ρ)
        for j in eachindex(ρ)
            ρp = copy(ρ); ρp[j] += h
            ρm = copy(ρ); ρm[j] -= h
            sp_p, _ = _score_grad(X, y, pen, ρp, family, link, w, off;
                method = method, scale = scale, want_grad = false)
            sp_m, _ = _score_grad(X, y, pen, ρm, family, link, w, off;
                method = method, scale = scale, want_grad = false)
            fd[j] = (sp_p - sp_m) / (2h)
        end

        relerr = maximum(abs.(analytic .- fd) ./ max.(abs.(fd), 1e-8))
        @test relerr < tol
        return relerr
    end

    @testset "known scale (φ = 1) — exact" begin
        # Poisson/log and Bernoulli/logit exercise the implicit dW/dρ terms;
        # a wrong or missing implicit term shows up here immediately.
        @test _check(Poisson(), LogLink(); tol = 1e-6) < 1e-6
        @test _check(Poisson(), LogLink(); tensor = true, tol = 1e-6) < 1e-6
        @test _check(Bernoulli(), LogitLink(); tol = 1e-6) < 1e-6
        @test _check(Bernoulli(), LogitLink(); tensor = true, tol = 1e-6) < 1e-6
        # Prior weights and offsets must not perturb the identity
        @test _check(Poisson(), LogLink(); use_weights = true, tol = 1e-6) < 1e-6
        @test _check(Poisson(), LogLink(); use_offset = true, tol = 1e-6) < 1e-6
        # ML shares the gradient path (only the Mp constant differs)
        @test _check(Poisson(), LogLink(); method = :ML, tol = 1e-6) < 1e-6
    end

    @testset "scale supplied explicitly — exact" begin
        @test _check(Normal(), IdentityLink(); scale = 0.0625, tol = 1e-6) < 1e-6
        @test _check(Normal(), IdentityLink(); scale = 0.0625, tensor = true,
            tol = 1e-6) < 1e-6
        @test _check(Gamma(), LogLink(); scale = 0.25, tol = 1e-5) < 1e-5
        @test _check(Normal(), IdentityLink(); scale = 0.0625, method = :ML,
            tol = 1e-6) < 1e-6
    end

    @testset "scale estimated internally — documented approximation" begin
        # The dσ̂²/dρ chain term is omitted (see the reml_score docstring).
        # These bounds record the size of the gap; they are NOT a claim of
        # correctness. If a future change profiles the scale analytically,
        # these should collapse to the exact regime above and the tolerances
        # should be tightened accordingly.
        gauss = _check(Normal(), IdentityLink(); tol = 1e-1)
        gamma = _check(Gamma(), LogLink(); tol = 1e-1)
        @test gauss > 1e-5   # gap is real, not numerical noise
        @test gamma > 1e-5

        # Supplying the REML-profiling scale σ̂² = Dp/(n − Mp) restores the
        # envelope-theorem condition for Gaussian, and the error collapses.
        X, y, pen = _grad_case(Normal(), IdentityLink())
        n, p = size(X)
        w = ones(n); off = zeros(n)
        ρ = [0.4]
        Mp = p - sum(b.rank for b in pen.blocks; init = 0)
        prof_scale = let S = GAM.total_penalty(pen, ρ, p)
            r = GAM.pirls(X, y, S, Normal(), IdentityLink();
                weights = w, offset = off, control = _tight)
            (r.deviance + dot(r.coefficients, S * r.coefficients)) / (n - Mp)
        end
        prof_err = _check(Normal(), IdentityLink(); scale = prof_scale, tol = 1e-6)
        @test prof_err < gauss   # profiling strictly improves agreement
    end
end
