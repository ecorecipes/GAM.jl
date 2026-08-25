using GAM: BamDesign, DenseDesign, bam_design, ncols, nrows, intercept_col,
    mul_eta!, accumulate_XtWX_XtWz!, accumulate_XtWX!, design_finalize,
    pirls_bam, outer_iteration_bam, pirls_finalize, setup_penalties,
    _accumulate_XtWX_XtWz_chunked!, _accumulate_XtWX_chunked!

@testset "BamDesign interface" begin
    rng_d = StableRNG(4242)
    n, p = 600, 7
    X = hcat(ones(n), randn(rng_d, n, p - 1))
    w = 0.5 .+ rand(rng_d, n)
    z = randn(rng_d, n)
    b = randn(rng_d, p)

    @testset "construction and accessors" begin
        D = bam_design(X)
        @test D isa DenseDesign
        @test D isa BamDesign
        @test ncols(D) == p
        @test nrows(D) == n
        @test size(D) == (n, p)
        # The wrapper must not copy: a design is created per fit and X is large.
        @test D.X === X
    end

    @testset "intercept_col: found, memoised, absent" begin
        D = bam_design(X)
        # Not scanned until asked (the scan is O(n*p) and warm-started inner
        # solves never need it).
        @test D.icpt[] == -2
        @test intercept_col(D) == 1
        @test D.icpt[] == 1
        # Second call is served from the memo.
        @test intercept_col(D) == 1

        # Intercept not in column 1 — the whole reason this is a scan and not
        # an assumption.
        X2 = hcat(randn(rng_d, n, 2), ones(n), randn(rng_d, n, 2))
        @test intercept_col(bam_design(X2)) == 3

        # No intercept column at all returns 0, not `nothing`, so the caller
        # stays type-stable.
        Dn = bam_design(randn(rng_d, n, 3))
        @test intercept_col(Dn) === 0
    end

    @testset "kernels agree exactly with the chunked functions" begin
        D = bam_design(X)

        # mul_eta! == mul!(eta, X, beta), bitwise
        eta_d = zeros(n)
        eta_r = zeros(n)
        mul_eta!(eta_d, D, b)
        mul!(eta_r, X, b)
        @test eta_d == eta_r

        for cs in (37, 1024, 10_000, 10^7)
            XtWX_d = zeros(p, p); XtWz_d = zeros(p)
            XtWX_r = zeros(p, p); XtWz_r = zeros(p)
            accumulate_XtWX_XtWz!(XtWX_d, XtWz_d, D, w, z; chunk_size = cs)
            _accumulate_XtWX_XtWz_chunked!(XtWX_r, XtWz_r, X, w, z, cs)
            @test XtWX_d == XtWX_r
            @test XtWz_d == XtWz_r

            XtWX_d2 = zeros(p, p); XtWX_r2 = zeros(p, p)
            accumulate_XtWX!(XtWX_d2, D, w; chunk_size = cs)
            _accumulate_XtWX_chunked!(XtWX_r2, X, w, cs)
            @test XtWX_d2 == XtWX_r2
        end
    end

    @testset "design_finalize agrees with pirls_finalize" begin
        D = bam_design(X)
        XtWX = zeros(p, p)
        accumulate_XtWX!(XtWX, D, w)
        A_chol = cholesky(Symmetric(XtWX + 1e-6 * I))

        for chd in (true, false)
            e_d, h_d, R_d = design_finalize(D, w, XtWX, A_chol;
                compute_hat_diag = chd)
            e_r, h_r, R_r = pirls_finalize(X, w, XtWX, A_chol;
                compute_hat_diag = chd)
            @test e_d == e_r
            @test h_d == h_r
            @test R_d == R_r
        end
        # Declining hat_diag is the documented escape hatch for designs that
        # cannot form an O(n*p^2) leverage sweep cheaply.
        _, h_off, _ = design_finalize(D, w, XtWX, A_chol; compute_hat_diag = false)
        @test h_off == Float64[]
    end

    @testset "type stability" begin
        D = bam_design(X)
        eta = zeros(n)
        XtWX = zeros(p, p); XtWz = zeros(p)
        @test @inferred(ncols(D)) == p
        @test @inferred(nrows(D)) == n
        @test @inferred(intercept_col(D)) == 1
        @test @inferred(size(D)) == (n, p)
        @inferred mul_eta!(eta, D, b)
        @inferred accumulate_XtWX_XtWz!(XtWX, XtWz, D, w, z)
        @inferred accumulate_XtWX!(XtWX, D, w)
        A_chol = cholesky(Symmetric(XtWX + 1e-6 * I))
        @inferred design_finalize(D, w, XtWX, A_chol)
    end

    @testset "Matrix wrappers dispatch to the design path identically" begin
        nn = 300
        x1 = rand(rng_d, nn)
        yg = sin.(2π .* x1) .+ 0.3 .* randn(rng_d, nn)
        yp = Float64.(rand.(rng_d, Poisson.(exp.(0.5 .* sin.(2π .* x1) .+ 1))))
        df = DataFrame(x1 = x1, yg = yg, yp = yp)

        # Build the same design matrix bam would, then drive both entry points.
        y_g, Xg, _, smooths_g, npar_g = GAM.setup_gam(
            GAM.@formula(yg ~ s(x1)), df; family = Normal())
        S0 = zeros(size(Xg, 2), size(Xg, 2))

        r_mat = pirls_bam(Xg, y_g, S0, Normal(), IdentityLink())
        r_des = pirls_bam(bam_design(Xg), y_g, S0, Normal(), IdentityLink())
        @test r_mat.coefficients == r_des.coefficients
        @test r_mat.deviance == r_des.deviance
        @test r_mat.edf_vec == r_des.edf_vec
        @test r_mat.hat_diag == r_des.hat_diag

        pen_a = setup_penalties(smooths_g, npar_g)
        pen_b = setup_penalties(smooths_g, npar_g)
        sp_mat, res_mat = outer_iteration_bam(Xg, y_g, smooths_g, pen_a,
            Normal(), IdentityLink())
        sp_des, res_des = outer_iteration_bam(bam_design(Xg), y_g, smooths_g,
            pen_b, Normal(), IdentityLink())
        @test sp_mat == sp_des
        @test res_mat.coefficients == res_des.coefficients
        @test res_mat.edf_vec == res_des.edf_vec

        # Non-Gaussian exercises the P-IRLS loop rather than the fast path.
        y_p, Xp, _, smooths_p, npar_p = GAM.setup_gam(
            GAM.@formula(yp ~ s(x1)), df; family = Poisson())
        pen_c = setup_penalties(smooths_p, npar_p)
        pen_d = setup_penalties(smooths_p, npar_p)
        spp_mat, resp_mat = outer_iteration_bam(Xp, y_p, smooths_p, pen_c,
            Poisson(), LogLink())
        spp_des, resp_des = outer_iteration_bam(bam_design(Xp), y_p, smooths_p,
            pen_d, Poisson(), LogLink())
        @test spp_mat == spp_des
        @test resp_mat.coefficients == resp_des.coefficients
        @test resp_mat.deviance == resp_des.deviance
    end
end
