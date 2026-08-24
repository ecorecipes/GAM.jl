# Stability of log|Σⱼ λⱼ Sⱼ|₊ within multi-penalty blocks (te/ti/t2, adaptive).
#
# Before the similarity-transform port, `_log_penalty_det` factored out λmax per
# block. That is exact across blocks but not within one: once the within-block λ
# ratio approached 1/eps the sub-dominant components fell below the eigenvalue
# threshold and their contribution was lost silently. Measured against a 256-bit
# reference, a `te` smooth lost 0.1 nats at ratio 1e16 and 108 nats at 1e24, `t2`
# lost 96 nats from 1e16 on, and the adaptive basis returned NaN at 1e24.
# Within-`te` marginal ratios of 1e8–1e16 arise routinely when one marginal is
# driven toward linearity, and `select = true` pushes further.
#
# The port (`_stable_penalty_factor`) reproduces mgcv 1.9-4's `gam.reparam` to
# all printed digits — verified out-of-band by feeding identical projected
# penalties to `mgcv:::gam.reparam`: for a te block, 271.98989335 / 548.30010407
# / 824.61031523 at ratios 1e8 / 1e16 / 1e24 against our 271.98989335 /
# 548.30010407 / 824.61031523; and for a 5-penalty adaptive block 109.45701204 /
# 275.24308305 / 441.02920975, matching ours exactly.
#
# Note mgcv is itself NOT exact at the most extreme ratios: at 1e24 both it and
# this port sit ~109 nats below the true pseudo-determinant, because the
# transform discards the dominant term's own tail below `r.tol`. Parity with
# mgcv is the target here, so the assertions below check exactness only where
# the algorithm is exact (ratios up to ~1e8), and check finiteness, rank
# consistency and gradient agreement everywhere.

@testset "Penalty log-determinant stability" begin
    _rng = StableRNG(20240824)
    _n = 250
    _data = (x = rand(_rng, _n), z = rand(_rng, _n), u = rand(_rng, _n),
             t = collect(range(0, 1; length = _n)))

    function _setup(specs)
        sms = [GAM.smooth_construct(sp, _data) for sp in specs]
        off = 1
        for sm in sms
            sm.first_para = off + 1
            sm.last_para = off + size(sm.X, 2)
            off = sm.last_para
        end
        GAM.setup_penalties(sms, 1)
    end

    # 256-bit reference for log|Σλⱼ Sⱼ|₊ over the (λ-independent) range space.
    function _reference(Ss, lsp)
        Z = GAM._total_range_basis(Ss)
        setprecision(BigFloat, 300) do
            Zb = BigFloat.(Z)
            A = zeros(BigFloat, size(Zb, 2), size(Zb, 2))
            for (j, S) in enumerate(Ss)
                A .+= exp(BigFloat(lsp[j])) .* (Zb' * BigFloat.(S) * Zb)
            end
            Float64(logabsdet(A)[1])
        end
    end

    @testset "single-penalty blocks are unchanged" begin
        # The multi-penalty branch must not perturb the ordinary path: with one
        # term, factoring out λmax is already exact.
        for specs in ([GAM.s(:x; k = 10)],
                      [GAM.s(:x; bs = :cr, k = 12)],
                      [GAM.s(:x; k = 10), GAM.s(:z; bs = :ps, k = 8)])
            pen = _setup(specs)
            nsp = length(pen.sp)
            for _ in 1:15
                lsp = randn(_rng, nsp) .* 8.0
                got = GAM._log_penalty_det(pen, lsp)
                # closed form: Σ log(non-zero eigenvalues of Sⱼ) + rankⱼ·log λⱼ
                want = 0.0
                for (j, block) in enumerate(pen.blocks)
                    ev = eigvals(Symmetric(Matrix(block.S[1])))
                    thresh = eps() * maximum(abs.(ev))
                    for e in ev
                        e > thresh && (want += log(e) + lsp[j])
                    end
                end
                @test got ≈ want atol = 1e-9
                @test isfinite(got)
            end
        end
    end

    multi = [("te", [GAM.te(:x, :z; k = 5)]),
             ("t2", [GAM.t2(:x, :z; k = 5)]),
             ("ad", [GAM.s(:t; bs = :ad, k = 20)])]

    @testset "$name: finite and monotone across extreme λ ratios" for (name, specs) in multi
        pen = _setup(specs)
        nsp = length(pen.sp)
        prev = -Inf
        for r in (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0)
            lsp = zeros(nsp)
            lsp[end] = r * log(10)
            ld = GAM._log_penalty_det(pen, lsp)
            # The 1e24 adaptive case used to return NaN outright.
            @test isfinite(ld)
            # log|S| is increasing in every λ, so it must increase with r.
            @test ld > prev
            prev = ld
        end
    end

    @testset "$name: exact where the algorithm is exact (ratio ≤ 1e8)" for (name, specs) in multi
        pen = _setup(specs)
        blk = pen.blocks[1]
        Ss = Matrix{Float64}[Matrix(S) for S in blk.S]
        nsp = length(Ss)
        for r in (0.0, 4.0, 8.0)
            lsp = zeros(nsp)
            lsp[end] = r * log(10)
            @test GAM._log_penalty_det(pen, lsp) ≈ _reference(Ss, lsp) rtol = 1e-6
        end
    end

    @testset "$name: derivatives sum to the penalty rank" for (name, specs) in multi
        pen = _setup(specs)
        blk = pen.blocks[1]
        nsp = length(blk.S)
        for r in (0.0, 6.0, 12.0, 20.0)
            lsp = zeros(nsp)
            lsp[end] = r * log(10)
            d = GAM._stable_block_logdet_derivs(blk, lsp)
            @test all(isfinite, d)
            @test all(>=(-1e-8), d)                 # each λⱼ·tr(S⁺Sⱼ) ≥ 0
            @test sum(d) ≈ blk.rank rtol = 1e-8
        end
    end

    @testset "$name: gradient agrees with finite differences" for (name, specs) in multi
        pen = _setup(specs)
        blk = pen.blocks[1]
        nsp = length(blk.S)
        for base in (0.0, 2.0, -3.0)
            lsp = fill(base, nsp)
            nsp > 1 && (lsp[end] += 8.0)
            an = GAM._stable_block_logdet_derivs(blk, lsp)
            h = 1e-6
            for j in 1:nsp
                lp = copy(lsp); lm = copy(lsp)
                lp[j] += h; lm[j] -= h
                fd = (GAM._log_penalty_det(pen, lp) -
                      GAM._log_penalty_det(pen, lm)) / (2h)
                @test an[j] ≈ fd rtol = 1e-5 atol = 1e-6
            end
        end
    end

    @testset "similarity transform is orthogonal and determinant preserving" begin
        pen = _setup([GAM.te(:x, :z; k = 5)])
        blk = pen.blocks[1]
        Ss = Matrix{Float64}[Matrix(S) for S in blk.S]
        lsp = [0.0, 16.0 * log(10)]
        st = GAM._stable_penalty_factor(Ss, lsp)
        @test st !== nothing
        d = size(st.Qf, 1)
        @test norm(st.Qf' * st.Qf - I) < 1e-12          # Qf orthogonal
        @test norm(st.Z' * st.Z - I) < 1e-12            # range basis orthonormal
        # St = Qf' (Z' S_λ Z) Qf
        A = zeros(d, d)
        for (j, S) in enumerate(Ss)
            A .+= exp(lsp[j]) .* (st.Z' * S * st.Z)
        end
        @test norm(st.Qf' * A * st.Qf - st.St) / norm(st.St) < 1e-12
    end

    @testset "empty penalty is handled" begin
        Ss = [zeros(4, 4), zeros(4, 4)]
        @test GAM._total_range_basis(Ss) == zeros(4, 0)
        @test GAM._stable_penalty_factor(Ss, [0.0, 0.0]) === nothing
    end
end
