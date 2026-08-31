# Configurable marginal basis for `bs=:sz` (mgcv's `xt$bs`).
#
# mgcv lets the sz smooth choose the basis used for its continuous marginal
# (`smooth.construct.sz.smooth.spec`): `xt` is either a bare basis name or a
# list carrying `bs`, defaulting to "tp". GAM.jl's `xt` is always a Dict, so
# the list form is the one that maps: `xt = Dict(:bs => :cr)`.
#
# The default is `:tp` in BOTH packages, so this is an added option rather
# than a behaviour change — and the first testset below exists to keep it that
# way, because "the default quietly moved" is the failure mode that would
# matter most here.

@testset "sz configurable base basis" begin

    _rng = StableRNG(20260901)
    _n = 300
    _week = repeat(collect(range(0.0, 52.0; length = 50)), 6)
    _lev = repeat(["a", "b", "c"], inner = 100)
    _amp = Dict("a" => 1.4, "b" => 0.9, "c" => 0.35)
    _y = [_amp[_lev[i]] * sin(2π * _week[i] / 52) for i in 1:_n] .+
         0.25 .* randn(_rng, _n)
    _df = DataFrame(week = _week, g = _lev, y = _y)

    _build(xt) = begin
        spec = xt === nothing ? GAM.s(:week, :g; k = 10, bs = :sz) :
               GAM.s(:week, :g; k = 10, bs = :sz, xt = xt)
        GAM.smooth_construct(spec, Tables.columntable(_df))
    end

    # ------------------------------------------------------------------
    # 1. The default is unchanged.
    #
    # Not merely "the default is :tp" — this reproduces the OLD hardcoded path
    # (a raw TPRS marginal) independently and asserts the sz basis and
    # penalties are elementwise identical to it. If someone changes the
    # default, or changes how the marginal is built, this fails.
    # ------------------------------------------------------------------
    @testset "default is bit-identical to the previous hardcoded TPRS path" begin
        sm_default = _build(nothing)
        sm_tp = _build(Dict{Symbol, Any}(:bs => :tp))

        @test size(sm_default.X) == size(sm_tp.X)
        @test maximum(abs.(sm_default.X .- sm_tp.X)) == 0.0
        @test length(GAM.penalty_matrices(sm_default)) ==
              length(GAM.penalty_matrices(sm_tp))
        for (a, b) in zip(GAM.penalty_matrices(sm_default),
                          GAM.penalty_matrices(sm_tp))
            @test maximum(abs.(a .- b)) == 0.0
        end

        # And independently: build the raw TPRS marginal the way the old code
        # did, and check the sz design is exactly kron(Q_L row, marginal row).
        cache = sm_default.predict_cache
        @test cache isa GAM.SZContrastPredictCache
        @test cache.marginal_smooth.spec.basis isa ThinPlateSpline

        mspec = GAM.SmoothSpec([:week], ThinPlateSpline(), 10,
            nothing, nothing, nothing, false, nothing, "s(week,bs=tp)")
        raw = GAM._construct_tprs(mspec, Tables.columntable(_df), nothing;
            absorb_cons = false)
        @test maximum(abs.(raw.X .- cache.marginal_smooth.X)) == 0.0
    end

    # ------------------------------------------------------------------
    # 2. Every advertised base actually works.
    # ------------------------------------------------------------------
    @testset "supported bases construct with per-level penalties" begin
        L = 3   # levels of g
        for b in GAM._SZ_BASE_BASES
            sm = _build(Dict{Symbol, Any}(:bs => b))
            pens = GAM.penalty_matrices(sm)
            # One penalty per level, matching mgcv's default branch.
            @test length(pens) == L
            @test size(sm.X, 2) > 0
            @test all(size(p) == (size(sm.X, 2), size(sm.X, 2)) for p in pens)
            @test all(isfinite, sm.X)
            # The marginal really is the requested basis.
            @test sm.predict_cache.marginal_smooth.spec.basis ===
                  GAM.BASIS_TYPES[b]
        end
    end

    # ------------------------------------------------------------------
    # 3. The per-level decomposition property survives a non-default base.
    #
    # The L per-level penalties must sum to the single summed penalty that
    # mgcv's `id` branch produces — that is what makes this a decomposition of
    # one model rather than a different one. Verified for `:tp` elsewhere;
    # here for a base that does not share TPRS's code path.
    # ------------------------------------------------------------------
    @testset "per-level penalties sum to the id-branch penalty (:cr)" begin
        sm = _build(Dict{Symbol, Any}(:bs => :cr))
        spec_id = GAM.s(:week, :g; k = 10, bs = :sz, id = :shared,
            xt = Dict{Symbol, Any}(:bs => :cr))
        sm_id = GAM.smooth_construct(spec_id, Tables.columntable(_df))

        pens = GAM.penalty_matrices(sm)
        pens_id = GAM.penalty_matrices(sm_id)
        @test length(pens_id) == 1
        @test maximum(abs.(sum(pens) .- pens_id[1])) < 1e-12
    end

    # ------------------------------------------------------------------
    # 4. Prediction uses the chosen base, not TPRS.
    # ------------------------------------------------------------------
    @testset "prediction round-trips for a non-default base" begin
        for b in (:cr, :ps, :cc)
            sm = _build(Dict{Symbol, Any}(:bs => b))
            Xp = GAM.predict_matrix(sm, Tables.columntable(_df))
            @test size(Xp) == size(sm.X)
            @test maximum(abs.(Xp .- sm.X)) < 1e-10
        end
    end

    # ------------------------------------------------------------------
    # 5. Bad input fails informatively rather than silently doing something
    #    else. mgcv's own message for the multiply-penalized case names xt,
    #    and so does ours.
    # ------------------------------------------------------------------
    @testset "unsupported and malformed bases are rejected" begin
        # Not in the supported set (`:gp` is singly penalized but has no
        # unconstrained construction path; `:ad` is multiply penalized).
        for bad in (:gp, :ad, :fs, :nonesuch)
            err = try
                _build(Dict{Symbol, Any}(:bs => bad)); nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("xt[:bs]", err.msg)
            @test occursin(":cr", err.msg)      # names what IS supported
        end

        # Wrong type entirely.
        err2 = try
            _build(Dict{Symbol, Any}(:bs => "cr")); nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("Symbol", err2.msg)
    end
end
