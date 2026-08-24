@testset "Vector sp for multi-penalty smooths" begin
    using Random, DataFrames, Serialization
    using StatsAPI: fitted, coef

    Random.seed!(11)
    n = 300
    xv = collect(range(0, 1; length = n))
    zv = rand(n)
    gv = string.("L", repeat(1:4, inner = n ÷ 4))
    yv = sin.(6π .* xv .^ 2) ./ (1 .+ 4 .* xv) .+ 0.5 .* zv .+ 0.2 .* randn(n)
    dfv = DataFrame(x = xv, z = zv, g = gv, y = yv)

    _fit(spec) = gam(GAM.GamFormula(:y, Symbol[], true, [spec]), dfv)

    @testset "construction accepts and normalizes a vector" begin
        sa = GAM.s(:x; bs = :ad, k = 20, sp = [1.0, 2.0, 3.0, 4.0, 5.0])
        @test sa.sp == [1.0, 2.0, 3.0, 4.0, 5.0]
        @test sa.sp isa Vector{Float64}

        # integer input is coerced, not rejected
        si = GAM.s(:x; bs = :ad, k = 20, sp = [1, 2, 3, 4, 5])
        @test si.sp isa Vector{Float64}
        @test si.sp == [1.0, 2.0, 3.0, 4.0, 5.0]

        @test GAM.t2(:x, :z; k = 5, sp = [1.0, 2.0, 3.0]).sp == [1.0, 2.0, 3.0]
        @test GAM.s(:x, :g; bs = :fs, k = 6, sp = [1.0, 2.0, 3.0]).sp == [1.0, 2.0, 3.0]
    end

    @testset "scalar sp is unchanged" begin
        # Stored as a bare Float64, exactly as before vector support existed.
        sc = GAM.s(:x; k = 10, sp = 2.0)
        @test sc.sp === 2.0
        @test GAM.s(:x; bs = :ad, k = 20, sp = 3).sp === 3.0
        @test GAM.s(:x; k = 10).sp === nothing

        # A scalar still fixes EVERY penalty of a multi-penalty smooth at that
        # value, and larger sp still means more smoothing. This is the
        # broadcast semantics that predates vector support.
        edfs = [sum(GAM.edf(_fit(GAM.s(:x; bs = :ad, k = 20, sp = v))))
                for v in (0.01, 1.0, 100.0)]
        @test issorted(edfs; rev = true)
        @test edfs[1] > edfs[3]
    end

    @testset "invalid sp is rejected at construction" begin
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = -1.0)
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = 0.0)
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = NaN)
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = Inf)
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = Float64[])
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = [1.0, -2.0])
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = [1.0, NaN])
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = "0.5")
        # sp= and fx=true stay incompatible for the vector form too
        @test_throws ArgumentError GAM.s(:x; k = 10, sp = [1.0, 2.0], fx = true)
        @test_throws ArgumentError GAM.t2(:x, :z; k = 5, sp = [1.0, 2.0, 3.0], fx = true)
    end

    @testset "length is validated against the constructed penalty count" begin
        # The count is only known after construction, so this fires at fit time
        # rather than in s(). The message must name the smooth and both numbers.
        err = try
            _fit(GAM.s(:x; bs = :ad, k = 20, sp = [1.0, 2.0]))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("2 entries", msg)
        @test occursin("5 penalties", msg)
        @test occursin("bs=ad", msg)

        @test_throws ArgumentError _fit(GAM.t2(:x, :z; k = 5, sp = [1.0, 2.0]))
        # too many is rejected as well as too few
        @test_throws ArgumentError _fit(GAM.s(:x; bs = :ad, k = 20,
                                              sp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    end

    @testset "sp round-trips within GAM.jl" begin
        # Free-fit, then feed the model its own per-penalty sp back as fixed.
        # Recovering the same fit is the property that makes a reported sp
        # vector meaningful; it fails if the vector is applied in the wrong
        # order or broadcast instead of indexed.
        cases = (
            ("ad", () -> GAM.s(:x; bs = :ad, k = 20),
                   sp -> GAM.s(:x; bs = :ad, k = 20, sp = sp)),
            ("t2", () -> GAM.t2(:x, :z; k = 5),
                   sp -> GAM.t2(:x, :z; k = 5, sp = sp)),
            ("fs", () -> GAM.s(:x, :g; bs = :fs, k = 6),
                   sp -> GAM.s(:x, :g; bs = :fs, k = 6, sp = sp)),
        )
        for (lbl, free, fixed) in cases
            m = _fit(free())
            spv = exp.(m.sp)
            @test length(spv) > 1                      # genuinely multi-penalty
            m2 = _fit(fixed(spv))
            @test maximum(abs.(coef(m) .- coef(m2))) < 1e-8
            @test maximum(abs.(fitted(m) .- fitted(m2))) < 1e-8
            @test abs(m.edf_total - m2.edf_total) < 1e-8
        end
    end

    @testset "entries are applied per penalty, not broadcast" begin
        # Distinct entries must produce a different fit from any single value,
        # otherwise indexing has silently degenerated to broadcasting.
        v = [1e-4, 1e2, 1.0, 1e-2, 10.0]
        m_vec = _fit(GAM.s(:x; bs = :ad, k = 20, sp = v))
        for c in v
            m_sc = _fit(GAM.s(:x; bs = :ad, k = 20, sp = c))
            @test abs(m_vec.edf_total - m_sc.edf_total) > 1e-6
        end
        # Permuting the entries changes the fit (they address distinct penalties)
        m_perm = _fit(GAM.s(:x; bs = :ad, k = 20, sp = v[[2, 1, 4, 3, 5]]))
        @test abs(m_vec.edf_total - m_perm.edf_total) > 1e-6
    end

    @testset "vector sp survives serialization" begin
        spec = GAM.s(:x; bs = :ad, k = 20, sp = [1.0, 2.0, 3.0, 4.0, 5.0])
        buf = IOBuffer(); serialize(buf, spec); seekstart(buf)
        spec2 = deserialize(buf)
        @test spec2.sp == spec.sp
        @test spec2.sp isa Vector{Float64}

        m = _fit(spec)
        buf2 = IOBuffer(); serialize(buf2, m); seekstart(buf2)
        m2 = deserialize(buf2)
        @test fitted(m) == fitted(m2)
        @test m.sp == m2.sp
    end

    @testset "show renders a model with vector sp" begin
        m = _fit(GAM.s(:x; bs = :ad, k = 20, sp = [1.0, 2.0, 3.0, 4.0, 5.0]))
        io = IOBuffer()
        @test (show(io, m); true)
        @test (show(io, MIME"text/plain"(), m); true)
        @test !isempty(String(take!(io)))
    end
end
