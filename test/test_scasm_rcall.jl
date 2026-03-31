using Test, GAM, DataFrames, Random, Statistics, StatsAPI, RCall

R"library(mgcv)"

function _fit_julia_constrained(x, y, xt, bs::Symbol; family = Normal(), intercept::Bool = true)
    df = DataFrame(x = x, y = y)

    if intercept
        f = @gam_formula(y ~ s(x, bs = bs, xt = xt, k = 10))
    else
        f = @gam_formula(y ~ 0 + s(x, bs = bs, xt = xt, k = 10))
    end

    if family isa Normal
        return gam(f, df)
    elseif family isa Poisson
        return gam(f, df; family = Poisson())
    else
        throw(ArgumentError("Unsupported family for scasm comparison: $(typeof(family))"))
    end
end

function _check_scasm_fit(x, y, xt::Vector{String}, bs::Symbol; threshold::Float64 = 0.99,
                          family = Normal(), intercept::Bool = true)
    bs_r = bs == :scad ? "scad" : "sc"
    @rput x y xt bs_r intercept
    if family isa Normal
        R"""
        form <- if (intercept) y ~ s(x, bs=bs_r, xt=xt, k=10) else y ~ 0 + s(x, bs=bs_r, xt=xt, k=10)
        m_r <- scasm(form, family=gaussian(), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
    elseif family isa Poisson
        R"""
        form <- if (intercept) y ~ s(x, bs=bs_r, xt=xt, k=10) else y ~ 0 + s(x, bs=bs_r, xt=xt, k=10)
        m_r <- scasm(form, family=poisson(), data=data.frame(x=x, y=y))
        fitted_r <- fitted(m_r)
        """
    else
        throw(ArgumentError("Unsupported family for scasm comparison: $(typeof(family))"))
    end

    m_jl = _fit_julia_constrained(x, y, xt, bs; family = family, intercept = intercept)
    fitted_r = rcopy(R"fitted_r")
    @test cor(m_jl.fitted_values, fitted_r) > threshold
end

@testset "SCASM R comparison" begin
    rng = MersenneTwister(42)
    n = 200
    x = sort(rand(rng, n))

    @testset "Gaussian constrained fits match mgcv::scasm" begin
        cases = [
            ("m+ -> :mpi", ["m+"], :mpi, 3.0 .* x .+ 0.1 .* randn(rng, n), 0.999),
            ("m- -> :mpd", ["m-"], :mpd, 3.0 .- 3.0 .* x .+ 0.1 .* randn(rng, n), 0.999),
            ("c+ -> :cx", ["c+"], :cx, x .^ 2 .+ 0.1 .* randn(rng, n), 0.998),
            ("c- -> :cv", ["c-"], :cv, sqrt.(x) .+ 0.1 .* randn(rng, n), 0.995),
            ("m+ c+ -> :micx", ["m+", "c+"], :micx, x .^ 2 .+ 0.1 .* randn(rng, n), 0.995),
            ("m+ c- -> :micv", ["m+", "c-"], :micv, sqrt.(x) .+ 0.1 .* randn(rng, n), 0.995),
            ("m- c+ -> :mdcx", ["m-", "c+"], :mdcx, (1 .- x) .^ 2 .+ 0.1 .* randn(rng, n), 0.995),
            ("m- c- -> :mdcv", ["m-", "c-"], :mdcv, .-x .^ 2 .+ 0.1 .* randn(rng, n), 0.995),
        ]

        for (label, xt, bs, y, threshold) in cases
            @testset "$label" begin
                _check_scasm_fit(x, y, xt, bs; threshold = threshold)
            end
        end
    end

    @testset "Poisson monotone increasing fit matches mgcv::scasm" begin
        eta = 1 .+ 0.8 .* x
        mu = exp.(eta)
        y = Float64.([rand(rng, Poisson(μ)) for μ in mu])
        _check_scasm_fit(x, y, ["m+"], :mpi; threshold = 0.99, family = Poisson())
    end

    @testset "Explicit linear-constraint bases match mgcv::scasm" begin
        cases = [
            ("sc m+", ["m+"], :sc, 3.0 .* x .+ 0.08 .* randn(rng, n), 0.995, true),
            ("sc c+", ["c+"], :sc, 0.3 .+ x .^ 2 .+ 0.05 .* randn(rng, n), 0.995, true),
            ("sc +", ["+"], :sc, 0.4 .+ 0.8 .* x .+ 0.03 .* randn(rng, n), 0.995, false),
            ("scad m+", ["m+"], :scad, 3.0 .* x .+ 0.08 .* randn(rng, n), 0.99, true),
            ("scad c+", ["c+"], :scad, 0.3 .+ x .^ 2 .+ 0.05 .* randn(rng, n), 0.99, true),
        ]

        for (label, xt, bs, y, threshold, intercept) in cases
            @testset "$label" begin
                _check_scasm_fit(x, y, xt, bs; threshold = threshold, intercept = intercept)
            end
        end
    end

    @testset "Poisson sc basis matches mgcv::scasm" begin
        eta = 0.8 .+ 0.9 .* x
        mu = exp.(eta)
        y = Float64.([rand(rng, Poisson(μ)) for μ in mu])
        _check_scasm_fit(x, y, ["m+"], :sc; threshold = 0.99, family = Poisson())
    end
end
