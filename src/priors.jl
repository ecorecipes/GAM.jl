# Prior specification for Bayesian GAMs
#
# Lightweight container mapping parameter classes/names to Distributions.jl
# prior distributions. Used by the Turing.jl backend when priors= is passed
# to gam(), gamlss(), scam(), etc.

"""
    PriorSpec

Specification of priors for Bayesian GAM fitting. Priors are looked up
hierarchically: specific parameter name > parameter class > default.

# Constructor
```julia
PriorSpec(;
    b = Normal(0, 10),                    # default for all fixed effects
    sds = truncated(TDist(3); lower=0),  # default for smooth SDs
    sigma = truncated(Cauchy(0, 2.5); lower=0),      # residual SD
    specific = Dict{String, Distribution}(),          # per-parameter overrides
)
```

# Parameter classes
- `b`: fixed/parametric effect coefficients (intercept, linear terms)
- `sds`: standard deviations of smooth random effects (controls smoothness)
- `sigma`: residual standard deviation (for Gaussian family)
- `phi`: precision/dispersion (for Beta, NegBin families)

# Specific overrides
Use the `specific` dictionary to set priors on individual parameters:
```julia
PriorSpec(
    sds = Exponential(1.0),
    specific = Dict(
        "sds_s(x2)" => Exponential(0.5),   # tighter prior on s(x2) smoothness
        "b_(Intercept)" => Normal(0, 100),  # wide prior on intercept
    )
)
```

# Examples
```julia
# Default priors (brms-like)
priors = PriorSpec()

# Custom priors
priors = PriorSpec(sds=Exponential(1.0), sigma=InverseGamma(2, 3))

# Use in gam()
m = gam(formula, data, Normal(); priors=priors, sampler=NUTS(), nsamples=2000)
```
"""
struct PriorSpec
    b::Distribution           # prior for fixed-effect coefficients
    sds::Distribution         # prior for smooth SD parameters
    sigma::Distribution       # prior for residual SD (Gaussian)
    phi::Distribution         # prior for dispersion/precision
    specific::Dict{String, Distribution}  # per-parameter overrides
end

function PriorSpec(;
    b::Distribution = Normal(0, 10),
    sds::Distribution = truncated(TDist(3); lower = 0),
    sigma::Distribution = truncated(Cauchy(0, 2.5); lower = 0),
    phi::Distribution = truncated(Cauchy(0, 5); lower = 0),
    specific::Dict{String, <:Distribution} = Dict{String, Distribution}(),
)
    return PriorSpec(b, sds, sigma, phi, specific)
end

"""
    get_prior(ps::PriorSpec, class::Symbol, name::String="") -> Distribution

Look up a prior: first checks specific[name], then falls back to class default.
"""
function get_prior(ps::PriorSpec, class::Symbol, name::String = "")
    # Check specific override first
    key = isempty(name) ? string(class) : "$(class)_$(name)"
    if haskey(ps.specific, key)
        return ps.specific[key]
    end
    if haskey(ps.specific, name)
        return ps.specific[name]
    end
    # Fall back to class default
    if class == :b
        return ps.b
    elseif class == :sds
        return ps.sds
    elseif class == :sigma
        return ps.sigma
    elseif class == :phi
        return ps.phi
    else
        error("Unknown prior class: $class")
    end
end

"""
    default_priors(family) -> PriorSpec

Sensible default priors for a given family, following brms conventions.
"""
function default_priors(family)
    return PriorSpec()
end

function Base.show(io::IO, ::MIME"text/plain", ps::PriorSpec)
    println(io, "PriorSpec:")
    println(io, "  b (fixed effects):    ", ps.b)
    println(io, "  sds (smooth SDs):     ", ps.sds)
    println(io, "  sigma (residual SD):  ", ps.sigma)
    println(io, "  phi (dispersion):     ", ps.phi)
    if !isempty(ps.specific)
        println(io, "  Specific overrides:")
        for (k, v) in sort(collect(ps.specific); by = first)
            println(io, "    ", k, " → ", v)
        end
    end
end

function Base.show(io::IO, ps::PriorSpec)
    n_spec = length(ps.specific)
    print(io, "PriorSpec(sds=", ps.sds)
    if n_spec > 0
        print(io, ", ", n_spec, " specific")
    end
    print(io, ")")
end
