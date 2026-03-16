# Pretty-printing for GAM types

function Base.show(io::IO, ::MIME"text/plain", m::GamModel)
    println(io, "Generalized Additive Model")
    println(io)

    if m.formula !== nothing
        println(io, "Formula: ", m.formula)
        println(io)
    end

    if m.family isa ExtendedFamily
        println(io, "Family: ", _family_name(m.family))
    else
        println(io, "Family: ", nameof(typeof(m.family)))
    end
    println(io, "Link:   ", nameof(typeof(m.link)))
    println(io, "Method: ", m.method)
    println(io)

    # Parametric coefficients
    println(io, "Parametric coefficients:")
    ct = coeftable(m)
    show(io, MIME("text/plain"), ct)
    println(io)
    println(io)

    # Smooth terms summary
    if m.n_smooth > 0
        println(io, "Approximate significance of smooth terms:")
        println(io, "─" ^ 50)
        @printf(io, "%-20s %8s %8s\n", "Smooth", "edf", "Ref.df")
        println(io, "─" ^ 50)
        for (i, sm) in enumerate(m.smooths)
            k_eff = size(sm.X, 2)
            @printf(io, "%-20s %8.2f %8d\n",
                sm.spec.label, m.edf[i], k_eff)
        end
        println(io, "─" ^ 50)
        println(io)
    end

    @printf(io, "R² (adj) = %.3f", adjr2(m))
    dev_expl = deviance_explained(m) * 100
    @printf(io, "   Deviance explained = %.1f%%\n", dev_expl)
    if _needs_scale_estimate(m.family)
        @printf(io, "Scale est. = %.4f   ", m.scale)
    end
    if m.family isa NegBinFamily
        @printf(io, "Theta est. = %.4f   ", m.family.theta)
    elseif m.family isa BetaFamily
        @printf(io, "Phi est. = %.4f   ", m.family.phi)
    elseif m.family isa TweedieFamily
        @printf(io, "Power = %.4f   ", m.family.p)
    end
    @printf(io, "n = %d\n", nobs(m))

    return nothing
end

function Base.show(io::IO, m::GamModel)
    print(io, "GamModel(")
    print(io, "n_smooth=$(m.n_smooth), ")
    print(io, "edf=$(round(m.edf_total; digits=1)), ")
    print(io, "deviance=$(round(m.deviance_val; digits=2))")
    print(io, ")")
end

function Base.show(io::IO, spec::SmoothSpec)
    print(io, spec.label)
end

function Base.show(io::IO, ::MIME"text/plain", spec::SmoothSpec)
    println(io, "SmoothSpec: ", spec.label)
    println(io, "  Variables: ", join(string.(spec.term_vars), ", "))
    println(io, "  Basis: ", nameof(typeof(spec.basis)))
    println(io, "  k: ", spec.k)
    spec.by !== nothing && println(io, "  by: ", spec.by)
    spec.fx && println(io, "  Fixed (unpenalized)")
end

function Base.show(io::IO, sm::ConstructedSmooth)
    k = size(sm.X, 2)
    print(io, "ConstructedSmooth($(sm.spec.label), k=$k, rank=$(sm.rank))")
end
