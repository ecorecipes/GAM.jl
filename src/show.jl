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

    # Smooth terms summary (Wood 2013 approximate test)
    if m.n_smooth > 0
        at = try
            anova_gam(m)
        catch
            nothing
        end
        println(io, "Approximate significance of smooth terms:")
        println(io, "─" ^ 66)
        stat_name = at !== nothing && at.test_type == :F ? "F" : "Chi.sq"
        @printf(io, "%-20s %8s %8s %10s %10s\n",
            "Smooth", "edf", "Ref.df", stat_name, "p-value")
        println(io, "─" ^ 66)
        for (i, sm) in enumerate(m.smooths)
            if at !== nothing
                t = at.smooth_table
                @printf(io, "%-20s %8.2f %8.2f %10.3f %10.4g\n",
                    sm.spec.label, t.edf[i], t.ref_df[i],
                    t.statistic[i], t.p_value[i])
            else
                @printf(io, "%-20s %8.2f %8d\n",
                    sm.spec.label, m.edf[i], size(sm.X, 2))
            end
        end
        println(io, "─" ^ 66)
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

    if !m.converged
        println(io)
        println(io, "WARNING: fit did not converge — estimates may be unreliable")
    end

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
