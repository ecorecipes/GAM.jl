# Pretty-printing for GAM types

"""
    _signif_stars(p) -> String

mgcv's significance codes: `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1.
"""
function _signif_stars(p::Real)
    isfinite(p) || return " "
    p < 0.001 && return "***"
    p < 0.01 && return "**"
    p < 0.05 && return "*"
    p < 0.1 && return "."
    return " "
end

"""
    _criterion_footer(m::GamModel) -> Union{Nothing, String}

The smoothness-selection criterion line, mirroring mgcv's `-REML = …` /
`GCV = …` footer. Returns `nothing` when no criterion value was recorded.
"""
function _criterion_footer(m::GamModel)
    if m.method in (:REML, :ML) && isfinite(m.reml)
        return @sprintf("-%s = %.4g", string(m.method), m.reml)
    elseif m.method in (:GCV, :UBRE) && isfinite(m.criterion)
        return @sprintf("%s = %.4g", string(m.method), m.criterion)
    end
    return nothing
end

function _show_gam(io::IO, m::GamModel)
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
        @printf(io, "%-20s %8s %8s %10s %10s %-4s\n",
            "Smooth", "edf", "Ref.df", stat_name, "p-value", "")
        println(io, "─" ^ 66)
        any_stars = false
        for (i, sm) in enumerate(m.smooths)
            if at !== nothing
                t = at.smooth_table
                stars = _signif_stars(t.p_value[i])
                any_stars |= stars != " "
                @printf(io, "%-20s %8.2f %8.2f %10.3f %10.4g %-4s\n",
                    sm.spec.label, t.edf[i], t.ref_df[i],
                    t.statistic[i], t.p_value[i], stars)
            else
                @printf(io, "%-20s %8.2f %8d\n",
                    sm.spec.label, m.edf[i], size(sm.X, 2))
            end
        end
        println(io, "─" ^ 66)
        if any_stars
            println(io, "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        end
        println(io)
    end

    @printf(io, "R² (adj) = %.3f", adjr2(m))
    dev_expl = deviance_explained(m) * 100
    @printf(io, "   Deviance explained = %.1f%%\n", dev_expl)
    crit = _criterion_footer(m)
    crit === nothing || print(io, crit, "   ")
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

Base.show(io::IO, ::MIME"text/plain", m::GamModel) = _show_gam(io, m)

"""
    GamSummary

Displayable model summary returned by `summary(::GamModel)`. Printing it
produces the same report as displaying the model itself.
"""
struct GamSummary
    model::GamModel
end

Base.show(io::IO, ::MIME"text/plain", s::GamSummary) = _show_gam(io, s.model)
Base.show(io::IO, s::GamSummary) = _show_gam(io, s.model)

"""
    summary(m::GamModel) -> GamSummary

Model summary in the style of mgcv's `summary.gam`: family and link, the
parametric coefficient table, approximate significance of smooth terms
(edf, Ref.df, test statistic, p-value, significance codes), and a footer
with the smoothness-selection criterion, R²(adj), deviance explained,
scale estimate and `n`.

This is the same report shown when a fitted model is displayed; `summary`
exists so that the habit carried over from mgcv works.

# Example
```julia
m = gam(@formula(y ~ s(x)), df)
summary(m)
```
"""
Base.summary(m::GamModel) = GamSummary(m)

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
