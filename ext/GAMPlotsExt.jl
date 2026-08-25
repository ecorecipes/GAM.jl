module GAMPlotsExt

using GAM
using Plots: @recipe, @series
using LinearAlgebra: diag

# ─── Smooth effect plot recipe ───────────────────────────────────────────────

@recipe function f(gp::GAM.GamPlot)
    m = gp.model
    n_smooth = length(m.smooths)
    n_smooth > 0 || return nothing

    idx_range = gp.select === nothing ? (1:n_smooth) : [gp.select]

    layout --> (1, length(idx_range))
    legend --> false

    for (panel, si) in enumerate(idx_range)
        sm = m.smooths[si]
        spec = sm.spec
        length(spec.term_vars) == 1 || continue

        var = spec.term_vars[1]
        sm_idx = sm.first_para:sm.last_para
        beta_s = m.coefficients[sm_idx]

        # `sm.X` is bitwise `m.X[:, sm_idx]` for a dense fit, but under
        # `discrete=true` it is the reduced `m x k` basis with `sm.rowmap`
        # giving each row's source. `smooth_matrix` returns the n-row form
        # either way; reading it still avoids reassembling the whole model
        # matrix once per plotted panel.
        X_col = GAM.smooth_matrix(sm)
        x_orig = _extract_x(m, sm)
        x_lo, x_hi = extrema(x_orig)
        x_grid = range(x_lo, x_hi; length=gp.n_grid) |> collect

        newdata = NamedTuple{(var,)}((x_grid,))
        X_pred = GAM.predict_matrix(sm, newdata)
        f_hat = X_pred * beta_s

        @series begin
            subplot := panel
            xguide --> string(var)
            yguide --> string("s(", var, ",", round(m.edf[si]; digits=1), ")")
            seriestype := :line
            linewidth --> 2
            linecolor --> :steelblue

            if gp.se
                Vp_s = m.Vp[sm_idx, sm_idx]
                se_grid = sqrt.(max.(diag(X_pred * Vp_s * X_pred'), 0.0))
                ribbon --> 2 .* se_grid
                fillalpha --> 0.2
                fillcolor --> :steelblue
            end

            x_grid, f_hat
        end

        if gp.residuals
            @series begin
                subplot := panel
                seriestype := :scatter
                markersize --> 2
                markeralpha --> 0.4
                markercolor --> :gray40
                markerstrokewidth --> 0

                # WORKING residuals (y-μ)·dη/dμ, so residuals live on the
                # link scale like the smooth term itself
                f_vals = X_col * beta_s
                wresid = similar(m.y)
                for i in eachindex(m.y)
                    dm = GAM.GLM.mueta(m.link, m.linear_predictor[i])
                    dm = abs(dm) < eps() ? eps() : dm
                    wresid[i] = (m.y[i] - m.fitted_values[i]) / dm
                end
                partial_resid = f_vals .+ wresid
                x_orig, partial_resid
            end
        end
    end
end

# ─── GamModel recipe (convenience) ──────────────────────────────────────────

@recipe function f(m::GAM.GamModel)
    GAM.gamplot(m; se=true, residuals=false)
end

# ─── 2D contour plot recipe ─────────────────────────────────────────────────

@recipe function f(gcp::GAM.GamContourPlot)
    m = gcp.model
    sm = m.smooths[gcp.select]
    spec = sm.spec

    length(spec.term_vars) == 2 ||
        error("gamcontour requires a 2D smooth, got $(length(spec.term_vars))D")

    var1, var2 = spec.term_vars
    sm_idx = sm.first_para:sm.last_para
    beta_s = m.coefficients[sm_idx]

    # Evaluate the smooth surface on a grid over the two covariates
    n_grid = 50
    x1_orig = _extract_var(m, var1)
    x2_orig = _extract_var(m, var2)
    x1_grid = collect(range(extrema(x1_orig)...; length = n_grid))
    x2_grid = collect(range(extrema(x2_orig)...; length = n_grid))
    # x1 varies fastest (column-major fill of Z)
    x1_full = repeat(x1_grid, n_grid)
    x2_full = repeat(x2_grid; inner = n_grid)
    newdata = NamedTuple{(var1, var2)}((x1_full, x2_full))
    f_hat = GAM.predict_matrix(sm, newdata) * beta_s
    Z = reshape(f_hat, n_grid, n_grid)  # rows: x1, cols: x2

    seriestype --> :contourf
    xguide --> string(var1)
    yguide --> string(var2)
    title --> "$(spec.label), edf=$(round(m.edf[gcp.select]; digits=1))"
    colorbar_title --> "Effect"
    fillalpha --> 0.8
    # Plots' contour expects z[j, i] over (x[i], y[j]): transpose so rows
    # index x2 (y-axis) and columns index x1 (x-axis)
    x1_grid, x2_grid, Z'
end

# ─── PartialResiduals recipe ────────────────────────────────────────────────

@recipe function f(pr::GAM.PartialResiduals)
    unique_smooths = unique(pr.smooth)
    n_panels = length(unique_smooths)

    layout --> (1, n_panels)
    legend --> false

    for (panel, label) in enumerate(unique_smooths)
        mask = pr.smooth .== label
        @series begin
            subplot := panel
            seriestype := :scatter
            markersize --> 2
            markeralpha --> 0.5
            title --> label
            xguide --> first(pr.xname[mask])
            yguide --> "partial residual"
            pr.x[mask], pr.residual[mask]
        end
    end
end

# ─── SmoothEstimates recipe ─────────────────────────────────────────────────

@recipe function f(se::GAM.SmoothEstimates)
    unique_smooths = unique(se.smooth)
    n_panels = length(unique_smooths)

    layout --> (1, n_panels)
    legend --> false

    for (panel, label) in enumerate(unique_smooths)
        mask = se.smooth .== label
        est = se.estimate[mask]
        se_vals = se.se[mask]

        # Covariate vectors are row-aligned with the estimates table (NaN
        # for rows whose smooth does not use that covariate) — pick the
        # first covariate fully populated for this smooth's rows
        x_vals = nothing
        x_name = ""
        for (k, v) in se.covariates
            vals = v[mask]
            if length(vals) == length(est) && !any(isnan, vals)
                x_vals = vals
                x_name = string(k)
                break
            end
        end
        x_vals === nothing && continue

        @series begin
            subplot := panel
            xguide --> x_name
            yguide --> label
            seriestype := :line
            linewidth --> 2
            linecolor --> :steelblue
            ribbon --> 2 .* se_vals
            fillalpha --> 0.2
            fillcolor --> :steelblue
            x_vals, est
        end
    end
end

# ─── DerivativeEstimates recipe ──────────────────────────────────────────────

@recipe function f(de::GAM.DerivativeEstimates)
    unique_smooths = unique(de.smooth)
    n_panels = length(unique_smooths)

    layout --> (1, n_panels)
    legend --> false

    for (panel, label) in enumerate(unique_smooths)
        mask = de.smooth .== label
        x_vals = de.x[mask]
        d_vals = de.derivative[mask]
        lo = de.lower[mask]
        hi = de.upper[mask]

        @series begin
            subplot := panel
            xguide --> split(label, "(")[end] |> s -> rstrip(s, ')')
            yguide --> "d/dx $(label)"
            seriestype := :line
            linewidth --> 2
            linecolor --> :steelblue
            ribbon --> (d_vals .- lo, hi .- d_vals)
            fillalpha --> 0.2
            fillcolor --> :steelblue
            x_vals, d_vals
        end

        # Reference line at zero
        @series begin
            subplot := panel
            seriestype := :hline
            linecolor --> :gray50
            linestyle --> :dash
            linewidth --> 1
            [0.0]
        end
    end
end

# ─── AppraiseData recipe (4-panel diagnostic) ───────────────────────────────

@recipe function f(ad::GAM.AppraiseData)
    layout --> (2, 2)
    legend --> false

    # QQ plot
    @series begin
        subplot := 1
        xguide --> "Theoretical quantiles"
        yguide --> "Sample quantiles"
        title --> "QQ plot"
        seriestype := :scatter
        markersize --> 2
        markercolor --> :steelblue
        markeralpha --> 0.6
        markerstrokewidth --> 0
        ad.qq_theoretical, ad.qq_sample
    end
    @series begin
        subplot := 1
        seriestype := :line
        linecolor --> :red
        linewidth --> 1
        rng = range(minimum(ad.qq_theoretical), maximum(ad.qq_theoretical); length=2)
        collect(rng), collect(rng)
    end

    # Residuals vs linear predictor
    @series begin
        subplot := 2
        xguide --> "Linear predictor"
        yguide --> "Deviance residuals"
        title --> "Residuals vs linear predictor"
        seriestype := :scatter
        markersize --> 2
        markercolor --> :steelblue
        markeralpha --> 0.6
        markerstrokewidth --> 0
        ad.linear_predictor, ad.residuals_deviance
    end
    @series begin
        subplot := 2
        seriestype := :hline
        linecolor --> :red
        linestyle --> :dash
        [0.0]
    end

    # Histogram of residuals
    @series begin
        subplot := 3
        xguide --> "Deviance residuals"
        yguide --> "Frequency"
        title --> "Histogram of residuals"
        seriestype := :histogram
        fillcolor --> :steelblue
        fillalpha --> 0.6
        linecolor --> :white
        ad.residuals_deviance
    end

    # Observed vs fitted
    @series begin
        subplot := 4
        xguide --> "Fitted values"
        yguide --> "Observed"
        title --> "Observed vs fitted"
        seriestype := :scatter
        markersize --> 2
        markercolor --> :steelblue
        markeralpha --> 0.6
        markerstrokewidth --> 0
        ad.fitted, ad.observed
    end
    @series begin
        subplot := 4
        seriestype := :line
        linecolor --> :red
        linewidth --> 1
        rng = range(minimum(ad.fitted), maximum(ad.fitted); length=2)
        collect(rng), collect(rng)
    end
end

# ─── RootogramData recipe ────────────────────────────────────────────────────

@recipe function f(rd::GAM.RootogramData)
    xguide --> "Count"
    yguide --> "√Frequency"
    title --> "Rootogram"
    legend --> true

    @series begin
        seriestype := :bar
        fillcolor --> :steelblue
        fillalpha --> 0.5
        linecolor --> :steelblue
        label --> "Observed"
        rd.count, rd.sqrt_observed
    end
    @series begin
        seriestype := :line
        linecolor --> :red
        linewidth --> 2
        label --> "Expected"
        rd.count, rd.sqrt_expected
    end
end

# ─── Helpers ─────────────────────────────────────────────────────────────────

function _extract_x(m::GAM.GamModel, sm::GAM.ConstructedSmooth)
    return _extract_var(m, sm.spec.term_vars[1])
end

function _extract_var(m::GAM.GamModel, varname::Symbol)
    if m.data !== nothing
        ct = GAM.Tables.columntable(m.data)
        if varname in GAM.Tables.columnnames(ct)
            return Float64.(collect(GAM.Tables.getcolumn(ct, varname)))
        end
    end
    error("Covariate :$varname is not available in the model's stored data")
end

# ─── VisGamData recipe (2D surface / contour) ───────────────────────────────

@recipe function f(vgd::GAM.VisGamData)
    seriestype --> :contourf
    xguide --> vgd.x1_label
    yguide --> vgd.x2_label
    title --> vgd.smooth_label
    colorbar_title --> vgd.z_label
    fillalpha --> 0.8
    vgd.x1, vgd.x2, vgd.z'
end

end # module
