# GAM diagnostics

"""
    gam_check(m::GamModel)

Print diagnostic information about a fitted GAM, including basis dimension
adequacy checks for each smooth term.
"""
function gam_check(m::GamModel)
    println("GAM checking results")
    println("====================")
    println()

    # Convergence
    if m.converged
        println("✓ Model converged")
    else
        println("✗ Model did NOT converge")
    end

    println()
    println("Method: $(m.method)")
    println("Scale est. = $(@sprintf("%.4f", m.scale))")
    println("n = $(nobs(m))")
    println()

    # Per-smooth diagnostics
    println("Basis dimension (k) checking results:")
    println("─" ^ 60)
    @printf("%-20s %8s %8s %8s\n", "Smooth", "k'", "edf", "k-index")
    println("─" ^ 60)

    for (i, sm) in enumerate(m.smooths)
        k_eff = size(sm.X, 2)
        edf_i = m.edf[i]
        k_index = edf_i / k_eff  # rough adequacy measure
        @printf("%-20s %8d %8.2f %8.3f\n",
            sm.spec.label, k_eff, edf_i, k_index)
    end
    println("─" ^ 60)
    println()

    if any(e / size(sm.X, 2) > 0.9 for (e, sm) in zip(m.edf, m.smooths))
        println("⚠ Some smooth terms have edf close to k'. Consider increasing k.")
    end
    println()

    # Overall fit
    dev_expl = r2(m) * 100
    @printf("Deviance explained = %.1f%%\n", dev_expl)
    if _needs_scale_estimate(m.family)
        @printf("Scale (σ²) = %.4f\n", m.scale)
    end

    return nothing
end

"""
    k_check(m::GamModel)

Check whether basis dimensions are adequate for each smooth term.
Returns a vector of (smooth_label, k', edf, p_value) tuples.

A significant p-value suggests the basis dimension may be too small.
"""
function k_check(m::GamModel)
    results = NamedTuple{(:label, :k, :edf, :k_ratio), Tuple{String, Int, Float64, Float64}}[]

    for (i, sm) in enumerate(m.smooths)
        k_eff = size(sm.X, 2)
        edf_i = m.edf[i]
        ratio = edf_i / k_eff
        push!(results, (label = sm.spec.label, k = k_eff, edf = edf_i, k_ratio = ratio))
    end

    return results
end

"""
    concurvity(m::GamModel; full=true)

Measure concurvity (analogue of collinearity) between smooth terms.
If `full=true`, returns worst-case concurvity for each smooth.
If `full=false`, returns pairwise concurvity matrix.
"""
function concurvity(m::GamModel; full::Bool = true)
    n_smooth = m.n_smooth
    n_smooth >= 1 || return Float64[]

    # Work in QR space like mgcv — more stable and correct for "worst" measure
    R_full = Matrix(qr(m.X).R)
    p = size(R_full, 2)

    # Smooth column ranges
    starts = [m.smooths[i].first_para for i in 1:n_smooth]
    stops = [m.smooths[i].last_para for i in 1:n_smooth]

    # Include parametric terms as first block
    has_para = minimum(starts) > 1
    if has_para
        para_stop = minimum(starts) - 1
        all_starts = vcat(1, starts)
        all_stops = vcat(para_stop, stops)
    else
        all_starts = starts
        all_stops = stops
    end
    mt = length(all_starts)
    offset = has_para ? 1 : 0

    if full
        conc = zeros(mt)
        for i in 1:mt
            idx_i = all_starts[i]:all_stops[i]
            other_idx = setdiff(1:p, idx_i)
            Xi = R_full[:, other_idx]
            Xj = R_full[:, idx_i]
            r = size(Xi, 2)

            # QR of [Xi | Xj]: R factor decomposes Xj into cross + residual parts
            R_comb = Matrix(qr(hcat(Xi, Xj)).R)
            RR = R_comb[:, (r + 1):end]       # last columns = Xj part
            R_cross = RR[1:r, :]               # cross part (projection of Xj onto Xi)
            Rt = Matrix(qr(RR).R)              # re-QR for full Xj factor

            # Worst-case: max eigenvalue of Rt^{-T} R_cross' (squared)
            z = Rt' \ R_cross'
            s_vals = svd(z).S
            conc[i] = length(s_vals) > 0 ? s_vals[1]^2 : 0.0
        end
        return conc[(offset + 1):end]
    else
        # Pairwise worst-case concurvity
        conc_mat = zeros(n_smooth, n_smooth)
        for i in 1:n_smooth, j in 1:n_smooth
            if i == j
                conc_mat[i, j] = 1.0
                continue
            end
            idx_i = starts[i]:stops[i]
            idx_j = starts[j]:stops[j]
            Xi = R_full[:, idx_i]
            Xj = R_full[:, idx_j]
            r = size(Xi, 2)

            R_comb = Matrix(qr(hcat(Xi, Xj)).R)
            RR = R_comb[:, (r + 1):end]
            R_cross = RR[1:r, :]
            Rt = Matrix(qr(RR).R)

            z = Rt' \ R_cross'
            s_vals = svd(z).S
            conc_mat[i, j] = length(s_vals) > 0 ? s_vals[1]^2 : 0.0
        end
        return conc_mat
    end
end

# ============================================================================
# ANOVA for GAMs — Wood (2013) smooth significance & deviance tests
# ============================================================================

"""
    AnovaGamResult

Result of `anova_gam`. Contains either a single-model smooth significance
table or a multi-model deviance comparison table.

# Fields
- `smooth_table`: named tuple of vectors (label, edf, ref_df, statistic, p_value) for smooth terms
- `parametric_table`: named tuple of vectors (term, df, statistic, p_value) for parametric terms
- `test_type`: `:F` or `:Chisq`
- `model_table`: named tuple of vectors for multi-model comparison
"""
struct AnovaGamResult
    smooth_table::Union{Nothing, NamedTuple{(:label, :edf, :ref_df, :statistic, :p_value),
        Tuple{Vector{String}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}
    parametric_table::Union{Nothing, NamedTuple{(:term, :df, :statistic, :p_value),
        Tuple{Vector{String}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}
    test_type::Symbol
    model_table::Union{Nothing, NamedTuple{(:resid_df, :resid_dev, :df, :deviance, :statistic, :p_value),
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}}}
end

"""
    anova_gam(m::GamModel)

Approximate significance of smooth terms using the Bayesian test from
Wood (2013, Biometrika). For each smooth, computes a Wald-type test
statistic on the Bayesian posterior covariance, with reference degrees
of freedom based on the effective degrees of freedom.

Returns an `AnovaGamResult` with the smooth significance table.
"""
function anova_gam(m::GamModel)
    use_f = _needs_scale_estimate(m.family) ||
            (m.family isa ExtendedFamily && _estimates_scale(m.family))
    test_type = use_f ? :F : :Chisq
    resid_df = dof_residual(m)

    labels = String[]
    edfs = Float64[]
    ref_dfs = Float64[]
    stats = Float64[]
    p_vals = Float64[]

    β = coef(m)
    Vp = m.Vp

    for (i, sm) in enumerate(m.smooths)
        edf_i = m.edf[i]
        T_stat, ref_df = _wood_test_statistic(m, i)

        if use_f
            F_stat = T_stat / ref_df
            p_val = ccdf(FDist(ref_df, resid_df), F_stat)
            push!(stats, F_stat)
        else
            push!(stats, T_stat)
            p_val = ccdf(Chisq(ref_df), T_stat)
        end

        push!(labels, sm.spec.label)
        push!(edfs, edf_i)
        push!(ref_dfs, ref_df)
        push!(p_vals, p_val)
    end

    smooth_table = (label=labels, edf=edfs, ref_df=ref_dfs,
                    statistic=stats, p_value=p_vals)

    return AnovaGamResult(smooth_table, nothing, test_type, nothing)
end

"""
    _wood_test_statistic(m, i) -> (T_stat, ref_df)

Wood (2013, Biometrika) test statistic for smooth term `i`. The test is
performed on the function scale: with `R` the triangular factor of the
smooth's model-matrix columns, `f ∝ R β` has Bayesian covariance
`V_f = R V_β R'`. The statistic is `Tᵣ = (Rβ)' V_f⁻ᵣ (Rβ)` where `V_f⁻ᵣ`
is the eigendecomposition pseudo-inverse truncated at rank
`r ≈ round(edf)`. Truncation at the effective (not nominal) dimension is
what makes the χ²_r / F(r, ·) reference distribution approximately valid;
inverting the full positive-definite block instead is anti-conservative
for strongly penalized smooths.
"""
function _wood_test_statistic(m::GamModel, i::Int)
    sm = m.smooths[i]
    idx = sm.first_para:sm.last_para
    β_i = coef(m)[idx]
    V_i = Symmetric(Matrix(m.Vp[idx, idx]))
    edf_i = m.edf[i]

    # Function-space projection via QR of the model-matrix columns
    R = Matrix(qr(m.X[:, idx]).R)
    Vf = Symmetric(R * V_i * R')
    bf = R * β_i

    eg = eigen(Vf)
    # Descending order
    vals = reverse(eg.values)
    vecs = reverse(eg.vectors; dims = 2)
    tol = max(vals[1], 0.0) * eps()^0.9
    r_est = count(>(tol), vals)
    r_est >= 1 || return (0.0, 1.0)

    # mgcv rounding rule: floor(edf), +1 when the fractional part is
    # non-negligible (we use integer truncation; mgcv additionally treats
    # the fractional eigenvalue specially)
    frac = edf_i - floor(edf_i)
    r = Int(floor(edf_i)) + (frac > 0.05 ? 1 : 0)
    r = clamp(r, 1, r_est)

    d = vecs[:, 1:r]' * bf
    T_stat = sum(abs2, d ./ sqrt.(vals[1:r]))
    return (T_stat, Float64(r))
end

"""
    anova_gam(m1::GamModel, m2::GamModel, models::GamModel...; test=:auto)

Sequential deviance comparison of two or more nested GAM models.
Models are sorted by increasing total EDF. For scale-estimated
families an F-test is used; for known-scale families a χ² test is used.

`test` may be `:F`, `:Chisq`, or `:auto` (default).
"""
function anova_gam(m1::GamModel, m2::GamModel, models::GamModel...; test::Symbol=:auto)
    all_models = [m1, m2, models...]

    # Sort by total EDF (ascending = simplest first)
    perm = sortperm([m.edf_total for m in all_models])
    all_models = all_models[perm]

    n_models = length(all_models)

    # Determine test type from largest model
    ref_model = all_models[end]
    if test == :auto
        use_f = _needs_scale_estimate(ref_model.family) ||
                (ref_model.family isa ExtendedFamily && _estimates_scale(ref_model.family))
        test_type = use_f ? :F : :Chisq
    else
        test_type = test
        use_f = (test == :F)
    end

    # Scale from the largest model
    scale = ref_model.scale
    resid_df_ref = dof_residual(ref_model)

    resid_dfs = Float64[dof_residual(m) for m in all_models]
    resid_devs = Float64[deviance(m) for m in all_models]

    df_diff = fill(NaN, n_models)
    dev_diff = fill(NaN, n_models)
    stat_vals = fill(NaN, n_models)
    p_vals = fill(NaN, n_models)

    for i in 2:n_models
        Δdf = all_models[i].edf_total - all_models[i-1].edf_total
        Δdev = deviance(all_models[i-1]) - deviance(all_models[i])
        df_diff[i] = Δdf
        dev_diff[i] = Δdev

        if Δdf > 0
            if use_f
                F_stat = (Δdev / Δdf) / scale
                stat_vals[i] = F_stat
                p_vals[i] = ccdf(FDist(Δdf, resid_df_ref), F_stat)
            else
                stat_vals[i] = Δdev
                p_vals[i] = ccdf(Chisq(Δdf), Δdev)
            end
        end
    end

    model_table = (resid_df=resid_dfs, resid_dev=resid_devs,
                   df=df_diff, deviance=dev_diff,
                   statistic=stat_vals, p_value=p_vals)

    return AnovaGamResult(nothing, nothing, test_type, model_table)
end

function Base.show(io::IO, r::AnovaGamResult)
    if r.smooth_table !== nothing
        _show_smooth_table(io, r)
    end
    if r.model_table !== nothing
        _show_model_table(io, r)
    end
end

function _show_smooth_table(io::IO, r::AnovaGamResult)
    st = r.smooth_table
    stat_name = r.test_type == :F ? "F" : "Chi.sq"

    println(io, "Approximate significance of smooth terms:")
    println(io, "─" ^ 70)
    @printf(io, "%-20s %8s %8s %10s %12s\n",
            "", "edf", "Ref.df", stat_name, "p-value")
    println(io, "─" ^ 70)

    for i in eachindex(st.label)
        p_str = st.p_value[i] < 2e-16 ? "< 2e-16" :
                st.p_value[i] < 0.001 ? @sprintf("%.2e", st.p_value[i]) :
                @sprintf("%.4f", st.p_value[i])
        @printf(io, "%-20s %8.3f %8.3f %10.3f %12s\n",
                st.label[i], st.edf[i], st.ref_df[i], st.statistic[i], p_str)
    end
    println(io, "─" ^ 70)
end

function _show_model_table(io::IO, r::AnovaGamResult)
    mt = r.model_table
    stat_name = r.test_type == :F ? "F" : "Chi.sq"
    n = length(mt.resid_df)

    println(io, "Analysis of Deviance Table")
    println(io)
    println(io, "Model comparison (sequential):")
    println(io, "─" ^ 80)
    @printf(io, "%-8s %10s %12s %8s %12s %10s %12s\n",
            "Model", "Resid.Df", "Resid.Dev", "Df", "Deviance", stat_name, "p-value")
    println(io, "─" ^ 80)

    for i in 1:n
        if i == 1
            @printf(io, "%-8s %10.2f %12.3f\n",
                    "$(i)", mt.resid_df[i], mt.resid_dev[i])
        else
            p_str = isnan(mt.p_value[i]) ? "" :
                    mt.p_value[i] < 2e-16 ? "< 2e-16" :
                    mt.p_value[i] < 0.001 ? @sprintf("%.2e", mt.p_value[i]) :
                    @sprintf("%.4f", mt.p_value[i])
            stat_str = isnan(mt.statistic[i]) ? "" : @sprintf("%.3f", mt.statistic[i])
            @printf(io, "%-8s %10.2f %12.3f %8.2f %12.3f %10s %12s\n",
                    "$(i)", mt.resid_df[i], mt.resid_dev[i],
                    mt.df[i], mt.deviance[i], stat_str, p_str)
        end
    end
    println(io, "─" ^ 80)
end
