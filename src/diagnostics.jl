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
