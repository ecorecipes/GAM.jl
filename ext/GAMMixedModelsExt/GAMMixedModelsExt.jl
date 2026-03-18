# GAMMixedModelsExt — GAMM fitting via MixedModels.jl
#
# This package extension is loaded when the user does `using GAM, MixedModels`.
# It provides _fit_gamm_mm() which converts GAM smooth terms to mixed-model
# random effects via smooth2random() and fits using MixedModels.jl's LMM/GLMM.
#
# Analogous to mgcv's gamm() using nlme/lme4 as backend.

module GAMMixedModelsExt

using GAM
using MixedModels
using LinearAlgebra
using Statistics
using Tables
using DataFrames: DataFrame
using GLM: Link, IdentityLink, LogLink, LogitLink, InverseLink
using Distributions: Normal, Poisson, Bernoulli, Binomial, Gamma, InverseGaussian

import GAM: _fit_gamm_mm

"""
    _fit_gamm_mm(y, X, smooths, n_parametric, random_effects,
                 formula, data, family, link, method, weights, control)

Fit a GAMM using MixedModels.jl as the backend.

Strategy:
1. Apply `smooth2random()` to each smooth → (Xf, Zs) mixed-model form
2. Build a DataFrame with synthetic columns for fixed effects (parametric +
   smooth null spaces) and random effects (smooth penalized parts + explicit REs)
3. Construct a MixedModels.jl formula and fit
4. Use `MixedModels.fitted()` for predictions (no back-transform needed)
5. Back-transform coefficients to original smooth basis for storage

Each smooth's penalized part becomes a random-effect block with a
single "group" level (the smooth basis applies globally to all obs).
Explicit random effects (e.g., `(1|subject)`) use their natural
grouping structure.
"""
function GAM._fit_gamm_mm(y, X, smooths, n_parametric, random_effects,
    formula, data, family, link, method, weights, control)

    n = length(y)

    # Apply smooth2random to each smooth
    sm_mixed = [GAM.smooth2random(sm) for sm in smooths]

    # --- Build parametric (fixed) part ---
    X_para = X[:, 1:n_parametric]

    # Collect smooth fixed (null space) parts
    Xf_parts = Matrix{Float64}[X_para]
    for smm in sm_mixed
        if size(smm.Xf, 2) > 0
            push!(Xf_parts, smm.Xf)
        end
    end
    X_fixed = hcat(Xf_parts...)
    n_fixed = size(X_fixed, 2)

    # --- Build DataFrame with synthetic columns ---
    df = DataFrame()
    df[!, :y] = y

    # Fixed-effect columns
    fixed_names = Symbol[]
    for j in 1:n_fixed
        nm = Symbol("xf_$j")
        df[!, nm] = X_fixed[:, j]
        push!(fixed_names, nm)
    end

    # --- Smooth random-effect blocks ---
    # Each block gets a dummy group factor (single level, since basis is global)
    # and columns for each basis function
    re_formula_parts = String[]
    block_info = NamedTuple[]  # track block metadata

    block_idx = 0
    for (si, smm) in enumerate(sm_mixed)
        for (zi, Z) in enumerate(smm.Zs)
            block_idx += 1
            grp = Symbol("sg_$block_idx")
            df[!, grp] = fill("g1", n)  # single level

            col_names = Symbol[]
            for j in 1:size(Z, 2)
                nm = Symbol("zs_$(block_idx)_$j")
                df[!, nm] = Z[:, j]
                push!(col_names, nm)
            end

            terms_str = join(string.(col_names), " + ")
            push!(re_formula_parts, "zerocorr(0 + $terms_str | $grp)")
            push!(block_info, (type = :smooth, smooth_idx = si, z_idx = zi,
                dim = size(Z, 2), grp = grp, cols = col_names))
        end
    end

    # --- Explicit random effects ---
    # Use natural MixedModels formula syntax (1|group), (x|group), etc.
    # instead of passing the Z indicator columns (which would double-encode)
    t = Tables.columntable(data)
    for (ri, cre) in enumerate(random_effects)
        block_idx += 1
        grp = cre.spec.grouping
        grp_col = Tables.getcolumn(t, grp)
        df[!, grp] = string.(grp_col)

        # Build LHS of RE term from spec
        re_lhs_parts = String[]
        if cre.spec.has_intercept
            push!(re_lhs_parts, "1")
        end
        for term_sym in cre.spec.terms
            # Add the covariate column to the DataFrame
            cov_col = Tables.getcolumn(t, term_sym)
            df[!, term_sym] = Float64.(cov_col)
            push!(re_lhs_parts, string(term_sym))
        end
        re_lhs = join(re_lhs_parts, " + ")
        if isempty(re_lhs_parts)
            re_lhs = "1"  # fallback to random intercept
        end

        push!(re_formula_parts, "($re_lhs | $grp)")
        push!(block_info, (type = :re, re_idx = ri, dim = cre.n_levels * cre.n_terms,
            grp = grp, n_levels = cre.n_levels, n_terms = cre.n_terms))
    end

    # --- Build and fit MixedModels formula ---
    fixed_str = join(string.(fixed_names), " + ")
    re_str = join(re_formula_parts, " + ")
    formula_str = isempty(re_formula_parts) ?
        "y ~ 0 + $fixed_str" :
        "y ~ 0 + $fixed_str + $re_str"

    mm_formula = eval(Meta.parse("MixedModels.@formula($formula_str)"))

    is_gaussian = family isa Normal && link isa IdentityLink

    if !is_gaussian
        @warn "MixedModels.jl backend currently best suited for Gaussian family. " *
              "Non-Gaussian uses Gaussian approximation. For better results, " *
              "use backend=:LAMS (default) or priors= for Bayesian."
    end

    mm_model = fit(MixedModel, mm_formula, df; REML = (method == :REML), progress = false)

    # --- Extract results ---
    β_mm_fixed = MixedModels.fixef(mm_model)
    σ_res = MixedModels.sdest(mm_model)

    # Use MixedModels.fitted() for predictions — avoids back-transform errors
    fitted_vals = MixedModels.fitted(mm_model)

    # Extract random effects per block from raneftables
    re_tables = MixedModels.raneftables(mm_model)

    random_coefs_all = Vector{Float64}[]
    random_vars_all = Float64[]

    for bi in block_info
        grp = bi.grp

        if bi.type == :smooth
            # Smooth blocks: single group level, extract using synthetic column names
            if haskey(re_tables, grp)
                tbl = re_tables[grp]
                rows = collect(tbl)
                row = rows[1]  # single group level
                coefs = Float64[row[bi.cols[j]] for j in 1:bi.dim]
                push!(random_coefs_all, coefs)
            else
                push!(random_coefs_all, zeros(bi.dim))
            end

            # Extract variance for this smooth block from MixedModels reterms
            σ2_block = 0.0
            try
                for rt in mm_model.reterms
                    if Symbol(rt.fname) == grp
                        λ = rt.λ
                        σ2_block = sum(diag(λ * λ')) * σ_res^2
                        break
                    end
                end
            catch
                σ2_block = var(random_coefs_all[end]) + 1e-10
            end
            push!(random_vars_all, max(σ2_block / max(bi.dim, 1), 1e-10))

        elseif bi.type == :re
            # Explicit RE: multiple group levels, extract BLUPs per level
            if haskey(re_tables, grp)
                tbl = re_tables[grp]
                rows = collect(tbl)
                # Extract only numeric BLUP columns (skip group level identifier)
                coefs = Float64[]
                all_keys = keys(rows[1])
                numeric_keys = [k for k in all_keys if rows[1][k] isa Number]
                for row in rows
                    for cn in numeric_keys
                        push!(coefs, Float64(row[cn]))
                    end
                end
                push!(random_coefs_all, coefs)
            else
                push!(random_coefs_all, zeros(bi.dim))
            end

            # Extract RE variance from MixedModels
            σ2_block = 0.0
            try
                for rt in mm_model.reterms
                    if Symbol(rt.fname) == grp
                        λ = rt.λ
                        σ2_block = sum(diag(λ * λ')) * σ_res^2
                        break
                    end
                end
            catch
                σ2_block = var(random_coefs_all[end]) + 1e-10
            end
            push!(random_vars_all, max(σ2_block, 1e-10))
        end
    end

    # --- Back-transform smooth coefficients to original basis ---
    # smooth2random: X_new = X_orig * U * diag(D)
    # X_new columns: [Zs (random, 1:p_rank) | Xf (fixed, p_rank+1:k)]
    # Back-transform: β_orig = U * (D .* [β_r; β_f])
    # Note: β_r comes FIRST, β_f comes LAST — matching column order of X_new
    β_full = zeros(size(X, 2))
    β_full[1:n_parametric] = β_mm_fixed[1:n_parametric]

    fixed_offset = n_parametric
    smooth_block_idx = 0
    for (i, smm) in enumerate(sm_mixed)
        nf = size(smm.Xf, 2)
        β_f = nf > 0 ? β_mm_fixed[fixed_offset+1:fixed_offset+nf] : Float64[]
        fixed_offset += nf

        # Gather random parts for this smooth
        β_r = Float64[]
        for Z in smm.Zs
            smooth_block_idx += 1
            if smooth_block_idx <= length(random_coefs_all)
                append!(β_r, random_coefs_all[smooth_block_idx])
            end
        end

        # Back-transform to original basis
        sm_orig = smooths[i]
        k_orig = size(sm_orig.X, 2)
        idx_s = sm_orig.first_para
        idx_e = sm_orig.last_para

        if smm.trans_U !== nothing && (length(β_f) + length(β_r) > 0)
            # CORRECT ordering: random first, then fixed (matches X_new column order)
            β_combined = vcat(β_r, β_f)

            if length(β_combined) == length(smm.trans_D)
                # Apply D scaling and U rotation: β_orig = U * (D .* β_combined)
                β_scaled = smm.trans_D .* β_combined
                β_full[idx_s:idx_e] = smm.trans_U * β_scaled
            end
        end
    end

    # --- Build GamModel from coefficients ---
    wts = weights === nothing ? ones(n) : Float64.(weights)
    scale_est = σ_res^2

    # Compute deviance
    if family isa Normal
        dev = sum((y .- fitted_vals) .^ 2)
    else
        dev = -2.0 * sum(Distributions.logpdf.(family, fitted_vals, y))
    end

    # EDF per smooth: approximate from variance components
    # EDF_j ≈ nf_j + sum_b(dim_b * σ²_b / (σ²_b + σ²_res))
    edf_per_smooth = Float64[]
    smooth_block_counter = 0
    for (i, smm) in enumerate(sm_mixed)
        edf_i = Float64(size(smm.Xf, 2))
        for Z in smm.Zs
            smooth_block_counter += 1
            if smooth_block_counter <= length(random_vars_all)
                σ2_block = random_vars_all[smooth_block_counter]
                edf_i += size(Z, 2) * σ2_block / (σ2_block + scale_est + 1e-10)
            end
        end
        push!(edf_per_smooth, edf_i)
    end
    edf_total = Float64(n_parametric) + sum(edf_per_smooth)

    # Penalty and sp (needed for GamModel struct)
    penalty = GAM.setup_penalties(smooths, n_parametric)
    log_sp = zeros(length(penalty.sp))

    # Approximate variance matrices
    Vp = zeros(size(X, 2), size(X, 2))
    Ve = zeros(size(X, 2), size(X, 2))
    hat_diag = zeros(n)
    R = zeros(0, 0)

    null_dev = sum((y .- mean(y)) .^ 2)

    # Linear predictor from smooth contributions only (without explicit REs)
    lp_smooth = X * β_full

    gam_model = GAM.GamModel(
        formula,
        y, X,
        β_full,
        fitted_vals,
        lp_smooth,
        wts,
        family, link,
        smooths,
        penalty,
        log_sp,
        edf_per_smooth,
        edf_total,
        scale_est,
        dev,
        null_dev,
        NaN,  # REML from MixedModels
        method,
        Vp, Ve,
        hat_diag,
        R,
        true,  # converged
        0,
        length(smooths),
        n_parametric,
        control,
        Tables.columntable(data),
    )

    # Extract explicit RE coefficients and variances
    re_coef_list = Vector{Float64}[]
    re_var_list = Float64[]
    n_smooth_blocks = smooth_block_idx
    for (i, cre) in enumerate(random_effects)
        re_block = n_smooth_blocks + i
        if re_block <= length(random_coefs_all)
            push!(re_coef_list, random_coefs_all[re_block])
            push!(re_var_list, random_vars_all[re_block])
        else
            push!(re_coef_list, zeros(size(cre.Z, 2)))
            push!(re_var_list, 0.0)
        end
    end

    return GAM.GammModel(gam_model, random_effects, re_coef_list, re_var_list)
end

end # module GAMMixedModelsExt
