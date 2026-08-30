# Docstring-attachment guard for the public API.
#
# In Julia a docstring binds to the expression that IMMEDIATELY follows it.
# Insert anything between the two — a helper function, its own docstring, even
# a pair of blank lines — and the docstring silently detaches: it becomes a
# free-floating string literal, `?name` shows nothing, and the binding is
# undocumented. Nothing about the code stops working, so tests pass.
#
# This has happened three times in this package:
#
#   1. `@gamm_formula`, `cqcheck` and `check_qgam` had their docstrings sitting
#      on the private `_`-prefixed helpers rather than the public bindings, so
#      `?cqcheck` was empty. Surfaced only when the names were registered on an
#      API page and Documenter complained.
#   2. A dangling docstring in `bam.jl` broke GAM precompilation outright, and
#      was noticed because it took an unrelated docs build down with it.
#   3. Inserting `_normalize_m` (and its docstring) between the `s` docstring
#      and `function s` orphaned the docstring for `s` — the single most-used
#      function in the package. The full test suite passed; only the docs build
#      caught it, and only because `s` happens to be listed in an `@docs` block.
#
# The docs build is a real guard, but a slow and partial one: it runs in a
# separate CI job and only checks names that some `.md` file explicitly lists.
# This checks every exported binding the package actually owns, in-suite.
#
# Re-exported names (StatsModels' `AbstractTerm`, GLM's link types,
# Distributions' families) are excluded — they are documented upstream and
# their docstrings are not this package's to maintain.

@testset "Public bindings have attached docstrings" begin

    # A name is "ours" when the value it refers to was defined in GAM. This is
    # what separates `gam` (ours) from `Binomial` (Distributions', re-exported).
    function _gam_owned(n::Symbol)
        try
            v = getfield(GAM, n)
            if v isa Function || v isa Type
                return parentmodule(v) === GAM
            end
            return false
        catch
            return false
        end
    end

    exported = filter(n -> n !== :GAM, names(GAM))
    owned = filter(_gam_owned, exported)

    # Guard the guard: if the ownership filter ever breaks (a refactor moving
    # definitions behind a wrapper module, say), `owned` could silently empty
    # and every assertion below would pass vacuously. 199 owned at time of
    # writing; the bound is loose enough to allow ordinary API growth or
    # pruning, tight enough that a filter collapse fails here.
    @test length(owned) > 150

    undocumented = String[]
    for n in owned
        doc = string(eval(:(@doc GAM.$n)))
        occursin("No documentation found", doc) && push!(undocumented, String(n))
    end

    # Logged before asserting so a break says WHICH binding lost its docstring,
    # rather than only that a count changed.
    if !isempty(undocumented)
        @info "Exported bindings with no attached docstring" undocumented
    end
    @test undocumented == String[]

    # The orphaning failure mode specifically: `s` is the most-used entry point
    # and has had its docstring detached once already, so pin it by content
    # rather than by mere presence — a stub would satisfy the check above.
    s_doc = string(@doc GAM.s)
    @test occursin("vars...", s_doc)
    @test occursin("bs", s_doc)

    # Same for the other smooth constructors, which share the helper whose
    # insertion caused the detachment.
    for f in (:te, :ti, :t2)
        d = string(eval(:(@doc GAM.$f)))
        @test !occursin("No documentation found", d)
    end
end
