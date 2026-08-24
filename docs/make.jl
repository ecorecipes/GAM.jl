ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Documenter
using GAM

DocMeta.setdocmeta!(GAM, :DocTestSetup, :(using GAM); recursive = true)

makedocs(;
    modules = [GAM],
    sitename = "GAM.jl",
    # `:missing_docs` is intentional: ~340 internal helpers carry docstrings
    # that deliberately stay out of the manual.
    # `:cross_references` remains for two reasons: `statsbase.jl` links to
    # `r2`, which is StatsAPI's binding and cannot be documented from this
    # module; and `s_nest`'s docstring links to
    # `trans_linear` / `trans_nexpsm` / `trans_mgks`, whose docstrings are
    # attached to the types (`TransLinear`, …) rather than the constructor
    # functions. Add a docstring to each constructor in `src/nested.jl` and
    # this entry can be dropped, making broken links a hard build error.
    warnonly = [:missing_docs, :cross_references],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://ecorecipes.github.io/GAM.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "tutorial.md",
        "Smooth Terms" => "smooths.md",
        "Formula Syntax" => "formulas.md",
        "Families & Models" => "families.md",
        "GAMLSS" => "gamlss.md",
        "Nested Effects" => "nested.md",
        "Shape Constraints (SCAM)" => "scam.md",
        "Quantile Regression (QGAM)" => "qgam.md",
        "Extreme Values (evgam)" => "evgam.md",
        "Large Data (BAM)" => "bam.md",
        "Mixed Models (GAMM)" => "gamm.md",
        "Bayesian Inference" => "bayesian.md",
        "Diagnostics" => "diagnostics.md",
        "Comparison with mgcv" => "mgcv.md",
        "Vignettes" => "vignettes.md",
        "API Reference" => [
            "Core" => "api.md",
            "Model Types" => "api_models.md",
            "Diagnostics & Plotting" => "api_diagnostics.md",
        ],
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/GAM.jl.git",
    devbranch = "main",
)
