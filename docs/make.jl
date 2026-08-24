ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Documenter
using GAM

DocMeta.setdocmeta!(GAM, :DocTestSetup, :(using GAM); recursive = true)

makedocs(;
    modules = [GAM],
    sitename = "GAM.jl",
    # `:missing_docs` is intentional: ~340 internal helpers carry docstrings
    # that deliberately stay out of the manual.
    # `:cross_references` was dropped once both of its causes were fixed: the
    # `r2` link in `statsbase.jl` (StatsAPI's binding, not documentable from
    # this module) is now plain code formatting, and `trans_linear` /
    # `trans_nexpsm` / `trans_mgks` carry docstrings on the constructor
    # functions rather than only on their types. Broken links are now a hard
    # build error; keep it that way.
    warnonly = [:missing_docs],
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
