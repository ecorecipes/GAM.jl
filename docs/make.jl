ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Documenter
using GAM

DocMeta.setdocmeta!(GAM, :DocTestSetup, :(using GAM); recursive = true)

makedocs(;
    modules = [GAM],
    sitename = "GAM.jl",
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
        "Shape Constraints (SCAM)" => "scam.md",
        "Quantile Regression (QGAM)" => "qgam.md",
        "Extreme Values (evgam)" => "evgam.md",
        "Large Data (BAM)" => "bam.md",
        "Mixed Models (GAMM)" => "gamm.md",
        "Bayesian Inference" => "bayesian.md",
        "Diagnostics" => "diagnostics.md",
        "Comparison with mgcv" => "mgcv.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/ecorecipes/GAM.jl.git",
    devbranch = "main",
)
