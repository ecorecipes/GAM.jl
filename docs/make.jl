using Documenter
using GAM

DocMeta.setdocmeta!(GAM, :DocTestSetup, :(using GAM); recursive = true)

makedocs(;
    modules = [GAM],
    sitename = "GAM.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/sdwfrost/GAM.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "tutorial.md",
        "Smooth Terms" => "smooths.md",
        "Formula Syntax" => "formulas.md",
        "Extended Families" => "families.md",
        "Comparison with mgcv" => "mgcv.md",
        "API Reference" => "api.md",
    ],
)
