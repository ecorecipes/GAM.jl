# Vignettes

Each numbered directory contains a Quarto vignette (`.qmd`) for GAM.jl, its
dataset(s) as CSV files, and an `R/` subdirectory with a companion vignette
that runs the equivalent analysis with the corresponding R package (mgcv,
scam, qgam, gamlss, evgam) on the **same** CSV data.

Rendered outputs (`.md`, `.html`, `.pdf`, `*_files/`) are build artifacts and
are not checked in — render them locally when needed.

## Rendering

From this directory (requires [Quarto](https://quarto.org) and an instantiated
Julia environment):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
quarto render 01_introduction/01_introduction.qmd
```

The R companions are rendered the same way from each `R/` subdirectory and
require the relevant R packages to be installed.

## Data

Datasets whose data-generating process is stated in the vignette text are
produced by `generate_data.jl` (fixed seeds), so the data can never drift from
the narrative:

```bash
julia --project=. generate_data.jl
```

Currently that covers `06_extreme_values/data_gpd.csv`,
`07_shape_constraints/data_cx.csv`, `07_shape_constraints/data_micv.csv`,
`10_gamm/data_gaussian_gamm.csv`, and `10_gamm/data_poisson_gamm.csv`; the
remaining CSVs predate the script and already match their vignettes.
The reference posterior CSVs in `11_bayesian_gam/` were produced by the R code
in `11_bayesian_gam/R/`.
