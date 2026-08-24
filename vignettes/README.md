# Vignettes

Each numbered directory contains a Quarto vignette (`.qmd`) for GAM.jl, its
dataset(s) as CSV files, and an `R/` subdirectory with a companion vignette
that runs the equivalent analysis with the corresponding R package (mgcv,
scam, qgam, gamlss, evgam) on the **same** CSV data.

GFM renders (`.md` plus `*_files/` figures) are checked in so the vignettes
can be browsed on GitHub; re-render them after changing a `.qmd` or the
package. HTML/PDF renders are local-only build artifacts.

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
`10_gamm/data_gaussian_gamm.csv`, `10_gamm/data_poisson_gamm.csv`,
`12_nested_effects/data_si.csv`, `12_nested_effects/data_expsm.csv`,
`13_migrating_from_mgcv/data.csv`, `14_model_selection/data.csv`, and
`15_large_and_spatial/{data_large,data_mrf,nb,data_sphere}.csv`; the
remaining CSVs predate the script and already match their vignettes.
The reference posterior CSVs in `11_bayesian_gam/` were produced by the R code
in `11_bayesian_gam/R/`.

### A caveat on reproducibility

A fixed seed pins the RNG *stream*, not the values drawn from it. Sampling a
non-uniform distribution consumes a variable, implementation-defined number of
draws, so a generator that interleaves `rand(rng, Uniform(...))` with
`rand(rng, Poisson(...))` can produce a different CSV after a Distributions.jl
upgrade even though nothing in this repository changed.

`gen_poisson_gamm()` has exactly that shape and no longer reproduces the
checked-in `10_gamm/data_poisson_gamm.csv` byte-for-byte under the current
Distributions.jl. The checked-in file remains the one the vignette narrates and
was generated from the documented DGP; re-running the script will overwrite it
with a different, equally valid sample. If you regenerate, either re-render
`10_gamm` or restore the CSV. New generators should draw each variable in its
own vectorized pass rather than interleaving distributions, which keeps the
uniform covariate stream stable across versions.
