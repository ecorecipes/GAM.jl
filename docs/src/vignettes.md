# [Vignettes](@id vignettes)

Twelve worked vignettes live in the repository's
[`vignettes/`](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes)
directory. Each is a Quarto document with a rendered Markdown version
checked in, and most have an **R companion** in an `R/` subdirectory that
runs the equivalent analysis with the corresponding R package (mgcv, scam,
qgam, gamlss, evgam, gamFactory) on the *same* CSV data, so results can be
compared side by side.

| Vignette | Topic |
|---|---|
| [01_introduction](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/01_introduction) | First GAM: fitting, smoothing selection, prediction |
| [02_basis_types](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/02_basis_types) | Tour of the smooth basis types |
| [03_multiple_smooths](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/03_multiple_smooths) | Multiple smooths, side constraints, concurvity |
| [04_families](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/04_families) | GLM and extended families |
| [05_diagnostics](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/05_diagnostics) | gratia-style diagnostics and visualization |
| [06_extreme_values](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/06_extreme_values) | GEV/GPD extreme-value GAMs (evgam) |
| [07_shape_constraints](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/07_shape_constraints) | Monotone/convex SCAM smooths |
| [08_quantile_regression](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/08_quantile_regression) | Quantile GAMs (qgam) |
| [09_gamlss](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/09_gamlss) | Location-scale-shape models |
| [10_gamm](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/10_gamm) | Mixed-model GAMs with random effects |
| [11_bayesian_gam](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/11_bayesian_gam) | Bayesian GAMs via Turing.jl |
| [12_nested_effects](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/12_nested_effects) | Nested effects (gamFactory-style), with a direct R comparison |
| [13_migrating_from_mgcv](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/13_migrating_from_mgcv) | Translation table from mgcv, measured parity, and every documented divergence |
| [14_model_selection](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/14_model_selection) | End-to-end selection and diagnostics workflow: `select=true`, `k_check`, concurvity, influence |
| [15_large_and_spatial](https://github.com/ecorecipes/GAM.jl/tree/main/vignettes/15_large_and_spatial) | Measured `gam` vs `bam` crossover, Markov random fields, and splines on the sphere, each against mgcv |

Datasets whose data-generating process is stated in a vignette are produced
by `vignettes/generate_data.jl` with fixed seeds, so data and narrative
cannot drift apart. See `vignettes/README.md` for rendering instructions.
