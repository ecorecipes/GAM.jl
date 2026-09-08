cran <- c("mgcv", "scam", "qgam", "gamlss", "evd", "evgam", "gratia", "nlme", "remotes")
repo <- Sys.getenv("RSPM", unset = "https://cloud.r-project.org")
options(repos = c(CRAN = repo))
install.packages(cran, Ncpus = 2)

# Neither package is on CRAN. Pin the sources used by the comparison job.
github_token <- Sys.getenv("GITHUB_PAT")
if (!nzchar(github_token)) github_token <- NULL
remotes::install_github(
    "mfasiolo/gamFactory@2658ced9985668ccc3ad27225db6729a82acda04",
    auth_token = github_token, upgrade = "never", dependencies = NA,
    build_vignettes = FALSE
)
remotes::install_github(
    "sdwfrost/egpd@d59f09e2a478eb89cbf31306d27428542dde59da",
    auth_token = github_token, upgrade = "never", dependencies = NA,
    build_vignettes = FALSE
)

required <- c(setdiff(cran, "remotes"), "gamFactory", "egpd")
available <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
for (package in required) {
    version <- if (available[[package]]) as.character(packageVersion(package)) else "MISSING"
    cat(sprintf("%-12s %s\n", package, version))
}
if (!all(available)) {
    stop("Required R comparison packages are missing: ",
         paste(required[!available], collapse = ", "))
}
if (!exists("scasm", envir = asNamespace("mgcv"), inherits = FALSE)) {
    stop("The installed mgcv does not provide scasm, required by the comparison suite")
}
