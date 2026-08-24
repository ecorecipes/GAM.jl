#!/usr/bin/env julia

@isdefined(run_benchmarks) || include(joinpath(@__DIR__, "benchmarks.jl"))

# Date stamp via Libc to avoid a Dates dependency in minimal environments
_today_stamp() = Libc.strftime("%Y-%m-%d", time())

# Environment stamp for the report header, so the snapshot records when and
# where it was produced (previously hand-edited and prone to going stale).
function environment_stamp()
    rver = try
        m = match(r"R version (\S+)", read(`R --version`, String))
        m === nothing ? "unknown" : m.captures[1]
    catch
        "unavailable"
    end
    arch = Sys.ARCH in (:aarch64, :arm64) ? "ARM64" : string(Sys.ARCH)
    os = Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : Sys.iswindows() ? "Windows" : string(Sys.KERNEL)
    "($(_today_stamp()), Julia $(VERSION), R $(rver), $(os) $(arch))"
end

function main()
    out_path = get(ENV, "GAM_BENCHMARK_RESULTS_PATH", joinpath(@__DIR__, "results.txt"))
    tmp_path, tmp_io = mktemp()
    try
        redirect_stdout(tmp_io) do
            run_benchmarks()
        end
        close(tmp_io)
        report = read(tmp_path, String)
        # Stamp the environment into the header line so it cannot silently go
        # stale: replace any existing "(...)" annotation after the suite title,
        # or append one if absent.
        stamp = environment_stamp()
        report = replace(report,
            r"GAM\.jl vs R Benchmark Suite(  \([^)]*\))?" =>
                "GAM.jl vs R Benchmark Suite  $(stamp)"; count = 1)
        write(out_path, report)
        print(report)
        println("\nWrote benchmark snapshot to $(out_path)")
    finally
        isopen(tmp_io) && close(tmp_io)
        rm(tmp_path; force = true)
    end
end

main()
