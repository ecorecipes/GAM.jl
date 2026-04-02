#!/usr/bin/env julia

@isdefined(run_benchmarks) || include(joinpath(@__DIR__, "benchmarks.jl"))

function main()
    out_path = get(ENV, "GAM_BENCHMARK_RESULTS_PATH", joinpath(@__DIR__, "results.txt"))
    tmp_path, tmp_io = mktemp()
    try
        redirect_stdout(tmp_io) do
            run_benchmarks()
        end
        close(tmp_io)
        report = read(tmp_path, String)
        write(out_path, report)
        print(report)
        println("\nWrote benchmark snapshot to $(out_path)")
    finally
        isopen(tmp_io) && close(tmp_io)
        rm(tmp_path; force = true)
    end
end

main()
