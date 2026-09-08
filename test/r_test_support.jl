module RTestSupport

using Test

const COMPARISON_FILES = (
    "test_rcall.jl",
    "test_sz_rcall.jl",
    "test_sz_base_rcall.jl",
    "test_duchon_rcall.jl",
    "test_evgam_rcall.jl",
    "test_egpd_rcall.jl",
    "test_qgam_rcall.jl",
    "test_scam_rcall.jl",
    "test_nested_rcall.jl",
    "test_scasm_rcall.jl",
    "test_gratia_rcall.jl",
    "test_scat_rcall.jl",
    "test_vector_sp_rcall.jl",
    "test_discrete_rcall.jl",
    "test_gamm_rcall.jl",
    "test_side_constraints_rcall.jl",
    "test_spde_rcall.jl",
)

function mode(env = ENV)
    skip = parse(Bool, get(env, "GAM_SKIP_RCALL", "false"))
    required = parse(Bool, get(env, "GAM_REQUIRE_RCALL", "false"))
    skip && required && throw(ArgumentError(
        "GAM_SKIP_RCALL and GAM_REQUIRE_RCALL cannot both be true"))
    return (; skip, required)
end

function probe_command()
    code = raw"""
    println("Julia: ", VERSION, " (", Sys.MACHINE, ")")
    println("R executable: ", Sys.which("R"))
    println("R_HOME: ", get(ENV, "R_HOME", "<unset>"))
    println("R_LIBS_USER: ", get(ENV, "R_LIBS_USER", "<unset>"))
    flush(stdout)
    using RCall
    println("RCall: ", Base.pkgversion(RCall))
    RCall.reval("library(mgcv); print(sessionInfo()); print(.libPaths())")
    """
    julia = Base.julia_cmd()
    project = Base.active_project()
    return project === nothing ?
        `$julia --startup-file=no -e $code` :
        `$julia --startup-file=no --project=$project -e $code`
end

function _probe(log, command; required, verbose, output)
    print(log, "Command: ")
    show(log, command)
    println(log)
    flush(log)
    process = try
        run(pipeline(ignorestatus(command); stdout = log, stderr = log))
    catch err
        err isa Base.IOError || rethrow()
        seekend(log)
        showerror(log, err)
        println(log)
        nothing
    end
    ok = process !== nothing && success(process)
    status = process === nothing ? "could not start" :
        "exit=$(process.exitcode), signal=$(process.termsignal)"
    seekend(log)
    println(log, "RCall subprocess probe: $status")
    flush(log)
    if !ok || verbose || required
        seekstart(log)
        print(output, read(log, String))
        flush(output)
    end
    if !ok
        required && error(
            "GAM_REQUIRE_RCALL=true, but the RCall/mgcv subprocess probe failed; " *
            "see the diagnostics above")
        @warn "Skipping R integration tests: the RCall/mgcv subprocess probe failed. " *
              "RCall will not be loaded in this process."
    end
    return ok
end

function probe(; required::Bool = false, verbose::Bool = false,
    log_path::Union{Nothing, AbstractString} = nothing,
    output::IO = stderr, command::Cmd = probe_command())
    if log_path === nothing
        return mktemp() do _, log
            _probe(log, command; required, verbose, output)
        end
    end
    path = abspath(log_path)
    mkpath(dirname(path))
    return open(path, "w+") do log
        _probe(log, command; required, verbose, output)
    end
end

function validate_comparison(testset, name; required::Bool)
    counts = Test.get_test_counts(testset)
    performed = counts.passes + counts.fails + counts.errors +
        counts.cumulative_passes + counts.cumulative_fails + counts.cumulative_errors
    skipped = counts.broken + counts.cumulative_broken
    if required && (performed == 0 || skipped > 0)
        error("Required R comparison $name ran $performed assertions " *
              "and skipped $skipped; empty or skipped comparisons are not allowed")
    end
    return performed > 0 && skipped == 0
end

function run_comparison(f, name; required::Bool, completed::Set{String})
    value = nothing
    testset = @testset "$name" begin
        value = f()
    end
    validate_comparison(testset, name; required) && push!(completed, name)
    return value
end

function require_complete(completed; expected = COMPARISON_FILES)
    missing = sort!(collect(setdiff(Set(expected), completed)))
    isempty(missing) || error(
        "Required R comparison files did not run: " * join(missing, ", "))
    return nothing
end

end
