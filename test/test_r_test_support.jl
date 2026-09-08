using Test

@testset "R comparison test infrastructure" begin
    @test RTestSupport.mode(Dict{String, String}()) == (skip=false, required=false)
    @test RTestSupport.mode(Dict("GAM_REQUIRE_RCALL"=>"true")).required
    @test RTestSupport.mode(Dict("GAM_SKIP_RCALL"=>"true")).skip
    @test_throws ArgumentError RTestSupport.mode(
        Dict("GAM_SKIP_RCALL"=>"true", "GAM_REQUIRE_RCALL"=>"true"))

    @testset "Subprocess diagnostics and required mode" begin
        mktempdir() do dir
            path = joinpath(dir, "probe.log")
            output = IOBuffer()
            good = `$(Base.julia_cmd()) --startup-file=no -e 'println("probe stdout"); println(stderr, "probe stderr")'`
            @test RTestSupport.probe(; command=good, required=true, log_path=path, output)
            @test "probe stdout" in split(read(path, String), '\n')
            @test "probe stderr" in split(String(take!(output)), '\n')

            bad = `$(Base.julia_cmd()) --startup-file=no -e 'println(stderr, "load failure detail"); exit(23)'`
            ok = @test_logs (:warn, r"Skipping R integration tests") RTestSupport.probe(
                ; command=bad, log_path=path, output)
            @test !ok
            @test "load failure detail" in split(String(take!(output)), '\n')
            @test_throws ErrorException RTestSupport.probe(
                ; command=bad, required=true, log_path=path, output)
            @test occursin("exit=23", String(take!(output)))

            missing = Cmd([joinpath(dir, "missing-executable")])
            @test_throws ErrorException RTestSupport.probe(
                ; command=missing, required=true, log_path=path, output)
            @test occursin("missing-executable", read(path, String))

            if !Sys.iswindows()
                # A terminated child must not terminate the test process.
                terminated = `$(Base.julia_cmd()) --startup-file=no --handle-signals=no -e 'ccall(:raise, Cint, (Cint,), 15)'`
                @test_throws ErrorException RTestSupport.probe(
                    ; command=terminated, required=true, log_path=path, output)
                @test occursin("signal=15", String(take!(output)))
            end
        end
    end

    @testset "Required comparison coverage" begin
        completed = Set{String}()
        value = RTestSupport.run_comparison("example"; required=true, completed) do
            @test true
            42
        end
        @test value == 42
        @test completed == Set(["example"])
        @test RTestSupport.require_complete(completed; expected=("example",)) === nothing
        @test_throws ErrorException RTestSupport.require_complete(
            completed; expected=("example", "missing"))

        empty = Test.DefaultTestSet("empty")
        @test !RTestSupport.validate_comparison(empty, "empty"; required=false)
        @test_throws ErrorException RTestSupport.validate_comparison(
            empty, "empty"; required=true)
        skipped = Test.DefaultTestSet("skipped")
        Test.record(skipped, Test.Broken(:skipped, :(false)))
        @test_throws ErrorException RTestSupport.validate_comparison(
            skipped, "skipped"; required=true)
    end
end
