using TOML
using DataFrames
using CSV
using Dates
using Printf
include("sim2.jl")

function runAgendaGLB(agendaFilepath, configBase; plotHeadless=false)
    if plotHeadless
    	ENV["GKSwstype"] = "nul"
    end

    basedir = dirname(agendaFilepath)
    data = read(agendaFilepath, String) |> TOML.parse

    log = DataFrame(runName = String[], N = Int[], potential = String[],
    dnn = Float64[], s=Float64[], d=Float64[], A=Float64[], dE = Float64[],
    runTime = Dates.AbstractTime[], dt = Float64[])
    display(log)

    potential = "glb"
    A = data["settings"]["A"]
    for config in data["settings"]["initialConditions"]
        for s in data["settings"]["sValues"]
            for d in data["settings"]["dValues"]

                println("\nRUNNING: ", config, " with potential \"", potential, "\" for ", data["settings"]["T"], " time units")

                runName = @sprintf("%s_%s_%.02f_d_%.02f", split(basename(config), ".")[1], potential, s, d)
                ret = runSimGLB(joinpath(configBase, config), potential, s, d, A, data["settings"]["T"], basedir, runName, data["settings"]["dt"])

                temp = basename(config) |> x->split(x, ".")[1] |> x->split(x, "-")
                N = parse(Int, split(temp[1], "_")[2])
                dnn = deserialize(joinpath(configBase, config))[end]
                row = vcat(runName, N, potential, dnn, s, d, A, ret...)
                push!(log, row)
                CSV.write(joinpath(basedir, "runlog.log"), log)
            end
        end
    end

    display(log)
    return log
end
