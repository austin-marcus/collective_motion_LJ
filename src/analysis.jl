using JLD2
using FileIO
using Printf
using TOML
using StatsBase
using DataFrames
# using Fontconfig
using CSV
include("physics.jl")
include("cluster.jl")
include("helper.jl")


# step is the number of samples to take
function analyze_energy(m, c, xs, vs, xB, yB, potFunc; step=1)
    Es = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    for i = 1:length(xs)÷step:length(xs)
        E, V, K = getEnergy(xs[i], vs[i], m, c, potFunc, xB, yB)
        push!(Es, E)
        push!(Vs, V)
        push!(Ks, K)
    end
    return (Es, Vs, Ks)
end

# WARNING normalizes by init
function RMSE(init, ts)
    # return ((ts .- init)/init*100).^2 |> sum |> x->x/length(ts) |> sqrt
    return ((ts .- init)).^2 |> sum |> x->x/length(ts) |> sqrt |> x->x/abs(init)
end

function analysis(stateFilename; step=1, plotHeadless=false, outputDir=nothing, checkIfComputed=true)
    if plotHeadless
    	ENV["GKSwstype"] = "nul"
    end

    data = loadData(stateFilename)
    analysisFilename = stateFilename * "-analysis.jld2"
    items = []
    if checkIfComputed && isfile(analysisFilename)
        items = load(analysisFilename) |> keys
    end
    results = Dict()

    paramFilename = stateFilename * "-param.jld2"
    params = load(paramFilename)["params"]
    # if glb, get parameters and encapsulate potential function
    # energy
    potFunc = (r, m1, m2, c1, c2)->V_glb(r, m1, m2, c1, c2, params["s"], params["d"], params["A"])
    println("\t\tComputing energy...")
    if !("energy" in items)
        es = analyze_energy(data["m"], data["c"], data["xs"], data["vs"], data["B"], data["B"], potFunc; step=step)
        results["energy"] = es
        results["energyRMSE"] = RMSE(es[1][1], es[1][2:end])
    end
    
    println("\t\tComputing momentum...")
    if !("momentum" in items)
        ps = []
        for i = 1:div(length(data["vs"]), step):length(data["vs"])
             push!(ps, getMomentum(data["vs"][i], data["m"]))
        end
        results["momentum"] = ps
        # display(ps)
        psMags = [ norm(ps[i]) for i=1:length(ps) ]
        results["momentumRMSE"] = RMSE(psMags[1], psMags[2:end])
    end

    # clustering profile
    println("\t\tComputing cluster profile for positions...")
    if !("clustProf_size_v2" in items)
        profile = clusterProfile_DBscan_size_v2(data["xs"], data["dim"])
        results["clustProf_size_v2"] = profile
    end

    if outputDir != nothing
        analysisFilename = joinpath(outputDir, basename(analysisFilename))
    end
    @printf("analysis outputting to %s\n", outputDir)
    jldopen(analysisFilename, "a+") do f
        for (name, result) ∈ results
            f[name] = result
        end
    end

    return results
end

function analyzeAgendaGLB(agendaFilename; step=1, plotHeadless=false, outputBase=nothing, checkIfComputed=true)
    inputBase = dirname(agendaFilename)
    if outputBase == nothing
        outputBase = inputBase 
    end
    @printf("input = %s, output = %s\n", inputBase, outputBase)
        
    data = read(agendaFilename, String) |> TOML.parse
    outputFilenames = data["settings"]["analysisConfs"] |> x->map(y->split(basename(y),".")[1]*"_", x)
    files = []
    @printf("processing %d files: \n", length(outputFilenames)*length(data["settings"]["dValues"])*length(data["settings"]["sValues"]))
    for f in outputFilenames
        for pot in data["settings"]["potentials"]
            for s in data["settings"]["sValues"]
                for d in data["settings"]["dValues"]
                    temp = @sprintf("%s_%.02f_d_%.02f", pot, s, d)
                    str = joinpath(inputBase, f * temp)
            	    @printf("\t%s\n", str)
                    push!(files, str)
                end
            end
        end
    end
    results = Dict()
    N = length(files)
    for (i, f) in enumerate(files)
        @printf("\tprocessing: %s: (%d / %d)\n", f, i, N)
        r = analysis(f; step=step, plotHeadless=plotHeadless, outputDir=outputBase, checkIfComputed=checkIfComputed)
        results[f] = r
    end
    return results
end

# profiles = list of profiles
function avgProfile(profiles)
    A = zeros(length(profiles[1]))
    for i = 1:length(profiles[1])
        A[i] = sum(x[i] for x in profiles)
    end
    A ./= length(profiles)
    return [A]
end

function clusterProfOutput(dir)
    files = readdir(dir; join=true) |> x->filter(y->contains(y, "analysis.jld2"), x)
    D = DataFrame(potential = String[], dnn = Float64[], dim=Float64[], s=Float64[], d=Float64[], energyRMSE=Float64[], momentumRMSE=Float64[], profile=[])
    for f in files
        println(basename(f))
        name = f[1:findlast("-", f)[1]]
        f2 = name * "param.jld2"
        analysis = load(f)
        param = load(f2)["params"]

    	if !("energyRMSE" in keys(analysis))
    		@printf("not found in %s", f)
    	end

        row = vcat(param["pot"], param["dnn"], param["dim"], param["s"], param["d"], analysis["energyRMSE"], analysis["momentumRMSE"], [analysis["clustProf_size_v2"]])
        push!(D, row)
    end
    return D
end

function produceCSV(D, filename)
    x = groupby(D, [:s, :d]) |> x->combine(x, :profile => avgProfile)
    CSV.write(filename, x)
end
