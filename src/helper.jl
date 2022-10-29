using JLD2
using FileIO
include("physics.jl")

function loadData(stateFilename)
    # load invariants over whole run
    params = load(stateFilename * "-param.jld2")["params"]
    c = params["c"]
    m = params["m"]
    dnn = params["dnn"]
    dim = params["dim"]
    potName = params["pot"]
    potFunc = Meta.parse("V_" * potName) |> eval

    # load changing vars over whole run
    states = load(stateFilename * "-state.jld2")
    xs = []
    vs = []
    for s in states |> values |> collect |> sort
        push!(xs, s[2])
        push!(vs, s[3])
    end

    B = [-dim/2, dim/2]
    return Dict("c"=>c, "m"=>m, "B"=>B, "dim"=>dim, "dnn"=>dnn, "potFunc"=>potFunc, "xs"=>xs, "vs"=>vs)
end

function pointsToDistance(xs, B)
    numP = size(xs)[1]
    rs = zeros(numP, numP)
    for i = 1:numP
        for j = i+1:numP
            r = periodDistance(xs[i,:], xs[j,:], B, B) |> norm
            rs[i,j] = r
            rs[j,i] = r
        end
    end
    return rs
end
