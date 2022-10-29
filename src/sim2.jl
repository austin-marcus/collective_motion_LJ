using LinearAlgebra
using ProgressMeter
using Plots
using Random
using Serialization
using Dates
using JLD2
using FileIO
using Printf
include("physics.jl")

function netAccels(x, m, c, pairForce, xB, yB)
    accels = zeros(size(x))
    Threads.@threads for i = 1:size(x)[1]
        for j = 1:size(x)[1]
            if i == j
                continue
            end
            r = periodDistance(x[i,:],  x[j,:], xB, yB)
            out = pairForce(r, m[i], m[j], c[i], c[j])
            accels[i,:] += out
        end
        accels[i,:] /= m[i]
    end
    return accels
end

function detectNan(l)
    return filter(isnan, l) |> x->length(x) > 0 ? true : false
end

function verlet(x,v,m,c,pairForce,xB,yB, dt)
    xMid = x + v*dt/2
    a = netAccels(xMid,m,c,pairForce,xB,yB)
    v2 = v + a*dt
    x2 = xMid + v2 * dt/2
    return x2, v2
end

function simStep(x, v, m, c, pairForce, xB, yB,dt)
    xNext, vNext = verlet(x,v,m,c,pairForce,xB,yB,dt)
    periodicBoundary!([xB, yB], xNext)
    return xNext, vNext
end

function periodicBoundary!(vecBounds, x; xPrev=nothing)
    function f(u, l, n)
        if n > u
            return l
        elseif n < l
            return u
        else
            return n
        end
    end
    for i = 1:size(x)[1]
        x[i, 1] = f(vecBounds[1][2], vecBounds[1][1], x[i, 1])
        x[i, 2] = f(vecBounds[2][2], vecBounds[2][1], x[i, 2])
        if xPrev != nothing
            xPrev[i, 1] = f(vecBounds[1][2], vecBounds[1][1], xPrev[i, 1])
            xPrev[i, 2] = f(vecBounds[2][2], vecBounds[2][1], xPrev[i, 2])
        end
    end
end

function plotState(x, c, time, dim)
    colors = :blue
    plt = scatter(x[:,1], x[:,2], markersize=800/dim/2, legend=false, xlim=(-dim/2,dim/2), ylim=(-dim/2,dim/2), title=time, size=(800,800), markercolor=colors)
    return(plt)
end

function detectSingularity(x, xB, yB; thresh=1)
    # compute pairwise distances
    for i = 1:size(x)[1]
        for j = i+1:size(x)[1]
            r = periodDistance(x[i,:], x[j,:], xB, yB) |> norm
            if r < thresh
                println(r)
                return true
            end
        end
    end
    return false
end

function incrementalSave!(filename, Hs)
    jldopen(filename, "a+") do f
        for el ∈ Hs
            f[@sprintf("%f", el[1])] = el
        end
    end
    deleteat!(Hs, 1:length(Hs))
end

function oneRun(;T, x, v, m, c, dim, F, V, outputDir, runName, dt=dt)
    t = 0
    ft = 0
    ani = Animation()
    E = []
    P = []
    pMult = 100
    prog = Progress(T*pMult, dt=1, desc="Running...", barlen=80, showspeed=true)
    at = 0

    xB = [-dim/2, dim/2]
    yB = [-dim/2, dim/2]

    E1 = getEnergy(x,v,m,c,V,xB,yB)

    tPf = 10^-1

    xPrev = nothing

    st = 0

    # state history
    N = 10^4
    simResultFileName = joinpath(outputDir, runName * "-state.jld2")
    Hs = []

    t1 = now()
    while t < T
        x, v = simStep(x,v,m,c, F, xB, yB, dt)

        t += dt
        ft += dt
        if ft > tPf
            plt = plotState(x, c, t, dim)
            frame(ani, plt)
            ft = 0
        end
        update!(prog, Int(floor(t*pMult)))

        # incrementally save state and allow for garbage collection
        push!(Hs, (t, x, v))
        if length(Hs) >= N
            incrementalSave!(simResultFileName, Hs)
        end
    end

    if length(Hs) >= 1
        incrementalSave!(simResultFileName, Hs)
    end

    println("Generating animation...")
    display(gif(ani, joinpath(outputDir, runName * ".mp4"), fps=5/tPf))

    t2 = now()

    # output percent energy change
    E2 = getEnergy(x,v,m,c,V,xB,yB)
    pe = (E2[1] - E1[1]) / E1[1] * 100

    return pe, Dates.canonicalize(t2-t1), dt
end

function runSimGLB(configPath, potName, s, d, A, T, outputDir, runName, dt)

    Ftemp = Meta.parse("F_" * potName) |> eval
    F = (r, m1, m2, c1, c2)->Ftemp(r, m1, m2, c1, c2, s, d, A)
    Vtemp = Meta.parse("V_" * potName) |> eval
    V = (r, m1, m2, c1, c2)->Vtemp(r, m1, m2, c1, c2, s, d, A)

    println(Meta.parse("V_" * potName))

    (x,v,m,c,dim,dnn) = deserialize(configPath)

    # save unchanging parameters in separate file
    simParamsFileName = joinpath(outputDir, runName * "-param.jld2")
    jldopen(simParamsFileName, "w") do f
        f["params"] = Dict("dnn"=>dnn, "dim"=>dim, "pot"=>potName, "m"=>m, "c"=>c, "s"=>s, "d"=>d, "A"=>A)
    end

    return oneRun(T=T, x=x, v=v, m=m, c=c, dim=dim, F=F, V=V, outputDir=outputDir, runName=runName, dt=dt)

end
