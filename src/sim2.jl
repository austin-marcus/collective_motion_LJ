using LinearAlgebra
using ProgressMeter
using Plots
using Random
using Serialization
using Revise
using Dates
using JLD2
using FileIO
using Printf
includet("physics.jl")

function netAccels(x, m, c, pairForce, xB, yB)
    accels = zeros(size(x))
    Threads.@threads for i = 1:size(x)[1]
        for j = 1:size(x)[1]
            if i == j
                continue
            end
            r = periodDistance(x[i,:],  x[j,:], xB, yB)
            out = pairForce(r, m[i], m[j], c[i], c[j])
            # if detectNan(out)
            #     println("$i,$j")
            #     display(x)
            # end
            accels[i,:] += out
            # display(accels)
        end
        # println(detectNan(accels))
        accels[i,:] /= m[i]
        # println(detectNan(accels))
    end
    return accels
end

# getDt(v, a; min=10^8, max=1) =

function RK2(x, v, m, pairForce, xB, yB; dt=nothing)
    a = netAccels(x, m, pairForce, xB, yB) # compute all forces
    if dt == nothing
        dt = getDt(v,a)
    end
    # dt = 10^-4

    xMid = x + v * (dt/2)
    vMid = v + a * (dt/2)

    xNext = x + vMid * dt

    # midState = stateUpdateXV(state, midPointXs, midPointVs)
    aMid = netAccels(xMid, m, pairForce, xB, yB) # compute all forces

    vNext = v + aMid * dt

    return xNext, vNext, dt
end

function detectNan(l)
    return filter(isnan, l) |> x->length(x) > 0 ? true : false
end

function verlet2(xPrev,x,v,m,pairForce,xB,yB; dt=nothing)
    # println("\nm = ", detectNan(m))
    # println("x = ", detectNan(x))
    a = netAccels(x,m,pairForce,xB,yB)
    # println("a = ", detectNan(a))
    if dt == nothing
        dt = 10^-3
        # dt = getDt(v,a)
    end
    # println("x = ", detectNan(x), "\txPrev = ", detectNan(xPrev))
    xNext = 2x - xPrev + a*dt^2
    # println("xNext = ", detectNan(xNext))
    vNext = (xNext - xPrev) / (2dt)
    return xNext, vNext, dt
end

function verlet(x,v,m,pairForce,xB,yB, dtPrev)
    # dtNew = 10^-3
    xMid = x + v*dtPrev/2
    a = netAccels(xMid,m,pairForce,xB,yB)
    dtNew = getDt(v,a)
    v2 = v + a*dtNew
    x2 = xMid + v2 * dtNew/2
    return x2, v2, dtNew
end

function verlet3(x,v,m,c,pairForce,xB,yB; dt)
    xMid = x + v*dt/2
    a = netAccels(xMid,m,c,pairForce,xB,yB)
    #dt = getDt(v,a)
    # dt = getDt2(x, 1, 10^-3, 10^-4, xB, yB)
    v2 = v + a*dt
    x2 = xMid + v2 * dt/2
    return x2, v2, dt
end

function getDt2(x, thresh, dt1, dt2, xB, yB)
    # compute min distance between two
    minr = Inf
    for i = 1:size(x)[1]
        for j = i+1:size(x)[1]
            r = periodDistance(x[i,:], x[j,:], xB, yB) |> norm
            if r < minr
                minr = r
            end
        end
    end
    if minr < thresh
        return dt2
    else
        return  dt1
    end
end

function getDt(v,a)
    maxA = 0
    maxV = 0
    for i = 1:size(a)[1]
        anorm = norm(a[i,:])
        vnorm = norm(v[i,:])
        if anorm > maxA
            maxA = anorm
        end
        if vnorm > maxV
            maxV = vnorm
        end
    end
    dt = 10^-3
    # dt = min(sqrt(10^-4 / maxA) / maxV, 10^-1) # this one works well!
    # dt = max(min(sqrt(10^-4 / maxA) / maxV, 10^-1), 10^-6)
    # dt = min(maxA^-6, .1) |> y->max(y, 10^-5) #dt3
    # dt = min(sqrt(10^-7 / maxA) / maxV, 10^-1)
    # dt = max(10^-9, min(sqrt(10^-5)/ maxA / maxV^2, 10^-1))
    # dt = max(10^-15, min(sqrt(10^-5)/ maxA^6 / maxV^6, 10^-1))
    # dt = min(1/(maxV + maxA)/1, 1)
    # dt = min(1/(maxV^2 + maxA^2)/1, 1)
    # dt = min(1/(maxV^2 + maxA^2)/5, 1)
    # dt = min(1/(maxV^2 + maxA^2)/10, 1)
    # dt = min(1/(maxV^2 + maxA^2)/100, 1)
    # println(dt)
    return dt
end

# function simStepVerlet(xPrev, x, v, m, pairForce, xB, yB)
#     dt = 10^-2
#     if xPrev == nothing
#         println("\nusing RK2 to get started")
#         xNext, vNext, dt = RK2(x,v,m,pairForce,xB,yB; dt=dt)
#     else
#         xNext, vNext, dt = verlet2(xPrev,x,v,m,pairForce,xB,yB; dt=dt)
#     end
#
#     # wallBoundary!([ (-vol, vol), (-vol, vol) ], xNext, vNext)
#     periodicBoundary!([xB, yB], xNext; xPrev=x)
#     return xNext, vNext, dt, x
# end

function simStep(x, v, m, c, pairForce, xB, yB, dtPrev)

    # xNext, vNext, dt = RK2(x,v,m,pairForce,xB,yB)
    xNext, vNext, dt = verlet3(x,v,m,c,pairForce,xB,yB; dt=dtPrev)
    # println(xNext)

    # nextState = stateUpdateXV(state, nextXs, nextVs)
    # wallBoundary!([ (-vol, vol), (-vol, vol) ], xNext, vNext)
    periodicBoundary!([xB, yB], xNext)
    return xNext, vNext, dt
end

function isWithin(vecBounds, testVec)
    # vecBounds = [ (x1,x2), (y1,y2) ]
    flips = ones(length(testVec))
    for i = 1:length(testVec)
        if !(vecBounds[i][1] < testVec[i] < vecBounds[i][2])
            flips[i] = -1
        end
    end
    return(flips)
end

function wallBoundary!(vecBounds, x, v)
    for i = 1:size(v)[1]
        flips = isWithin(vecBounds, x[i,:])
        v[i,:] .*= flips
    end
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

function plotState(x, c, time, dim; chargeColor=false)
    colors = :blue
    if chargeColor
        colors = [i == 1 ? :blue : :red for i in c]
    end
    plt = scatter(x[:,1], x[:,2], markersize=800/dim/2, legend=false, xlim=(-dim/2,dim/2), ylim=(-dim/2,dim/2), title=time, size=(800,800), markercolor=colors)
    return(plt)
end

function createGIF(animation, outputDir; animate=true, fps=50)
    if animate
        display(gif(animation, "$outputDir/animation.mp4", fps=fps))
    end
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
    # F, V = F_gra, V_gra
    # F, V = F_mol, V_mol

    xB = [-dim/2, dim/2]
    yB = [-dim/2, dim/2]

    E1 = getEnergy(x,v,m,c,V,xB,yB)

    tPf = 10^-1

    xPrev = nothing

    st = 0
    #dt = 10^-2

    # state history
    N = 10^4
    simResultFileName = joinpath(outputDir, runName * "-state.jld2")
    Hs = []

    t1 = now()
    singularity = false
    while t < T
        # global x, v, t, ft, T, ani, E, prog, at, F, V, P
        x, v, dt = simStep(x,v,m,c, F, xB, yB, dt)
        # x, v, dt, xPrev = simStepVerlet(xPrev, x,v,m, F, xB, yB)

        # println("dt = $dt")
        #st += dt
        #if st > 10^-2
        #    if detectSingularity(x, xB, yB; thresh=10^-2)
        #        println("\nTERMINATED: detected singularity")
        #        singularity = true
        #        break
        #    end
        #    st = 0
        #end

        # at += dt
        # if at > 0.5
        #     push!(E, getEnergy(x,v,m, V, xB, yB))
        #     push!(P, getMomentum(v, m))
        #     at = 0
        # end

        t += dt
        ft += dt
        if ft > tPf
            colorCharge = V == V_ele ? true : false
            plt = plotState(x, c, t, dim; chargeColor=colorCharge)
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

    # plot([i[1] for i in E]) |> display

    # output percent energy change
    # change = ([E[i][1] for i=1:length(E)] .|> abs) |> y->max(y...)
    # change = E[end][1]
    # pe = (E[1][1] - change) / E[1][1] * 100
    E2 = getEnergy(x,v,m,c,V,xB,yB)
    pe = (E2[1] - E1[1]) / E1[1] * 100

    return pe, Dates.canonicalize(t2-t1), dt, singularity
end

function runSim(configPath, potName, T, outputDir, runName, dt)

    F = Meta.parse("F_" * potName) |> eval
    V = Meta.parse("V_" * potName) |> eval

    (x,v,m,c,dim,dnn) = deserialize(configPath)
    #display(c)

    # save unchanging parameters in separate file
    simParamsFileName = joinpath(outputDir, runName * "-param.jld2")
    jldopen(simParamsFileName, "w") do f
        f["params"] = Dict("dnn"=>dnn, "dim"=>dim, "pot"=>potName, "m"=>m, "c"=>c)
    end

    return oneRun(T=T, x=x, v=v, m=m, c=c, dim=dim, F=F, V=V, outputDir=outputDir, runName=runName, dt=dt)

end

function runSimGLB(configPath, potName, s, d, A, T, outputDir, runName, dt)

    Ftemp = Meta.parse("F_" * potName) |> eval
    F = (r, m1, m2, c1, c2)->Ftemp(r, m1, m2, c1, c2, s, d, A)
    Vtemp = Meta.parse("V_" * potName) |> eval
    V = (r, m1, m2, c1, c2)->Vtemp(r, m1, m2, c1, c2, s, d, A)

    println(Meta.parse("V_" * potName))

    (x,v,m,c,dim,dnn) = deserialize(configPath)
    #display(c)

    # save unchanging parameters in separate file
    simParamsFileName = joinpath(outputDir, runName * "-param.jld2")
    jldopen(simParamsFileName, "w") do f
        f["params"] = Dict("dnn"=>dnn, "dim"=>dim, "pot"=>potName, "m"=>m, "c"=>c, "s"=>s, "d"=>d, "A"=>A)
    end

    return oneRun(T=T, x=x, v=v, m=m, c=c, dim=dim, F=F, V=V, outputDir=outputDir, runName=runName, dt=dt)

end
