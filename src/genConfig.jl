using Serialization
using Random
using LinearAlgebra
using Printf
include("physics.jl")

function avgDistToNearest(x)
    s = 0
    for i = 1:size(x)[1]
        # find nearest particle
        mind = Inf
        for j = 1:size(x)[1]
            if i == j
                continue
            end
            d = norm(x[i,:] - x[j,:])
            if d < mind
                mind = d
            end
        end
        s += mind
        if mind < 10^-3
            println("WARNING: singularity between $i and $j")
        end
    end
    return s/size(x)[1]
end

function genConfig(outputDir, N, n; minDist=1, tag="")
    mkdir(outputDir)

    numSets = n
    #N = 200
    dnn = [5 3 2.5 2]
    dims = sqrt(N) * dnn * 2
    V = 1/sqrt(2)
    for s in 1:numSets
    	for (i, dim) in enumerate(dims)
    	    B = [-dim/2,dim/2]
    	    println("ρ = ", N/dim^2)
    	    xRange = [-dim/2, dim/2]
    	    println("xRange = $xRange")
    	    xs = zeros(N,2)
    	    n = 0
    	    while n < N
    	        newX = rand(2) * (xRange[2]-xRange[1]) .+ xRange[1]
    	        ok = true
    	        for i = 1:n
    	            r = periodDistance(newX, xs[i,:], B, B) |> norm
    	            if r < minDist
    	                ok = false
    	                break
    	            end
    	        end
    	        if ok
    	            n+=1
    	            xs[n,:] = newX
    	        end
    	    end
    	    display(xs)
    	    println("Mindist = ", getMinR(xs, B, B))
    	    println("avgDist = ", avgDistToNearest(xs), "\tExpected avg dist = ", dim/sqrt(N)/2)

    	    vRange = [-V, V]
    	    vs = rand!(zeros(N, 2)) * (vRange[2]-vRange[1]) .+ vRange[1]
    	    ms = ones(N, 2)
    	    cs = rand!(zeros(N, 2)) .-0.5 .|> x->x/abs(x)
    	    outFilename = @sprintf("N_%d-D_%d%s-%d-U%0.2d.conf", N, Int(round(dnn[i]*10)), tag != "" ? "-"*tag : "", s, Int(floor(rand() * 99)))
    	    serialize(joinpath(outputDir, outFilename), (xs, vs, ms, cs, dim, dnn[i]))
    	end
    end
end

function extractConfig(filename)
    things = deserialize(filename)
    for i in things
        display(i)
    end
    return things
end

function getMinR(x, xB, yB)
    minr = Inf
    for i = 1:size(x)[1]
        for j = i+1:size(x)[1]
            r = periodDistance(x[i,:], x[j,:], xB, yB) |> norm
            if r < minr
                minr = r
            end
        end
    end
    return minr
end
