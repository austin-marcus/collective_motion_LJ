using LinearAlgebra
using Printf
using FileIO
using StatsBase
using ProgressMeter
# using Gadfly
using DataFrames
# using Cairo, Fontconfig
include("sim2.jl")
include("cluster.jl")



# for each scale, for all time
# create histogram of particle counts in regions OR just whether a particle is present or not
# create TPM of system states (sparse representation, string encoding of state)
# compute entropy rate of TPM

# OR
# for each scale (spatial and temporal)
# compute avg velocity of particles in region
# compute entropy of avg velocity distribution across regions
    # how to choose velocity bins?

function H(probs)
    probs = filter(x->x!=0, probs)
    return -sum(probs .* log2.(probs))
end

function cprofile(;xs, dim, vs, vdim, vsens, xscales, tscales, cFunc)
    D = DataFrame(xscale = Float64[], tscale = Float64[], c=Float64[])
    C = zeros(length(xscales), length(tscales))
    prog = Progress(length(xscales)*length(tscales), desc="Computing cprofile...", barlen=80, showspeed=true)

    # vscale = (xscale/dim) * vdim
    velEdges = -vdim-.1:vsens:vdim+.1
    for (j, tscale) in collect(enumerate(tscales)) # for each tscale
        tchunks = 1:1:length(xs) |> x->Iterators.partition(x, tscale) |> collect
        for (i, xscale) in enumerate(xscales) # for each xscale
            Ht = zeros(length(tchunks))
            xbins = -dim/2:xscale:dim/2
            Threads.@threads for (t, tchunk) in collect(enumerate(tchunks))
                Ht[t] = cFunc(xs, vs, xbins, tchunk, velEdges)
            end
            C[i,j] = sum(Ht)/length(Ht) # set complexity to avg of H over time chunks at this xscale
            push!(D, vcat(xscales[i], tscales[j], C[i,j]))
            next!(prog)
        end
    end

    return C, D
end

function cprofile_noT(;xs, xscales, cFunc, skip=1)
    D = DataFrame(xscale = Float64[], tscale = Float64[], c=Float64[])
    C = zeros(length(xscales))
    prog = Progress(length(xscales), desc="Computing cprofile...", barlen=80, showspeed=true)

    T = length(xs)
    times = 1:skip:T
    Threads.@threads for (i, xscale) in collect(enumerate(xscales)) # for each xscale
        for t = times
            C[i] += cFunc(xs[t], xscale)
        end
        C[i] /= length(times)
        push!(D, vcat(xscale, 1, C[i]))
        next!(prog)
    end

    return C, D
end

function _cprofile_avgDen(xs, vs, xbins, tbins, velEdges, cutoff)
    nx = length(xbins)
    avgDen = zeros(nx-1, nx-1)
    for (ti, t) in enumerate(tbins) # for each time
        xPos = xs[t][:,1]
        yPos = xs[t][:,2]
        posHist = fit(Histogram, (xPos, yPos), (xbins, xbins)).weights
        avgDen += posHist ./ sum(posHist)
    end
    # avg over time
    avgDen ./= length(tbins)
    # display(avgDen)

    # bin density values
    bins = 0:2/nx^2:1
    denHist = fit(Histogram, reshape(avgDen, prod(size(avgDen))), bins).weights
    # denHist = fit(Histogram, reshape(avgDen, prod(size(avgDen)))).weights
    # compute entropy of the variance in density values, not how spread out density is
    return H(denHist ./ sum(denHist))# / H(repeat([1/length(bins)], length(bins)))

    # nbins = length(xbins)^2
    # return H(avgDen) / H(repeat([1/nbins], nbins))
end

function _cprofile_avgVel(xs, vs, xbins, tbins, velEdges, cutoff)
    nx = length(xbins)
    avgVxs = zeros(nx, nx)
    avgVys = zeros(nx, nx)
    nt = zeros(nx, nx, length(tbins))
    for (ti, t) in enumerate(tbins) # for each time
        nP = zeros(nx, nx)
        tmpVxs = zeros(nx, nx)
        tmpVys = zeros(nx, nx)
        for p in 1:size(xs[t])[1] # for each particle
            # place in bin
            prtcle = xs[t][p,:]
            i = searchsortedfirst(xbins, prtcle[1]) |> x->x > length(xbins) ? length(xbins) : x
            j = searchsortedfirst(xbins, prtcle[2]) |> x->x > length(xbins) ? length(xbins) : x
            tmpVxs[i,j] += vs[t][p,1]
            tmpVys[i,j] += vs[t][p,2]
            nP[i,j] += 1
            nt[i,j,ti] = 1
        end
        for i = 1:nx
            for j = 1:nx
                if nP[i,j] != 0
                    tmpVxs[i,j] /= nP[i,j]
                    tmpVys[i,j] /= nP[i,j]
                end
            end
        end
        avgVxs += tmpVxs
        avgVys += tmpVys
    end
    # avg over time
    # NOTE: averaging so to exclude regions that had no particles during a certain time step. produces NaNs for those regions in the output
    ntSums = sum(nt, dims=3)
    avgVxs ./= ntSums
    avgVys ./= ntSums

    x = filter(z->!isnan(z), avgVxs) |> z->reshape(z, prod(size(z)))
    y = filter(z->!isnan(z), avgVys) |> z->reshape(z, prod(size(z)))

    hx = fit(Histogram, x, velEdges).weights
    hy = fit(Histogram, y, velEdges).weights
    # hx = fit(Histogram, x).weights
    # hy = fit(Histogram, y).weights
    Hx = hx ./ sum(hx) |> H
    Hy = hy ./ sum(hy) |> H
    # NOTE: NaN issue occuring when scale is big enough that only 1 region exists after filtering

    return (Hx+Hy)/2
end

function plotCprofile(D, xscales, tscales; name="")
    # p1 = Gadfly.plot(D, x=:t, y=Col.value(cols...), color=Col.index(cols...), Scale.x_log10, Geom.line, Guide.ylabel("Normalized MI"), Guide.xlabel("Simulation Duration"), Guide.colorkey(labels=["Random", "Coherent"]), Coord.Cartesian(ymin=0,ymax=1.2))
    p1 = Gadfly.plot(D, x=:xscale, y=:c, Geom.line, color=:tscale, Guide.xlabel("Spatial Scale"), Guide.ylabel("Shannon Entropy"))
    p2 = Gadfly.plot(D, x=:tscale, y=:c, Geom.line, color=:xscale, Guide.xlabel("Temporal Scale"), Guide.ylabel("Shannon Entropy"))
    p = vstack(p1, p2)
    display(p);
    draw(PNG(@sprintf("cprofile_test_%s.png", name), 8inch, 6inch, dpi=200), p)
end

function test_single(filename, vdim, vsens)
    data =loadData(filename)
    dim = data["dim"]
    # xs = data["xs"]
    # vs = data["vs"]
    # xscales = [dim÷2^i for i = 10:-1:0] |> collect |> x->filter(y->y!=0, x)
    # tscales = [length(data["xs"])÷2^i for i = 10:-1:0] |> collect |> x->filter(y->y!=0, x)
    xscales = range(0, dim, length=20) |> collect |> x->floor.(x) |> x->filter(y->y!=0, x)
    # tscales = range(0, length(data["xs"]), length=15) |> collect |> x->Int.(floor.(x)) |> x->filter(y->y!=0, x)
    # tscales = [length(data["xs"]) ÷ 2]
    T = length(data["xs"])
    tscales = [1, Int(T÷5)]
    # tscales = [i÷vdim÷5e-3 for i = xscales] |> x->filter(y->y<T, x) |> x->Int.(x)
    println(xscales)
    println(tscales)
    # C,D = cprofile(xs=data["xs"], vs=data["vs"], dim=dim, vdim=vdim, vsens=vsens, xscales=xscales, tscales=tscales, cFunc=_cprofile_avgClu)
    C,D = cprofile_noT(xs=data["xs"], xscales=xscales, cFunc=countClusters)
    display(C)
    display(D)
    plotCprofile(D, xscales, tscales; name=basename(filename))
    return C, D
end

function test(s, p, t, v)
    # xscales = [s÷2^i for i = 10:-1:0] |> collect |> x->filter(y->y!=0, x)
    xscales = range(0, s, length=40) |> collect |> x->floor.(x) |> x->filter(y->y!=0, x)
    display(xscales)
    tscales = [t÷2^i for i = 10:-1:0] |> collect |> x->filter(y->y!=0, x)
    display(tscales)
    vsens = v/10

    B = [-s/2, s/2]
    xsRand = [s*(rand(p,2).-0.5)]
    vsRand = [v*rand(p,2).-0.5 for i=1:t]
    for i = 2:t
        newX = xsRand[i-1] + vsRand[i-1]
        # vsRand = [ones(p,2) for i = 1:t]
        periodicBoundary!([B, B], newX)
        push!(xsRand, newX)
    end
    # xbins = -s/2:s/2
    # tbins = 1:t

    @printf("Random:\n")
    # println(_cprofile_avgDen(xsRand, vsRand, -s/2:xscales[1]:s/2, 10, nothing))
    # println(_cprofile_avgDen(xsRand, vsRand, -s/2:30:s/2, 10, nothing))
    velEdges = -1-.1:.1:1+.1
    # for xscale in xscales
    #     x, y = _cprofile(xsRand, vsRand, -s/2:xscale:s/2, 10)
    #     display(x)
    #     x = filter(z->!isnan(z), x) |> z->reshape(z, prod(size(z)))
    #     hx = fit(Histogram, x, velEdges).weights
    #     println(hx)
    #     entropy = H(hx./sum(hx))
    #     println(entropy)
    #     println(entropy*sum(hx))
    # end

    # return
    # display(y)
    cRand, dRand = cprofile(xs=xsRand, vs=vsRand, dim=s, vdim=v, vsens=vsens, xscales=xscales, tscales=tscales, cFunc=_cprofile_avgDen)
    display(cRand)
    plotCprofile(dRand, xscales, tscales; name="rand")
    return

    @printf("Coherent:\n")
    R = -s/2:s/sqrt(p):s/2
    p = length(R)^2
    c = zeros(p,2)
    # display(R)
    a = 1
    for i = R
        for j = R
            # c[i,:] = repeat([i/p-.5], 2) .* s
            c[a,:] = [i j]
            a+=1
        end
    end
    # xsCoh = [rand(20,2).*s .- s/2]
    xsCoh = [c]
    for i = 2:t
        newX = xsCoh[i-1] .+ 1
        periodicBoundary!([B, B], newX)
        # display(newX)
        push!(xsCoh, newX)
    end
    vsCoh = [v*ones(p,2) for i = 1:t]
    println(_cprofile_avgVel(xsCoh, vsCoh, -s/2:xscales[1]:s/2, 10, nothing))
    println(_cprofile_avgVel(xsCoh, vsCoh, -s/2:30:s/2, 10, nothing))
    return
    cCoh, dCoh = cprofile(xs=xsCoh, vs=vsCoh, dim=s, vdim=v, vsens=vsens, xscales=xscales, tscales=tscales, cFunc=_cprofile_avgDen)
    display(cCoh)
    plotCprofile(dCoh, xscales, tscales; name="coh")
    # display(xsCoh)
    # display(vsCoh)
    # x, y = _cprofile(xsCoh, vsCoh, xbins, tbins)
    # display(x)
    # display(y)
    return cRand, cCoh
end
