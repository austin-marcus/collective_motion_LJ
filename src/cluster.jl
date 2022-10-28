using Graphs
using Printf
using Plots
using SimpleWeightedGraphs
using Clustering
using StatsBase
include("helper.jl")

# xs = (np, 2) matrix, representing a SINGLE timestep
function countClusters(xs, cutoff)
    # compute pair-wise distance
    np = size(xs)[1]
    d = zeros(np, np)
    for i = 1:np
        for j = i+1:np
            r = norm(xs[i,:] - xs[j,:])
            d[i,j] = r
            # make symmetric entry too
            d[j,i] = r
        end
    end
    # display(d)
    # binarize
    f(x) = x < cutoff ? 1 : 0 |> Int
    gm = map(f, d)
    # display(gm)
    # connected components
    g = SimpleGraph(gm)
    cc = connected_components(g)
    ccSize = [length(i)^2 for i = cc] |> x->(sum(x)/length(x))
    @printf("ccSize = %.04f\n", ccSize)
    # return global_clustering_coefficient(g) #MAYBE?
    return length(cc) #NO
    # return (assortativity(g)) #MAYBE?
    # cps = core_periphery_deg(g)
    # core = filter(x->x==1, cps) |> sum |> x->x > 1 ? 1 : 0
    # peri = filter(x->x==2, cps) |> sum
    # return core + peri #NO
    # return local_clustering_coefficient(g, collect(vertices(g))) |> x->(sum(x)/length(x))
    # return modularity(g, [rand(1:2) for i = 1:length(vertices(g))])

end

function dbscanSizeH(dbscanResult)
    if length(dbscanResult) > 0
        sizes = Float64[]
        for el in dbscanResult
            push!(sizes, el.size)
        end
        binWeights = fit(Histogram, sizes).weights
        binProbs = binWeights ./ sum(binWeights)
        return H(binProbs)
    else
        return 0
    end
end

function clusterProfile_DBscan(xs, minDists; step=1000)
    scores = zeros(length(minDists))
    start = length(xs) |> x->x/2|> floor |> Int
    samples = start:step:length(xs)
    for t = samples
        xst = xs[t]'
        for (j, minDist) = enumerate(minDists)
            clusters = dbscan(xst, minDist, min_neighbors=3, min_cluster_size=3)
            # l = filter(x->x.size>1, clusters)
            # scores[j] += l |> length
            # display(clusters)
            # scores[j] += clusters |> length
            scores[j] += dbscanSizeH(clusters)
        end
    end
    scores ./= length(samples)
    return scores
end

function clusterProfile_DBscan_size_v2(xs, dim; clusterSizes=1:100, step=1000)
    scores = zeros(length(clusterSizes))
    start = length(xs) |> x->x/2|> floor |> Int
    samples = start:step:length(xs)
    B = [-dim/2, dim/2]
    for t = samples
        display(xs[t])
        xst = pointsToDistance(xs[t], B)
        display(xst)
        clusters = dbscan(xst, 2, 1).counts
        for (j, s) = enumerate(clusterSizes)
            l = filter(x->x>=s, clusters)
            # scores[j] += dbscanSizeH(l)
            scores[j] += l |> length
        end
    end
    scores ./= length(samples)
    return scores
end

function clusterProfile_DBscan_size(xs; clusterSizes=1:100, step=1000)
    scores = zeros(length(clusterSizes))
    start = length(xs) |> x->x/2|> floor |> Int
    samples = start:step:length(xs)
    for t = samples
        xst = xs[t]'
        clusters = dbscan(xst, 2, min_cluster_size=1)
        for (j, s) = enumerate(clusterSizes)
            l = filter(x->x.size>=s, clusters)
            # scores[j] += dbscanSizeH(l)
            scores[j] += l |> length
        end
    end
    scores ./= length(samples)
    return scores
end

function CPDB_test2(filenames, labels)
    p = Plots.plot(xaxis=:log, xlabel="Minimum Cluster Size", ylabel="# Clusters")
    results = []
    for (file, label) in zip(filenames, labels)
        data = loadData(file)
        xs = data["xs"]
        # prof = clusterProfile_DBscan_size2(xs, data["dim"])
        prof = clusterProfile_DBscan_size(xs)
        p =  Plots.plot!(p, prof, label=label)
        push!(results, prof)
    end
    display(p)
    # savefig(p, "cluster_min.png")
    return results
end

function CPDB_test(filenames, labels, minDists)
    p = Plots.plot(xlabel="Neighborhood Radius", ylabel="H of Cluster Size Distribution")
    results = []
    for (file, label) in zip(filenames, labels)
        xs = loadData(file)["xs"]
        prof = clusterProfile_DBscan(xs, minDists)
        p =  Plots.plot!(p, minDists, prof, label=label)
        push!(results, prof)
    end
    display(p)
    savefig(p, "cluster_H.png")
    return results
end

function clusterProfile(xs; clusterDist=3, step=1000, threshs=0:.01:.5)
    scores = zeros(length(threshs))
    samples = 1:step:length(xs)
    for t = samples
        # compute distance matrix
        xst = xs[t]
        np = size(xst)[1]
        d = zeros(np, np)
        for i = 1:np
            for j = 1:np
                r = norm(xst[i,:] - xst[j,:])
                d[i,j] = r
                d[j,i] = r
            end
        end

        # compute nodal densities
        densitys = zeros(size(d)[1])
        for i = 1:size(d)[1]
            densitys[i] = (d[i,:] |> x->filter(y->y!=0, x) .|> x->(1/x)) |> x->(sum(x)/length(x))
        end

        # compute information score across density thresholds
        for (j, thresh) = enumerate(threshs)
            keeps = map(x->(x>thresh ? 1 : 0), densitys)
            removedNum = filter(x->(x==0), keeps) |> length
            gm = map(x->(x<clusterDist ? 1 : 0), d)
            for (i, p) = enumerate(keeps)
                if p == 0
                    gm[i,:] .= 0
                    gm[:,i] .= 0
                end
                gm[i,i] = 0
            end
            g = SimpleGraph(gm)
            cc = connected_components(g)
            scores[j] += length(cc) - removedNum
            # push!(scores, length(cc) - removedNum)
        end
    end
    scores ./= length(samples)
    return scores
end

function test2(filenames)
    for filename in filenames
        data = loadData(filename)
        t = length(data["xs"])
        xs = data["xs"][t]

        np = size(xs)[1]
        d = zeros(np, np)
        sources::Vector{Int64} = []
        dests::Vector{Int64} = []
        weights::Vector{Float64} = []
        for i = 1:np
            for j = 1:np
                r = norm(xs[i,:] - xs[j,:])
                push!(sources, i)
                push!(dests, j)
                push!(weights, r)
            end
        end
        g = SimpleWeightedGraph(sources, dests, weights)
        # display(g)

        minM = +Inf
        for p = 1:50
            for i = 1:1000
                m = modularity(g, [rand(1:p) for i = 1:np])
                if m < minM
                    minM = m
                end
            end
        end
        @printf("%s ==> %.04f\n", filename, minM)
    end
end

# knockout nodes based on connection density to emualte effect of blurring ==> blurring
    # white space removes isolated nodes first
# then count number of connected components ==> filesize
function test3(filenames)
    output = []
    for filename in filenames
        data = loadData(filename)
        t = length(data["xs"])
        xs = data["xs"][t]

        np = size(xs)[1]
        d = zeros(np, np)
        sources::Vector{Int64} = []
        dests::Vector{Int64} = []
        weights::Vector{Float64} = []
        for i = 1:np
            for j = 1:np
                r = norm(xs[i,:] - xs[j,:])
                d[i,j] = r
                d[j,i] = r
                push!(sources, i)
                push!(dests, j)
                push!(weights, r)
            end
        end
        # g = SimpleWeightedGraph(sources, dests, weights)
        # display(g)
        display(d)

        densitys = zeros(size(d)[1])
        for i = 1:size(d)[1]
            densitys[i] = (d[i,:] |> x->filter(y->y!=0, x) .|> x->(1/x)) |> x->(sum(x)/length(x))
        end
        println(densitys)

        scores = []
        for thresh = 0:.01:.5
            keeps = map(x->(x>thresh ? 1 : 0), densitys)
            removedNum = filter(x->(x==0), keeps) |> length
            gm = map(x->(x<3 ? 1 : 0), d)
            for (i, p) = enumerate(keeps)
                if p == 0
                    gm[i,:] .= 0
                    gm[:,i] .= 0
                end
                gm[i,i] = 0
            end
            # if thresh == 0.5
            #     display(gm)
            #     return
            # end
            # display(gm)
            g = SimpleGraph(gm)
            cc = connected_components(g)
            push!(scores, length(cc) - removedNum)
        end

        push!(output, scores)
        # push!(output, [ length(filter(x->(x>thresh), densitys)) for thresh = 1:50 ])
        # return densitys
    end
    Plots.plot(output, xaxis=:log) |> display
    return output
end

function test(filenames)
    p = Plots.plot()
    for filename in filenames
        data = loadData(filename)
        t = length(data["xs"])
        xs = data["xs"][t]
        r = 1:.1:10
        # println(countClusters(xs, cutoff))
        hs = zeros(length(r))
        for (i, c) = enumerate(r)
            hs[i] = countClusters(xs, c)
        end
        p = Plots.plot!(hs)
    end
    Plots.plot(p) |> display
end
