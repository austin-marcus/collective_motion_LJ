# using Graphs
using Printf
# using SimpleWeightedGraphs
using Clustering
# using StatsBase
include("helper.jl")

function clusterProfile_DBscan_size_v2(xs, dim; clusterSizes=1:100, step=1000)
    scores = zeros(length(clusterSizes))
    start = length(xs) |> x->x/2|> floor |> Int
    samples = start:step:length(xs)
    B = [-dim/2, dim/2]
    for t = samples
        scores .+= _clusterPoints(xs[t], B, clusterSizes)
        # xst = pointsToDistance(xs[t], B)
        # clusters = dbscan(xst, 2, 1).counts
        # for (j, s) = enumerate(clusterSizes)
        #     l = filter(x->x>=s, clusters)
        #     scores[j] += l |> length
        # end
    end
    scores ./= length(samples)
    return scores
end

function _clusterPoints(x, B, clusterSizes)
    xDis = pointsToDistance(x, B)
    clusters = dbscan(xDis, 2, 1).counts
    scores = zeros(length(clusterSizes))
    for (j, s) = enumerate(clusterSizes)
        l = filter(x->x>=s, clusters)
        scores[j] = l |> length
    end
    return scores
end
