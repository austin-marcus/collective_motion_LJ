using Random
using Printf
using Plots
using DataFrames
using CSV
include("cluster.jl")
include("physics.jl")
include("sim2.jl")

function prob_clust_1(scale, m, g, dx=0.01)
	theta = (1-g) / (Float64(m)^(1-g)-1) 
	return theta * sum(Float64(x)^(-g)*dx for x=scale:dx:scale+1-dx)
end

# function pickCluster(size c

# generate particle distribution
function dist_powerlaw(N, g; m=100)
	cluster_probs = [prob_clust_1(x, m, g) for x=1:m]
	particle_probs = [cluster_probs[x]*x for x=1:m]
	theta = N / sum(particle_probs)
	particle_counts = particle_probs .* theta
	# print(sum(particle_counts))
	cluster_counts = [particle_counts[x]/x for x=1:m]
	cluster_cum_counts = zeros(m)
	total = 0
	for i = m:-1:1
		cluster_cum_counts[i] += cluster_counts[i] + total
		total += cluster_counts[i]
	end
	cluster_cum_counts = [Int(round(x)) for x = cluster_cum_counts]
	return cluster_cum_counts, cluster_counts, cluster_probs
end

function gen_cluster(size, dim, points)
	xB = [-dim/2, dim/2]
	center = zeros(2)
	for i = 1:100
		center = rand(Float64, 2)*dim .- dim/2
		# println(center)
		collision = false
		for x in 1:Base.size(points)[1]
			p = points[x,:]
			if norm(periodDistance(center, p, xB, xB)) < 7
					collision = true
				break
			end
			# println(norm(periodDistance(center, p, xB, xB)))
		end
		if collision == false
			# println(i)
			break
		end
	end
	xs = zeros(size, 2)
	xs[1, :] = center
	for i = 2:size
		intraSpread = 4
		offset = intraSpread.*(rand(Float64,2) .- 0.5)
		# println(offset)
		xs[i,:] = center .+ offset
	end
	return xs
end

function gen_particles(cluster_counts, dim; mult=1)
	xs = fill(NaN, (Int(round(sum(i*x for (i,x) = enumerate(cluster_counts)))), 2))
	# println(xs)
	numPart = 0
	for (size, count) = reverse(collect(enumerate(cluster_counts)))
		@printf("%d, %f\n", size, count*mult)
		for c = 1:(count*mult)
			clust = gen_cluster(size, dim, xs)
			# display(clust)
			xs[numPart+1:numPart+1+size-1, :] = clust
			numPart += size
		end
	end
	return xs
end

function plotParticles(xs, dim)
	colors = :blue
    p=Plots.plot(xs[:,1], xs[:,2], seriestype=:scatter, markersize=800/dim/2, legend=false, xlim=(-dim/2,dim/2), ylim=(-dim/2,dim/2), size=(800,800), markercolor=colors)
	return p
end

function getComplex1(;dim, g, m, mult=1, num=1)
	println(num)
	ccc, cc, cp = dist_powerlaw(200, g; m=m)
	xs = gen_particles(cc, dim; mult=mult)
	profile = _clusterPoints(xs, [-dim/2, dim/2], 1:100) # compute profile
	println(ccc .- profile[1:m])
	println(sum(isnan.(xs)[:,1]))
	println()

	p = plotParticles(xs, dim)
	# display(p)
	savefig(p, @sprintf("complex_%.02f_g_%.02f_m_%d_mult_%.02f_%d.png", dim, g, m, mult, num))
	# plt = plotState(xs, 0, 0, dim)
	error = sqrt(sum((ccc .- profile[1:m]).^2))
	# open(@sprintf("complex_%.02f_g_%.02f_m_%d_mult_%.02f_%d.txt", dim, g, m, mult, num), "w") do fd
	# 	write(fd, string(profile) * "\n" * string(error))
	# end
	return profile, error
end

function getCoherent(;dim)
	xs = gen_cluster(200, dim, [])
	display(xs)
	p = plotParticles(xs, dim)
	display(p)
	savefig(p, @sprintf("coherent.png"))
	return _clusterPoints(xs, [-dim/2, dim/2], 1:100)
end

function getRandom(;dim)
	step = dim/14
	xs = zeros(200, 2)
	for i = 0:13
		for j = 0:13
			xs[(i*14+j)+1, :] = [i*step-dim/2, j*step-dim/2]
		end
	end
	display(xs)
	p = plotParticles(xs, dim)
	display(p)
	savefig(p, @sprintf("random.png"))
	return _clusterPoints(xs, [-dim/2, dim/2], 1:100)
end

function generateSet(;dim, g, m, mult=1, num)
	data = DataFrame(type=String[], dim = Float64[], num=Int[], error=Float64[], profile=[])
	errs = zeros(num, 2)
	for i = 1:num
		profile, error = getComplex1(dim=dim, g=g, m=m, mult=mult, num=i)
		errs[i,:] = [i, error]
		row = vcat("powerlaw", dim, i, error, [profile])
		push!(data, row)
	end

	push!(data, vcat("coherent", dim, 1, NaN, [getCoherent(dim=dim)]))
	push!(data, vcat("random", dim, 1, NaN, [getRandom(dim=dim)]))

	CSV.write(@sprintf("set_%.02f_g_%.02f_m_%d_mult_%.02f_%d.csv", dim, g, m, mult, num), data)
	return errs[argmin(errs[:,2]), :]
end

