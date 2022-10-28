using JLD2
using FileIO
using Plots
# using MultiKDE
# using Cubature
using ProgressMeter
using Printf
using TOML
using StatsBase
using DataFrames
# using Gadfly
# using Cairo
using Fontconfig
using CSV
include("physics.jl")
include("profile.jl")
include("cluster.jl")
include("helper.jl")


# runtime estimate: 6 minutes on lab computer
# maxevals = 1e4 may be sufficient ≈ 230 sec
# for i = [1e1, 1e2, 1e3, 1e4]
#    @time x = diffEntropy(pdf, -dim/2, dim/2, 4; maxevals=i)
#    println(x)
# end
function diffEntropy(f, xmin, xmax, d; tol=1e-3, maxevals=0)
    x1 = repeat([xmin], d)
    x2 = repeat([xmax], d)
    function integrand(x)
        fx = f(x...)
        return fx*log2(fx)
    end
    ret = hcubature(integrand,x1, x2; reltol=tol, abstol=tol, maxevals=Int(maxevals))
    # println("error = ", ret[2])
    return -ret[1]
end

# convert to vector of points for particle i
# xs = all particle state positions over time
function extractParticlePoints(xs, i; step=1)
    return [xs[t][i,:] for t in 1:step:length(xs)]
end

# xis = vector of points for one particle over time
function getSingleParticlePDF(xis)
    n = length(xis)
    d = 2
    # scott's rule: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    # bw = n^(-1/(d+4))
    bw = 2
    kde = KDEMulti([ContinuousDim(), ContinuousDim()], [bw, bw], xis)
    return (x, y)->MultiKDE.pdf(kde, [x, y])
end

function getJointParticlePDF(xis1, xis2)
    bw = 2
    bws = repeat([bw], 4)
    dims = repeat([ContinuousDim()], 4)
    n = length(xis1)
    # xis12 = Array{Array{Float64, 1}, 1}(undef, n, 4)
    xis12 = []
    for (n, (i, j)) in enumerate(zip(xis1, xis2))
        # xis12[n] = [i..., j...]
        push!(xis12, [i..., j...])
    end
    kde = KDEMulti(dims, bws, xis12)
    return (w, x, y, z)->MultiKDE.pdf(kde, [w, x, y, z])
end

function MI_pos_disc(xs, dim; binsize=1, print=false)
    # compute 2D normalized histogram of particle positions over time
    n = size(xs[1])[1]
    histis = Array{Any}(undef, n)
    frames = length(xs)
    nbins = Int(ceil(dim/binsize))
    # r = reshape(xs, prod(size(xs)))
    # maxR = findmax(r)[1]
    # minR = findmin(r)[1]
    # nbins = (maxR-minR) / binsize
    # nbins = n^(1/3) |> floor |> Int
    xis = Array{Any}(undef, n)
    prog = Progress(n, desc="Computing single particle histograms...", barlen=80, showspeed=true)
    Threads.@threads for i = 1:n
        xis[i] = extractParticlePoints(xs, i)
        xisX = [j[1] for j in xis[i]]
        xisY = [j[2] for j in xis[i]]
        h1 = fit(Histogram, (xisX, xisY); nbins=nbins)
        h = h1.weights ./ frames
        # h = fit(Histogram, (xisX, xisY), Tuple(-dim/2:binsize:dim/2 for i=1:2)).weights ./ frames
        # if print == true
        #     display(h1)
        # end
        histis[i] = h
        next!(prog)
        # push!(histis, h)
    end

    prog = Progress(2*binomial(n,2) + n, desc="Computing single particle entropies...", barlen=80, showspeed=true)
    # compute entropy of each single distribution
    Hi = zeros(n)
    Threads.@threads for i = 1:n
        Hi[i] = H(histis[i])
        next!(prog)
    end

    # compute 4D normalized histogram of particle-pair positions over time
    prog = Progress(binomial(n,2), desc="Computing particle pair histograms and MI...", barlen=80, showspeed=true)
    Hij = zeros(n,n)
    MI = zeros(n,n)
    U = zeros(n,n)
    Threads.@threads for i = 1:n
        xisX = [j[1] for j in xis[i]]
        xisY = [j[2] for j in xis[i]]
        for j = i+1:n
            xjsX = [j[1] for j in xis[j]]
            xjsY = [j[2] for j in xis[j]]
            tup = (xisX, xisY, xjsX, xjsY)
            h = fit(Histogram, tup; nbins=nbins).weights ./ frames
            # h = fit(Histogram, tup, Tuple(-dim/2:binsize:dim/2 for i=1:4)).weights ./ frames
            Hij[i,j] = H(h)
            MI[i,j] = Hi[i] + Hi[j] - Hij[i,j]
            U[i,j] = 2MI[i,j] / (Hi[i] + Hi[j]) # symmetric uncertainty (Witten & Frank 2005)
            # U[i,j] = MI[i,j] / Hi[i]
            next!(prog)
        end
    end

    return MI, U, histis
end

function MI_pos_cont(xs, dim; step=1)
    # estimate position probability distribution for each particle from its positions over time
        # multivariate kernel density estimation
    numP = size(xs[1])[1]
    R = -dim/2:step:dim/2
    xis = Array{Any}(undef, numP)
    pdfis = Array{Any}(undef, numP)
    println("Computing pdfs...")
    Threads.@threads for i = 1:numP
        xis[i] = extractParticlePoints(xs, i; step=step)
        pdf = getSingleParticlePDF(xis[i])
        pdfis[i] = pdf
    end

    # compute the differential entropy of this continous distribution
    Hs = zeros(numP)
    prog = Progress(numP, desc="Computing H(X)...", barlen=80, showspeed=true)
    Threads.@threads for i = 1:numP
        Hs[i] = diffEntropy(pdfis[i], R[1], R[end], 2)
        next!(prog)
    end
    display(Hs)

    # do the same for each joint distribution
    println("Computing joint pdfs...")
    pdfijs = Array{Any}(undef, numP, numP)
    Threads.@threads for i = 1:(numP-1)
        for j = i+1:numP
            pdfij = getJointParticlePDF(xis[i], xis[j])
            pdfijs[i,j] = pdfij
        end
    end
    display(pdfijs)

    # compute MI for each pair
    MI = zeros(numP, numP)
    n = numP*(numP+1)/2-numP |> Int
    prog = Progress(n, desc="Computing MI...", barlen=80, showspeed=true)
    Threads.@threads for i = 1:numP-1
        for j = i+1:numP
            Hxy = diffEntropy(pdfijs[i,j], R[1], R[end], 4; maxevals=1e3)
            MI[i,j] = Hs[i] + Hs[j] - Hxy
            next!(prog)
        end
    end
    display(MI)
    # compute some summary statistic on the above
    return MI, Hs, pdfis
end

function analyze_temperature(Us, N; step=1)
    Ts = Float64[]
    for i = 1:step:length(Us)
        t = getTemperature(Us[i], N)
        push!(Ts, t)
    end
    return Ts
end

# step is the number of samples to take
function analyze_energy(m, c, xs, vs, xB, yB, potFunc; step=1)
    # println("E[1] = ", getEnergy(xs[1], vs[1], m, potFunc, xB, yB))
    # println("E[50] = ", getEnergy(xs[50], vs[50], m, potFunc, xB, yB))
    # println("E[100] = ", getEnergy(xs[100], vs[100], m, potFunc, xB, yB))
    Es = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    #println(length(xs))
    for i = 1:length(xs)÷step:length(xs)
        E, V, K = getEnergy(xs[i], vs[i], m, c, potFunc, xB, yB)
        #println("$E\t$V\t$K")
        push!(Es, E)
        push!(Vs, V)
        push!(Ks, K)
    end

    return (Es, Vs, Ks)
end

function SMSE(init, ts)
    return ((ts .- init)/init*100).^2 |> sum |> x->x/length(ts) |> sqrt
end

function maxErr(init, ts)
    return (ts .- init)./init .|> abs |> x->max(x...)
end

function analysis(stateFilename; step=1, plotHeadless=false, redoC=false)
    if plotHeadless
    	ENV["GKSwstype"] = "nul"
    end

    data = loadData(stateFilename)
    analysisFilename = stateFilename * "-analysis.jld2"
    items = []
    if isfile(analysisFilename)
        items = load(analysisFilename) |> keys
    end
    results = Dict()

    paramFilename = stateFilename * "-param.jld2"
    params = load(paramFilename)["params"]
    # if glb, get parameters and encapsulate potential function
    if params["pot"] == "glb"
        # energy
        potFunc = (r, m1, m2, c1, c2)->V_glb(r, m1, m2, c1, c2, params["s"], params["d"], params["A"])
        println("Computing energy...")
        if !("energy" in items)
            es = analyze_energy(data["m"], data["c"], data["xs"], data["vs"], data["B"], data["B"], potFunc; step=step)
            results["energy"] = es
            results["energySMSE"] = SMSE(es[1][1], es[1][2:end])
        end
    else
        println("WARNING: Energy calculation of non-glb potential is not implemented.")
    end

    # clustering profile
    println("Computing cluster profile for positions...")
    if !("clustProf_size_v2" in items)
        profile = clusterProfile_DBscan_size_v2(data["xs"], data["dim"])
        results["clustProf_size_v2"] = profile
    end

    # # mutual information
    # println("Computing MI over positions...")
    # if !("MI_pos" in items)
    #     MI, U, posHists = MI_pos_disc(data["xs"], data["dim"]; binsize=2, print=false)
    #     results["MI_pos"] = MI
    #     results["U_pos"] = U
    #     results["hists_pos"] = posHists
    # end
    #
    # println("Computing MI over velocities...")
    # if !("MI_vel" in items)
    #     MI, U, velHists = MI_pos_disc(data["vs"], 2*2; print=false, binsize=.2)
    #     results["MI_vel"] = MI
    #     results["U_vel"] = U
    #     results["hists_vel"] = velHists
    # end
    #
    # T = Int(length(data["xs"]))
    # tscales = [1, T÷10, T ÷ 5]
    # xscales = range(0, data["dim"]/2, length=30) |> collect |> x->floor.(x) |> x->filter(y->y!=0, x)
    # println("Computing complexity profile for avg vel...")
    # if !("cprofile_avgVel" in items) || redoC
    #     C, D = cprofile(xs=data["xs"], vs=data["vs"], dim=data["dim"], xscales=xscales, tscales=tscales, vdim=1, vsens=0.01, cFunc=_cprofile_avgVel)
    #     results["cprofile_avgVel"] = D
    # end
    #
    # println("Computing complexity profile for avg density...")
    # if !("cprofile_avgDen" in items) || redoC
    #     C, D = cprofile(xs=data["xs"], vs=data["vs"], dim=data["dim"], xscales=xscales, tscales=tscales, vdim=1, vsens=0.01, cFunc=_cprofile_avgDen)
    #     results["cprofile_avgDen"] = D
    # end
    #
    # println("Computing complexity profile for clusters...")
    # if !("cprofile_clusters" in items) || redoC
    #     C,D = cprofile_noT(xs=data["xs"], xscales=xscales, cFunc=countClusters, skip=step)
    #     results["cprofile_clusters"] = D
    # end

    # results = Dict( "MI"=> MI, "pos hists"=>posHists, "energy"=>es )
    jldopen(analysisFilename, "a+") do f
        if redoC
            Base.delete!(f, "cprofile_avgVel")
            Base.delete!(f, "cprofile_avgDen")
        end
        for (name, result) ∈ results
            f[name] = result
        end
    end

    return results
end

function analyzeAgenda(agendaFilename; step=1, plotHeadless=false, redoC=false)
    outputBase = dirname(agendaFilename)
    data = read(agendaFilename, String) |> TOML.parse
    outputFilenames = data["settings"]["analysisConfs"] |> x->map(y->split(basename(y),".")[1]*"_", x)
    files = []
    @printf("processing %d files: \n", length(outputFilenames)*length(data["settings"]["potentials"]))
    for f in outputFilenames
        for pot in data["settings"]["potentials"]
            # for type in ["state", "param"]
            #     s = f * pot * "-" * type * ".jld2"
            #     push!(files, f)
            # end
            s = joinpath(outputBase, f * pot)
	    @printf("\t%s\n", s)
            push!(files, s)
        end
    end

    results = Dict()
    for f in files
        @printf("\tprocessing: %s\n", f)
        r = analysis(f; step=step, plotHeadless=plotHeadless, redoC=redoC)
        results[f] = r
    end
    return results
end

function analyzeAgendaGLB(agendaFilename; step=1, plotHeadless=false, redoC=false)
    outputBase = dirname(agendaFilename)
    data = read(agendaFilename, String) |> TOML.parse
    outputFilenames = data["settings"]["analysisConfs"] |> x->map(y->split(basename(y),".")[1]*"_", x)
    files = []
    @printf("processing %d files: \n", length(outputFilenames)*length(data["settings"]["potentials"]))
    for f in outputFilenames
        for pot in data["settings"]["potentials"]
            for s in data["settings"]["sValues"]
                for d in data["settings"]["dValues"]
                    temp = @sprintf("%s_%.02f_d_%.02f", pot, s, d)
                    str = joinpath(outputBase, f * temp)
            	    @printf("\t%s\n", str)
                    push!(files, str)
                end
            end
            # for type in ["state", "param"]
            #     s = f * pot * "-" * type * ".jld2"
            #     push!(files, f)
            # end
        end
    end

    results = Dict()
    N = length(files)
    for (i, f) in enumerate(files)
        @printf("\tprocessing: %s: (%d / %d)\n", f, i, N)
        r = analysis(f; step=step, plotHeadless=plotHeadless, redoC=redoC)
        results[f] = r
    end
    return results
end

function outputResults(dir)
    files = readdir(dir; join=true) |> x->filter(y->contains(y, "analysis.jld2"), x)
    display(files)
    D = DataFrame(potential = String[], dnn = Float64[], dim=Float64[], avgMI_pos=Float64[], avgU_pos = Float64[], avgMI_vel=Float64[], avgU_vel = Float64[], energy_smse=Float64[])
    CD_vel = DataFrame(potential = String[], dnn=Float64[], xscale=Float64[], tscale=Float64[], c=Float64[])
    CD_den = DataFrame(potential = String[], dnn=Float64[], xscale=Float64[], tscale=Float64[], c=Float64[])
    CD_clu = DataFrame(potential = String[], dnn=Float64[], xscale=Float64[], tscale=Float64[], c=Float64[])
    for f in files
        println(basename(f))
        name = f[1:findlast("-", f)[1]]
        f2 = name * "param.jld2"
        analysis = load(f)
        param = load(f2)["params"]

        # energy
        E = analysis["energy"][1]
        esmse = SMSE(E[1], E[2:end])

        # MI
        MI_pos = analysis["MI_pos"]
        avgMI_pos = sum(MI_pos) / binomial(size(MI_pos)[1], 2)
        U_pos = analysis["U_pos"]
        avgU_pos = sum(U_pos) / binomial(size(U_pos)[1], 2)

        MI_vel = analysis["MI_vel"]
        avgMI_vel = sum(MI_vel) / binomial(size(MI_vel)[1], 2)
        U_vel = analysis["U_vel"]
        avgU_vel = sum(U_vel) / binomial(size(U_vel)[1], 2)

        # MI histogram
        # reshape(MI, prod(size(MI))) |> x->filter(y->y!=0, x) |> x->histogram(x, xlim=[5,9])
        # title!(basename(name))
        # savefig(name * "MI_pos_hist.png")

        # complexity profile
        prof = analysis["cprofile_avgVel"]
        for prof_row in eachrow(prof)
            row = vcat(param["pot"], param["dnn"], prof_row...)
            push!(CD_vel, row)
        end
        prof = analysis["cprofile_avgDen"]
        for prof_row in eachrow(prof)
            row = vcat(param["pot"], param["dnn"], prof_row...)
            push!(CD_den, row)
        end
        prof = analysis["cprofile_clusters"]
        for prof_row in eachrow(prof)
            row = vcat(param["pot"], param["dnn"], prof_row...)
            push!(CD_clu, row)
        end


        @printf("\tpot = %s\tdnn = %.02f\tdim = %.02f\n", param["pot"], param["dnn"], param["dim"])
        @printf("\tenergy smse = %.04f\n", esmse)
        @printf("\tavg MI pos = %.04f\tavg MI vel = %.04f\n", avgMI_pos, avgMI_vel)

        row = vcat(param["pot"], param["dnn"], param["dim"], avgMI_pos, avgU_pos, avgMI_vel, avgU_vel, esmse)
        push!(D, row)
    end

    # plot MI
    data2 = combine(groupby(D, [:potential, :dnn]), :avgMI_pos=>mean, :avgMI_vel=>mean, :avgU_pos=>mean, :avgU_vel=>mean)

    p = Gadfly.plot(data2, x=:potential, y=:avgMI_pos_mean, shape=:dnn, Geom.point, Guide.Title("Mutual Information of Particle Positions"))
    saveplot(p, joinpath(dir,"avgMI_pos"))
    display(p)

    p = Gadfly.plot(data2, x=:potential, y=:avgMI_vel_mean, shape=:dnn, Geom.point, Guide.Title("Mutual Information of Particle Velocities"))
    saveplot(p, joinpath(dir,"avgMI_vel"))
    display(p)

    p = Gadfly.plot(data2, x=:potential, y=:avgU_pos_mean, shape=:dnn, Geom.point, Guide.Title("Normalized MI of Particle Position"))
    saveplot(p, joinpath(dir,"avgU_pos"))
    display(p)

    p = Gadfly.plot(data2, x=:potential, y=:avgU_vel_mean, shape=:dnn, Geom.point, Guide.Title("Normalized MI of Particle Velocity"))
    saveplot(p, joinpath(dir,"avgU_vel"))
    display(p)

    #plot complexity profile
    cDatVel = combine(groupby(CD_vel, [:potential, :dnn, :xscale, :tscale]), :c=>mean, :c=>std)
    cDatVel[:, :ymin] = cDatVel[:, :c_mean] - cDatVel[:, :c_std]
    cDatVel[:, :ymax] = cDatVel[:, :c_mean] + cDatVel[:, :c_std]
    p = Gadfly.plot(cDatVel[cDatVel.dnn.==2,:], x=:xscale, y=:c_mean, color=:potential, ymin=:ymin, ymax=:ymax, Geom.subplot_grid(Geom.line, Geom.ribbon), xgroup=:tscale)
    saveplot(p, joinpath(dir,"cprofile_vel"))
    display(p)

    cDatDen = combine(groupby(CD_den, [:potential, :dnn, :xscale, :tscale]), :c=>mean, :c=>std)
    cDatDen[:, :ymin] = cDatDen[:, :c_mean] - cDatDen[:, :c_std]
    cDatDen[:, :ymax] = cDatDen[:, :c_mean] + cDatDen[:, :c_std]
    p = Gadfly.plot(cDatDen[cDatDen.dnn.==2,:], x=:xscale, y=:c_mean, color=:potential, ymin=:ymin, ymax=:ymax, Geom.subplot_grid(Geom.line, Geom.ribbon), xgroup=:tscale)
    saveplot(p, joinpath(dir,"cprofile_den"))
    display(p)

    cDatClu = combine(groupby(CD_clu, [:potential, :dnn, :xscale, :tscale]), :c=>mean, :c=>std)
    display(cDatClu)
    cDatClu[:, :ymin] = cDatClu[:, :c_mean] - cDatClu[:, :c_std]
    cDatClu[:, :ymax] = cDatClu[:, :c_mean] + cDatClu[:, :c_std]
    dnns = cDatClu[!,:dnn] |> unique
    for (i, x) in enumerate(groupby(cDatClu, [:dnn]))
        p = Gadfly.plot(x, x=:xscale, y=:c_mean, color=:potential, ymin=:ymin, ymax=:ymax, Geom.subplot_grid(Geom.line, Geom.ribbon), xgroup=:tscale, Scale.x_log10, Scale.y_log10)
        saveplot(p, joinpath(dir,@sprintf("cprofile_clu-%d", Int(floor(dnns[i]*10)))))
        display(p)
    end

    # plot energy error
    # WARNING: plotting across all dnn's ==> different distributions
    p = Gadfly.plot(D, x=:potential, y=:energy_smse, Geom.boxplot)
    saveplot(p, joinpath(dir,"energyErr"))
    display(p)

    return D, CD_den, CD_vel, CD_clu
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

function cps(profile)
    return sum([x > 1 ? log2(x)/i : 0 for (i, x) in enumerate(profile)])
end

function clusterProfOutput(dir)
    files = readdir(dir; join=true) |> x->filter(y->contains(y, "analysis.jld2"), x)
    display(files)
    D = DataFrame(potential = String[], dnn = Float64[], dim=Float64[], s=Float64[], d=Float64[], energySMSE=Float64[], profile=[])
    for f in files
        println(basename(f))
        name = f[1:findlast("-", f)[1]]
        f2 = name * "param.jld2"
        analysis = load(f)
        param = load(f2)["params"]

    	if !("energySMSE" in keys(analysis))
    		@printf("not found in %s", f)
    	end

        row = vcat(param["pot"], param["dnn"], param["dim"], param["s"], param["d"], analysis["energySMSE"], [analysis["clustProf_size_v2"]])
        print(row)
        push!(D, row)
    end
    return D
end

function produceCSV_1(D, filename)
    x = groupby(D, [:s, :d]) |> x->combine(x, :profile => avgProfile)
    CSV.write(filename, x)
end

function saveplot(p, name)
    draw(PNG(name*".png", 8inch, 6inch, dpi=200), p)
end
