include("physics.jl")
using CairoMakie
using Plots
using Printf
using CSV
using DataFrames
using LaTeXStrings

f(x; s=0.5, d=1, A=10) = V_glb(x, 1, 1, 1, 1, s, d, A, rThresh=1)
r = 0:0.01:5

# p1 = Plots.plot()
# ys = (-9, 5)
# s = 0.5
# for d = [-4, -1, 0.01, 1, 4]
#     Plots.plot!(p1, r, x->f(x; s=s, d=d), ylim=ys, label=nothing, c=cgrad(:balance), clims=(-5,5), line_z=d, w=4, colorbar_title="d")
# end
# title!(@sprintf("s = %.02f", s))

# p2 = Plots.plot()
# d = 4
# for s = [0.01, 0.15, 0.30, 0.45, 0.6, 0.75]
#     Plots.plot!(p2, r, x->f(x; s=s, d=d), ylim=ys, label=nothing, c=cgrad(:amp), clims=(0, 0.8), line_z=s, w=4, colorbar_title="s")
# end
# title!(@sprintf("d = %.02f", d))

# sizeInches = (7.5, 3)
# dpi = 300
# p = Plots.plot(p1, p2; layout=(1,2), xguide="Distance", yguide="Potential Energy", fontfamily="Times New Roman", guidefontsize=18, tickfontsize=12, colorbar_titlefontsize=18, colorbar_tickfontsize=12, dpi=dpi)
# Plots.plot!(size=sizeInches .* dpi, margin=10Plots.mm)
# println(p)
# display(p)

# savefig(p, "potential.png")

# data = CSV.read("./complexity_profile.csv", DataFrame)
# p1 = Plots.plot()
# title!("d < 0")
# d = 2
# cs = :rainbow
# for (i, row) in enumerate(eachrow(data))
#     if (row.d != -d)
#         continue
#     end
#     Plots.plot!(p1, r, x->f(x; s=row.s, d=row.d), ylim=(-4,0), label=nothing, w=2, c=cgrad(cs), line_z=row.cps, clims=(0.6, 1),colorbar_title="C", colorbar_titlefontrotation=90)
# end
# p2 = Plots.plot()
# title!("d > 0")
# for (i, row) in enumerate(eachrow(data))
#     if (row.d != d)
#         continue
#     end
#     Plots.plot!(p2, r, x->f(x; s=row.s, d=row.d), ylim=(-2,1), label=nothing, w=2, c=cgrad(cs), line_z=row.cps, clims=(0.6, 1), colorbar_title="C")
# end
# p = Plots.plot(p1, p2; layout=(1,2), xguide="Distance", yguide="Potential Energy", fontfamily="Times New Roman", guidefontsize=18, tickfontsize=12, dpi=300, size=(8,4).*300, margin=10Plots.mm,
#                 colorbar_titlefontsize=18, colorbar_tickfontsize=12)
# display(p)
# savefig(p, "max_complexity.png")


# # FIGURE
# dataMax = combine(sdf -> sdf[argmax(sdf.cps), :], groupby(data, :d)) |> x->sort(x,:d)
# dataMin = combine(sdf -> sdf[argmin(sdf.cps), :], groupby(data, :d)) |> x->sort(x,:d)
# cdata = vcat(dataMin, dataMax)
# p1 = Plots.plot()
# cs = :rainbow
# clims = (minimum(data.cps), maximum(data.cps))
# width = 3
# title!("d > 0")
# for row in eachrow(cdata)
#     if (row.d < 0)
#         continue
#     end
#     Plots.plot!(p1, r, x->f(x; s=row.s, d=row.d), ylim=(-4,0), label=nothing, w=width, c=cgrad(cs), line_z=row.cps, clims=clims,colorbar_title="C", colorbar_titlefontrotation=90)
# end
# p2 = Plots.plot()
# title!("d < 0")
# for row in eachrow(cdata)
#     if (row.d > 0)
#         continue
#     end
#     Plots.plot!(p2, r, x->f(x; s=row.s, d=row.d), ylim=(-4,0), label=nothing, w=width, c=cgrad(cs), line_z=row.cps, clims=clims,colorbar_title="C", colorbar_titlefontrotation=90)
# end
# p = Plots.plot(p1, p2; layout=(1,2), xguide="Distance", yguide="Potential Energy", fontfamily="Times New Roman", titlefontsize=22, guidefontsize=22, tickfontsize=22, dpi=300, size=(8,4).*300, margin=12Plots.mm,
#                 colorbar_titlefontsize=22, colorbar_tickfontsize=18)
# # display(p)
# savefig(p, "max_complexity2.png")

# companion figure for visualization of regimes
params = [
    (0.08, 2.5),
    (0.525, 2.5),
    (0.375, 0.01),
    (0.08, -2.5),
    (0.525, -2.5)
]
colors = ["red", "green", "blue", "red", "green"]
names = ["Coherent", "Complex", "Homogeneous", "Coherent", "Complex"]
p = Plots.plot()
for (i, set) in enumerate(params)
    c = colors[i]
    name = names[i] * " " * (set[2]<0 ? L"d<0" : L"d>0")
    style = set[2] < 0 ? :dash : :solid
    Plots.plot!(p, r, x->f(x; s=set[1], d=set[2]), ylim=(-4,4), xlim=(0, 3), label=name, c=c, w=3, linestyle=style, dpi=300)
end
Plots.plot!(p, size=(1,1).* 300)
savefig(p, "companion_potentials.png")

# p = Plots.plot()
# width=3
# c = 0.2
# for d = 0:0.5:4
#     s = c*d
#     Plots.plot!(p, r, x->f(x; s=s, d=d), ylim=(-4,0), label=nothing, w=width)
# end
# display(p)
# savefig(p, "same_ratio.png")
