include("physics.jl")
using CairoMakie
using Plots
using Printf
using CSV
using DataFrames

f(x; s=0.5, d=1, A=10) = V_glb(x, 1, 1, 1, 1, s, d, A, rThresh=1)
# f(x; s=0.5, d=1) = F_glb(x, 1, 1, 1, 1; s=s, d=d, A=10, rThresh=1) |> norm

# V = 1/sqrt(2)
# KE_avg = V^2 / 6
# r = 0:0.01:5
# # for d = (-2*KE_avg):0.1:(2*KE_avg)
# for d = [-4, -2, -1, 1, 2, 4]
#     println(d)
#     p = Plots.plot()
#     for s = [.01, .25, .5, .75, .99]
#         Plots.plot!(p, r, x->f(x; s=s, d=d), ylim=(-10,20), label=@sprintf("%.02f, %.02f", s, d))
#     end
#     display(p)
# end
# # lines(0:.01:10, f, axis=(yscale=log10))

r = 0:0.01:5
p1 = Plots.plot()
ys = (-9, 5)
s = 0.5
for d = [-4, -1, 0.01, 1, 4]
    Plots.plot!(p1, r, x->f(x; s=s, d=d), ylim=ys, label=nothing, c=cgrad(:balance), clims=(-5,5), line_z=d, w=4, colorbar_title="d")
    # Plots.plot!(p, r, x->f(x; s=s, d=d), ylim=(-10,20), label)
end
title!(@sprintf("s = %.02f", s))

p2 = Plots.plot()
d = 4
for s = [0.01, 0.15, 0.30, 0.45, 0.6, 0.75]
    Plots.plot!(p2, r, x->f(x; s=s, d=d), ylim=ys, label=nothing, c=cgrad(:amp), clims=(0, 0.8), line_z=s, w=4, colorbar_title="s")
end
title!(@sprintf("d = %.02f", d))

sizeInches = (7.5, 3)
dpi = 300
p = Plots.plot(p1, p2; layout=(1,2), xguide="Distance", yguide="Potential Energy", fontfamily="Times New Roman", guidefontsize=18, tickfontsize=12, colorbar_titlefontsize=18, colorbar_tickfontsize=12, dpi=dpi)
Plots.plot!(size=sizeInches .* dpi, margin=10Plots.mm)
display(p)

savefig(p, "potential.png")
# draw(PNG("potential.png", 12inch, 6inch, dpi=300), p)

params = [
    0.45 2
    0.525 -3.5
    .525 -4
    .6 -3.5
    .45 -2
    .525 -3
    .45 -3
    .525 -3
    .45 -3
    .375 1.5
    .15 3.5
]
data = CSV.read("./complexity_profile.csv", DataFrame)
p1 = Plots.plot()
title!("d < 0")
for (i, row) in enumerate(eachrow(data))
    if (row.d > 0)
        continue
    end
    Plots.plot!(p1, r, x->f(x; s=row.s, d=row.d), ylim=ys, label=nothing, w=1, c=cgrad(:amp), line_z=row.cps, clims=(0.6, 1),colorbar_title="C", colorbar_titlefontrotation=90)
end
p2 = Plots.plot()
title!("d > 0")
for (i, row) in enumerate(eachrow(data))
    if (row.d < 0)
        continue
    end
    Plots.plot!(p2, r, x->f(x; s=row.s, d=row.d), ylim=ys, label=nothing, w=1, c=cgrad(:amp), line_z=row.cps, clims=(0.6, 1), colorbar_title="C")
end
p = Plots.plot(p1, p2; layout=(1,2), xguide="Distance", yguide="Potential Energy", fontfamily="Times New Roman", guidefontsize=18, tickfontsize=12, dpi=300, size=(4,4).*300, margin=10Plots.mm,
                colorbar_titlefontsize=18, colorbar_tickfontsize=12)
# Plots.plot!(p1, r, x->f(x; s=0.45, d=2), ylim=ys, label="molecule", w=4)
# Plots.plot!(p1, r, x->f(x; s=0.525, d=-3.5), ylim=ys, label="gravity", w=4)
display(p)
savefig(p, "max_complexity.png")

# display(p1)
# display(p2)

# r = 0.5:0.01:2.5
# ys = (-10, 10)
# for (s, d, ys, name) in [ (0.2, -4, (-10, 0), "gra"), (0.45, 2, (-5, 10), "mol"), (0.75, 0.01, (-5, 10), "bil") ]
#     p = Plots.plot(r, x->f(x; s=s, d=d), ylim=ys, label=nothing, w=5)
#     Plots.hline!(p, [0], label=nothing, w=3, linestyle=:dash)
#     savefig(p, @sprintf("%s.png", name))
#     display(p)
# end
