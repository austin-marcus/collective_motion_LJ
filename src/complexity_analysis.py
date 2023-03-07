import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt

def cps(profile, norm=False):
    # C = sum( [np.log2(x)/(i+1)/np.log(2) if x > 1 else 0 for i, x in enumerate(profile)] ) # in paper 10/3/22
    C = sum( [np.log2(x)/np.log(2) if x > 1 else 0 for i, x in enumerate(profile)] ) # in paper 10/3/22
    # C = sum( [np.log2(x+1)/(i+1)/np.log(2) for i, x in enumerate(profile)] ) # in paper 10/3/22
    if norm:
        return C / countParticles(profile)
    else:
        return C

def cps2(profile):
    return sum( [ (np.log2(x)+1)/(i+1) if x > 0 else 0 for i, x in enumerate(profile)] )

def countParticles(clusterProfile):
    L = clusterProfile.copy()
    L.append(0)
    return(sum(scale*change*-1 for scale, change in zip(range(1, len(L)+1), np.diff(L))))

def particleProfileToCluster(L):
    L = L.copy()
    L = [x/i for i, x in enumerate(L, 1)]
    for i in range(len(L)-2, -1, -1):
        L[i] += L[i+1]
    return L

def clusterProfileToParticle(L):
    return [i*x for i, x in enumerate(L, 1)]

def complexityProfile(data, exclude=1, fn="test.png"):
    # M = max(data["cps"])
    M = 1
    data["cps"] = data["cps"].apply(lambda x:(x/M))
    data["sum"] = data["profile"].apply(lambda x:sum(x))
    data = data.sort_values("cps")
    print(data)

    fig, ax = plt.subplots(figsize=(6,6))
    color = plt.cm.plasma
    norm = plt.Normalize(vmin=min(data["cps"]), vmax=max(data["cps"]))
    for I, i in enumerate(data.iterrows()):
        if I % exclude != 0 and i[-1].source == "experiment":
            continue
        L = i[-1]
    #     print(L.cps)
    #     start = 1; end = 98
        start, end = 1, len(L.profile)
        plt.plot(
                 np.log2(range(start, end)), 
                 # range(start, end), 
                 [np.log2(x) if x>1 else 0 for x in L.profile[start-1:end-1]], 
                 linewidth=1, 
                 color=color(norm(L.cps)),
                 linestyle="-" if L.source=="experiment" else "--"
                 # alpha=0.2
                )
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_xlabel(r"$\sigma$", fontsize=22)
    ax.set_ylabel(r"$I(\sigma)$", fontsize=22, rotation=0, labelpad=30)
    sm = plt.cm.ScalarMappable(cmap=color, norm=norm)
    cbar = plt.gcf().colorbar(sm, shrink=0.75, ax=ax)
    cbar.set_label(r"$C$", fontsize=22, rotation=0, labelpad=20)
    cbar.ax.tick_params(labelsize=14) 
    plt.gcf().set_size_inches(6,6)
    fig.set_dpi(300)
    plt.savefig(fn)
    plt.show()

# D = pd.read_csv("../runs/full_run/0622_full_analysis.csv", converters={"Float64profile":ast.literal_eval})

def avgProfile(profiles):
    profiles = list(profiles)
    avg = []
    for i in range(len(profiles[0])):
        avg.append(np.mean([p[i] for p in profiles]))
    # print(avg)
    return avg

def classifyRegime(row):
    if row.s < 0.3 and row.d != 0.01: # coherent
        return "coherent"
    elif row.s >= 0.30 and row.d != 0.01: # complex
        return "complex"
    elif row.d == 0.01: # random
        return "random"

if __name__ == "__main__":
    data = pd.read_csv("../runs/full_run/0622_full_analysis.csv", converters={"Float64profile":ast.literal_eval})
    bench = pd.read_csv("set_56.00_g_1.10_m_25_mult_1.95_20.csv", converters={"profile":ast.literal_eval, "ccs":ast.literal_eval})
    data = data.rename(columns={"Float64profile": "profile"})
    data["cps"] = data["profile"].apply(cps)
    data["source"] = "experiment"
    bench["source"] = "bench"
    bench["cps"] = bench["profile"].apply(cps)

    # # exact profiles
    # data1 = pd.concat([data, bench])
    # print(data1)
    #     # fig, ax = plt.subplots()
    # # plt.yscale("log"); plt.xscale("log")
    # # for row in data1.iterrows():
    # #     ax.plot(np.diff(row[-1].profile)*-1)
    # ex = 24
    # complexityProfile(data1, exclude=ex, fn=f"profile_exact_exclude_{ex}.png")

    # average profiles
    data["regime"] = data.apply(classifyRegime, axis=1)
    dataAvg = data.groupby("regime", as_index=False)["profile"].agg(avgProfile)
    benchAvg = bench.groupby("type", as_index=False)["profile"].agg(avgProfile)
    dataAvg["source"] = "experiment"
    benchAvg["source"] = "bench"
    dataAvg["cps"] = dataAvg["profile"].apply(cps)
    benchAvg["cps"] = benchAvg["profile"].apply(cps)
    dataAvg = pd.concat([dataAvg, benchAvg])
    dataAvg["cps"] = dataAvg["cps"] / dataAvg["cps"].max()
    print(dataAvg)
    complexityProfile(dataAvg, fn="avg.png")
