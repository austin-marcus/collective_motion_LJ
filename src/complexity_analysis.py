import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import itertools
from matplotlib import colors
import matplotlib.image as mpimg
from sklearn import linear_model
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.lines as mlines

def cps(profile, norm=False):
    C = sum( [np.log2(x)/(i+1)/np.log(2) if x > 1 else 0 for i, x in enumerate(profile)] ) # in paper 10/3/22
    # C = sum( [np.log2(x)/np.log(2) if x > 1 else 0 for i, x in enumerate(profile)] ) # works better 3/7/23
    # C = sum( [np.log2(x+1)/(i+1)/np.log(2) for i, x in enumerate(profile)] )
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

def getColorForAvg(L):
    ex = {"complex": "green", "coherent": "red", "random": "blue", "powerlaw": "green"}
    if L.source == "experiment":
        return ex[L.regime]
    else:
        return ex[L.type]

color = plt.cm.plasma
def complexityProfile(data, exclude=1, fn="test.pdf", minmax=(None, None), lw=3, diff=False, avg=False):
    # M = max(data["cps"])
    M = 1
    # data["cps"] = data["cps"].apply(lambda x:(x/M))
    # data["sum"] = data["profile"].apply(lambda x:sum(x))
    data = data.sort_values("cps")
    # print(data)

    fig, ax = plt.subplots(figsize=(6,6))
    # norm = plt.Normalize(vmin=min(data["cps"]), vmax=max(data["cps"]))
    norm = plt.Normalize(vmin=minmax[0], vmax=minmax[1])
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
                 linewidth=lw, 
                 color=color(norm(L.cps)) if not avg else getColorForAvg(L),
                 linestyle="-" if L.source=="experiment" else "--"
                 # alpha=0.2
                )
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if diff:
        ax.set_ylabel(r"$\log_2 \left ( \frac{d}{d\sigma}L(x) \right )$", fontsize=22, labelpad=15)
    else:
        ax.set_ylabel(r"$I(\sigma)$", fontsize=22, rotation=0, labelpad=30)

    if not avg:
        sm = plt.cm.ScalarMappable(cmap=color, norm=norm)
        cbar = plt.gcf().colorbar(sm, shrink=0.75, ax=ax)
        cbar.set_label(r"$C$", fontsize=22, rotation=0, labelpad=20)
        cbar.ax.tick_params(labelsize=14) 
    else:
        complex = mlines.Line2D([], [], color='green', marker='s', ls='', label='Complex')
        random = mlines.Line2D([], [], color='blue', marker='s', ls='', label='Homogeneous')
        coherent = mlines.Line2D([], [], color='red', marker='s', ls='', label='Coherent')
        bench = mlines.Line2D([], [], color='black', marker='', ls='--', label='Benchmark')
        sim = mlines.Line2D([], [], color='black', marker='', ls='-', label='Simulation')

        # etc etc
        plt.legend(handles=[complex, random, coherent, bench, sim])

    ax.set_xlabel(r"$\sigma$", fontsize=22)
    plt.gcf().set_size_inches(6,6)
    fig.set_dpi(300)
    # plt.xlim(0, 24)
    plt.tight_layout()
    plt.savefig(fn)
    # plt.show()

def f_powerlaw(x, g, c, c0):
    return c0 + c * x**g

def truncate(l):
    for i, n in enumerate(l[1:], 1):
        if n < 1:
            print(l[:i+1])
            return l[:i+1]
    return l

def fitPowerlaws(data):
    data["powerlaw"] = data["profile"].apply(lambda prof: curve_fit(f_powerlaw, list(map(float, range(1,len(prof)+1))), prof, p0=[-1.0,1,0]))

def heatmap(data, minmax=(None, None), ax=None):
    ss = list(map(str, [0.01, .08, 0.15, .23, 0.3, .38, 0.45, .53, 0.6, .68, .75]))
    ds = list(map(str, [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.01, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        fig.set_dpi(300)
        standalone = True
    else:
        standalone = False
        
    pv = data.groupby(["s","d"]).mean().reset_index().pivot(index="d", columns="s", values="cps")
    norm = plt.Normalize(vmin=minmax[0], vmax=minmax[1])

    if standalone:
        pos = plt.pcolormesh(ss, ds, pv, cmap=plt.cm.plasma, shading="auto")
        ax.set_xlabel("s", fontsize=22)
        plt.xticks([e for i, e in enumerate(ss) if i%2==0], fontsize=14)
        ax.set_ylabel("d", fontsize=22, rotation="horizontal", labelpad=20)
        plt.yticks([e for i, e in enumerate(ds) if i%2==0], fontsize=14)
    else:
        pos = ax.pcolormesh(ss, ds, pv, cmap=plt.cm.plasma, shading="auto")
        ax.set_xlabel("$s$")
        ax.set_xticks([e for i, e in enumerate(ss) if i%2==0])
        ax.set_ylabel("$d$", rotation="horizontal")
        ax.set_yticks([e for i, e in enumerate(ds) if i%2==0])

    sm = plt.cm.ScalarMappable(cmap=color, norm=norm)
    cbar = plt.gcf().colorbar(sm, shrink=0.75, ax=ax)
    # cbar = fig.colorbar(pos, shrink=0.75)
    if standalone:
        cbar.set_label(r"$C$", fontsize=22, rotation=0, labelpad=20)
        cbar.ax.tick_params(labelsize=14) 
        plt.tight_layout()
        plt.savefig("heatmap.pdf")
    else:
        cbar.set_label(r"$C$", rotation=0, labelpad=5, fontsize=14)
        cbar.ax.tick_params(labelsize=12) 
    # plt.show()

# plot complexity as function of ratio of s and d
def c_rel(data):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    fig.set_dpi(300)
    data["s/d2"] = data["s"] / data["d"]
    meanPos = data[data["d"]>0].nlargest(10, "cps")["s/d2"].mean()
    meanNeg = data[data["d"]<0].nlargest(10, "cps")["s/d2"].mean()
    pos = ax.scatter(data["s/d2"], data["cps"], alpha=0.3, color="black")
    plt.xlim([-2,2])
    plt.xlabel(r"$s/d$", fontsize=22, labelpad=5)
    plt.ylabel(r"$C$", fontsize=22, rotation=0, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axvline(meanPos, linestyle="--", color="black", alpha=0.5)
    plt.axvline(meanNeg, linestyle="--", color="black", alpha=0.5)
    plt.tight_layout()
    plt.savefig("c_rel.pdf")

    sns.relplot(data, x="s", y="cps", hue="d", errorbar="ci", kind="line")
    plt.savefig("c_v_s.pdf")
    sns.relplot(data, x="d", y="cps", hue="s", errorbar="ci", kind="line")
    plt.savefig("c_v_d.pdf")
    # plt.show()

@mpl.rc_context({
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.markersize": 2,
    "axes.labelpad": 10,
})
def c_rel2(data):
    # fig, axes = plt.subplots(2,2, figsize=(12,12))
    # fig.set_dpi(300)

    data["s/d2"] = data["s"] / data["d"]
    meanPos = data[data["d"]>0].nlargest(10, "cps")["s/d2"].mean()
    meanNeg = data[data["d"]<0].nlargest(10, "cps")["s/d2"].mean()
    print(meanPos, meanNeg)

    sns.set_context("paper")
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    sns.scatterplot(data, x="s/d2", y="cps", ax=ax, alpha=0.3)
    ax.set_xlim(-2,2)
    ax.set_xlabel("$s/d$")
    ax.set_ylabel("$C$", rotation="horizontal")
    ax.vlines(meanPos, data["cps"].min(), data["cps"].max(), linestyles="dashed", linewidth=3, color="black")
    ax.vlines(meanNeg, data["cps"].min(), data["cps"].max(), linestyles="dashed", linewidth=3, color="black")
    plt.tight_layout()
    plt.savefig("ratio.pdf")
    fig.set_dpi(300)

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    sns.lineplot(data, x="s", y="cps", ax=ax, errorbar="sd")
    ax.set_xlabel("$s$")
    ax.set_ylabel("$C$", rotation="horizontal")
    ax.set_ylim(13, 21)
    plt.tight_layout()
    plt.savefig("s_v_C.pdf")
    fig.set_dpi(300)

    fig, ax = plt.subplots(1,1, figsize=(4,4))
    sns.lineplot(data, x="d", y="cps", ax=ax, errorbar="sd")
    ax.set_xlabel("$d$")
    ax.set_ylabel("$C$", rotation="horizontal")
    ax.set_ylim(13, 21)
    plt.tight_layout()
    plt.savefig("d_v_C.pdf")
    fig.set_dpi(300)

    # fig, ax = plt.subplots(1,1, figsize=(8,8))
    # heatmap(data, (data.cps.min(), data.cps.max()), ax=ax)
    # plt.tight_layout()
    # plt.savefig("heatmap.pdf")
    # fig.set_dpi(300)

    # plt.show()
    

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

# using: https://en.wikipedia.org/wiki/Welch's_t-test
@mpl.rc_context({
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})
def sigTest(data, groupVar, plot=False):
    groups = [(name, group) for name, group in data.groupby(groupVar)]
    results = {}
    for g1, g2 in itertools.combinations(groups, 2):
        data1 = list(g1[1]["cps"])
        data2 = list(g2[1]["cps"])
        print(data1)
        results[frozenset([g1[0], g2[0]])] = stats.ttest_ind(data1, data2, equal_var=False)
    for name, group in groups:
        results[name] = (group["cps"].mean(), group["cps"].std())

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        fig.set_dpi(300)
        sns.set_context("talk")
        sns.histplot(data, x="cps", hue="regime", palette=["red", "blue", "green"], ax=ax)
        sns.move_legend(ax, "upper left", labels=["Coherent", "Homogeneous", "Complex"][::-1], title=None)
        # sns.move_legend(ax, "upper left", title=None)
        ax.set_xlabel("$C$")
        # plt.margins(0.2)
        plt.tight_layout()
        # plt.legend(labels=["Coherent", "Homogeneous", "Complex"])
        plt.savefig("regime_hist.pdf")

    return results

def loadData():
    data = pd.read_csv("../runs/full_run/0622_full_analysis.csv", converters={"Float64profile":ast.literal_eval})
    bench = pd.read_csv("set_56.00_g_1.10_m_25_mult_1.95_20.csv", converters={"profile":ast.literal_eval, "ccs":ast.literal_eval})
    data = data.rename(columns={"Float64profile": "profile"})
    data["regime"] = data.apply(classifyRegime, axis=1)
    data["cps"] = data["profile"].apply(cps)
    data["source"] = "experiment"
    bench["source"] = "bench"
    bench["cps"] = bench["profile"].apply(cps)

    return data, bench

def dataToAvg(data, group):
    dataAvg = data.groupby(group, as_index=False)["profile"].agg(avgProfile)
    dataAvg["cps"] = dataAvg["profile"].apply(cps)
    return dataAvg
    
def doProfile():
    data, bench = loadData()
    # # exact profiles
    data1 = pd.concat([data, bench])
    data1.to_csv("cps.csv")
    # print(data1)
    #     # fig, ax = plt.subplots()
    # # plt.yscale("log"); plt.xscale("log")
    # # for row in data1.iterrows():
    # #     ax.plot(np.diff(row[-1].profile)*-1)
    # ex = 24
    # complexityProfile(data1, exclude=ex, fn=f"profile_exact_exclude_{ex}.pdf")

    plt.rcParams["font.family"] = "Times New Roman"
    # average profiles
    dataAvg = data.groupby("regime", as_index=False)["profile"].agg(avgProfile)
    benchAvg = bench.groupby("type", as_index=False)["profile"].agg(avgProfile)
    dataAvg["source"] = "experiment"
    benchAvg["source"] = "bench"
    dataAvg["cps"] = dataAvg["profile"].apply(cps)
    benchAvg["cps"] = benchAvg["profile"].apply(cps)
    dataAvg = pd.concat([dataAvg, benchAvg])
    data["cps"] = data["profile"].apply(cps)
    print(dataAvg)
    minmax = (data.cps.min(), data.cps.max())
    complexityProfile(dataAvg, fn="avg.pdf", minmax=minmax, avg=True)
    complexityProfile(dataAvg, fn="avg2.pdf", minmax=minmax, avg=False)
    complexityProfile(data, exclude=20, fn="raw_profile.pdf", minmax=minmax, lw=1)
    dataAvg["profile"] = dataAvg["profile"].apply(lambda x: -np.diff(x))
    print(dataAvg)
    complexityProfile(dataAvg, fn="avg_diff.pdf", minmax=minmax, diff=True)
    heatmap(data, minmax=minmax)
    c_rel2(data)

def particlePlots():
    files = [
        "coherent.png",
        "complex_56.00_g_1.10_m_25_mult_1.95_1.png",
        "random.png"
    ]
    names = ["Coherent", "Complex", "Homogeneous"]
    fig, axes = plt.subplots(1, 3, figsize=(6,3))
    for i in range(3):
        axes[i].imshow(mpimg.imread(files[i]))
        axes[i].set_title(names[i], fontsize=26)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.tight_layout()
    fig.set_dpi(100)
    # plt.show()
    fig.savefig("benchmark_plots.pdf")

# give a copy of data; modifies data
def cRegression(data):
    # normalize s and d
    data["s"] = data["s"] / data["s"].max()
    data["d"] = (data["d"] - data["d"].min()) 
    data["d"] = data["d"] / data["d"].max()
    data["d"] = abs(data["d"])
    print(set(data["s"]))
    print(set(data["d"]))

    # X = data.loc[:, ["s", "d"]]
    # y = data["cps"]
    # reg = linear_model.LinearRegression().fit(X, y)

    y = data["cps"]
    XS = np.array(data.loc[:, "s"]).reshape(-1,1)
    XD = np.array(data.loc[:, "d"]).reshape(-1,1)
    regS = linear_model.LinearRegression().fit(XS,y)
    regD = linear_model.LinearRegression().fit(XD,y)

    # fit = np.polyfit(X, y, 2)
    # print(fit)
    
    # return data, (reg.score(X,y), reg.coef_, reg.intercept_)
    return data, (regS.score(XS,y), regS.coef_), (regD.score(XD,y), regD.coef_)

if __name__ == "__main__":

    doProfile() # profiles and heatmap
    # plt.show()


    # statistics
    data, bench = loadData()
    # maxC = max(data.cps.max(), bench.cps.max())
    # data["cps"] = data["cps"] / maxC
    # bench["cps"] = bench["cps"] / maxC
    x = sigTest(data, "regime", plot=True)
    y = sigTest(bench, "type")

    reg = cRegression(data.copy())

    particlePlots()

    data["profile"] = abs(data["profile"].apply(np.diff))
    bench["profile"] = abs(bench["profile"].apply(np.diff))
    dataAvg = pd.concat([dataToAvg(data, "regime"), dataToAvg(bench, "type")])
    fitPowerlaws(dataAvg)

