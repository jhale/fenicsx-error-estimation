import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpltools import annotation

colors = [cm.get_cmap('tab20')(i / 20.) for i in range(20)]
darkblue = colors[0]
lightblue = colors[1]
darkorange = colors[2]
lightorange = colors[3]

plt.style.use("../plots.mplstyle")

def marker(x_data, y_datas, position, gap):
    middle = np.floor(len(x_data)/2.).astype(np.int32)
    anchor_1_1 = []
    anchor_2_1 = []

    for data in y_datas:
        anchor_1_1.append(data[middle])
        anchor_2_1.append(data[-1])
    anchor_1_1 = min(anchor_1_1)
    anchor_2_1 = min(anchor_2_1)

    anchor_1_0 = x_data[middle]
    anchor_2_0 = x_data[-1]

    anchor_1 = [anchor_1_0, anchor_1_1]
    anchor_2 = [anchor_2_0, anchor_2_1]
    marker_x = anchor_1[0]**position*anchor_2[0]**(1.-position)\
        * (anchor_2[1]/anchor_1[1])**gap
    marker_y = anchor_1[1]**position*anchor_2[1]**(1.-position)\
        * (anchor_1[0]/anchor_2[0])**gap
    return marker_x, marker_y

def slopes(xs, ys):
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T

    m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
    m = np.round(m,2)
    return m

cm.colors

slopes_results = {"method": [], "s": [], "rational adaptive": [], "total number solves": [], "total number dof": [], "slope error": [], "slope estimator": [], "efficiency": []}
for method in ["bp", "bura"]:
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        plt.figure()
        df = pd.read_csv(f"./results_{method}_{str(s)[-1]}.csv")
        df_ra_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_rational_adaptive.csv")

        xs = df["dof num"].values
        ys = df["total estimator"].values
        ys_err = df["total error"].values
        xs_ra_adaptive = df_ra_adaptive["dof num"].values
        ys_ra_adaptive = df_ra_adaptive["total estimator"].values
        ys_ra_adaptive_err = df_ra_adaptive["total error"].values
        slope = slopes(xs, ys)
        slope_err = slopes(xs, ys_err)
        slope_ra_adaptive = slopes(xs_ra_adaptive, ys_ra_adaptive)
        slope_ra_adaptive_err = slopes(xs_ra_adaptive, ys_ra_adaptive_err)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope estimator"].append(slope)
        slopes_results["slope error"].append(slope_err)

        total_number_solves = sum(df["num solves"].values)
        total_number_dof = sum(np.multiply(df["num solves"].values, df["dof num"].values))
        slopes_results["total number solves"].append(total_number_solves)
        slopes_results["total number dof"].append(total_number_dof)

        efficiency = np.mean(df["total estimator"].values[-5:]/df["total error"].values[-5:])
        slopes_results["efficiency"].append(efficiency)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["rational adaptive"].append(True)
        slopes_results["slope estimator"].append(slope_ra_adaptive)
        slopes_results["slope error"].append(slope_ra_adaptive_err)

        total_number_solves = sum(df_ra_adaptive["num solves"].values)
        total_number_dof = sum(np.multiply(df_ra_adaptive["num solves"].values, df_ra_adaptive["dof num"].values))
        slopes_results["total number solves"].append(total_number_solves)
        slopes_results["total number dof"].append(total_number_dof)

        efficiency = np.mean(df_ra_adaptive["total estimator"].values[-5:]/df_ra_adaptive["total error"].values[-5:])
        slopes_results["efficiency"].append(efficiency)

        plt.loglog(xs, ys, "^-", color=darkblue, label=f"s={s}")
        plt.loglog(xs, ys_err, "^--", color=lightblue, label=f"s={s} (exact)")
        # plt.loglog(xs_ra_adaptive, ys_ra_adaptive, "^-", color=darkorange, label=f"s={s} (FE adapt., ra adapt.)")
        # plt.loglog(xs_ra_adaptive, ys_ra_adaptive_err, "^--", color=lightorange, label=f"s={s} (FE adaptive, exact)")

        marker_x, marker_y = marker(xs, [ys_err], 0.2, 0.05)
        annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
        plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
        plt.xlabel("dof")
        plt.savefig(f"conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

df = pd.DataFrame(slopes_results)
df.to_csv("./slopes_results.csv")
print(df)

for method in ["bp", "bura"]:
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        plt.figure()
    
        df = pd.read_csv(f"./results_{method}_{str(s)[-1]}.csv")
        df_FE_ra_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_rational_adaptive.csv")

        xs = df["num solves"].values * df["dof num"].values
        ys = df["total estimator"].values
        ys_err = df["total error"].values
        xs_ra_adaptive = df_ra_adaptive["num solves"].values * df_ra_adaptive["dof num"].values
        ys_ra_adaptive = df_ra_adaptive["total estimator"].values
        ys_ra_adaptive_err = df_ra_adaptive["total error"].values

        slope = slopes(xs, ys)
        slope_ra_adaptive = slopes(xs_ra_adaptive, ys_ra_adaptive)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope estimator"].append(slope)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["rational adaptive"].append(True)
        slopes_results["slope estimator"].append(slope_ra_adaptive)

        plt.loglog(xs, ys_err, "o--", label=f"s={s} (exact)")
        plt.loglog(xs, ys, "o-", label=f"s={s}")
        plt.loglog(xs_ra_adaptive, ys_ra_adaptive_err, "^--", label=f"s={s} (ra adapt., exact)")
        plt.loglog(xs_ra_adaptive, ys_ra_adaptive, "^-", label=f"s={s} (ra adapt.)")

        plt.legend()
        plt.xlabel("total dof")
        #plt.ylabel(r"$\eta^{\mathrm{bw}}_N$")
        plt.savefig(f"cumulative_conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

for method in ["bp", "bura"]:
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        plt.figure()
        plt.title(method + f" s={str(s)}")

        df_ra_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_rational_adaptive.csv")

        xs_ra_adaptive = df_ra_adaptive["num solves"].values * df_ra_adaptive["dof num"].values
        ys_ra_adaptive_bw = df_ra_adaptive["L2 bw"].values
        ys_ra_adaptive_ra = df_ra_adaptive["rational estimator"].values

        plt.loglog(xs_ra_adaptive, ys_ra_adaptive_bw, "^--", label=f"s={s} (FE adapt., ra adapt., FE)")
        plt.loglog(xs_ra_adaptive, ys_ra_adaptive_ra, "^--", label=f"s={s} (FE adapt., ra adapt., ra)")

        plt.legend()
        plt.xlabel("total dof")
        plt.savefig(f"FE_vs_ra_{method}_{str(s)[-1]}.pdf")
