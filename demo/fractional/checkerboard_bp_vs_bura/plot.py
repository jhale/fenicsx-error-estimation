import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpltools import annotation

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
    return m

slopes_results = {"method": [], "s": [], "FE adaptive": [], "rational adaptive": [], "slope": []}
for method in ["bp", "bura"]:
    plt.figure()
    plt.title("2D Checkerboard " + f"{method}")
    for s in [0.3, 0.5, 0.7]:
        df = pd.read_csv(f"./results_{method}_{str(s)[-1]}.csv")
        df_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_FE_adaptive.csv")
        df_FE_ra_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_FE_adaptive_rational_adaptive.csv")

        xs = df["dof num"].values
        ys = df["L2 bw"].values
        xs_adaptive = df_adaptive["dof num"].values
        ys_adaptive = df_adaptive["total estimator"].values
        xs_FE_ra_adaptive = df_FE_ra_adaptive["dof num"].values
        ys_FE_ra_adaptive = df_FE_ra_adaptive["total estimator"].values
        slope = slopes(xs, ys)
        slope_adaptive = slopes(xs_adaptive, ys_adaptive)
        slope_FE_ra_adaptive = slopes(xs_FE_ra_adaptive, ys_FE_ra_adaptive)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(False)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope"].append(slope)
    
        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(True)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope"].append(slope_adaptive)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(True)
        slopes_results["rational adaptive"].append(True)
        slopes_results["slope"].append(slope_FE_ra_adaptive)

        plt.loglog(xs, ys, "o-", label=f"s={s}")
        plt.loglog(xs_adaptive, ys_adaptive, "^--", label=f"s={s} (FE adaptive)")
        plt.loglog(xs_FE_ra_adaptive, ys_FE_ra_adaptive, "^--", label=f"s={s} (FE adapt., ra adapt.)")

        marker_x, marker_y = marker(xs_FE_ra_adaptive, [ys_FE_ra_adaptive], 0.2, 0.1)
        annotation.slope_marker((marker_x, marker_y), (-2, 2), invert=True)
    plt.legend()
    plt.xlabel("dof")
    plt.ylabel(r"$\eta^{\mathrm{bw}}_N$")
    plt.savefig(f"conv_{method}.pdf")

df = pd.DataFrame(slopes_results)
df.to_csv("./slopes_results.csv")
print(df)

for method in ["bp", "bura"]:
    for s in [0.3, 0.5, 0.7]:
        plt.figure()
        plt.title(method + f" s={str(s)}")
    
        df = pd.read_csv(f"./results_{method}_{str(s)[-1]}.csv")
        df_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_FE_adaptive.csv")
        df_FE_ra_adaptive = pd.read_csv(f"./results_{method}_{str(s)[-1]}_FE_adaptive_rational_adaptive.csv")

        xs = df["num solves"].values * df["dof num"].values
        ys = df["L2 bw"].values
        xs_adaptive = df_adaptive["num solves"].values * df_adaptive["dof num"].values
        ys_adaptive = df_adaptive["total estimator"].values
        xs_FE_ra_adaptive = df_FE_ra_adaptive["num solves"].values * df_FE_ra_adaptive["dof num"].values
        ys_FE_ra_adaptive = df_FE_ra_adaptive["total estimator"].values

        slope = slopes(xs, ys)
        slope_adaptive = slopes(xs_adaptive, ys_adaptive)
        slope_FE_ra_adaptive = slopes(xs_FE_ra_adaptive, ys_FE_ra_adaptive)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(False)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope"].append(slope)
    
        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(True)
        slopes_results["rational adaptive"].append(False)
        slopes_results["slope"].append(slope_adaptive)

        slopes_results["method"].append(method)
        slopes_results["s"].append(s)
        slopes_results["FE adaptive"].append(True)
        slopes_results["rational adaptive"].append(True)
        slopes_results["slope"].append(slope_FE_ra_adaptive)

        plt.loglog(xs, ys, "o-", label=f"s={s}")
        plt.loglog(xs_adaptive, ys_adaptive, "^--", label=f"s={s} (FE adaptive)")
        plt.loglog(xs_FE_ra_adaptive, ys_FE_ra_adaptive, "^--", label=f"s={s} (FE adapt., ra adapt.)")

        plt.legend()
        plt.xlabel("total dof")
        plt.ylabel(r"$\eta^{\mathrm{bw}}_N$")
        plt.savefig(f"cumulative_conv_{method}_{str(s)[-1]}.pdf")
