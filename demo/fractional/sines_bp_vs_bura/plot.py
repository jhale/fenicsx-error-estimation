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

def slopes(xs, ys, method):
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T

    m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
    return m

slopes_results = {"s": [], "method": [], "slope": []}
for s in [0.3, 0.5, 0.7]:
    for method in ["bp", "bura"]:
        df = pd.read_csv(f"./results_{method}_{str(s)[-1]}.csv")

        xs = df["dof num"].values
        ys = df["L2 bw"].values
        err = df["total error"].values

        m = slopes(xs, ys, f"{method}, s={s}")

        slopes_results["s"].append(s)
        slopes_results["method"].append(method)
        slopes_results["slope"].append(m)

        plt.figure()
        plt.loglog(xs, ys, "^-", color="#3B75AF", linewidth=2.5, label=fr"$\eta^{{\mathrm{{bw}}}}_N$ ({method}, s={s})")
        plt.loglog(xs, err, "^--", color="#B3C6E5", linewidth=2.5, label=fr"$||u_{{Q^s_N, 1}} - u||_{{L^2(\Omega)}}$ ({method}, s={s})")

        marker_x, marker_y = marker(xs, [ys, err], 0.2, 0.1)
        annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
        plt.legend()
        plt.xlabel("dof")
        plt.ylabel(r"$\eta^{\mathrm{bw}}_N$")
        plt.savefig(f"conv_" + method + f"_{str(s)[-1]}.pdf")

        df_parametric = pd.read_csv(f"./output/{method}_{str(s)[-1]}/parametric_results_5.csv")
        df_rational_parameters = pd.read_csv(f"./output/{method}_{str(s)[-1]}/rational_parameters.csv")

        xs_parametric = df_parametric["parametric index"].values
        #xs_parametric = df_rational_parameters["c_1s"].values
        ys_parametric = df_parametric["parametric exact error"].values

        plt.figure()
        plt.semilogy(xs_parametric, ys_parametric, color="#3B75AF", linewidth=2.5)
        #plt.loglog(xs_parametric, ys_parametric, color="#3B75AF", linewidth=2.5)
        plt.xlabel(f"Parametric index $l$")
        #plt.xlabel(f"Parametric coefficient $b_l$")
        if method == "bura":
            plt.xticks(xs_parametric)
        else:
            plt.xticks(xs_parametric[::17])
        
        plt.ylabel(r"$||u_l - u_{l,1}||_{L^2}$")
        plt.savefig(f"parametric_errors_" + method + f"_{str(s)[-1]}.pdf")

df = pd.DataFrame(slopes_results)
print(df)
df.to_csv("slopes_results.csv")
