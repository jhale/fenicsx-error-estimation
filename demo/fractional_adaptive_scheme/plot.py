import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpltools import annotation

def marker(x_data, y_datas, position, gap):
    middle = np.floor(len(x_data)/2.).astype(int)
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
    xs = xs[-10:]
    ys = ys[-10:]
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T

    m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
    print(f"Estimator slope {method}: {m}")


df_bp = pd.read_csv(f"./results_bp_adaptive.csv")
df_bp_nap = pd.read_csv(f"./results_bp_non_adaptive.csv")
df_bura = pd.read_csv(f"./results_bura_adaptive.csv")
df_bura_nap = pd.read_csv(f"./results_bura_non_adaptive.csv")

xs_bp = np.multiply(df_bp["dof num"].values[:], df_bp["solves num"].values[:])
xs_bp = np.cumsum(xs_bp)
xs_bp_nap = np.multiply(df_bp_nap["dof num"].values[:], df_bp_nap["solves num"].values[:])
xs_bp_nap = np.cumsum(xs_bp_nap)
xs_bura = np.multiply(df_bura["dof num"].values[:], df_bura["solves num"].values[:])
xs_bura = np.cumsum(xs_bura)
xs_bura_nap = np.multiply(df_bura_nap["dof num"].values[:], df_bura_nap["solves num"].values[:])
xs_bura_nap = np.cumsum(xs_bura_nap)


# Total estimator ~ L2 BW + ||f|| * rational_error (for checkerboard ||f|| = 0.5)
ys_bp_UB = df_bp["L2 bw"].values + 0.5 * df_bp["rational error"]
ys_bp_nap_UB = df_bp_nap["L2 bw"].values + 0.5 * df_bp_nap["rational error"]
ys_bura_UB = df_bura["L2 bw"].values + 0.5 * df_bura["rational error"]
ys_bura_nap_UB = df_bura_nap["L2 bw"].values + 0.5 * df_bp_nap["rational error"]

ys_bp_LB = np.abs(df_bp["L2 bw"].values - 0.5 * df_bp["rational error"])
ys_bp_nap_LB = np.abs(df_bp_nap["L2 bw"].values - 0.5 * df_bp_nap["rational error"])
ys_bura_LB = np.abs(df_bura["L2 bw"].values - 0.5 * df_bura["rational error"])
ys_bura_nap_LB = np.abs(df_bura_nap["L2 bw"].values - 0.5 * df_bp_nap["rational error"])

plt.figure()

method = "bura"

if method == "bp":
    slopes(xs_bp_nap, ys_bp_nap_UB, "BP (non adap)")
    slopes(xs_bp, ys_bp_UB, "BP")
    plt.loglog(xs_bp_nap, ys_bp_nap_UB, "^-", label="Total est. upper bound (BP non adap)")
    plt.loglog(xs_bp_nap, ys_bp_nap_LB, "^-", label="Total est. lower bound (BP non adap)")
    plt.loglog(xs_bp, ys_bp_UB, "o--", label="Total est. upper bound (BP adap)")
    plt.loglog(xs_bp, ys_bp_LB, "o--", label="Total est. lower bound (BP adap)")
    plt.loglog(xs_bp, xs_bp**(-0.75) * 2, "--", color="black", label="slope -0.75")

elif method == "bura":
    slopes(xs_bura_nap, ys_bura_nap_UB, "BURA (non adap)")
    slopes(xs_bura, ys_bura_UB, "BURA (adap)")
    plt.loglog(xs_bura_nap, ys_bura_nap_UB, "^-", label="Total est. upper bound (BURA non adap)")
    plt.loglog(xs_bura_nap, ys_bura_nap_LB, "^-", label="Total est. lower bound (BURA non adap)")
    plt.loglog(xs_bura, ys_bura_UB, "o--", label="Total est. upper bound (BURA adap)")
    plt.loglog(xs_bura, ys_bura_LB, "o--", label="Total est. lower bound (BURA adap)")
    plt.loglog(xs_bura, xs_bura**(-0.75), "--", color="black", label="slope -0.75")

plt.legend()
plt.xlabel("total number of dof")
plt.ylabel("Estimators")
plt.savefig(f"conv_ap_vs_nap_{method}.pdf")