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
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T

    m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
    print(f"Estimator slope ({method}): {m}")


df_bp = pd.read_csv(f"./results_bp.csv")
df_bura = pd.read_csv(f"./results_bura.csv")

xs_bp = df_bp["dof num"].values
ys_bp = df_bp["L2 bw"].values

xs_bura = df_bura["dof num"].values
ys_bura = df_bura["L2 bw"].values

plt.figure()
slopes(xs_bp, ys_bp, "BP")
slopes(xs_bura, ys_bura, "BURA")
plt.loglog(xs_bp, ys_bp, "^--", label="L2 bw (BP)")
plt.loglog(xs_bura, ys_bura, "^-.", label="L2 bw (BURA)")
plt.loglog(xs_bp, xs_bp**(-1.), "-", label="slope -1")
plt.legend()
plt.xlabel("dof")
plt.ylabel("L2 bw")
plt.savefig(f"conv.pdf")
