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
    print(f"Estimator slope ({method}): {m}")

for s in [0.3, 0.5, 0.7]:
    df_bp = pd.read_csv(f"./results_bp_{str(s)[-1]}.csv")
    df_bura = pd.read_csv(f"./results_bura_{str(s)[-1]}.csv")

    xs_bp = df_bp["dof num"].values
    ys_bp = df_bp["L2 bw"].values
    err_bp = df_bp["exact error"].values

    xs_bura = df_bura["dof num"].values
    ys_bura = df_bura["L2 bw"].values
    err_bura = df_bura["exact error"].values

    # Total estimator ~ L2 BW + ||f|| * rational_error (for checkerboard ||f|| = 0.5)
    '''
    ys_bp_UB = df_bp["L2 bw"].values + 2. * df_bp["rational error"]
    ys_bura_UB = df_bura["L2 bw"].values + 2. * df_bura["rational error"]

    ys_bp_LB = np.maximum(df_bp["L2 bw"].values, 2. * df_bp["rational error"])
    ys_bura_LB = np.maximum(df_bura["L2 bw"].values, 2. * df_bura["rational error"])
    '''

    slopes(xs_bp, ys_bp, f"BP, s={s}")
    slopes(xs_bura, ys_bura, f"BURA, s={s}")

    plt.figure()
    plt.loglog(xs_bp, ys_bp, "o-", label=fr"$\eta^{{\mathrm{{bw}}}}_N$ (BP, s={s})")
    plt.loglog(xs_bp, err_bp, "o-", label=fr"$||u_{{Q^s_N, 1}} - u||_{{L^2(\Omega)}}$ (BP, s={s})")

    plt.loglog(xs_bura, ys_bura, "^--", label=fr"$\eta^{{\mathrm{{bw}}}}_N$ (BURA, s={s})")
    plt.loglog(xs_bura, err_bura, "^--", label=fr"$||u_{{Q^s_N, 1}} - u||_{{L^2(\Omega)}}$ (BURA, s={s})")

    '''
    plt.loglog(xs_bp, ys_bp_LB, "s-.", label=f"Total error LB")
    plt.loglog(xs_bp, ys_bp_UB, "s-.", label=f"Total error UB")

    plt.loglog(xs_bp, ys_bura_LB, "s-.", label=f"Total error LB")
    plt.loglog(xs_bp, ys_bura_UB, "s-.", label=f"Total error UB")
    '''

    marker_x, marker_y = marker(xs_bp, [ys_bp, err_bp], 0.2, 0.1)
    annotation.slope_marker((marker_x, marker_y), (-2, 2), invert=True)
    plt.legend()
    plt.xlabel("dof")
    plt.ylabel(r"$\eta^{\mathrm{bw}}_N$")
    plt.savefig(f"conv_" + f"{str(s)[-1]}.pdf")
