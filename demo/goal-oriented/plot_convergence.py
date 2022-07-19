# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpltools import annotation


def marker(coefficients, anchors_x, anchors_y):
    marker_x = np.power(anchors_x[0], coefficients[0])\
        * np.power(anchors_x[1], 1. - coefficients[0])
    marker_y = np.power(anchors_y[0], coefficients[1])\
        * np.power(anchors_y[1], 1. - coefficients[1])
    return marker_x, marker_y

estimators = [("bw", 1, 2),
              ("res", None, None),
              ("zz", None, None)]

colors = [cm.get_cmap('Greys')(1. - i / 4.) for i in range(3)]
styles = ["^", "x", "o"]

# height = 3.50394 / 1.608
# width = 3.50394
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'lines.markersize': 4})
plt.rcParams.update({'lines.linewidth': 1.5})
# plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})

plt.figure()

dofs = []
exact_errors = []
for (estimator, param1, param2), style in zip(estimators, styles):
    if estimator == "bw":
        estimator_path = "bw_" + str(param1) + "_" + str(param2)
        label = f"$\eta_{{\mathrm{{{estimator}}}}}^{{{str(param2)}, {str(param1)}}}$"
    else:
        estimator_path = estimator
        label = f"$\eta_{{\mathrm{{{estimator}}}}}$"

    df = pd.read_pickle(os.path.join("output", estimator_path, "results.pkl"))
    num_dofs = df["num_dofs"][11:]
    est_u = df["error_" + estimator + "_u"][11:]
    est_z = df["error_" + estimator + "_z"][11:]
    est_w = df["error_" + estimator + "_w"][11:]

    plt.loglog(num_dofs, est_u, style + "-", label=label + " (primal)",
               color="black")
    plt.loglog(num_dofs, est_z, style + "--", label=label + " (dual)",
               color="gray")
    plt.loglog(num_dofs, est_w, style + ":", label=label + " (wgo)",
               color="silver")

    dofs.append(num_dofs.values.tolist())
    exact_errors.append(df["exact_error"][11:])

dofs = np.ndarray.flatten(np.array(dofs))
exact_errors = np.ndarray.flatten(np.array(exact_errors))
A = np.vstack([np.log(dofs), np.ones_like(dofs)]).T
m, c = np.linalg.lstsq(A, np.log(exact_errors), rcond=None)[0]

sorted_dofs = np.sort(dofs)
plt.loglog(sorted_dofs, np.exp(c) * sorted_dofs**m, color='black', label="lstsq error")
# Added these useless loglog plots in order to the items to be properly stored
# in legend
plt.loglog(sorted_dofs, np.exp(c) * sorted_dofs**m, color='black', alpha=0., label=" ")
plt.loglog(sorted_dofs, np.exp(c) * sorted_dofs**m, color='black', alpha=0., label=" ")

plt.xlabel("Number of dof")
plt.ylabel(r"$\eta$")

'''
marker_x, marker_y = marker([0.5, 0.2],
                            [df["num_dofs"].median(), df["num_dofs"].tail(1).item()],
                            [df["error_bw_u"].median(), df["error_bw_u"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-1, 2), invert=True)
marker_x, marker_y = marker([0.5, 0.3],
                            [df["num_dofs"].median(), df["num_dofs"].tail(1).item()],
                            [df["error_bw_w"].median(), df["error_bw_w"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-2, 2), invert=True)
marker_x, marker_y = marker([0.5, 0.3],
                            [df["num_dofs"].median(), df["num_dofs"].tail(1).item()],
                            [df["exact_error"].median(), df["exact_error"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-2, 2), invert=True)
'''
df = pd.read_pickle("output/bw_1_2/results.pkl")
num_dofs = df["num_dofs"].values.tolist()[11:]
est_z = df["error_bw_z"].values.tolist()[11:]

marker_x, marker_y = marker([0.5, 0.2],
                            [np.median(num_dofs), num_dofs[-1:]],
                            [np.median(est_z), est_z[-1:]])
annotation.slope_marker((marker_x[0], marker_y[0]), (-1, 2), invert=True)

df = pd.read_pickle("output/bw_1_2/results.pkl")
num_dofs = df["num_dofs"].values.tolist()[11:]
est_z = df["error_bw_w"].values.tolist()[11:]

marker_x, marker_y = marker([0.5, 0.3],
                            [np.median(num_dofs), num_dofs[-1:]],
                            [np.median(est_z), est_z[-1:]])
annotation.slope_marker((marker_x[0], marker_y[0]), (-2, 2), invert=True)

df = pd.read_pickle("output/res/results.pkl")
num_dofs = df["num_dofs"].values.tolist()[11:]
est_z = df["error_res_w"].values.tolist()[11:]

marker_x, marker_y = marker([0.5, 0.3],
                            [np.median(num_dofs), num_dofs[-1:]],
                            [np.median(est_z), est_z[-1:]])
annotation.slope_marker((marker_x[0], marker_y[0]), (-2, 2), invert=True)

df = pd.read_pickle("output/res/results.pkl")
num_dofs = df["num_dofs"].values.tolist()[11:]
est_z = df["error_res_w"].values.tolist()[11:]

marker_x, marker_y = marker([0.5, 0.3],
                            [np.median(num_dofs), num_dofs[-1:]],
                            [np.median(est_z), est_z[-1:]])
annotation.slope_marker((marker_x[0], marker_y[0]), (-2, 2), invert=True)

marker_x, marker_y = marker([0.5, 0.3],
                            [np.median(dofs), dofs[-1:]],
                            [np.median(exact_errors), exact_errors[-1:]])
annotation.slope_marker((marker_x[0], marker_y[0]), (-2, 2), invert=True)

plt.legend(ncol=4, bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", loc="lower left")
plt.savefig("plot.pdf", bbox_inches="tight")
