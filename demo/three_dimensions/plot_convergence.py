# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
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


dark_map = [cm.get_cmap('Dark2')(i / 8.) for i in range(8)]

df = pd.read_pickle("output/results.pkl")
print(df)

height = 3.50394 / 1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.loglog(df["num_dofs"], df["error_bw"], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[2])
plt.xlabel("Number of dof")
plt.ylabel(r"$\eta$")
marker_x, marker_y = marker([0.5, 0.2], [df["num_dofs"].median(), df["num_dofs"].tail(1).item()], [
                            df["error_bw"].median(), df["error_bw"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-1, 3), invert=True)
plt.legend()
plt.savefig("output/error.pdf")
