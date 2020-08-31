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
plt.loglog(df["num_dofs"], df["error_hu"], '^-',
           label=r"$\eta_{u}$", color=dark_map[3])
plt.loglog(df["num_dofs"], df["error_hz"], '^-',
           label=r"$\eta_{z}$", color=dark_map[4])
plt.loglog(df["num_dofs"], df["error_hw"], '^-',
           label=r"$\eta_{w}$", color=dark_map[2])
plt.loglog(df["num_dofs"], df["error"], '^--',
           label=r"$\eta_{e}$", color=dark_map[0])
plt.xlabel("Number of dofs")
plt.ylabel(r"$\eta$")
marker_x, marker_y = marker([0.5, 0.2], [df["num_dofs"].median(), df["num_dofs"].tail(1).item()], [
                            df["error_hw"].median(), df["error_hw"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
marker_x, marker_y = marker([0.5, 0.2], [df["num_dofs"].median(), df["num_dofs"].tail(1).item()], [
                            df["error_hz"].median(), df["error_hz"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-0.5, 1), invert=True)
plt.legend(loc=(1.04, 0.25))
plt.savefig("output/error.pdf")
