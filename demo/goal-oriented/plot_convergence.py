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

height = 3.50394 / 1.608
width = 3.50394
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'lines.markersize': 3})
plt.rcParams.update({'lines.linewidth': 1})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.loglog(df["num_dofs"], df["error_bw_w"], '^-',
           label=r"$\eta_{\mathrm{bw},w}$", color=dark_map[3])
plt.loglog(df["num_dofs"], df["error_res_w"], '^-',
           label=r"$\eta_{\mathrm{res},w}$", color=dark_map[5])
plt.loglog(df["num_dofs"], df["error_zz_w"], '^-',
           label=r"$\eta_{\mathrm{zz},w}$", color=dark_map[6])
plt.loglog(df["num_dofs"], df["exact_error"], '^--',
           label=r"$\eta_{\mathrm{err}}$", color=dark_map[2])
plt.loglog(df["num_dofs"], df["error_bw_u"], '^--',
           label=r"$\eta_{\mathrm{bw},u}$", color=dark_map[1])
plt.loglog(df["num_dofs"], df["error_res_u"], '^--',
           label=r"$\eta_{\mathrm{res},u}$", color=dark_map[4])
plt.loglog(df["num_dofs"], df["error_zz_u"], '^--',
           label=r"$\eta_{\mathrm{zz},u}$", color=dark_map[7])
plt.xlabel("Number of dofs")
plt.ylabel(r"$\eta$")
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
marker_x, marker_y = marker([0.5, 0.3],
                            [df["num_dofs"].median(), df["num_dofs"].tail(1).item()],
                            [df["error_res_w"].median(), df["error_res_w"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-2, 2), invert=True)
plt.legend(ncol=3)
plt.savefig("plot.pdf")

'''
fig = plt.figure()
eff_bw = np.divide(df["error"].values, df["error_bw"].values)
x = np.arange(len(eff_bw))

height = 3.50394 / 1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.plot(x, eff_bw, '^-', label="Efficiency BW")
plt.legend()
plt.savefig("output/efficiency.pdf")
'''
