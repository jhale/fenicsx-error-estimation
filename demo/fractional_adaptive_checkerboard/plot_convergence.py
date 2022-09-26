# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from mpltools import annotation


def marker(coefficients, anchors_x, anchors_y):
    marker_x = np.power(anchors_x[0], coefficients[0])\
        * np.power(anchors_x[1], 1. - coefficients[0])
    marker_y = np.power(anchors_y[0], coefficients[1])\
        * np.power(anchors_y[1], 1. - coefficients[1])
    return marker_x, marker_y


dark_map = [cm.get_cmap('Dark2')(i / 8.) for i in range(8)]

df = pd.read_csv("./output/results.csv")

height = 2.8
width = 4.0
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.loglog(df["dof num"], df["L2 bw"], '^-',
           label=r"$\eta^{\mathrm{bw}}_{\kappa}$", color=dark_map[3])
plt.xlabel("dofs")
plt.ylabel(r"$\eta^{\mathrm{bw}}_{\kappa}$")
marker_x, marker_y = marker([0.5, 0.2], [df["dof num"].median(), df["dof num"].tail(1).item()], [
                            df["L2 bw"].median(), df["L2 bw"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
print(np.polyfit(np.log10(df["dof num"])[:-5:-1], np.log10(df["L2 bw"])[:-5:-1], 1)) 
#plt.legend()
plt.tight_layout()
plt.savefig("output/error.pdf")
