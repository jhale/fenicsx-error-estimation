# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.cm as cm
dark_map = [cm.get_cmap('Dark2')(i / 8.) for i in range(8)]

df = pd.read_pickle("output/results.pkl")

height = 3.50394 / 1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.plot(df["error"] / df["error_hw"], '^-', color=dark_map[0])
plt.xlabel("Adaptive refinement step")
plt.ylabel("efficiency")
plt.savefig("output/efficiency.pdf")
