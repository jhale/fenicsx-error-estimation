import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpltools import annotation


def marker(coefficients, anchors_x, anchors_y):
    marker_x = np.power(anchors_x[0], coefficients[0])\
        * np.power(anchors_x[1], 1.-coefficients[0])
    marker_y = np.power(anchors_y[0], coefficients[1])\
        * np.power(anchors_y[1], 1.-coefficients[1])
    return marker_x, marker_y


import matplotlib.cm as cm
dark_map = [cm.get_cmap('Dark2')(i/8.) for i in range(8)]

df = pd.read_pickle("output/results.pkl")
print(df)

fig = plt.figure()
plt.loglog(df["num_dofs"], df["estimated_error"], '^-',
           label=r"$\eta_{\mathrm{BW}}$", color=dark_map[2])
plt.loglog(df["num_dofs"], df["error"], '^-',
           label=r"$|J(u) - J(u_h)|$", color=dark_map[0])
plt.xlabel("Number of dofs")
plt.ylabel("$\eta$")
marker_x, marker_y = marker([0.5, 0.35], [df["num_dofs"].median(), df["num_dofs"].tail(1).item()], [df["estimated_error"].median(), df["estimated_error"].tail(1).item()])
annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
plt.legend()
plt.savefig("output/error.pdf")
