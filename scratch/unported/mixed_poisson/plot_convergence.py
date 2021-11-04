import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpltools import annotation

k = 1

if k == 1:
    ncol = 3
else:
    ncol = 2

if k == 3:
    ymax = 20
else:
    ymax = 10


def marker(x_data, y_datas, position, gap):
    middle = np.floor(len(x_data) / 2.).astype(int)
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
    marker_x = anchor_1[0]**position * anchor_2[0]**(1. - position)\
        * (anchor_2[1] / anchor_1[1])**gap
    marker_y = anchor_1[1]**position * anchor_2[1]**(1. - position)\
        * (anchor_1[0] / anchor_2[0])**gap
    return marker_x, marker_y


dark_map = [cm.get_cmap("Dark2")(i / 8.) for i in range(8)]

df_bw = pd.read_pickle('output/results.pkl')

print('Results:\n')
print(df_bw)

x = np.log(df_bw['num_dofs'].values[-5:])
y = np.log(df_bw['error_bw'].values[-5:])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print('BW slope =', m)

height = 3.50394 / 1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
plt.figure()

plt.loglog(df_bw["num_dofs"], df_bw['error_bw'], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[0])


plt.xlabel("Number of dof")
plt.ylabel(r"$\eta$")

marker_x, marker_y = marker(df_bw["num_dofs"].values, [
                            df_bw["error_bw"].values], 0.4, 0.1)

annotation.slope_marker((marker_x, marker_y), (-k, 2), invert=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=ncol, mode="expand", borderaxespad=0.)
plt.savefig('output/error.pdf')
