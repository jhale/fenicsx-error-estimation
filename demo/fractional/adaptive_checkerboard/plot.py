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

df = pd.read_csv("./results.csv")

plt.figure()
slope = 1

xs = df["dof num"].values
ys = df["L2 bw"].values

A = np.vstack([np.log(xs), np.ones_like(xs)]).T

m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
print(f"Estimator slope: {m}")

last_ys = ys[::-5]
print(last_ys)
plt.loglog(xs, ys, "^--", label="L2 bw")
plt.loglog(xs, xs**(-slope), "-", label=f"slope {slope}")

plt.legend()
plt.xlabel("dof")
plt.ylabel("L2 bw")
plt.savefig("conv.pdf")
