import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpltools import annotation

pd.set_option('display.expand_frame_repr', False)
colors = [cm.get_cmap('tab20c')(i / 20.) for i in range(20)]
darkblue =      colors[0]
middleblue =    colors[1]
lightblue =     colors[2]
darkorange =    colors[8]
middleorange =  colors[9]
lightorange =   colors[10]

plt.style.use("./plots.mplstyle")

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

def slopes(xs, ys):
    xs = xs[-8:]
    ys = ys[-8:]
    A = np.vstack([np.log(xs), np.ones_like(xs)]).T

    m = np.linalg.lstsq(A, np.log(ys), rcond=None)[0][0]
    m = np.round(m,2)
    return m

def results_table(dr):
    disp_results = {"method":                   [],
                    "power":                    [],
                    "adapt. method":            [],
                    "iterations num.":          [],
                    "solves total num.":        [],
                    "dof total num.":           [],
                    "total error slope":        [],
                    "total estimator slope":    [],
                    "efficiency":               []}

    for method in ["bp", "bura"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for adaptation in ["None", "ra", "FE", "FE + ra"]:
                if adaptation == "FE":
                    adapt_str = "_FE_adaptive"
                elif adaptation == "ra":
                    adapt_str = "_rational_adaptive"
                elif adaptation == "FE + ra":
                    adapt_str = "_FE_adaptive_rational_adaptive"
                else:
                    adapt_str = ""

                try:
                    df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}{adapt_str}.csv")
                except FileNotFoundError:
                    continue

                disp_results["method"].append(method)
                disp_results["power"].append(s)
                disp_results["adapt. method"].append(adaptation)

                iterations_num = len(df["dof num"].values)
                disp_results["iterations num."].append(iterations_num)

                total_number_solves = sum(df["num solves"].values)
                disp_results["solves total num."].append(total_number_solves)

                total_number_dof = sum(np.multiply(df["num solves"].values, df["dof num"].values))
                disp_results["dof total num."].append(total_number_dof)

                xs = df["dof num"].values
                ys = df["total estimator"].values
                slope = slopes(xs, ys)

                disp_results["total estimator slope"].append(slope)

                if "total error" in df:
                    ys_err = df["total error"].values
                    slope_err = slopes(xs, ys_err)
                    efficiency = np.mean(ys[-5:]/ys_err[-5:])
                else:
                    slope_err = None
                    efficiency = None
                
                disp_results["total error slope"].append(slope_err)
                disp_results["efficiency"].append(efficiency)

    df_results = pd.DataFrame(disp_results)
    df_results.to_csv(f"./{dr}/results/results.csv")
    print(df_results)

def convergence_plots(dr):
    for method in ["bp", "bura"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}.csv")

            plt.figure()

            xs = df["dof num"].values
            ys = df["total estimator"].values

            plt.loglog(xs, ys, "^-", color=darkblue)

            y_datas = [ys]
            if "total error" in df:
                ys_err = df["total error"].values
                plt.loglog(xs, ys_err, "^--", color=lightblue)
                y_datas.append(ys_err)
            
            marker_x, marker_y = marker(xs, y_datas, 0.2, 0.05)
            annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.xlabel("dof")
            plt.savefig(f"{dr}/results/conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def cumulative_convergence_plots(dr):
    styles = ["^-", "o--", "o--", "s-."]
    colors = [[darkblue, darkorange], 
              [middleblue, middleorange],
              [lightblue, lightorange]]
    adaptation_methods = ["None", "ra", "FE", "FE + ra"]
    for method in ["bura", "bp"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            plt.figure()
            i=0
            for adaptation, style in zip(adaptation_methods, styles):
                if adaptation == "FE":
                    adapt_str = "_FE_adaptive"
                elif adaptation == "ra":
                    adapt_str = "_rational_adaptive"
                elif adaptation == "FE + ra":
                    adapt_str = "_FE_adaptive_rational_adaptive"
                else:
                    adapt_str = ""
                
                try:
                    df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}{adapt_str}.csv")
                except FileNotFoundError:
                    continue

                xs = df["num solves"].values * df["dof num"].values
                ys = df["total estimator"].values
                y_datas = [ys]
                
                plt.loglog(xs, ys, style, color=colors[i][0], label=f"est. ({adaptation})")

                if "total error" in df:
                    ys_err = df["total error"].values
                    y_datas.append(ys_err)
                    plt.loglog(xs, ys_err, style, color=colors[i][1], label=f"err. ({adaptation})")
                
                i += 1
                
            plt.xlabel(r"dof num. $\times$ solves num.")
            #marker_x, marker_y = marker(xs, y_datas, 0.1, 0.05)
            #annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.legend(loc=3)
            plt.savefig(f"{dr}/results/cumulative_conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def comparison_FE_ra(dr):
    for method in ["bp", "bura"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            plt.figure()
            plt.title(method + f" s={str(s)}")

            try:
                df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}_rational_adaptive.csv")
            except FileNotFoundError:
                df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}_FE_adaptive_rational_adaptive.csv")

            xs = df["num solves"].values * df["dof num"].values
            ys_bw = df["L2 bw"].values
            ys_ra = df["rational estimator"].values

            plt.loglog(xs, ys_bw, "^--", label=f"FE")
            plt.loglog(xs, ys_ra, "^--", label=f"ra")

            plt.legend(loc=3)
            plt.xlabel("total dof")
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.savefig(f"{dr}/results/FE_vs_ra_{method}_{str(s)[-1]}.pdf")

if __name__=="__main__":
    dirs = ["sines_bp_vs_bura", "checkerboard_bp_vs_bura"]

    for dr in dirs:
        print(dr)
        results_table(dr)
        convergence_plots(dr)
        cumulative_convergence_plots(dr)
        comparison_FE_ra(dr)