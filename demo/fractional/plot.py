import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from marker import marker

pd.set_option('display.expand_frame_repr', False)
colors = [cm.get_cmap('tab20c')(i / 20.) for i in range(20)]
darkblue =      colors[0]
middleblue =    colors[1]
lightblue =     colors[2]
darkorange =    colors[8]
middleorange =  colors[9]
lightorange =   colors[10]

plt.style.use("./plots.mplstyle")

def slopes(xs, ys):
    xs = xs[-10:]
    ys = ys[-10:]
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
                    "total est. efficiency":    [],
                    "ra est. efficiency":       [],
                    "FE est. efficiency":       []}

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
                    total_efficiency = np.mean(ys[-5:]/ys_err[-5:])

                    ra_est = df["rational estimator"].values
                    ra_err = df["rational error"].values
                    ra_efficiency = np.mean(ra_est[-5:]/ra_err[-5:])

                    FE_est = df["L2 bw"].values
                    FE_err = df["FE error"].values
                    FE_efficiency = np.mean(FE_est[-5:]/FE_err[-5:])
                else:
                    slope_err = None
                    total_efficiency = None
                    ra_efficiency = None
                    FE_efficiency = None
                
                disp_results["total error slope"].append(slope_err)
                disp_results["total est. efficiency"].append(total_efficiency)
                disp_results["ra est. efficiency"].append(ra_efficiency)
                disp_results["FE est. efficiency"].append(FE_efficiency)

    df_results = pd.DataFrame(disp_results)
    df_results.to_csv(f"./{dr}/results/results.csv")
    print(df_results)

def convergence_plots(dr):
    for method in ["bp", "bura"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}.csv")

            figure = plt.figure()

            xs = df["dof num"].values
            ys = df["total estimator"].values

            plt.loglog(xs, ys, "^-", color=darkblue)

            y_datas = [ys]
            if "total error" in df:
                ys_err = df["total error"].values
                plt.loglog(xs, ys_err, "^--", color=lightblue)
                y_datas.append(ys_err)
            
            marker(figure, xs, y_datas, 0.2, 1., slope=1, loglog=True)
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.xlabel("dof")
            plt.savefig(f"{dr}/results/conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def cumulative_convergence_plots(dr):
    styles_1 = ["^-", "^--", "o-", "s-"]
    styles_2 = ["o-", "o--", "o-.", ""]
    colors = [[darkblue, darkorange], 
              [middleblue, middleorange],
              [lightblue, lightorange]]
    adaptation_methods = ["None", "ra", "FE", "FE + ra"]
    for method in ["bura", "bp"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            figure = plt.figure()
            i=0
            y_datas = []
            for adaptation, style_1, style_2 in zip(adaptation_methods, styles_1, styles_2):
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
                #print(f"./{dr}/results/results_{method}_{str(s)[-1]}{adapt_str}.csv")
                #print(df)

                xs = df["num solves"].values * df["dof num"].values
                ys = df["total estimator"].values
                y_datas.append(ys)
                
                plt.loglog(xs, ys, style_1, color=colors[i][0], label=f"est. ({adaptation})")

                if "total error" in df:
                    ys_err = df["total error"].values
                    plt.loglog(xs, ys_err, style_2, color=colors[i][1], label=f"err. ({adaptation})")
                    y_datas.append(ys_err)
                i += 1

            plt.xlabel(r"dof num. $\times$ solves num.")
            #plt.legend(loc=3)
            marker(figure, xs, y_datas, 0.2, 0.8, slope=1, loglog=True)
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.savefig(f"{dr}/results/cumulative_conv_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def comparison_FE_ra(dr):
    for method in ["bp", "bura"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            plt.figure()

            try:
                df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}_rational_adaptive.csv")
            except FileNotFoundError:
                df = pd.read_csv(f"./{dr}/results/results_{method}_{str(s)[-1]}_FE_adaptive_rational_adaptive.csv")

            xs = df["num solves"].values * df["dof num"].values
            ys_bw = df["L2 bw"].values
            ys_ra = df["rational estimator"].values

            plt.loglog(xs, ys_bw, "^--", label=f"FE")
            plt.loglog(xs, ys_ra, "^--", label=f"ra")

            # plt.legend(loc=3)
            plt.xlabel(r"dof num. $\times$ solves num.")
            plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.95), xycoords="axes fraction", size=13, horizontalalignment='right')
            plt.savefig(f"{dr}/results/FE_vs_ra_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def evolution_dof(dr):
    styles = ["^-", "o--", "o--", "s-."]
    colors = [[darkblue, darkorange], 
              [middleblue, middleorange],
              [lightblue, lightorange]]
    adaptation_methods = ["None", "ra", "FE", "FE + ra"]
    for method in ["bura", "bp"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            figure = plt.figure()
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

                ys = np.cumsum(df["num solves"].values * df["dof num"].values)
                xs = np.arange(len(ys))
                
                plt.semilogy(xs, ys, style, color=colors[i][0], label=f"dof num ({adaptation})")
                i += 1

            plt.xlabel(r"ref. step")
            plt.ylabel(r"$\eta^{\mathrm{bw}}_{\mathcal{Q}_s}$")
            plt.legend(loc=2)
            plt.savefig(f"{dr}/results/evolution_dof_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

def parametric_errors(dr):
    for method in ["bura", "bp"]:
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
            figure = plt.figure()
            plt.style.use("./parametric_plots.mplstyle")
            df = pd.read_csv(f"./{dr}/output/{method}_{str(s)[-1]}/parametric_results_{method}_5.csv")

            ys = df["parametric exact error"].values
            ys /= np.max(ys)
            xs = df["parametric index"].values

            plt.plot(xs, ys)
            if method=="bp":
                if s==0.1:
                    plt.annotate(method.upper() + f", s={s}", xy=(0.01,0.03), xycoords="axes fraction", size=18, horizontalalignment='left')
                else:
                    plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.93), xycoords="axes fraction", size=18, horizontalalignment='right')
            else:
                plt.annotate(method.upper() + f", s={s}", xy=(0.99,0.03), xycoords="axes fraction", size=18, horizontalalignment='right')
            plt.xlabel(r"Parametric index $l$")
            # plt.ylabel(r"$e_l/\max_{-M\leqslant l \leqslant N}(e_l)$")
            plt.savefig(f"{dr}/results/parametric_error_{method}_{str(s)[-1]}.pdf", bbox_inches="tight")

if __name__=="__main__":
    dirs = ["sines_bp_vs_bura", "checkerboard_bp_vs_bura"]

    for dr in dirs:
        #print(dr)
        results_table(dr)
        convergence_plots(dr)
        cumulative_convergence_plots(dr)
        comparison_FE_ra(dr)
        evolution_dof(dr)

    parametric_errors("sines_bp_vs_bura")