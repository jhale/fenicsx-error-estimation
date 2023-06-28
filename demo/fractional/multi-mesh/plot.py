import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
import os

def parametric_details(s, results_dir_str):
    meshes_num_refinement                   = np.load(os.path.join(results_dir_str, "meshes_num_refinement.npy"))
    global_weighted_parametric_est_history  = np.load(os.path.join(results_dir_str, "global_weighted_parametric_est_history.npy"))
    global_parametric_est_history           = np.load(os.path.join(results_dir_str, "global_parametric_est_history.npy"))
    ls_c_diff                               = np.load(os.path.join(results_dir_str, "ls_c_diff.npy"))
    ls_c_react                              = np.load(os.path.join(results_dir_str, "ls_c_react.npy"))
    ls_weights                              = np.load(os.path.join(results_dir_str, "ls_weights.npy"))

    param_pbm_nums = np.arange(len(ls_c_diff))
    ref_step_max = len(global_parametric_est_history[:,0])

    fig, axs = plt.subplots(2)
    fig.suptitle(rf"\Large  \textbf{{Multi-mesh adaptive refinement algorithm ($s={s}$)}}.")
    cmap_full = plt.get_cmap("Reds")
    cmap = colors.ListedColormap(cmap_full(np.linspace(0.3, 1, 256)))
    for ref_step in range(ref_step_max):
        color_index = (ref_step/(ref_step_max-1))
        color = cmap(color_index)
        axs[0].plot(param_pbm_nums, global_weighted_parametric_est_history[ref_step,:], color=color)
    axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axs[0].set_xlim(0,len(ls_c_diff) - 1)
    axs[0].set(xlabel="Mesh number $l$", ylabel=r"$a_l\eta^{\mathrm{bw}}_{\mathcal T_l}$")
    axs[0].set_title("Weighted parametric estimators")
    norm = plt.Normalize(0,ref_step_max-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Customize the colorbar position and appearance
    cbar_ax = axs[0].inset_axes([0.59, 0.9, 0.4, 0.4])
    cbar_ax.set_box_aspect(0.01)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    cbar_ax.spines['top'].set_visible(False)
    cbar_ax.spines['bottom'].set_visible(False)
    cbar_ax.spines['left'].set_visible(False)
    cbar_ax.spines['right'].set_visible(False)
    cbar_ax.xaxis.set_ticks([])
    cbar_ax.yaxis.set_ticks([])
    cbar_ax.set_facecolor('none')
    cbar = fig.colorbar(sm, ax=cbar_ax, orientation="horizontal", label="Ref. step")
    # Customize the colorbar ticks
    tick_locator = ticker.MaxNLocator(nbins=8)
    cbar.locator = tick_locator
    cbar.update_ticks()
    axs[1].bar(param_pbm_nums, meshes_num_refinement)
    axs[1].set(xlabel="Mesh number $l$", ylabel="Num. ref.")
    axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    axs[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
    axs[1].set_xlim(0,len(ls_c_diff) - 1)
    axs[1].set_title("Number of times each mesh is refined")

    fig.subplots_adjust(hspace=0.5)
    fig.savefig(os.path.join(results_dir_str, "plot.pdf"), bbox_inches="tight")

def convergence(s, results_dir_str):
    ls_frac_global_est = np.load(os.path.join(results_dir_str, "frac_global_est.npy"))
    ls_total_dof_num   = np.load(os.path.join(results_dir_str, "total_dof_num.npy"))

    fig = plt.figure()

    fig.suptitle(rf"\Large \textbf{{Convergence of $\eta^{{\mathrm{{bw}}}}_{{\mathcal U}}$ ($s={s}$)}}")
    ax = fig.add_subplot(1, 1, 1)

    # Linear regression on the 10 last values
    fit = np.polyfit(np.log(ls_total_dof_num[-10:]), np.log(ls_frac_global_est[-10:]), 1)
    print(fit)

    ax.loglog(ls_total_dof_num, ls_frac_global_est, "^-", color="red", label=r"$\eta^{\mathrm{bw}}_{\mathcal U}$")
    ax.loglog(ls_total_dof_num, np.exp(fit[1]) * ls_total_dof_num ** fit[0] , "--", color="black", label=rf"$\mathrm{{dof}}^{{{np.round(fit[0],2)}}}$")

    ax.legend()
    ax.set_xlabel("dof")
    ax.set_ylabel(r"$\eta^{\mathrm{bw}}_{\mathcal U}$")
    fig.savefig(os.path.join(results_dir_str, "plot_global_est.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    plt.style.use('ggplot')
    params = {"text.usetex"     : True,
            "font.family"     : "serif",
            "font.serif"      : ["Computer Modern Serif"],
            "ytick.color"     : "black",
            "xtick.color"     : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor"  : "black",}
    plt.rcParams.update(params)

    for s in [0.3, 0.5, 0.7]:
        results_dir_str = f"results_frac_pw_{str(int(10*s))}"
        parametric_details(s, results_dir_str)
        convergence(s, results_dir_str)

