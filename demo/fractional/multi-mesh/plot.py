import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
import os

s = 0.3
results_dir_str = f"results_frac_pw_{str(int(10*s))}"

meshes_num_refinement                   = np.load(os.path.join(results_dir_str, "meshes_num_refinement.npy"))
global_weighted_parametric_est_history  = np.load(os.path.join(results_dir_str, "global_weighted_parametric_est_history.npy"))
global_parametric_est_history           = np.load(os.path.join(results_dir_str, "global_parametric_est_history.npy"))
ls_c_diff                               = np.load(os.path.join(results_dir_str, "ls_c_diff.npy"))
ls_c_react                              = np.load(os.path.join(results_dir_str, "ls_c_react.npy"))
ls_weights                              = np.load(os.path.join(results_dir_str, "ls_weights.npy"))

param_pbm_nums = np.arange(len(ls_c_diff))
ref_step_max = len(global_parametric_est_history[:,0])

fig, axs = plt.subplots(2)
fig.suptitle(rf"Multi-mesh adaptive refinement algorithm (frac. power $s={s}$).", fontweight="bold")
cmap_full = plt.get_cmap("Blues")
cmap = colors.ListedColormap(cmap_full(np.linspace(0.3, 1, 256)))
for ref_step in range(ref_step_max):
    color_index = (ref_step/(ref_step_max-1))
    color = cmap(color_index)
    axs[0].plot(param_pbm_nums, global_weighted_parametric_est_history[ref_step,:], color=color)
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
axs[0].set_xlim(0,len(ls_c_diff) - 1)
axs[0].set(xlabel="Mesh number", ylabel=r"Weighted $\eta^{\mathrm{bw}}_l$")
axs[0].set_title("Weighted parametric estimators")
norm = plt.Normalize(0,ref_step_max-1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# Customize the colorbar position and appearance
cbar_ax = axs[0].inset_axes([0.59, 0.9, 0.4, 0.4])
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
axs[1].set(xlabel="Mesh number", ylabel="Num. ref.")
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(5))
axs[1].set_xlim(0,len(ls_c_diff) - 1)
axs[1].set_title("Number of refinement steps for each mesh")

fig.subplots_adjust(hspace=0.5)
fig.savefig(os.path.join(results_dir_str, "plot.pdf"), bbox_inches="tight")