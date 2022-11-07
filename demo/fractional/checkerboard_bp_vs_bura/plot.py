import sys
sys.path.append("../")

from plot_utils import (convergence_results,
                        plot_convergence,
                        plot_cumulative_convergence,
                        plot_FE_vs_ra)
output_dir = "./"

convergence_results(output_dir)
plot_convergence(output_dir)
plot_cumulative_convergence(output_dir)
plot_FE_vs_ra(output_dir)