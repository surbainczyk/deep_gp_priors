import numpy as np
import pickle

from MCMC_2D_fractional_op.plotting import save_flattened_image


def load_statistics(file_name):
    with open(file_name, 'rb') as f:
        stats = pickle.load(f)
    
    return stats


def get_correlation_length(sqrt_diag, alpha):
    # compute correlation lengths
    nu = alpha - 1
    rho = np.sqrt(2 * nu) / sqrt_diag

    return rho


# Script for plotting comparison from saved files
problem_dir_name = 'interpolation'

plots_dir = f"MCMC_2D_fractional_op/{problem_dir_name}/plots_comp/"

directories = [f'pCN_det_free_2D_rational/{problem_dir_name}/plots_a1_5/',
               f'MCMC_2D_fractional_op/{problem_dir_name}/plots_a2/',
               f'pCN_det_free_2D_rational/{problem_dir_name}/plots_a2_5/',
               f'pCN_det_free_2D_rational/{problem_dir_name}/plots/',
               f'pCN_det_free_2D_rational/{problem_dir_name}/plots_a3_5/',
               f'MCMC_2D_fractional_op/{problem_dir_name}/plots/']

alpha_vals = [1.5, 2, 2.5, 3, 3.5, 4]

corr_length_list = []
for j, dir in enumerate(directories):
    # load statistics
    file_name = dir + "stats.pickle"
    print(f'Loading {file_name}...')
    stats = load_statistics(file_name)
    
    sqrt_diag = stats["layer_0_F_sqrt_mean"]
    corr_length = get_correlation_length(sqrt_diag, alpha_vals[j])

    corr_length_list.append(corr_length)

print("Plotting...")
vmin = min([np.amin(cl) for cl in corr_length_list])
vmax = max([np.amax(cl) for cl in corr_length_list])
vrange = (vmin, vmax)

for j, cl in enumerate(corr_length_list):
    alpha_appdx = '_a' + str(alpha_vals[j]).replace('.', '_') + '.pdf'
    save_flattened_image(cl, plots_dir + 'corr_length' + alpha_appdx, figsize=(2.8, 2.8), vrange=vrange,
                         ticks_off=True, plot_cbar_to=plots_dir + 'corr_length_cbar.pdf')

print("Finished.")
