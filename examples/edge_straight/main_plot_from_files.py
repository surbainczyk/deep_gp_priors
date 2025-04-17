import numpy as np
import pickle
from datetime import datetime

from examples.edge_straight.parameters import parameters
from run_edge_experiment import initialise_deep_gp_and_sampler
from plotting import *

# Plot from files saved during previous run of main_edge_detection.py

plots_dir = "examples/edge_straight/plots/"

keys = ["true_img", "true_edges", "forward_op", "obs", "obs_noise_std", "burn_in", "plot_obs", "plot_uq"]
(true_img, true_edges, forward_op, obs, obs_noise_std, burn_in, plot_obs, plot_uq) = [parameters[key] for key in keys]

deep_gp, mcmc_solver = initialise_deep_gp_and_sampler(parameters)
mcmc_solver.initialise_observations(obs, forward_op, obs_noise_std)

# recover data from saved files
with open(plots_dir + 'stats.pickle', 'rb') as stats_f:
    statistics = pickle.load(stats_f)
with np.load(plots_dir + 'iterates.npz') as prop_f:
    mcmc_solver.prop_array = prop_f['prop_array']
try:
    with np.load(plots_dir + 'class_samples.npz') as class_f:
        class_samples = class_f['class_samples']
except FileNotFoundError:
    class_samples = None

figsize = (2.8, 2.8)
if plot_obs:
    save_flattened_image_with_obs_locations(true_img, forward_op, plots_dir + "true_img_with_obs.pdf", figsize=figsize)
else:
    save_flattened_image(true_img, plots_dir + "true_img.pdf", figsize=figsize)
save_flattened_image(true_edges, plots_dir + "true_edges.pdf", figsize=figsize)
save_image_with_errors(statistics["top_layer_mean_scaled"], true_img, plots_dir + "output_img.pdf", figsize=figsize)
keys = ["layer_0_F_sqrt_mean", "deep_gp_sol", "deep_gp_sol_ls", "deep_gp_edges", "deep_gp_ls_edges", "sobel_img"]
for key in keys:
    save_flattened_image(statistics[key], plots_dir + key + ".pdf", figsize=figsize)

plot_edge_errors(statistics, plots_dir)

if plot_uq:
    plot_edge_uq_results(true_img, mcmc_solver, burn_in, plots_dir, figsize=figsize, class_samples=class_samples)

print("Finished. The time is:")
print(str(datetime.now()).split(".")[0])
