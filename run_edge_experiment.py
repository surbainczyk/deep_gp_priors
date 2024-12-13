import numpy as np
import os
import pprint
from datetime import datetime

from pCN_nonfractional.GP_models.DeepGP import DeepGP as DeepGPNonfractional
from pCN_nonfractional.pCNSampler import pCNSampler
from pCN_det_free_fractional.GP_models.DeepGP import DeepGP as DeepGPFractional
from pCN_det_free_fractional.pCNDetFreeSampler import pCNDetFreeSampler
from pCN_nonfractional.GP_models.StandardGP import StandardGP
from estimate_corr_length import estimate_corr_length
from utils import compute_statistics, augment_edge_statistics, print_and_save_label_errs, save_statistics
from plotting import plot_edge_reconstruction_result, save_flattened_image, save_flattened_image_with_obs_locations, \
    save_image_with_errors, plot_potential_values, plot_acceptance_history, plot_edge_errors


def initialise_deep_gp_and_sampler(params):
    keys = ["n_dof", "layers", "F", "alpha", "base_diag", "sigma"]
    n_dof, layers, F, alpha, base_diag, sigma = [params[key] for key in keys]

    if (params['alpha'] / 2).is_integer():
        dgp = DeepGPNonfractional(layers=layers, layer_n_dof=n_dof, F=F, alpha=alpha, base_diag=base_diag, sigma=sigma)
        mcmc = pCNSampler(dgp)
    else:
        dgp = DeepGPFractional(layers=layers, layer_n_dof=n_dof, F=F, alpha=alpha, base_diag=base_diag, sigma=sigma)
        mcmc = pCNDetFreeSampler(dgp, fix_precond=params["fix_precond"])

    return dgp, mcmc


def standard_gp_regression(gp, mcmc_solver, rho):
    diag = (2 * gp.nu / rho ** 2) * np.ones(gp.n_dof)
    try:
        Q, diag_mat, log_det_QDQ = gp.compute_rat_approx_C_inv(diag)
        _, gp_mean = mcmc_solver.potential_and_regression(Q, log_det_QDQ, diag_mat)
    except AttributeError:
        gp_mean = mcmc_solver.regression(diag)

    return gp_mean


def run_edge_experiment(parameters):
    # unpack parameters
    print("Parameters:")
    pprint.pprint(parameters)

    keys = ["n_dof", "true_img", "true_edges", "forward_op", "obs", "obs_noise_std", "rho_vals", "its", "burn_in", "plot_obs", "plots_dir"]
    (n_dof, true_img, true_edges, forward_op, obs, obs_noise_std, rho_vals, its, burn_in, plot_obs, plots_dir) = [parameters[key] for key in keys]

    deep_gp, mcmc_solver = initialise_deep_gp_and_sampler(parameters)
    mcmc_solver.initialise_observations(obs, forward_op, obs_noise_std)

    print("Running MCMC...")
    mcmc_solver.run_mcmc(its=its, burn_in=burn_in, initial_state=None, beta=0.005, accept_rate=0.25, beta_split=int(its / 1000))
    statistics = compute_statistics(mcmc_solver, n_dof)

    print("Computing additional statistics...")
    # estimate edge location with GPs
    if np.floor(parameters['alpha']) == np.ceil(parameters['alpha']):
        standard_gp = StandardGP(n_dof, int(parameters['alpha']), parameters['sigma'])
        opt_rho = estimate_corr_length(standard_gp, parameters['obs'], forward_op, parameters['obs_noise_std'] ** 2)
        rho_vals.append(opt_rho)
        print(f"Estimated optimal rho was: {opt_rho}")
        
        standard_gp_vals = [standard_gp.regression_mean(rho, parameters['obs'], forward_op, parameters['obs_noise_std'] ** 2)
                            for rho in rho_vals]
    else:
        standard_gp_vals = [standard_gp_regression(deep_gp.top_layer, mcmc_solver, rho) for rho in rho_vals]

    # compare and plot results
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

    augment_edge_statistics(statistics, true_img, true_edges, standard_gp_vals, rho_vals)
    print_and_save_label_errs(statistics, plots_dir + "errors.csv")

    print(f"Saving results to {plots_dir}...")
    save_statistics(statistics, plots_dir + "stats.pickle")

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
    
    plot_edge_reconstruction_result(statistics, plots_dir + "result.pdf", edge_coords=[[0.5, 0.5], [0, 1]])

    attr_names = ["potential_vals", "aux_potential_vals", "comb_potential_vals", "beta_vals", "logdet_diffs", "potential_its", "auxiliary_its"]
    for name in attr_names:
        try:
            plot_potential_values(getattr(mcmc_solver, name), plots_dir + name + ".pdf")
        except AttributeError:
            pass
    try:
        plot_acceptance_history(mcmc_solver.accepted_hist, plots_dir + "acceptance.pdf")
        middle_idx = int(n_dof / 2 + np.sqrt(n_dof) / 2)
        plot_potential_values(mcmc_solver.proposals_array[middle_idx, :], plots_dir + "trace_of_proposal.pdf", logscale=False)
    except AttributeError:
        pass

    print("Finished. The time is:")
    print(str(datetime.now()).split(".")[0])
