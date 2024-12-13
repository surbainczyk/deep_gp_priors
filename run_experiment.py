import numpy as np
import os
import pprint
from datetime import datetime

from pCN_nonfractional.GP_models.DeepGP import DeepGP as DeepGPNonfractional
from pCN_nonfractional.pCNSampler import pCNSampler
from pCN_det_free_fractional.GP_models.DeepGP import DeepGP as DeepGPFractional
from pCN_det_free_fractional.pCNDetFreeSampler import pCNDetFreeSampler
from pCN_nonfractional.GP_models.StandardGP import StandardGP
from forward_operators import apply_forward_operator
from estimate_corr_length import estimate_corr_length
from utils import compute_statistics, augment_statistics, save_statistics, print_and_save_error_metrics, save_run_time
from plotting import (save_flattened_image, save_flattened_image_with_obs_locations, save_image_with_errors,
                                            plot_mcmc_result, plot_multilayer_mcmc_result, plot_potential_values, plot_acceptance_history,
                                            save_radon_observations, plot_errors)


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


def standard_gp_regression(gp, mcmc_solver, rho, scale, shift):
    diag = (2 * gp.nu / rho ** 2) * np.ones(gp.n_dof)
    try:
        Q, diag_mat, log_det_QDQ = gp.compute_rat_approx_C_inv(diag)
        _, gp_mean_norm = mcmc_solver.potential_and_regression(Q, log_det_QDQ, diag_mat)
    except ValueError:    # using a gp and mcmc_solver without logdet output/input
        Q, diag_mat = gp.compute_rat_approx_C_inv(diag)
        _, gp_mean_norm = mcmc_solver.potential_and_regression(Q, diag_mat)
    except AttributeError:
        gp_mean_norm = mcmc_solver.regression(diag)
    gp_mean = scale * gp_mean_norm + shift

    return gp_mean


def run_experiment(parameters):
    # unpack parameters
    keys = ["n_dof", "true_img", "forward_op", "obs_shape", "obs_noise_std",
            "rho_vals", "its", "burn_in", "plots_dir", "plot_obs", "set_vrange", "rescale_obs"]
    (n_dof, true_img, forward_op, obs_shape, obs_noise_std,
     rho_vals, its, burn_in, plots_dir, plot_obs, set_vrange, rescale_obs) = [parameters[key] for key in keys]
    
    deep_gp, mcmc_solver = initialise_deep_gp_and_sampler(parameters)

    # normalise observations
    np.random.seed(0)
    observations = apply_forward_operator(forward_op, true_img, obs_noise_std)
    if rescale_obs:
        scale = np.std(observations)
        shift = np.mean(observations)
        normalised_obs = (observations - shift) / scale
    else:
        scale = 1; shift = 0; normalised_obs = observations

    # set up MCMC variables
    print("Parameters:")
    pprint.pprint(parameters)

    mcmc_solver.initialise_observations(normalised_obs, forward_op, obs_noise_std / scale)
    try:
        mcmc_solver.old_solver.tol = mcmc_solver.prop_solver.tol = parameters["lsqr_tol"]
    except:
        pass

    # run MCMC
    print("Running MCMC...")
    mcmc_solver.run_mcmc(its=its, burn_in=burn_in, beta=0.005, accept_rate=0.25, beta_split=int(its / 1000))

    # perform standard GP regression
    if np.floor(parameters['alpha']) == np.ceil(parameters['alpha']):
        standard_gp = StandardGP(n_dof, int(parameters['alpha']), parameters['sigma'])
        opt_rho = estimate_corr_length(standard_gp, normalised_obs, forward_op, (obs_noise_std / scale) ** 2)
        rho_vals.append(opt_rho)
        print(f"Estimated optimal rho was: {opt_rho}")
        
        standard_gp_vals = [scale * standard_gp.regression_mean(rho, normalised_obs, forward_op, (obs_noise_std / scale) ** 2) + shift
                            for rho in rho_vals]
    else:
        standard_gp_vals = [standard_gp_regression(deep_gp.top_layer, mcmc_solver, rho, scale, shift) for rho in rho_vals]

    # compute errors/statistics
    print("Computing errors/statistics...")
    statistics = compute_statistics(mcmc_solver, n_dof, shift, scale)
    diag_mean  = deep_gp.F(statistics[f"layer_{deep_gp.layers-1}_mean"])
    try:
        Q_mean, diag_mat_mean, log_det_QDQ_mean = deep_gp.top_layer.compute_rat_approx_C_inv(diag_mean)
        _, gp_vals_mean_norm = mcmc_solver.potential_and_regression(Q_mean, log_det_QDQ_mean, diag_mat_mean)
    except ValueError:    # using a gp and mcmc_solver without logdet output/input
        Q_mean, diag_mat_mean = deep_gp.top_layer.compute_rat_approx_C_inv(diag_mean)
        _, gp_vals_mean_norm = mcmc_solver.potential_and_regression(Q_mean, diag_mat_mean)
    except AttributeError:
        gp_vals_mean_norm = mcmc_solver.regression(diag_mean)
    gp_vals_mean = scale * gp_vals_mean_norm + shift
    augment_statistics(statistics, true_img, standard_gp_vals, rho_vals, gp_vals_mean, observations)
    top_mean = scale * statistics["top_layer_mean"] + shift
    top_map  = scale * statistics["top_layer_MAP"] + shift
    
    # save results
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)

    print_and_save_error_metrics(true_img, standard_gp_vals + [top_mean, top_map], rho_vals, plots_dir + "errors.csv")
    save_run_time(mcmc_solver.run_time, plots_dir + "run_time.txt")

    print(f"Saving results to {plots_dir}...")
    save_statistics(statistics, plots_dir + "stats.pickle")
    if set_vrange:
        vmin = min([np.amin(true_img), np.amin(np.array(standard_gp_vals)),
                    np.amin(gp_vals_mean), np.amin(statistics["top_layer_mean_scaled"])])
        vmax = max([np.amax(true_img), np.amax(np.array(standard_gp_vals)),
                    np.amax(gp_vals_mean), np.amax(statistics["top_layer_mean_scaled"])])
        vrange = (vmin, vmax)
    else:
        vrange = None

    save_flattened_image(true_img, plots_dir + "true_img.pdf", vrange=vrange)
    if plot_obs:
        save_flattened_image_with_obs_locations(true_img, forward_op, plots_dir + "true_img_with_obs.pdf", vrange=vrange)
    
    if 'radon' in plots_dir:
        save_radon_observations(observations, plots_dir + "observations.pdf", shape=obs_shape)
    else:
        save_flattened_image(observations, plots_dir + "observations.pdf", shape=obs_shape)

    save_image_with_errors(statistics["top_layer_mean_scaled"], true_img, plots_dir + "output_img.pdf", vrange=vrange)
    save_flattened_image(statistics["layer_0_F_sqrt_mean"], plots_dir + "inner_F_sqrt_mean.pdf")
    save_flattened_image(statistics["top_layer_std"], plots_dir + "top_layer_std.pdf")
    save_flattened_image(statistics["layer_0_F_std"], plots_dir + "inner_F_std.pdf")
    save_image_with_errors(statistics["top_layer_MAP_scaled"], true_img, plots_dir + "output_img_MAP.pdf", vrange=vrange)
    save_flattened_image(statistics["layer_0_F_sqrt_MAP"], plots_dir + "inner_F_sqrt_MAP.pdf")

    incl_or = np.floor(parameters['alpha']) == np.ceil(parameters['alpha'])
    plot_errors(statistics['true_img'], statistics['top_layer_mean_scaled'], statistics['standard_gp_vals'], statistics['rho_vals'],
                plots_dir, include_opt_rho=incl_or)
    
    if deep_gp.layers == 1:
        plot_mcmc_result(statistics, plots_dir + "result.pdf")
    else:
        plot_multilayer_mcmc_result(statistics, deep_gp.layers, plots_dir + "result.pdf")

    attr_names = ["potential_vals", "aux_potential_vals", "comb_potential_vals", "beta_vals", "logdet_diffs", "potential_its", "auxiliary_its"]
    for name in attr_names:
        try:
            plot_potential_values(getattr(mcmc_solver, name), plots_dir + name + ".pdf", burn_in=burn_in)
        except AttributeError:
            pass
    
    try:
        plot_acceptance_history(mcmc_solver.accepted_hist, plots_dir + "acceptance.pdf")
        middle_idx = int(n_dof / 2 + np.sqrt(n_dof) / 2)
        plot_potential_values(mcmc_solver.proposals_array[middle_idx, :], plots_dir + "trace_of_proposal.pdf", logscale=False)
        print(f"Mean iteration counts for potential evaluation: {np.mean(mcmc_solver.potential_its):.4f} (StD: {np.std(mcmc_solver.potential_its):.4f})")
        print(f"Mean iteration counts for auxiliary variable:   {np.mean(mcmc_solver.auxiliary_its):.4f} (StD: {np.std(mcmc_solver.auxiliary_its):.4f})")
    except AttributeError:
        pass
    
    for i, rho in enumerate(rho_vals):
        gp_name = "gp_mean_" + f"{rho:.2f}"[2:] + ".pdf"
        save_image_with_errors(standard_gp_vals[i],  true_img, plots_dir + gp_name, vrange=vrange)

    print("Finished. The time is:")
    print(str(datetime.now()).split(".")[0])
