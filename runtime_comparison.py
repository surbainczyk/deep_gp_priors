import os
import pprint
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from forward_operators import apply_forward_operator
from plotting import plot_comparison_result


def plot_from_ind_results(alpha_vals, iter_counts, plots_dir):
    errors_dict = {"alpha_vals": []}
    run_times_dict = {}

    for alpha in alpha_vals:
        errors_list = []
        run_times_list = []
        id_no = 0
        while True:
            res_file = plots_dir + f"ind_result_a{alpha:.1f}_{id_no}.pickle"
            run_file = plots_dir + f"ind_run_times_a{alpha:.1f}_{id_no}.pickle"
            try:
                with open(res_file, 'rb') as res_f:
                    res_obj = pickle.load(res_f)
            except FileNotFoundError:
                break
            try:
                with open(run_file, 'rb') as run_f:
                    run_obj = pickle.load(run_f)
            except FileNotFoundError:
                break

            try:
                (regr_results, _, true_img) = res_obj
            except ValueError:    # old format
                (regr_results, true_img) = res_obj
            errors = compute_errors(regr_results, true_img, iter_counts)
            errors_list.append(errors)
            run_times_list.append(run_obj)

            id_no += 1
        
        err_means, err_std, run_means, run_std = process_errors_and_times(errors_list, run_times_list)
            
        errors_dict["alpha_vals"].append(alpha)
        
        errors_dict[f"means_a{alpha:.1f}"] = err_means
        errors_dict[f"std_a{alpha:.1f}"] = err_std
        
        run_times_dict[f"means_a{alpha:.1f}"] = run_means
        run_times_dict[f"std_a{alpha:.1f}"] = run_std

    print("Plotting...")
    plot_comparison_result(errors_dict, run_times_dict, plots_dir)

    print("Finished.")


def compute_errors(regr_results, true_img, iter_counts):
    errors = {}
    for i in iter_counts:
        sol = regr_results[i]

        l1 = np.mean(np.abs(sol - true_img))
        l2 = np.sqrt(mean_squared_error(true_img, sol))
        p  = peak_signal_noise_ratio(true_img, sol, data_range=true_img.max() - true_img.min())
        s  = structural_similarity(true_img, sol, data_range=true_img.max() - true_img.min())

        errors[i] = {"l1": l1, "l2": l2, "p": p, "s": s}
    
    return errors


def process_errors_and_times(errors_list, run_times_list):
    err_means = {it: {} for it in errors_list[0].keys()}
    err_std = {it: {} for it in errors_list[0].keys()}
    for it in errors_list[0].keys():
        for err_type in ["l1", "l2", "p", "s"]:
            err_means[it][err_type] = np.mean([errors_list[i][it][err_type] for i in range(len(errors_list))])
            err_std[it][err_type] = np.std([errors_list[i][it][err_type] for i in range(len(errors_list))])

    run_means = {}
    run_std = {}
    for it in run_times_list[0].keys():
        run_means[it] = np.mean([run_times_list[i][it] for i in range(len(run_times_list))])
        run_std[it] = np.std([run_times_list[i][it] for i in range(len(run_times_list))])
    
    return err_means, err_std, run_means, run_std


def run_ind_experiment(DeepGPClass, MCMCSolverClass, parameters, iter_counts, id_no, plots_dir):
    # unpack parameters
    keys = ["true_img", "forward_op", "obs_noise_std", "burn_in", "rescale_obs"]
    (true_img, forward_op, obs_noise_std, burn_in, rescale_obs) = [parameters[key] for key in keys]

    print("Parameters:")
    pprint.pprint(parameters)
    print(f"\nID number: {id_no}")

    deep_gp = DeepGPClass(layers=parameters["layers"], layer_n_dof=parameters["n_dof"], F=parameters["F"],
                          alpha=parameters['alpha'], base_diag=parameters["base_diag"], sigma=parameters['sigma'])
    try:
        mcmc_solver = MCMCSolverClass(deep_gp, fix_precond=parameters["fix_precond"])
    except:
        mcmc_solver = MCMCSolverClass(deep_gp)
    
    np.random.seed(0)
    observations = apply_forward_operator(forward_op, true_img, obs_noise_std)
    # normalise observations
    if rescale_obs:
        scale = np.std(observations)
        shift = np.mean(observations)
        normalised_obs = (observations - shift) / scale
    else:
        scale = 1; shift = 0; normalised_obs = observations
    
    mcmc_solver.initialise_observations(normalised_obs, forward_op, obs_noise_std / scale)
    try:
        mcmc_solver.old_solver.tol = mcmc_solver.prop_solver.tol = parameters["lsqr_tol"]
    except:
        pass
    
    np.random.seed(id_no)    # fix MCMC samples for each ID number
    mcmc_solver.run_mcmc(its=max(iter_counts), burn_in=burn_in, beta=0.005, accept_rate=0.25,
                        beta_split=20, compute_ess=False, breakpoint_its=iter_counts)
    
    print("Postprocessing...")
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    
    alpha = parameters['alpha']
    save_ind_result(mcmc_solver, true_img, iter_counts, alpha, scale, shift, plots_dir + f"ind_result_a{alpha:.1f}_{id_no}.pickle")
    save_run_times_dict(mcmc_solver, iter_counts, plots_dir + f"ind_run_times_a{alpha:.1f}_{id_no}.pickle")

    print("Finished. The time is:")
    print(str(datetime.now()).split(".")[0])


def save_ind_result(mcmc_solver, true_img, iter_counts, alpha, scale, shift, pck_save_as):
    inner_means  = {}
    regr_results = {}
    is_fractional = np.floor(alpha / 2) != alpha / 2
    for i in iter_counts:
        F_mean = mcmc_solver.breakpoint_results[i][1]
        if is_fractional:
            sol = mcmc_solver.regression(F_mean)
        else:
            sol = mcmc_solver.breakpoint_results[i][0]
        regr = scale * sol + shift
        
        inner_means[i]  = F_mean
        regr_results[i] = regr
    
    pck_obj = (regr_results, true_img)
    with open(pck_save_as, 'wb') as f:
        pickle.dump(pck_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to: {pck_save_as}")


def save_run_times_dict(mcmc_solver, iter_counts, pck_save_as):
    run_dict = {it: mcmc_solver.breakpoint_results[it][-1] for it in iter_counts}

    with open(pck_save_as, 'wb') as f:
        pickle.dump(run_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved run times to: {pck_save_as}")
