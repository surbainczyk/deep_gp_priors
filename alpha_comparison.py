import os
import pprint
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from MCMC_2D_fractional_op.forward_operators import apply_forward_operator
from MCMC_2D_fractional_op.plotting import plot_comparison_result


def plot_from_dictionaries(alpha_vals, plots_dir):
    errors_dict = {"alpha_vals": []}
    run_times_dict = {}

    print("Opening and processing files...")
    for alpha in alpha_vals:
        err_file = plots_dir + f"errors_a{alpha:.1f}.pickle"
        run_file = plots_dir + f"run_times_a{alpha:.1f}.pickle"
        try:
            with open(err_file, 'rb') as err_f:
                err_obj = pickle.load(err_f)
        except FileNotFoundError:
            print(f"File {err_file} was missing.")
            continue
        try:
            with open(run_file, 'rb') as run_f:
                run_obj = pickle.load(run_f)
        except FileNotFoundError:
            print(f"File {run_file} was missing.")
            continue
            
        errors_dict["alpha_vals"].append(alpha)
        
        errors_dict[f"means_a{alpha:.1f}"] = err_obj[0]
        errors_dict[f"std_a{alpha:.1f}"] = err_obj[1]
        
        run_times_dict[f"means_a{alpha:.1f}"] = run_obj[0]
        run_times_dict[f"std_a{alpha:.1f}"] = run_obj[1]

    print("Plotting...")
    plot_comparison_result(errors_dict, run_times_dict, plots_dir)

    print("Finished plotting.")


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


def run_comp_cost_experiment(alpha, sigma, DeepGPClass, MCMCSolverClass, parameters, iter_counts, reps, plots_dir):
    # unpack parameters
    keys = ["true_img", "forward_op", "obs_noise_std", "burn_in", "rescale_obs"]
    (true_img, forward_op, obs_noise_std, burn_in, rescale_obs) = [parameters[key] for key in keys]
    np.random.seed(0)

    print(f"\nRunning with alpha={alpha:.1f}...")

    deep_gp = DeepGPClass(layers=parameters["layers"], layer_n_dof=parameters["n_dof"], F=parameters["F"],
                    alpha=alpha, base_diag=parameters["base_diag"], sigma=sigma)
    try:
        mcmc_solver = MCMCSolverClass(deep_gp, fix_precond=parameters["fix_precond"])
    except:
        mcmc_solver = MCMCSolverClass(deep_gp)
    
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
    
    errors_list = []
    run_times_list = []
    is_fractional = np.floor(alpha / 2) != alpha / 2
    
    for j in range(reps):
        print(f"\nStarting {j+1}/{reps} runs...")
        mcmc_solver.run_mcmc(its=max(iter_counts), burn_in=burn_in, beta=0.005, accept_rate=0.25,
                            beta_split=int(max(iter_counts) / 1000), compute_ess=False, breakpoint_its=iter_counts)
        
        if is_fractional:
            regr_results = {i: mcmc_solver.regression(mcmc_solver.breakpoint_results[i][0]) for i in iter_counts}
        else:
            regr_results = {i: mcmc_solver.breakpoint_results[i][0] for i in iter_counts}
        
        errors = compute_errors(regr_results, true_img, iter_counts)
        errors_list.append(errors)
        run_times_list.append({it: mcmc_solver.breakpoint_results[it][-1] for it in iter_counts})

    print("Postprocessing...")
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    err_means, err_std, run_means, run_std = process_errors_and_times(errors_list, run_times_list)
    save_errors(err_means, err_std, plots_dir + f"errors_a{alpha:.1f}.csv", plots_dir + f"errors_a{alpha:.1f}.pickle")
    save_run_times(run_means, run_std, plots_dir + f"run_times_a{alpha:.1f}.txt", plots_dir + f"run_times_a{alpha:.1f}.pickle")

    print("Finished. The time is:")
    print(str(datetime.now()).split(".")[0])


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


def save_errors(err_means, err_std, csv_save_as, pck_save_as):
    pck_obj = (err_means, err_std)
    with open(pck_save_as, 'wb') as f:
        pickle.dump(pck_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    iter_counts = err_means.keys()

    l1 = [err_means[it]["l1"] for it in iter_counts]
    l2 = [err_means[it]["l2"] for it in iter_counts]
    p  = [err_means[it]["p"]  for it in iter_counts]
    s  = [err_means[it]["s"]  for it in iter_counts]

    l1_std = [err_std[it]["l1"] for it in iter_counts]
    l2_std = [err_std[it]["l2"] for it in iter_counts]
    p_std  = [err_std[it]["p"]  for it in iter_counts]
    s_std  = [err_std[it]["s"]  for it in iter_counts]

    df_index = [it for it in iter_counts]
    errors_df = pd.DataFrame({"L1 error": l1, "L1 error StD": l1_std, "L2 error": l2, "L2 error StD": l2_std,
                                "PSNR": p, "PSNR StD": p_std, "SSIM": s, "SSIM StD": s_std}, index=df_index).T
    errors_df.to_csv(csv_save_as)


def save_run_times(run_means, run_std, txt_save_as, pck_save_as):
    pck_obj = (run_means, run_std)
    with open(pck_save_as, 'wb') as f:
        pickle.dump(pck_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    is_first_write = True
    iter_counts = run_means.keys()
    for it in iter_counts:
        run_time = run_means[it]
        run_time_std = run_std[it]

        rem_time = run_time
        days = int(np.floor(rem_time / 60 / 60 /24))
        rem_time -= 24 * 60 * 60 * days
        hrs = int(np.floor(rem_time / 60 / 60))
        rem_time -= 60 * 60 * hrs
        mins = int(np.floor(rem_time / 60))
        rem_time -= 60 * mins
        secs = int(rem_time)

        if is_first_write:
            is_first_write = False
            with open(txt_save_as, "w") as file:
                file.write(f"Run time for it={it}: {days} days, {hrs}:{mins}:{secs}\n")
                file.write(f"Exact time in seconds: {run_time} (StD: {run_time_std})\n")
        else:
            with open(txt_save_as, "a") as file:
                file.write(f"\nRun time for it={it}: {days} days, {hrs}:{mins}:{secs}\n")
                file.write(f"Exact time in seconds: {run_time} (StD: {run_time_std})\n")


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
