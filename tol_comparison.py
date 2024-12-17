import os
import pprint
import pickle
import numpy as np
from datetime import datetime
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from forward_operators import apply_forward_operator
from plotting import plot_tol_comparison_result


def plot_from_tol_results(tol_vals, plots_dir):
    errors_dict = {"tol_vals": []}

    for tol in tol_vals:
        errors_list = []
        id_no = 0
        while True:
            res_file = plots_dir + f"ind_result_tol_{tol:.0e}_{id_no}.pickle"
            try:
                with open(res_file, 'rb') as res_f:
                    res_obj = pickle.load(res_f)
            except FileNotFoundError:
                break

            errors = res_obj[0]
            errors_list.append(errors)

            id_no += 1
        
        err_means, err_std = process_errors(errors_list)
            
        errors_dict["tol_vals"].append(tol)
        
        errors_dict[f"means_tol_{tol:.0e}"] = err_means
        errors_dict[f"std_tol_{tol:.0e}"] = err_std
    
    err_types = ["l1", "l2", "p", "s"]
    piv_errors = {et: {} for et in err_types}
    for et in err_types:
        piv_errors[et]["mean"]  = np.array([errors_dict[f"means_tol_{tol:.0e}"][et] for tol in tol_vals])
        piv_errors[et]["std"]   = np.array([errors_dict[f"std_tol_{tol:.0e}"][et] for tol in tol_vals])
        piv_errors[et]["lower"] = piv_errors[et]["mean"] - 2 * piv_errors[et]["std"]
        piv_errors[et]["upper"] = piv_errors[et]["mean"] + 2 * piv_errors[et]["std"]

    print("Plotting...")
    plot_tol_comparison_result(piv_errors, tol_vals, plots_dir)

    print("Finished.")


def process_errors(errors_list):
    err_means = {}
    err_std = {}
    for err_type in ["l1", "l2", "p", "s"]:
        err_means[err_type] = np.mean([errors_list[i][err_type] for i in range(len(errors_list))])
        err_std[err_type] = np.std([errors_list[i][err_type] for i in range(len(errors_list))])
    
    return err_means, err_std


def run_tol_experiment(mcmc_solver, parameters, tol, id_no, plots_dir):
    # unpack parameters
    keys = ["true_img", "forward_op", "obs_noise_std", "its", "burn_in", "rescale_obs"]
    (true_img, forward_op, obs_noise_std, its, burn_in, rescale_obs) = [parameters[key] for key in keys]

    print("Parameters:")
    pprint.pprint(parameters)
    print(f"\nLSQR tol:  {tol:.0e}")
    print(f"ID number:     {id_no}")
    
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
    mcmc_solver.old_solver.tol = mcmc_solver.prop_solver.tol = tol
    
    np.random.seed(id_no)    # fix MCMC samples for each ID number
    mcmc_solver.run_mcmc(its=its, burn_in=burn_in, beta=0.005, accept_rate=0.25, beta_split=int(its / 1000))
    
    print("Postprocessing...")
    if not os.path.isdir(plots_dir):
        os.mkdir(plots_dir)
    
    compute_and_save_errors(mcmc_solver, true_img, scale, shift, plots_dir + f"ind_result_tol_{tol:.0e}_{id_no}.pickle")

    print("Finished. The time is:")
    print(str(datetime.now()).split(".")[0])


def compute_and_save_errors(mcmc_solver, true_img, scale, shift, save_as):
    sol = mcmc_solver.regression(mcmc_solver.F_mean)
    sol = scale * sol + shift

    l1 = np.mean(np.abs(sol - true_img))
    l2 = np.sqrt(mean_squared_error(true_img, sol))
    p  = peak_signal_noise_ratio(true_img, sol, data_range=true_img.max() - true_img.min())
    s  = structural_similarity(true_img, sol, data_range=true_img.max() - true_img.min())

    errors = {"l1": l1, "l2": l2, "p": p, "s": s}
    
    pck_obj = (errors, sol, true_img)
    with open(save_as, 'wb') as f:
        pickle.dump(pck_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved results to: {save_as}")
