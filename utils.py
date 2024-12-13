import numpy as np
import numba as nb
import pandas as pd
import pickle

from scipy.optimize import minimize_scalar
from skimage import filters
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom


@nb.njit((nb.i4[:], nb.i4[:], nb.f8[:]), parallel=True)
def numba_sort_indices(indptr, indices, data):
    for i in nb.prange(indptr.size - 1):
        col_indices = indices[indptr[i]:indptr[i+1]]
        col_data = data[indptr[i]:indptr[i+1]]

        idx = np.argsort(col_indices)
        indices[indptr[i]:indptr[i+1]] = col_indices[idx]
        data[indptr[i]:indptr[i+1]] = col_data[idx]


def interpolation_initial_state(observations, deep_gp):
    obs_width = int(np.sqrt(observations.size))
    obs_img = np.reshape(observations, (obs_width, obs_width))
    edges = filters.roberts(obs_img)

    scaling = np.sqrt(deep_gp.layer_n_dof / observations.size)
    scaled_grad_norm = zoom(edges, scaling).flatten()

    # perform forward pass with base layer's diff. operator
    desired_u_0 = 10 * scaled_grad_norm    # using 10 to obtain larger values
    diag_vector = 400 * np.ones(deep_gp.layer_n_dof)
    initial_state = deep_gp.base_layer.evaluate_forward_pass(diag_vector, desired_u_0)

    return initial_state


def compute_statistics(mcmc_obj, layer_n_dof, shift=0.0, scale=1.0):
    stats = {}

    layers = mcmc_obj.deep_gp.layers
    F = mcmc_obj.deep_gp.F

    # compute mean of u_l and resulting values of F for each layer
    for i in range(layers):
        i_lower = i * layer_n_dof
        i_upper = (i + 1) * layer_n_dof
        stats[f"layer_{i}_mean"] = mcmc_obj.u_mean[i_lower:i_upper]
        
        stats[f"layer_{i}_mean_F_sqrt"] = np.sqrt(F(stats[f"layer_{i}_mean"]))

    # compute statistics of F(u_l) for each layer
    for i in range(layers):
        i_lower = i * layer_n_dof
        i_upper = (i + 1) * layer_n_dof
        stats[f"layer_{i}_F_sqrt_mean"] = np.sqrt(mcmc_obj.F_mean[i_lower:i_upper])
        stats[f"layer_{i}_F_sqrt_MAP"]  = np.sqrt(mcmc_obj.F_map[i_lower:i_upper])
        stats[f"layer_{i}_F_var"] = mcmc_obj.F_var[i_lower:i_upper]
        stats[f"layer_{i}_F_std"] = np.sqrt(mcmc_obj.F_var[i_lower:i_upper])

    # compute statistics of top layer mean values
    try:
        stats["top_layer_mean"] = mcmc_obj.regr_mean
        stats["top_layer_MAP"]  = mcmc_obj.regr_map
        stats["top_layer_mean_scaled"] = scale * mcmc_obj.regr_mean + shift
        stats["top_layer_MAP_scaled"]  = scale * mcmc_obj.regr_map  + shift
        stats["top_layer_var"] = mcmc_obj.regr_var
        stats["top_layer_std"] = np.sqrt(mcmc_obj.regr_var)
    except AttributeError:
        stats["top_layer_mean"] = mcmc_obj.regression(mcmc_obj.F_mean[i_lower:i_upper])
        stats["top_layer_MAP"]  = mcmc_obj.regression(mcmc_obj.F_map[i_lower:i_upper])
        stats["top_layer_mean_scaled"] = scale * stats["top_layer_mean"] + shift
        stats["top_layer_MAP_scaled"]  = scale * stats["top_layer_MAP"]  + shift
        stats["top_layer_var"] = np.zeros(1)
        stats["top_layer_std"] = np.zeros(1)

    try:
        stats["last_state"] = mcmc_obj.last_state
    except AttributeError:
        pass
    
    return stats


def augment_statistics(stats, true_img, standard_gp_vals, rho_vals, gp_mean, observations):
    stats["true_img"] = true_img
    stats["standard_gp_vals"]  = standard_gp_vals
    stats["rho_vals"] = rho_vals
    stats["gp_mean"]  = gp_mean
    stats["observations"]  = observations


def save_statistics(stats, save_as):
    with open(save_as, "wb") as file:
        pickle.dump(stats, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_run_time(run_time, save_as):
    rem_time = run_time
    days = int(np.floor(rem_time / 60 / 60 /24))
    rem_time -= 24 * 60 * 60 * days
    hrs = int(np.floor(rem_time / 60 / 60))
    rem_time -= 60 * 60 * hrs
    mins = int(np.floor(rem_time / 60))
    rem_time -= 60 * mins
    secs = int(rem_time)
    with open(save_as, "w") as file:
        file.write(f"MCMC finished after {days} days, {hrs}:{mins}:{secs}\n")
        file.write(f"Exact time in seconds: {run_time}\n")


def H_1_error(ground_truth, sol):
    sq_l2_err = np.mean((sol - ground_truth) ** 2)    # integrating over 1x1 square

    width = int(np.sqrt(ground_truth.size))

    err = np.reshape(sol - ground_truth, (width, width))
    grad_x = (err[:, 1:] - err[:, :-1]) * width
    grad_x = 0.5 * grad_x[1:, :] + 0.5 * grad_x[:-1, :]    # interpolate in y-direction
    grad_y = (err[1:, :] - err[:-1, :]) * width
    grad_y = 0.5 * grad_y[:, 1:] + 0.5 * grad_y[:, :-1]    # interpolate in x-direction
    sq_h01_err = np.mean(grad_x ** 2 + grad_y ** 2)

    h_1_err = np.sqrt(sq_l2_err + sq_h01_err)

    return h_1_err


def print_and_save_error_metrics(ground_truth, sol_list, rho_vals, save_as):
    l1_err = []
    mse  = []
    psnr = []
    ssim = []

    for sol in sol_list:
        l1 = np.mean(np.abs(sol - ground_truth))    # integrating over 1x1 square
        l1_err.append(l1)
        
        m = mean_squared_error(ground_truth, sol)
        mse.append(m)
        
        p = peak_signal_noise_ratio(ground_truth, sol, data_range=ground_truth.max() - ground_truth.min())
        psnr.append(p)
        
        s = structural_similarity(ground_truth, sol, data_range=ground_truth.max() - ground_truth.min())
        ssim.append(s)
    
    l2_err = [np.sqrt(err) for err in mse]
    
    # print results
    first_line = 12 * " " + "".join([f"GP (rho={rho:.2f}) | " for rho in rho_vals]) + "deep GP (mean) | deep GP (MAP)"
    l1_line   = "L1 error: " + "".join([f"{err:>15.4f} " for err in l1_err])
    l2_line   = "L2 error: " + "".join([f"{err:>15.4f} " for err in l2_err])
    mse_line  = "MSE:      " + "".join([f"{err:>15.4f} " for err in mse])
    psnr_line = "PSNR:     " + "".join([f"{err:>15.4f} " for err in psnr])
    ssim_line = "SSIM:     " + "".join([f"{err:>15.4f} " for err in ssim])

    line_list = [first_line, l1_line, l2_line, mse_line, psnr_line, ssim_line]

    # add missing space
    for i in range(len(line_list) - 1):
        line = line_list[i + 1]
        line_list[i + 1] = line[:-32] + "|" + line[-32:]

    for line in line_list:
        print(line)
    
    df_index = [f"GP (rho={rho:.2f})" for rho in rho_vals] + ["deep GP (mean)", "deep GP (MAP)"]
    errors_df = pd.DataFrame({"L1 error": l1_err, "L2 error": l2_err, "MSE": mse, "PSNR": psnr, "SSIM": ssim}, index=df_index).T
    errors_df.to_csv(save_as)
    
    tex_save_as = save_as[:-4] + "_tex.csv"
    tex_l1_err = l1_err[:-3] + [l1_err[-2]]
    tex_l2_err = l2_err[:-3] + [l2_err[-2]]
    tex_psnr = psnr[:-3] + [psnr[-2]]
    tex_ssim = ssim[:-3] + [ssim[-2]]
    tex_df_index = [f"GP (rho={rho:.2f})" for rho in rho_vals[:-1]] + ["deep GP (mean)"]
    tex_errors_df = pd.DataFrame({"L1 error": tex_l1_err, "L2 error": tex_l2_err, "PSNR": tex_psnr, "SSIM": tex_ssim}, index=tex_df_index)
    tex_errors_df.to_csv(tex_save_as)


def augment_edge_statistics(stats, true_img, true_edges, standard_gp_vals, rho_vals):
    top_layer_mean = stats["top_layer_mean"]
    width = int(np.sqrt(top_layer_mean.size))
    sobel_img = filters.sobel(top_layer_mean.reshape(width, width)).flatten()

    deep_gp_sol = threshold(top_layer_mean)
    deep_gp_sol_ls = threshold_with_length_scale(stats["layer_0_F_sqrt_mean"])
    gp_sols = [threshold(gp_mean) for gp_mean in standard_gp_vals]

    # compute optimal edge maps w.r.t. F-score
    deep_gp_edges, deep_gp_f_score = optimal_f_score_edge_map(top_layer_mean, true_edges)
    deep_gp_ls_edges, deep_gp_ls_f_score = optimal_f_score_edges_from_length_scale(stats["layer_0_F_sqrt_mean"], true_edges)
    gp_f_sols = [optimal_f_score_edge_map(gp_mean, true_edges) for gp_mean in standard_gp_vals]
    gp_edges = [gp_f_sol[0] for gp_f_sol in gp_f_sols]
    gp_f_scores = [gp_f_sol[1] for gp_f_sol in gp_f_sols]

    # compute optimal edge maps w.r.t. classification score
    deep_gp_class_edges, deep_gp_class = optimal_classification_edge_map(top_layer_mean, true_edges)
    deep_gp_ls_class_edges, deep_gp_ls_class = optimal_classification_edges_from_length_scale(stats["layer_0_F_sqrt_mean"], true_edges)
    gp_f_class_sols = [optimal_classification_edge_map(gp_mean, true_edges) for gp_mean in standard_gp_vals]
    gp_class_edges = [gp_sol[0] for gp_sol in gp_f_class_sols]
    gp_class = [gp_sol[1] for gp_sol in gp_f_class_sols]

    keys = ["true_img", "true_edges", "deep_gp_sol", "deep_gp_edges", "deep_gp_f_score", "deep_gp_sol_ls", "deep_gp_ls_edges",
            "deep_gp_ls_f_score", "gp_sols", "gp_edges", "gp_f_scores", "standard_gp_vals", "rho_vals", "sobel_img",
            "deep_gp_class_edges", "deep_gp_class", "deep_gp_ls_class_edges", "deep_gp_ls_class", "gp_class_edges", "gp_class"]
    vals = [ true_img ,  true_edges ,  deep_gp_sol ,  deep_gp_edges ,  deep_gp_f_score ,  deep_gp_sol_ls ,  deep_gp_ls_edges ,
             deep_gp_ls_f_score ,  gp_sols ,  gp_edges ,  gp_f_scores ,  standard_gp_vals ,  rho_vals ,  sobel_img ,
             deep_gp_class_edges ,  deep_gp_class ,  deep_gp_ls_class_edges ,  deep_gp_ls_class ,  gp_class_edges ,  gp_class]
    for key, val in zip(keys, vals):
        stats[key] = val


def optimal_f_score_edge_map(img, true_edges):
    def neg_f_score_from_thresh(t):
        edge_map = threshold(sobel_img, thresh=t, v_range=(0, 1))
        neg_fs = - f_score(edge_map, true_edges)
        
        return neg_fs

    width = int(np.sqrt(img.size))
    sobel_img = filters.sobel(img.reshape(width, width)).flatten()
    result = minimize_scalar(neg_f_score_from_thresh, bounds=(sobel_img.min(), sobel_img.max()))
    if result.success:
        opt_t = result.x
    else:
        raise ValueError("minimize_scalar did not converge.")
    
    opt_edges = threshold(sobel_img, thresh=opt_t, v_range=(0, 1))
    fs = f_score(opt_edges, true_edges)

    return opt_edges, fs


def optimal_f_score_edges_from_length_scale(ls_img, true_edges):
    def neg_f_score_from_thresh(t):
        edge_map = threshold(ls_img, thresh=t, v_range=(0, 1))
        neg_fs = - f_score(edge_map, true_edges)
        
        return neg_fs

    result = minimize_scalar(neg_f_score_from_thresh, bounds=(ls_img.min(), ls_img.max()))
    if result.success:
        opt_t = result.x
    else:
        raise ValueError("minimize_scalar did not converge.")
    
    opt_edges = threshold(ls_img, thresh=opt_t, v_range=(0, 1))
    fs = f_score(opt_edges, true_edges)

    return opt_edges, fs


def optimal_classification_edge_map(img, true_edges):
    def class_score_from_thresh(t):
        edge_map = threshold(sobel_img, thresh=t, v_range=(0, 1))
        cs = classification_error(edge_map, true_edges)
        
        return cs

    width = int(np.sqrt(img.size))
    sobel_img = filters.sobel(img.reshape(width, width)).flatten()
    result = minimize_scalar(class_score_from_thresh, bounds=(sobel_img.min(), sobel_img.max()))
    if result.success:
        opt_t = result.x
    else:
        raise ValueError("minimize_scalar did not converge.")
    
    opt_edges = threshold(sobel_img, thresh=opt_t, v_range=(0, 1))
    cs = classification_error(opt_edges, true_edges)

    return opt_edges, cs


def optimal_classification_edges_from_length_scale(ls_img, true_edges):
    def class_score_from_thresh(t):
        edge_map = threshold(ls_img, thresh=t, v_range=(0, 1))
        cs = classification_error(edge_map, true_edges)
        
        return cs

    result = minimize_scalar(class_score_from_thresh, bounds=(ls_img.min(), ls_img.max()))
    if result.success:
        opt_t = result.x
    else:
        raise ValueError("minimize_scalar did not converge.")
    
    opt_edges = threshold(ls_img, thresh=opt_t, v_range=(0, 1))
    cs = classification_error(opt_edges, true_edges)

    return opt_edges, cs


def f_score(edge_map, true_edges):
    tp = ((edge_map == 1) * (true_edges == 1)).sum()
    fp = ((edge_map == 1) * (true_edges == 0)).sum()
    fn = ((edge_map == 0) * (true_edges == 1)).sum()

    f_score = 2 * tp / (2 * tp + fp + fn)

    return f_score


def print_and_save_label_errs(stats, save_as):
    true_img = stats["true_img"]
    deep_gp_sol = stats["deep_gp_sol"]
    deep_gp_sol_ls = stats["deep_gp_sol_ls"]
    gp_sols = stats["gp_sols"]
    rho_vals = stats["rho_vals"]

    # classification errors
    deep_gp_err = classification_error(deep_gp_sol, true_img)
    deep_gp_err_ls = classification_error(deep_gp_sol_ls, true_img)
    err_list = [deep_gp_err, deep_gp_err_ls]

    # line location
    deep_gp_loc = line_location(deep_gp_sol)
    deep_gp_loc_ls = line_location(deep_gp_sol_ls)
    loc_list = [deep_gp_loc, deep_gp_loc_ls]

    # OIS F-scores
    deep_gp_f_score = stats["deep_gp_f_score"]
    deep_gp_ls_f_score = stats["deep_gp_ls_f_score"]
    gp_f_scores = stats["gp_f_scores"]
    fsc_list = [deep_gp_f_score, deep_gp_ls_f_score] + gp_f_scores

    # OIS classification errors
    deep_gp_class = stats["deep_gp_class"]
    deep_gp_ls_class = stats["deep_gp_ls_class"]
    gp_class_errs = stats["gp_class"]
    cla_list = [deep_gp_class, deep_gp_ls_class] + gp_class_errs

    print("Errors:")
    print( "Reconstruction type    | Class. error | Edge location | OIS F-score | OIS class. error")
    print(f"Deep GP                |       {deep_gp_err:>4.4f} |        {deep_gp_loc:>4.4f} |      {deep_gp_f_score:>4.4f} |           {deep_gp_class:>4.4f}")
    print(f"Deep GP (length scale) |       {deep_gp_err_ls:>4.4f} |        {deep_gp_loc_ls:>4.4f} |      {deep_gp_ls_f_score:>4.4f} |           {deep_gp_ls_class:>4.4f}")
    for i in range(len(gp_sols)):
        err = classification_error(gp_sols[i], true_img)
        err_list.append(err)
        loc = line_location(gp_sols[i])
        loc_list.append(loc)
        f_score = gp_f_scores[i]
        gp_class = gp_class_errs[i]
        print(f"Standard GP (rho={rho_vals[i]:.2f}) |       {err:>4.4f} |        {loc:>4.4f} |      {f_score:>4.4f} |           {gp_class:>4.4f}")
    
    df_index = ["deep GP (thresh)", "deep GP (hidden)"] + [f"GP (rho={rho:.2f})" for rho in rho_vals]
    errors_df = pd.DataFrame({"Class. error": err_list, "Line location": loc_list, "Line location error": [abs(loc - 0.5) for loc in loc_list],
                              "OIS F-score": fsc_list, "OIS class. error": cla_list}, index=df_index)
    errors_df.to_csv(save_as)


def classification_error(sol, true_sol):
    err = (true_sol != sol).sum() / true_sol.size

    return err


def line_location(sol):
    width = int(np.sqrt(sol.size))
    sol_square = np.reshape(sol, (width, width))

    idx = np.argmax(sol_square, axis=1)
    loc = 1 / width * idx
    mean_loc = np.mean(loc)

    return mean_loc


def threshold(img, thresh=0.0, v_range=(-1, 1)):
    sol = img.copy()
    sol[img <= thresh] = min(v_range)
    sol[img > thresh]  = max(v_range)

    return sol


def threshold_with_length_scale(ls_img, v_range=(-1, 1)):
    width = int(np.sqrt(len(ls_img)))
    square_img = np.reshape(ls_img, (width, width))
    x_idx = np.argmax(square_img, axis=1)
    x_idx[x_idx == 0] = 1
    x_idx[x_idx == width - 1] = width - 2
    x_idx = x_idx + (square_img[np.arange(width), x_idx + 1] > square_img[np.arange(width), x_idx - 1])

    square_sol = v_range[1] * np.ones(square_img.shape)
    for i in range(width):
        square_sol[i, :x_idx[i] + 1] = v_range[0]
    
    sol = square_sol.flatten()

    return sol
