import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from utils import line_location, classification_error, H_1_error


plt.rcParams['figure.dpi'] = 600
plt.rc('axes',  labelsize=18)
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)


def save_flattened_image(img, save_as, shape=None, figsize=None, vrange=None, ticks_off=False, cbar_ticks=None, plot_cbar_to=None):
    if shape is None:
        width = int(np.sqrt(img.size))
        shape = (width, width)
    if figsize is None:
        figsize = (2.8, 2.8)
    
    plt.figure(figsize=figsize)
    square_img = np.reshape(img, shape)
    vmin = vrange[0] if vrange else None
    vmax = vrange[1] if vrange else None

    im = plt.imshow(square_img, cmap="viridis", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    if ticks_off:
        plt.xticks([], [])
        plt.yticks([], [])
    if plot_cbar_to:
        plt.savefig(save_as, bbox_inches="tight")

    clb = plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)
    if cbar_ticks:
        clb.set_ticks(cbar_ticks)
    
    if plot_cbar_to:
        plt.gca().set_visible(False)
        plt.savefig(plot_cbar_to, bbox_inches="tight")
    else:
        plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def save_flattened_image_with_obs_locations(img, obs_operator, save_as, figsize=None, vrange=None):
    if figsize is None:
        figsize = (3.5, 3.5)
    
    plt.figure(figsize=figsize)
    width = int(np.sqrt(img.size))
    square_img = np.reshape(img, (width, width))
    vmin = vrange[0] if vrange else None
    vmax = vrange[1] if vrange else None

    im = plt.imshow(square_img, cmap="viridis", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)

    obs_indices = np.sum(obs_operator.toarray(), axis=0, dtype=bool)
    x, y = np.unravel_index(np.arange(img.size)[obs_indices], (width, width))
    plt.scatter((x + 0.5) / width, (y + 0.5) / width, s=6, color="r", marker="x")

    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def save_image_with_errors(img, true_img, save_as, figsize=None, shape=None, vrange=None, log_err=False):
    if shape is None:
        width = int(np.sqrt(img.size))
        shape = (width, width)
    if figsize is None:
        figsize = (3.5, 3.5)
    
    plt.figure(figsize=figsize)
    square_img = np.reshape(img, shape)
    vmin = vrange[0] if vrange else None
    vmax = vrange[1] if vrange else None

    im = plt.imshow(square_img, cmap="viridis", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=figsize)
    err = np.reshape(img - true_img, shape)
    err_save_as = save_as.split(".")[0] + "_err." + save_as.split(".")[1]

    im = plt.imshow(err, cmap="viridis", extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)
    plt.savefig(err_save_as, bbox_inches="tight")
    plt.close()

    if log_err:
        plt.figure(figsize=figsize)
        log_err = np.log(abs(err))
        log_err_save_as = save_as.split(".")[0] + "_log_err." + save_as.split(".")[1]

        im = plt.imshow(log_err, cmap="viridis", extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04)
        plt.savefig(log_err_save_as, bbox_inches="tight")
        plt.close()


def save_radon_observations(img, save_as, shape=None, figsize=None, vrange=None):
    if shape is None:
        width = int(np.sqrt(img.size))
        shape = (width, width)
    if figsize is None:
        figsize = (2.8, 2.8)
    
    plt.figure(figsize=figsize)
    square_img = np.reshape(img, shape)
    vmin = vrange[0] if vrange else None
    vmax = vrange[1] if vrange else None

    plt.imshow(square_img, cmap="viridis", extent=[0, shape[1], 0, shape[0]], vmin=vmin, vmax=vmax)
    if shape[0] == shape[1]:
        plt.xticks(np.arange(0, shape[1]+1, step=shape[1]/2), [r"0", r"$\pi/2$", r"$\pi$"])
    else:
        plt.xticks(np.arange(0, shape[1]+1, step=shape[1]), [r"0", r"$\pi$"])
    plt.yticks(np.arange(0, shape[0]+1, step=shape[0]/4, dtype=int))
    plt.colorbar()
    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def plot_errors(ground_truth, deep_gp_sol, gp_list, rho_vals, plots_dir, include_opt_rho=False):
    deep_l1 = np.mean(np.abs(deep_gp_sol - ground_truth))    # integrating over 1x1 square
    deep_m  = mean_squared_error(ground_truth, deep_gp_sol)
    deep_l2 = np.sqrt(deep_m)
    deep_p  = peak_signal_noise_ratio(ground_truth, deep_gp_sol, data_range=ground_truth.max() - ground_truth.min())
    deep_s  = structural_similarity(ground_truth, deep_gp_sol, data_range=ground_truth.max() - ground_truth.min())
    deep_h  = H_1_error(ground_truth, deep_gp_sol)

    l1_err = []
    mse  = []
    psnr = []
    ssim = []
    h1_err = []

    if include_opt_rho:
        rho_opt = rho_vals[-1]
        sorted_idx = np.argsort(rho_vals)
        rho_vals_sorted = [rho_vals[i] for i in sorted_idx]
        gp_list_sorted  = [gp_list[i] for i in sorted_idx]
        opt_line_color = '0.5'
    else:
        rho_vals_sorted = rho_vals[:-1]
        gp_list_sorted  = gp_list[:-1]
    
    for sol in gp_list_sorted:
        l1 = np.mean(np.abs(sol - ground_truth))    # integrating over 1x1 square
        l1_err.append(l1)
        
        m = mean_squared_error(ground_truth, sol)
        mse.append(m)
        
        p = peak_signal_noise_ratio(ground_truth, sol, data_range=ground_truth.max() - ground_truth.min())
        psnr.append(p)
        
        s = structural_similarity(ground_truth, sol, data_range=ground_truth.max() - ground_truth.min())
        ssim.append(s)

        h = H_1_error(ground_truth, sol)
        h1_err.append(h)
    
    l2_err = [np.sqrt(err) for err in mse]
    
    # plot errors
    plt.style.use('seaborn-v0_8-ticks')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    markers = ["x", "o", "*", "d", "s", "v", "1"]

    # plot L1 and L2 errors in one figure
    plt.figure(figsize=(5, 3.75))
    if include_opt_rho:
        plt.axvline(rho_opt, ls=':', color=opt_line_color)
    ax1 = plt.gca()
    lns1 = ax1.plot(rho_vals_sorted, l1_err, label=r"$L^1$-error", color=colors[0], marker=markers[0])
    d_lns1 = ax1.plot([rho_vals_sorted[0], rho_vals_sorted[-1]], [deep_l1, deep_l1], ls='--', color='k', label="deep GP\n"+r"($L^1$- \& $L^2$-error)")
    ax1.set_xlabel(r"$\rho$")
    # ax1.set_yscale('log')
    ax1.set_ylabel(r"$L^1$-error")

    ax2 = ax1.twinx()
    lns2 = ax2.plot(rho_vals_sorted, l2_err, label=r"$L^2$-error", color=colors[1], marker=markers[1])
    ax2.plot([rho_vals_sorted[0], rho_vals_sorted[-1]], [deep_l2, deep_l2], ls='--', color='k')
    # ax2.set_yscale('log')
    ax2.set_ylabel(r"$L^2$-error")

    rescale_y_axis(ax1, l1_err[2:] + [deep_l1])
    rescale_y_axis(ax2, l2_err[2:] + [deep_l2])
    adjust_y_axis_limits(ax1, ax2, deep_l1, deep_l2)

    lns = d_lns1 + lns1 + lns2
    labs = [l.get_label() for l in lns]
    # plt.legend(lns, labs)
    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.legend(lns, labs, loc='upper right', bbox_to_anchor=(1.0, 1))
    plt.savefig(plots_dir + "rho_comparison_l1_l2.pdf", bbox_inches="tight")
    plt.close()

    # plot PSNR and SSIM in one figure
    plt.figure(figsize=(5, 3.75))
    if include_opt_rho:
        plt.axvline(rho_opt, ls=':', color=opt_line_color)
    ax3 = plt.gca()
    lns3 = ax3.plot(rho_vals_sorted, psnr, label=r"PSNR", color=colors[2], marker=markers[2])
    d_lns3 = ax3.plot([rho_vals_sorted[0], rho_vals_sorted[-1]], [deep_p, deep_p], ls='--', color='k', label="deep GP\n"+r"(PSNR \& SSIM)")
    ax3.set_xlabel(r"$\rho$")
    ax3.set_ylabel(r"PSNR")

    ax4 = ax3.twinx()
    lns4 = ax4.plot(rho_vals_sorted, ssim, label=r"SSIM", color=colors[3], marker=markers[3])
    ax4.plot([rho_vals_sorted[0], rho_vals_sorted[-1]], [deep_s, deep_s], ls='--', color='k')
    ax4.set_ylabel(r"SSIM")

    rescale_y_axis(ax3, psnr[2:] + [deep_p])
    rescale_y_axis(ax4, ssim[2:] + [deep_s])
    adjust_y_axis_limits(ax3, ax4, deep_p, deep_s)

    lns = lns3 + lns4 + d_lns3
    labs = [l.get_label() for l in lns]
    # plt.legend(lns, labs)
    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.legend(lns, labs, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.savefig(plots_dir + "rho_comparison_p_s.pdf", bbox_inches="tight")
    plt.close()

    # plot H^1 error
    plt.figure(figsize=(5, 3.75))
    if include_opt_rho:
        plt.axvline(rho_opt, ls=':', color=opt_line_color)
    plt.plot(rho_vals_sorted, h1_err, label=r"$H^1$-error", color=colors[4], marker=markers[4])
    plt.plot([rho_vals_sorted[0], rho_vals_sorted[-1]], [deep_h, deep_h], ls='--', color='k', label="deep GP")
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$H^1$-error")

    rescale_y_axis(plt.gca(), h1_err[2:] + [deep_h])

    plt.legend()
    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.savefig(plots_dir + "rho_comparison_h.pdf", bbox_inches="tight")
    plt.close()


def plot_edge_errors(stats, plots_dir, include_opt_rho=False):
    true_img = stats["true_img"]
    deep_gp_sol = stats["deep_gp_sol"]
    deep_gp_sol_ls = stats["deep_gp_sol_ls"]
    gp_sols = stats["gp_sols"]
    rho_vals = stats["rho_vals"]

    if include_opt_rho:
        rho_opt = rho_vals[-1]
        sorted_idx = np.argsort(rho_vals)
        rho_vals_sorted = [rho_vals[i] for i in sorted_idx]
        gp_list_sorted  = [gp_sols[i] for i in sorted_idx]
        gp_f_scores = [stats["gp_f_scores"][i] for i in sorted_idx]
        opt_line_color = '0.5'
    else:
        rho_vals_sorted = rho_vals[:-1]
        gp_list_sorted  = gp_sols[:-1]
        gp_f_scores = stats["gp_f_scores"][:-1]

    # classification errors
    deep_gp_err = classification_error(deep_gp_sol, true_img)
    deep_gp_err_ls = classification_error(deep_gp_sol_ls, true_img)
    gp_errors = [classification_error(g, true_img) for g in gp_list_sorted]

    # OIS F-scores
    deep_gp_f_score = stats["deep_gp_f_score"]
    deep_gp_ls_f_score = stats["deep_gp_ls_f_score"]

    # plot errors
    plt.style.use('seaborn-v0_8-ticks')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    markers = ["x", "o", "*", "d", "s", "v", "1"]
    max_rho_idx = np.amax(np.argwhere([r <= 0.25 for r in rho_vals_sorted]))

    # plot classification error
    plt.figure(figsize=(5, 3.75))
    if include_opt_rho:
        plt.axvline(rho_opt, ls=':', color=opt_line_color)
    plt.plot(rho_vals_sorted[:max_rho_idx+1], gp_errors[:max_rho_idx+1], label=r"GP class. error", color=colors[0], marker=markers[0])
    plt.plot([rho_vals_sorted[0], rho_vals_sorted[max_rho_idx]],
             [deep_gp_err, deep_gp_err], ls='--', color='k', label=r"deep GP (top layer)")
    plt.plot([rho_vals_sorted[0], rho_vals_sorted[max_rho_idx]],
             [deep_gp_err_ls, deep_gp_err_ls], ls=(0, (3, 1, 1, 1)), color='k', label=r"deep GP (bottom layer)")
    
    rescale_y_axis(plt.gca(), gp_errors[1:max_rho_idx+1] + [deep_gp_err, deep_gp_err_ls])

    plt.xlabel(r"$\rho$")
    plt.ylabel(r"Class. error")
    # plt.legend()
    plt.legend(loc='lower right', bbox_to_anchor=(0.96, 0.4))
    # plt.legend(loc='lower right', bbox_to_anchor=(1.02, 0.5))
    plt.savefig(plots_dir + "rho_comparison_class_error.pdf", bbox_inches="tight")
    plt.close()

    # plot F-score
    plt.figure(figsize=(5, 3.75))
    if include_opt_rho:
        plt.axvline(rho_opt, ls=':', color=opt_line_color)
    plt.plot(rho_vals_sorted[:max_rho_idx+1], gp_f_scores[:max_rho_idx+1], label=r"GP F-score", color=colors[1], marker=markers[1])
    plt.plot([rho_vals_sorted[0], rho_vals_sorted[max_rho_idx]],
             [deep_gp_f_score, deep_gp_f_score], ls='--', color='k', label=r"deep GP (top layer)")
    plt.plot([rho_vals_sorted[0], rho_vals_sorted[max_rho_idx]],
             [deep_gp_ls_f_score, deep_gp_ls_f_score], ls=(0, (3, 1, 1, 1)), color='k', label=r"deep GP (bottom layer)")
    
    rescale_y_axis(plt.gca(), gp_f_scores[1:max_rho_idx+1] + [deep_gp_f_score, deep_gp_ls_f_score])

    plt.xlabel(r"$\rho$")
    plt.ylabel(r"F-score")
    # plt.legend()
    plt.legend(loc='lower right', bbox_to_anchor=(0.96, 0.0))
    plt.savefig(plots_dir + "rho_comparison_f_score.pdf", bbox_inches="tight")
    plt.close()


def rescale_y_axis(ax, val_list, margin=0.15):
    # scale y limits using given values
    top = max(val_list)
    bottom = min(val_list)
    range = top - bottom
    ax.set_ylim(bottom = bottom - margin * range, top = top + margin * range)


def adjust_y_axis_limits(ax1, ax2, val1, val2):
    # adjust y limits on ax1 and ax2 so that val1 and val2 coincide on joint axes
    ax1_lim = ax1.axes.get_ylim()
    ax2_lim = ax2.axes.get_ylim()
    if (ax2_lim[1] - val2) * (val1 - ax1_lim[0]) < (ax1_lim[1] - val1) * (val2 - ax2_lim[0]):
        if (val1 - ax1_lim[0]) > 0:
            y2_top = (ax1_lim[1] - val1) / (val1 - ax1_lim[0]) * (val2 - ax2_lim[0]) + val2
            ax2.set_ylim(top=y2_top)
        else:
            y1_low = val1 - (ax1_lim[1] - val1) * (val2 - ax2_lim[0]) / (ax2_lim[1] - val2)
            ax1.set_ylim(bottom=y1_low)
    elif (ax2_lim[1] - val2) * (val1 - ax1_lim[0]) > (ax1_lim[1] - val1) * (val2 - ax2_lim[0]):
        if (ax1_lim[1] - val1) > 0:
            y2_low = val2 - (ax2_lim[1] - val2) * (val1 - ax1_lim[0]) / (ax1_lim[1] - val1)
            ax2.set_ylim(bottom=y2_low)
        else:
            y1_top = (ax2_lim[1] - val2) / (val2 - ax2_lim[0]) * (val1 - ax1_lim[0]) + val1
            ax1.set_ylim(top=y1_top)


def plot_mcmc_result(stats, save_as, avoid_oom=True):
    num_gps = len(stats["standard_gp_vals"])
    if avoid_oom and num_gps > 15:
        gp_idx = np.linspace(0, num_gps, 15, endpoint=False, dtype=int)
        gp_list  = [stats['standard_gp_vals'][i] for i in gp_idx]
        rho_list = [stats['rho_vals'][i] for i in gp_idx]
        num_standard_gps = 15
    else:
        gp_list  = stats['standard_gp_vals']
        rho_list = stats["rho_vals"]
        num_standard_gps = num_gps
    
    fig, axs = plt.subplots(4 + num_standard_gps, 3, figsize=(15, 4 * (4 + num_standard_gps)))
    fig.set_dpi(300)

    # plot mean estimate results
    top_layer_error = stats["top_layer_mean_scaled"] - stats["true_img"]
    l1e_mean = np.mean(np.abs(top_layer_error))
    plot_2D_function(axs[0, 0], top_layer_error, x_label=r"Error of $\mathrm{Mean}\left(u_1\right)$, L1 error=" + f"{l1e_mean:.4f}")

    top_layer_mean = stats["top_layer_mean_scaled"]
    plot_2D_function(axs[0, 1], top_layer_mean, x_label=r"$\mathrm{Mean}\left(u_1\right)$")

    F_mean = stats["layer_0_F_sqrt_mean"]
    plot_2D_function(axs[0, 2], F_mean, x_label=r"$\mathrm{Mean}\left(F\left(u_0\right)^{1/2}\right)$")

    # plot MAP estimate results
    top_layer_map_error = stats["top_layer_MAP_scaled"] - stats["true_img"]
    l1e_map = np.mean(np.abs(top_layer_map_error))
    plot_2D_function(axs[1, 0], top_layer_map_error, x_label=r"Error of $\mathrm{MAP}\left(u_1\right)$, L1 error=" + f"{l1e_map:.4f}")

    top_layer_map = stats["top_layer_MAP_scaled"]
    plot_2D_function(axs[1, 1], top_layer_map, x_label=r"$u_1^\mathrm{MAP}$")

    F_map = stats["layer_0_F_sqrt_MAP"]
    plot_2D_function(axs[1, 2], F_map, x_label=r"$F\left(u_0^\mathrm{MAP}\right)^{1/2}$")

    # plot GP regression results/errors
    for i in range(num_standard_gps):
        rho = rho_list[i]
        gp_error = gp_list[i] - stats["true_img"]
        l1e_long = np.mean(np.abs(gp_error))
        plot_2D_function(axs[i + 2, 0], gp_error, x_label=rf"Error of GP ($\rho={rho:.2f}$), L1 error=" + f"{l1e_long:.4f}")

        gp = gp_list[i]
        plot_2D_function(axs[i + 2, 1], gp, x_label=rf"GP ($\rho={rho:.2f}$)")

        plot_placeholder(axs[i + 2, 2])

    # plot GP regression results/errors with length scale from mean of deep GP layer
    gp_mean_error = stats["gp_mean"] - stats["true_img"]
    l1e_gp_mean = np.mean(np.abs(gp_mean_error))
    plot_2D_function(axs[-2, 0], gp_mean_error, x_label=r"Error of GP (mean), L1 error=" + f"{l1e_gp_mean:.4f}")

    gp_mean = stats["gp_mean"]
    plot_2D_function(axs[-2, 1], gp_mean, x_label=r"GP (mean)")

    F_gp_mean = stats["layer_0_mean_F_sqrt"]
    plot_2D_function(axs[-2, 2], F_gp_mean, x_label=r"$F\left(\mathrm{Mean}\left(u_0\right)\right)^{1/2}$")
    
    # # plot gradient errors for top layer, and both GP examples
    # top_layer_grad = compute_gradient_norm(top_layer_error)
    # plot_2D_function(axs[-2, 0], top_layer_grad, logscale=True,
    #                  x_label=r"Gradient error of $\mathrm{Mean}\left(u_1\right)$, MSE=" + f"{np.mean(top_layer_grad):.4f}")

    # long_grad = compute_gradient_norm(gp_long_error)
    # plot_2D_function(axs[-2, 1], long_grad, logscale=True,
    #                  x_label=r"Gradient error of GP (long), MSE=" + f"{np.mean(long_grad):.4f}")

    # short_grad = compute_gradient_norm(gp_short_error)
    # plot_2D_function(axs[-2, 2], short_grad, logscale=True,
    #                  x_label=r"Gradient error of GP (short), MSE=" + f"{np.mean(short_grad):.4f}")
    
    # plot standard deviation of top and hidden layer
    plot_placeholder(axs[-1, 0])

    top_layer_std = stats["top_layer_std"]
    plot_2D_function(axs[-1, 1], top_layer_std, x_label=r"$\mathrm{StD}\left(u_1\right)$", logscale=True)

    F_std = stats["layer_0_F_std"]
    plot_2D_function(axs[-1, 2], F_std, x_label=r"$\mathrm{StD}\left(F\left(u_0\right)^{1/2}\right)$", logscale=True)

    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def plot_multilayer_mcmc_result(stats, layers, save_as, avoid_oom=True):
    num_gps = len(stats["standard_gp_vals"])
    if avoid_oom and num_gps > 15:
        gp_idx = np.linspace(0, num_gps, 15, endpoint=False, dtype=int)
        gp_list  = [stats['standard_gp_vals'][i] for i in gp_idx]
        rho_list = [stats['rho_vals'][i] for i in gp_idx]
        num_standard_gps = 15
    else:
        gp_list  = stats['standard_gp_vals']
        rho_list = stats["rho_vals"]
        num_standard_gps = num_gps
    rows = 4 + num_standard_gps
    cols = 2 + layers
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.set_dpi(300)

    # plot mean estimate results
    top_layer_error = stats["top_layer_mean_scaled"] - stats["true_img"]
    l1e_mean = np.mean(np.abs(top_layer_error))
    plot_2D_function(axs[0, 0], top_layer_error,
                     x_label=r"Error of $\mathrm{Mean}" + rf"\left(u_{layers}\right)$, L1 error={l1e_mean:.4f}")

    top_layer_mean = stats["top_layer_mean_scaled"]
    plot_2D_function(axs[0, 1], top_layer_mean, x_label=r"$\mathrm{Mean}" + rf"\left(u_{layers}\right)$")

    for i in range(layers - 1, -1, -1):
        F_mean = stats[f"layer_{i}_mean_F_sqrt"]
        plot_2D_function(axs[0, 1 + layers - i], F_mean, x_label=r"$F\left(\mathrm{Mean}\left(" + rf"u_{i}" + r"\right)^{1/2}\right)$")

    # plot MAP estimate results
    top_layer_map_error = stats["top_layer_MAP_scaled"] - stats["true_img"]
    l1e_map = np.mean(np.abs(top_layer_map_error))
    plot_2D_function(axs[1, 0], top_layer_map_error,
                     x_label=r"Error of $\mathrm{MAP}" + rf"\left(u_{layers}\right)$, L1 error={l1e_map:.4f}")

    top_layer_map = stats["top_layer_MAP_scaled"]
    plot_2D_function(axs[1, 1], top_layer_map, x_label=rf"$u_{layers}" + r"^\mathrm{MAP}$")

    for i in range(layers):
        plot_placeholder(axs[1, 2 + i])    # we don't have MAP results on these layers

    # plot GP regression results/errors
    for i in range(num_standard_gps):
        rho = rho_list[i]
        gp_error = gp_list[i] - stats["true_img"]
        l1e_long = np.mean(np.abs(gp_error))
        plot_2D_function(axs[i + 2, 0], gp_error, x_label=rf"Error of GP ($\rho={rho:.2f}$), L1 error=" + f"{l1e_long:.4f}")

        gp = gp_list[i]
        plot_2D_function(axs[i + 2, 1], gp, x_label=rf"GP ($\rho={rho:.2f}$)")

        for j in range(layers):
            plot_placeholder(axs[i + 2, 2 + j])

    # plot GP regression results/errors with length scale from mean of deep GP layer
    gp_mean_error = stats["gp_mean"] - stats["true_img"]
    l1e_gp_mean = np.mean(np.abs(gp_mean_error))
    plot_2D_function(axs[-2, 0], gp_mean_error, x_label=r"Error of GP (mean), L1 error=" + f"{l1e_gp_mean:.4f}")

    gp_mean = stats["gp_mean"]
    plot_2D_function(axs[-2, 1], gp_mean, x_label=r"GP (mean)")

    F_gp_mean = stats[f"layer_{layers-1}_mean_F_sqrt"]
    plot_2D_function(axs[-2, 2], F_gp_mean, x_label=r"$F\left(\mathrm{Mean}\left(" + rf"u_{layers-1}" + r"\right)\right)^{1/2}$")

    for i in range(layers - 1):
        plot_placeholder(axs[-2, 3 + i])
    
    # plot standard deviation of top layer
    plot_placeholder(axs[-1, 0])

    top_layer_std = stats["top_layer_std"]
    plot_2D_function(axs[-1, 1], top_layer_std, x_label=r"$\mathrm{StD}\left(" + rf"u_{layers}\right)$", logscale=True)

    for i in range(layers - 1, -1, -1):
        F_std = stats[f"layer_{i}_F_std"]
        plot_2D_function(axs[-1, 1 + layers - i], F_std,
                         x_label=r"$\mathrm{StD}\left(F\left(" + rf"u_{i}" + r"\right)^{1/2}\right)$", logscale=True)

    for i in range(layers - 2):
        plot_placeholder(axs[-1, 4 + i])

    plt.savefig(save_as, bbox_inches="tight")
    plt.close()


def plot_2D_function(ax, data, x_label=None, logscale=False):
    n_side = int(np.sqrt(len(data)))
    value_matrix = np.reshape(data, (n_side, n_side))

    logscale_valid = np.any(data > 0)
    norm = "log" if logscale and logscale_valid else None
    im = ax.imshow(value_matrix, cmap="viridis", extent=[0, 1, 0, 1], norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if x_label:
        ax.set_xlabel(x_label)


def plot_line(ax, line_coords, style="k--"):
    ax.plot(line_coords[0], line_coords[1], style)


def plot_placeholder(ax):
    text_kwargs = dict(ha='center', va='center', fontsize=24, color='C1')

    ax.text(0.5, 0.5, 'Hier könnte\nIhre Werbung\nstehen!', **text_kwargs)
    ax.set_aspect('equal', adjustable='box')


def compute_gradient_norm(pointwise_error):
    # for input of N**2 vector, returns (N-1)x(N-1) vector
    N = int(np.sqrt(len(pointwise_error)))
    err = np.reshape(pointwise_error, (N, N))
    grad_x = err[:, 1:] - err[:, :-1]
    grad_x = 0.5 * grad_x[1:, :] + 0.5 * grad_x[:-1, :]    # interpolate in y-direction
    grad_y = err[1:, :] - err[:-1, :]
    grad_y = 0.5 * grad_y[:, 1:] + 0.5 * grad_y[:, :-1]    # interpolate in x-direction
    norm = np.sqrt(grad_x ** 2 + grad_y ** 2).ravel()

    return norm


def plot_potential_values(pot_vals, save_as, logscale=True, burn_in=None):
    plt.figure()
    plt.plot(pot_vals)
    if burn_in:    # plot line at burn-in value
        plt.axvline(burn_in, ls=':', color='k')
    if logscale:
        plt.yscale("log")

    plt.savefig(save_as, bbox_inches="tight")


def plot_acceptance_history(acc_hist, save_as):
    def moving_avg(array, k=3):
        return np.convolve(array, np.ones(k) / k, 'valid')

    averaged_hist = moving_avg(acc_hist, k=100)
    acc_rate = np.mean(acc_hist)
    start = int(0.2 * len(acc_hist))
    clean_acc_rate = np.mean(acc_hist[start:])

    plt.figure()
    plt.plot(averaged_hist)
    plt.title(f"Overall acceptance rate: {acc_rate:.4f} (after burn-in: {clean_acc_rate:.4f})")

    plt.savefig(save_as, bbox_inches="tight")


def plot_comparison_result(errors_dict, run_times_dict, plots_dir):
    plt.style.use('seaborn-v0_8-ticks')

    markers = ["x", "o", "*", "d", "s", "v", "1"]

    error_types = ["l1", "l2", "p", "s"]
    error_names = {"l1": r"$L^1$-error", "l2": r"$L^2$-error", "p": "PSNR", "s": "SSIM"}
    lines = []
    for err in error_types:
        plt.figure(figsize=(5, 3.75))
        for j, alpha in enumerate(errors_dict["alpha_vals"]):
            iter_counts = errors_dict[f"means_a{alpha:.1f}"].keys()
            errors = np.array([errors_dict[f"means_a{alpha:.1f}"][it][err] for it in iter_counts])
            errors_std = np.array([errors_dict[f"std_a{alpha:.1f}"][it][err] for it in iter_counts])
            errors_lower = errors - 2 * errors_std
            errors_upper = errors + 2 * errors_std
            times = [run_times_dict[f"means_a{alpha:.1f}"][it] for it in iter_counts]
        
            lns = plt.plot(times, errors, label=rf"$\alpha={alpha:.1f}$", marker=markers[j])
            lines.append(lns[0])
            plt.fill_between(times, errors_lower, errors_upper, alpha=0.2)
        
        bottom, top = plt.ylim()
        diff = top - bottom
        plt.ylim(bottom - 0.05 * diff, top + 0.05 * diff)
        
        plt.xlabel("Run times [s]")
        plt.ylabel(error_names[err])
        if err == error_types[0]:
            plt.legend(loc='center left', bbox_to_anchor=(-0.75, 0.5))
        plt.savefig(plots_dir + f"comparison_{err}.pdf", bbox_inches="tight")
        plt.close()


def plot_tol_comparison_result(errors_dict, tol_vals, plots_dir):
    plt.style.use('seaborn-v0_8-ticks')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    markers = ["x", "o", "*", "d", "s", "v", "1"]

    # plot L1 and L2 errors in one figure
    plt.figure(figsize=(5, 3.75))
    ax1 = plt.gca()
    ax1.set_xscale("log")
    lns1 = ax1.plot(tol_vals, errors_dict["l1"]["mean"], label=r"$L^1$-error", color=colors[0], marker=markers[0])
    ax1.fill_between(tol_vals, errors_dict["l1"]["lower"], errors_dict["l1"]["upper"], color=colors[0], alpha=0.2)
    ax1.set_xlabel("LSQR tolerance")
    ax1.set_ylabel(r"$L^1$-error")

    ax2 = ax1.twinx()
    lns2 = ax2.plot(tol_vals, errors_dict["l2"]["mean"], label=r"$L^2$-error", color=colors[1], marker=markers[1])
    ax2.fill_between(tol_vals, errors_dict["l2"]["lower"], errors_dict["l2"]["upper"], color=colors[1], alpha=0.2)
    ax2.set_ylabel(r"$L^2$-error")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper left')
    plt.savefig(plots_dir + "tol_comparison_l1_l2.pdf", bbox_inches="tight")

    # plot PSNR and SSIM in one figure
    plt.figure(figsize=(5, 3.75))
    ax3 = plt.gca()
    ax3.set_xscale("log")
    lns3 = ax3.plot(tol_vals, errors_dict["p"]["mean"], label=r"PSNR", color=colors[2], marker=markers[2])
    ax3.fill_between(tol_vals, errors_dict["p"]["lower"], errors_dict["p"]["upper"], color=colors[2], alpha=0.2)
    ax3.set_xlabel("LSQR tolerance")
    ax3.set_ylabel(r"PSNR")

    ax4 = ax3.twinx()
    lns4 = ax4.plot(tol_vals, errors_dict["s"]["mean"], label=r"SSIM", color=colors[3], marker=markers[3])
    ax4.fill_between(tol_vals, errors_dict["s"]["lower"], errors_dict["s"]["upper"], color=colors[3], alpha=0.2)
    ax4.set_ylabel(r"SSIM")

    lns = lns3 + lns4
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='lower left')
    plt.savefig(plots_dir + "tol_comparison_p_s.pdf", bbox_inches="tight")


def plot_edge_reconstruction_result(stats, save_as, edge_coords=None, avoid_oom=True):
    num_gps = len(stats["standard_gp_vals"])
    if avoid_oom and num_gps > 10:
        gp_idx = np.linspace(0, num_gps, 10, endpoint=False, dtype=int)
        num_standard_gps = 10
    else:
        gp_idx = np.arange(num_gps)
        num_standard_gps = num_gps
    
    true_img = stats["true_img"]
    fig, axs = plt.subplots(2 + num_standard_gps, 5, figsize=(25, 4 * (2 + num_standard_gps)))
    fig.set_dpi(300)

    # plot top layer results
    deep_gp_err = (true_img != stats["deep_gp_sol"]).sum() / true_img.size
    plot_2D_function(axs[0, 0], stats["deep_gp_sol"], x_label=f"Deep GP reconstruction (err={deep_gp_err:.4f}).")
    if edge_coords:
        plot_line(axs[0, 0], edge_coords)
        loc = line_location(stats["deep_gp_sol"])
        plot_line(axs[0, 0], [[loc, loc], [0, 1]], style="r-.")

    top_layer_mean = stats["top_layer_mean_scaled"]
    plot_2D_function(axs[0, 1], top_layer_mean, x_label=r"$\mathrm{Mean}\left(u_1\right)$.")

    top_layer_grad = compute_gradient_norm(stats["top_layer_mean_scaled"])
    plot_2D_function(axs[0, 2], top_layer_grad, x_label=r"Gradient norm of $\mathrm{Mean}\left(u_1\right)$.")

    top_layer_edges = stats["deep_gp_edges"]
    fs = stats["deep_gp_f_score"]
    plot_2D_function(axs[0, 3], top_layer_edges, x_label=rf"Reconstructed edge map (OIS F-score {fs:.4f}).")

    top_layer_class_edges = stats["deep_gp_class_edges"]
    cs = stats["deep_gp_class"]
    plot_2D_function(axs[0, 4], top_layer_class_edges, x_label=rf"Reconstructed edge map (OIS class. error {cs:.4f}).")

    # plot hidden layer results
    deep_gp_err_ls = (true_img != stats["deep_gp_sol_ls"]).sum() / true_img.size
    plot_2D_function(axs[1, 0], stats["deep_gp_sol_ls"], x_label=f"Deep GP rec. from length scale (err={deep_gp_err_ls:.4f}).")
    if edge_coords:
        plot_line(axs[1, 0], edge_coords)
        loc = line_location(stats["deep_gp_sol_ls"])
        plot_line(axs[1, 0], [[loc, loc], [0, 1]], style="r-.")

    F_mean = stats["layer_0_F_sqrt_mean"]
    fs = stats["deep_gp_ls_f_score"]
    plot_2D_function(axs[1, 1], F_mean, x_label=r"$\mathrm{Mean}\left(F\left(u_0\right)^{1/2}\right)$.")

    plot_placeholder(axs[1, 2])

    length_scale_edges = stats["deep_gp_ls_edges"]
    plot_2D_function(axs[1, 3], length_scale_edges, x_label=rf"Reconstructed edge map (OIS F-score {fs:.4f}).")

    length_scale_class_edges = stats["deep_gp_ls_class_edges"]
    cs = stats["deep_gp_ls_class"]
    plot_2D_function(axs[1, 4], length_scale_class_edges, x_label=rf"Reconstructed edge map (OIS class. error {cs:.4f}).")

    # plot GP regression results/errors
    for j in range(num_standard_gps):
        i = gp_idx[j]
        rho = stats["rho_vals"][i]
        gp_err = (true_img != stats["gp_sols"][i]).sum() / true_img.size
        plot_2D_function(axs[j + 2, 0], stats["gp_sols"][i], x_label=rf"GP reconstruction ($\rho={rho:.2f}$, err={gp_err:.4f}).")
        if edge_coords:
            plot_line(axs[j + 2, 0], edge_coords)
            loc = line_location(stats["gp_sols"][i])
            plot_line(axs[j + 2, 0], [[loc, loc], [0, 1]], style="r-.")

        gp = stats["standard_gp_vals"][i]
        plot_2D_function(axs[j + 2, 1], gp, x_label=rf"GP ($\rho={rho:.2f}$).")

        gp_grad = compute_gradient_norm(stats["standard_gp_vals"][i])
        plot_2D_function(axs[j + 2, 2], gp_grad, x_label=rf"Gradient norm of GP ($\rho={rho:.2f}$).")

        gp_edges = stats["gp_edges"][i]
        fs = stats["gp_f_scores"][i]
        plot_2D_function(axs[j + 2, 3], gp_edges, x_label=rf"Reconstructed edge map (OIS F-score {fs:.4f}).")

        gp_class_edges = stats["gp_class_edges"][i]
        cs = stats["gp_class"][i]
        plot_2D_function(axs[j + 2, 4], gp_class_edges, x_label=rf"Reconstructed edge map (OIS class. error {cs:.4f}).")

    plt.savefig(save_as, bbox_inches="tight")
