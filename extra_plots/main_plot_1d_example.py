import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv as modified_bessel, gamma


plt.style.use('seaborn-v0_8-ticks')
plt.rc('axes',  labelsize=18)
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)


# Script for plotting example of how a stationary correlation length fails to represent a 1D signal
def matern_kernel(r, nu, rho):
    kappa = np.sqrt(2 * nu) / rho
    b = modified_bessel(nu, kappa * abs(r))
    b[r == 0] = 0
    cov = 1 / (2 ** (nu - 1) * gamma(nu)) * (kappa * abs(r)) ** nu * b
    cov[r == 0] = 1

    return cov


def regression(prediction_loc, obs_loc):
    C_pn = kernel(prediction_loc[:, np.newaxis] - obs_loc[np.newaxis, :])
    C_nn = kernel(obs_loc[:, np.newaxis] - obs_loc[np.newaxis, :])

    inner_mat = C_nn + var_noise * np.eye(n_obs)
    regr_mean = C_pn @ (np.linalg.solve(inner_mat, obs))

    return regr_mean


def plot_regression(regr_results, ground_truth, domain, extended_obs, obs_idx, save_truth_as, save_regr_as):
    mid_idx = int(np.argmax(abs(ground_truth[1:] - ground_truth[:-1]))) + 1
    
    figsize = (5, 3.75)
    plt.figure(figsize=figsize)
    plt.plot(domain[:mid_idx], ground_truth[:mid_idx], 'k:', label='ground truth')
    plt.plot(domain[mid_idx:], ground_truth[mid_idx:], 'k:')
    plt.scatter(domain[obs_idx], extended_obs[obs_idx], s=6, color="r", marker="x", label='observation')
    plt.xlabel(r'$x$')
    plt.legend(loc='center left', bbox_to_anchor=(-0.025, 0.5))
    plt.savefig(save_truth_as, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=figsize)
    plt.plot(domain[:mid_idx], ground_truth[:mid_idx], 'k:', label='ground truth')
    plt.plot(domain[mid_idx:], ground_truth[mid_idx:], 'k:')
    for key in list(regr_results.keys()):
        plt.plot(domain, regr_results[key], label=rf'$\rho={key}$')
    plt.xlabel(r'$x$')
    plt.legend(loc='center left', bbox_to_anchor=(-0.025, 0.5))
    plt.savefig(save_regr_as, bbox_inches="tight")
    plt.close()


print('Setup...')
np.random.seed(0)
n_dof = int(2e2)
domain = np.linspace(0, 1, n_dof)

ground_truth = 2 * np.array(domain > 0.5, dtype=float) - 1

n_obs = int(50)
observation_idx = np.linspace(0, n_dof, n_obs, endpoint=False, dtype=int)

std_noise = 0.05
var_noise = std_noise ** 2
obs_noise = std_noise * np.random.randn(n_obs)
obs = ground_truth[observation_idx] + obs_noise

plots_dir = 'extra_plots/'
ext_obs = np.zeros_like(domain)
ext_obs[observation_idx] = obs

print('GP regression...')
rho_vals = [0.3, 0.03]
results  = {}

for rho in rho_vals:
    kernel = lambda r: matern_kernel(r, nu=5/2, rho=rho)
    regr_mean = regression(domain, domain[observation_idx])
    results[rho] = regr_mean

print('Plotting GP regression results...')
plot_regression(results, ground_truth, domain, ext_obs, observation_idx,
                plots_dir+'1d_ground_truth.pdf', plots_dir+'1d_regr_result.pdf')

print('Finished.')
