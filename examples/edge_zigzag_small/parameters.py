import numpy as np
from scipy.special import gamma

from forward_operators import construct_random_obs_matrix
from true_images import small_zigzag_edge


def F(u):
    F_minus = 50
    F_plus = 100 ** 2
    a = 200
    b = 1

    return np.minimum(F_minus + a * np.exp(b * u), F_plus)


# set up problem
n_dof = 128 ** 2
n_obs = int(n_dof / 50)
alpha = 3
obs_sigma = 0.02

true_img, true_edges = small_zigzag_edge(n_dof)

np.random.seed(0)
forward_op = construct_random_obs_matrix(n_dof, n_obs)

obs = forward_op @ true_img    # image is discrete, so noise would not make sense

rho_vals = list(np.array(np.arange(0.01, 0.26, 0.01), dtype=np.float32)) + \
        list(np.array(np.arange(0.3, 0.55, 0.05), dtype=np.float32))

parameters = {"n_dof": n_dof,
              "true_img": true_img,
              "true_edges": true_edges,
              "forward_op": forward_op,
              "obs_noise_std": obs_sigma,
              "obs": obs,

              "layers": 1,
              "F": F,
              "alpha": 3,
              "base_diag": 1500,
              "sigma": np.sqrt(gamma(alpha) * 4 * np.pi / gamma(alpha - 1)),    # corresponds to marginal variance ~1

              "rho_vals": rho_vals,

              "its": int(2e4),
              "burn_in": int(1e4),
              "fix_precond": 100,
              "plot_obs": True,
              "plot_uq": False,
              "set_vrange": False}
