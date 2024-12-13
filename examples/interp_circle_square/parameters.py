import numpy as np
from scipy.special import gamma

from forward_operators import construct_observation_matrix
from true_images import square_and_circle


def F(u):
    F_minus = 50
    F_plus = 100 ** 2
    a = 200
    b = 1

    return np.minimum(F_minus + a * np.exp(b * u), F_plus)


# reconstruct image from few observations using MCMC and a deep GP prior
n_dof = 128 ** 2
n_obs = int(n_dof / 16)
alpha = 3

parameters = {"n_dof": n_dof,
              "true_img": square_and_circle(n_dof),
              "forward_op": construct_observation_matrix(n_dof, n_obs),
              "obs_shape": (int(np.sqrt(n_obs)), int(np.sqrt(n_obs))),
              "obs_noise_std": 0.02,

              "layers": 1,
              "F": F,
              "alpha": alpha,
              "base_diag": 1500,
              "sigma": np.sqrt(gamma(alpha) * 4 * np.pi / gamma(alpha - 1)),    # corresponds to marginal variance ~1

              "rho_vals": [r * 0.005 for r in range(2, 6)] + [r * 0.01 for r in range(3, 21)],

              "its": int(4e4),
              "burn_in": int(1e4),
              "fix_precond": 100,
              "plot_obs": False,
              "set_vrange": False,
              "rescale_obs": True}
