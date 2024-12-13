import numpy as np
from scipy.special import gamma

from forward_operators import get_radon_transform_matrix
from true_images import shepp_logan


def F(u):
    F_minus = 50
    F_plus = 100 ** 2
    a = 200
    b = 1

    return np.minimum(F_minus + a * np.exp(b * u), F_plus)


# reconstruct image from few observations using MCMC and a deep GP prior
n_dof = 128 ** 2
n_angles = int(np.sqrt(n_dof) / 4)

angles = np.linspace(0, 180, n_angles, endpoint=False)
alpha = 3

forward_op = get_radon_transform_matrix(n_dof, angles)

parameters = {"n_dof": n_dof,
              "true_img": shepp_logan(n_dof),
              "forward_op": forward_op,
              "obs_shape": (int(np.sqrt(n_dof)), n_angles),
              "obs_noise_std": 0.02,

              "layers": 1,
              "F": F,
              "alpha": alpha,
              "base_diag": 1500,
              "sigma": 0.25 * np.sqrt(gamma(alpha) * 4 * np.pi / gamma(alpha - 1)),    # corresponds to marginal variance ~0.25

              "rho_vals": [r * 0.005 for r in range(1, 6)] + [r * 0.01 for r in range(3, 21)],

              "its": int(4e4),
              "burn_in": int(3e4),
              "fix_precond": 1000,
              "lsqr_tol": 5e-4,
              "plot_obs": False,
              "set_vrange": False,
              "rescale_obs": False}
