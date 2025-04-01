import numpy as np
from scipy.special import gamma

from pCN_det_free_fractional.GP_models.GPLayer import GPLayer
from plotting import save_flattened_image


# Script for plotting sample of 2-layer deep Gaussian process
F_minus = 50
F_plus = 100 ** 2
a = 1500
b = 1


def my_F(u):

    return np.minimum(F_minus + a * np.exp(b * u), F_plus)


layer_n_dof = 128 ** 2
n_layers = 2
np.random.seed(2)

# compute deep GP sample
alpha = 3
sigma = np.sqrt(gamma(alpha) * 4 * np.pi / gamma(alpha - 1))
layer = GPLayer(layer_n_dof, alpha=alpha, sigma=sigma)

diag_vector = 200 * np.ones(layer_n_dof)
u_list = []

for _ in range(n_layers):
    xi = np.random.randn(layer_n_dof)
    u = layer.evaluate(diag_vector, xi)
    print(f"StD was {np.std(u)}.")
    # u /= np.std(u)
    u_list.append(u)
    diag_vector = my_F(u)

# plots for paper
xi = np.random.randn(layer_n_dof)
u_F_min = layer.evaluate(F_minus * np.ones_like(diag_vector), xi)
u_F_min /= np.std(u_F_min)

xi = np.random.randn(layer_n_dof)
u_F_max = layer.evaluate(F_plus * np.ones_like(diag_vector), xi)
u_F_max /= np.std(u_F_max)

plots_dir = 'extra_plots/'
save_flattened_image(u_list[0], plots_dir+'u_0.pdf', figsize=(2.8, 2.8))
save_flattened_image(u_list[1], plots_dir+'u_1.pdf', figsize=(2.8, 2.8))

save_flattened_image(u_F_min, plots_dir+'u_F_min.pdf', figsize=(2.8, 2.8))
save_flattened_image(u_F_max, plots_dir+'u_F_max.pdf', figsize=(2.8, 2.8))

print('Finished plotting.')
