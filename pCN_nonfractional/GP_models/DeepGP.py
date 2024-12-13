import numpy as np

from .GPLayer import GPLayer


class DeepGP:
    def __init__(self, layers, layer_n_dof, F, alpha=4, base_diag=400, sigma=2.0):
        self.alpha = alpha
        self.base_layer = GPLayer(n_dof=layer_n_dof, alpha=alpha, sigma=sigma, is_base_layer=True)
        self.middle_layer = GPLayer(n_dof=layer_n_dof, alpha=alpha, sigma=sigma)
        self.top_layer = GPLayer(n_dof=layer_n_dof, alpha=alpha, sigma=sigma)
        self.layers = layers
        self.layer_n_dof = layer_n_dof
        self.n_dof = self.layers * self.layer_n_dof
        self.base_diag = base_diag
        self.alpha_scaling = (2 * alpha - 2) / 6

        self.F = F
    
    def evaluate(self, random_sample):
        u = np.zeros(self.n_dof)

        # sample base layer
        base_sample = random_sample[:self.layer_n_dof]
        diag_vector = self.alpha_scaling * self.base_diag * np.ones(self.layer_n_dof)

        u_prev = self.base_layer.evaluate(diag_vector, base_sample)
        u[:self.layer_n_dof] = u_prev

        # sample other layers
        for i in range(1, self.layers):
            idx_1 = i * self.layer_n_dof
            idx_2 = (i + 1) * self.layer_n_dof
            layer_sample = random_sample[idx_1:idx_2]
            diag_vector = self.alpha_scaling * self.F(u_prev)

            u_prev = self.middle_layer.evaluate(diag_vector, layer_sample)
            u[idx_1:idx_2] = u_prev
        
        # evaluate precision matrix arising from top layer values
        u_top_slice = u[-self.layer_n_dof:]
        diag_vector = self.alpha_scaling * self.F(u_top_slice)

        Q, middle_mat, log_det_QDQ = self.top_layer.compute_C_inv(diag_vector)
        
        return u, Q, middle_mat, log_det_QDQ, diag_vector
