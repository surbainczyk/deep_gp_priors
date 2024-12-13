import numpy as np
import math
from scipy.sparse import eye

from .FESolver import FESolver


class GPLayer:
    def __init__(self, n_dof, alpha, sigma=None, is_base_layer=False):
        self.n_dof = n_dof
        self.nu = alpha - 1

        if sigma is None:
            sigma_squared = 1 / (math.gamma(self.nu) / (2 * np.sqrt(np.pi) * math.gamma(self.nu + 1)))
            self.sigma = np.sqrt(sigma_squared)
        else:
            self.sigma = sigma
        
        self.is_base_layer = is_base_layer

        self.fe_solver = FESolver(n_dof, power=alpha / 2)

    def evaluate(self, diag_vector, random_sample):
        w = self.fe_solver.compute_random_rhs(random_sample)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(random_sample.shape) == 1:
            x = self.sigma * diag_vector ** (self.nu / 2) * w
        else:
            x = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * w

        # solve for operator (P + Gamma)
        y = self.fe_solver.solve_with_operator(diag_vector, x, is_base_layer=self.is_base_layer)

        return y
    
    def evaluate_T(self, diag_vector, vec):
        # solve for operator (P + Gamma).T
        y = self.fe_solver.solve_with_operator_T(diag_vector, vec, is_base_layer=self.is_base_layer)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(vec.shape) == 1:
            x = self.sigma * diag_vector ** (self.nu / 2) * y
        else:
            x = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * y

        w = self.fe_solver.compute_random_rhs_T(x)

        return w

    def evaluate_inv(self, diag_vector, vec):
        # multiply with operator (P + Gamma)
        y = self.fe_solver.solve_with_operator_inv(diag_vector, vec)

        # solve for Gamma ^ {nu/2} * sigma
        if len(vec.shape) == 1:
            x = y / (self.sigma * diag_vector ** (self.nu / 2))
        else:
            x = y / (self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2))

        w = self.fe_solver.compute_random_rhs_inv(x)

        return w
    
    def compute_C_inv(self, diag_vector):
        diag_mat = eye(self.n_dof)
        diag_mat.setdiag(diag_vector ** (- self.nu / 2) / self.sigma)

        Q = self.fe_solver.compute_polynomial_matrix(diag_vector)
        middle_mat = diag_mat @ self.fe_solver.inv_mass.power(-1) @ diag_mat

        log_det_QDQ = 2 * self.fe_solver.log_det_Q + 2 * np.sum(np.log(diag_mat.diagonal())) + self.fe_solver.log_det_mass

        return Q, middle_mat, log_det_QDQ
