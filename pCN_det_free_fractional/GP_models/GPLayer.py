import numpy as np
import math

from .FESolver import FESolver


class GPLayer:
    def __init__(self, n_dof, alpha, sigma=None, add_fe_solver=True):
        self.n_dof = n_dof
        self.nu = alpha - 1

        if sigma is None:
            sigma_squared = 1 / (math.gamma(self.nu) / (2 * np.sqrt(np.pi) * math.gamma(self.nu + 1)))
            self.sigma = np.sqrt(sigma_squared)
        else:
            self.sigma = sigma

        if add_fe_solver:
            self.fe_solver = FESolver(n_dof, fraction=alpha / 2)

        self.ignore_diag_vector = False
    
    def copy(self):
        copied_gp_layer = GPLayer(self.n_dof, self.nu + 1, sigma=self.sigma, add_fe_solver=False)
        copied_gp_layer.fe_solver = self.fe_solver.copy()
        copied_gp_layer.ignore_diag_vector = self.ignore_diag_vector

        return copied_gp_layer

    def evaluate(self, diag_vector, random_sample, fix_diag=False):
        w = self.fe_solver.compute_random_rhs(random_sample)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(random_sample.shape) == 1:
            x = self.sigma * diag_vector ** (self.nu / 2) * w
        else:
            x = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * w

        # solve for operator (P + Gamma)
        if self.ignore_diag_vector:
            y = self.fe_solver.solve_with_fractional_operator(None, x)
        else:
            y = self.fe_solver.solve_with_fractional_operator(diag_vector, x)

        self.ignore_diag_vector = fix_diag

        return y
    
    def evaluate_T(self, diag_vector, vec, fix_diag=False):
        # solve for operator (P + Gamma).T
        if self.ignore_diag_vector:
            y = self.fe_solver.solve_with_fractional_operator_T(None, vec)
        else:
            y = self.fe_solver.solve_with_fractional_operator_T(diag_vector, vec)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(vec.shape) == 1:
            x = self.sigma * diag_vector ** (self.nu / 2) * y
        else:
            x = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * y

        w = self.fe_solver.compute_random_rhs_T(x)

        self.ignore_diag_vector = fix_diag

        return w

    def evaluate_inv(self, diag_vector, vec):
        # multiply with operator (P + Gamma)
        y = self.fe_solver.solve_with_fractional_operator_inv(diag_vector, vec)

        # solve for Gamma ^ {nu/2} * sigma
        if len(vec.shape) == 1:
            x = y / (self.sigma * diag_vector ** (self.nu / 2))
        else:
            x = y / (self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2))

        w = self.fe_solver.compute_random_rhs_inv(x)

        return w

    def evaluate_inv_T(self, diag_vector, vec):
        # solve for Gamma ^ {nu/2} * sigma
        y = self.fe_solver.compute_random_rhs_inv_T(vec)

        if len(vec.shape) == 1:
            x = y / (self.sigma * diag_vector ** (self.nu / 2))
        else:
            x = y / (self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2))

        # multiply with operator (P + Gamma).T
        w = self.fe_solver.solve_with_fractional_operator_inv_T(diag_vector, x)

        return w
    
    def apply_C(self, diag_vector, vec, fix_diag=False):
        # solve for operator (P + Gamma).T
        if self.ignore_diag_vector:
            y = self.fe_solver.solve_with_fractional_operator_T(None, vec)
        else:
            y = self.fe_solver.solve_with_fractional_operator_T(diag_vector, vec)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(vec.shape) == 1:
            x = self.sigma * diag_vector ** (self.nu / 2) * y
        else:
            x = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * y

        w = self.fe_solver.solve_with_mass(x)

        # multiply with Gamma ^ {nu/2} * sigma
        if len(w.shape) == 1:
            v = self.sigma * diag_vector ** (self.nu / 2) * w
        else:
            v = self.sigma * diag_vector[:, np.newaxis] ** (self.nu / 2) * w

        # solve for operator (P + Gamma)
        u = self.fe_solver.solve_with_fractional_operator(None, v)

        self.ignore_diag_vector = fix_diag

        return u
