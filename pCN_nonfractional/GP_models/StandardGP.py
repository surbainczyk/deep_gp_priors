import numpy as np
import math
from scipy.sparse import diags
from sksparse.cholmod import cholesky
from sparse_dot_mkl import dot_product_mkl as dot_mkl
from fenics import *


parameters["reorder_dofs_serial"] = False
set_log_level(30)


class StandardGP:
    def __init__(self, n_dof, alpha, sigma=None):
        self.n_dof = n_dof
        self.alpha = alpha
        self.nu = alpha - 1

        if sigma is None:
            sigma_squared = 1 / (math.gamma(self.nu) / (2 * np.sqrt(np.pi) * math.gamma(self.nu + 1)))
            self.sigma = np.sqrt(sigma_squared)
        else:
            self.sigma = sigma

        self.set_up_fenics_variables()
        self.compute_mass_matrix()
        self.compute_laplacian_matrix()
    
    def set_up_fenics_variables(self):
        n_edge_elements = int(np.sqrt(self.n_dof)) - 1
        self.mesh = UnitSquareMesh(n_edge_elements, n_edge_elements)
        self.function_space = FunctionSpace(self.mesh, "CG", 1)
        self.u = TrialFunction(self.function_space)
        self.v = TestFunction(self.function_space)
    
    def compute_mass_matrix(self):
        a = self.u * self.v * dx
        A = assemble(a, tensor=EigenMatrix())
        self.mass_matrix = A.sparray()
        mass_vector = np.array(self.mass_matrix.sum(axis=0)).flatten()

        self.inv_mass = diags(1 / mass_vector, format="csr")
    
    def compute_laplacian_matrix(self):
        s = dot(grad(self.u), grad(self.v)) * dx
        S = assemble(s, tensor=EigenMatrix())
        self.laplacian_matrix = S.sparray()
    
    def compute_operator_matrix(self, kappa): 
        scaled_mass = kappa ** 2 * self.mass_matrix
        op_matrix = self.laplacian_matrix + scaled_mass

        return op_matrix
    
    def compute_precision_matrix(self, alpha, op_matrix, log_det_op):
        # following Lindgren et al., 2011, 'An explicit link between Gaussian fields and Gaussian Markov random fields'
        if alpha < 1 or not isinstance(alpha, int):
            raise ValueError('alpha needs to be a positive natural number.')
        if alpha <= 2:
            if alpha == 1:
                precision = op_matrix
                log_det_P = log_det_op
            else:    # alpha == 2
                precision = dot_mkl(op_matrix, dot_mkl(self.inv_mass, op_matrix))
                log_det_P = 2 * log_det_op + np.sum(np.log(self.inv_mass.diagonal()))
        else:
            small_alpha_P, log_det_P = self.compute_precision_matrix(alpha - 2, op_matrix, log_det_op)
            precision = dot_mkl(op_matrix, dot_mkl(self.inv_mass, dot_mkl(small_alpha_P, dot_mkl(self.inv_mass, op_matrix))))
            log_det_P = 2 * (log_det_op + np.sum(np.log(self.inv_mass.diagonal()))) + log_det_P
        
        return precision, log_det_P

    def regression_mean(self, rho, obs, A, noise_var):
        high_dim_data  = A.T @ obs
        high_dim_noise = dot_mkl(A.T, A, cast=True) / noise_var

        kappa = np.sqrt(2 * self.alpha - 2) / rho
        op_matrix = self.compute_operator_matrix(kappa)
        P, _ = self.compute_precision_matrix(self.alpha, op_matrix, 0)
        P = P / (self.sigma * kappa ** self.nu) ** 2
        P_noisy = P + high_dim_noise
        
        try:
            self.chol_factor.cholesky_inplace(P_noisy.T)
        except AttributeError:
            self.chol_factor = cholesky(P_noisy.T)
        
        regr_mean = self.chol_factor.solve_A(high_dim_data) / noise_var

        return regr_mean
    
    def likelihood(self, rho, obs, A, noise_var):
        high_dim_data  = A.T @ obs
        high_dim_noise = dot_mkl(A.T, A, cast=True) / noise_var

        kappa = np.sqrt(2 * self.alpha - 2) / rho
        op_matrix = self.compute_operator_matrix(kappa)
        log_det_op = cholesky(op_matrix.T).logdet()
        P, log_det_P = self.compute_precision_matrix(self.alpha, op_matrix, log_det_op)
        P = P / (self.sigma * kappa ** self.nu) ** 2
        log_det_P = log_det_P - 2 * self.n_dof * np.log(self.sigma * kappa ** self.nu)
        P_noisy = P + high_dim_noise
        
        try:
            self.chol_factor.cholesky_inplace(P_noisy.T)
        except AttributeError:
            self.chol_factor = cholesky(P_noisy.T)
        
        regr_mean = self.chol_factor.solve_A(high_dim_data) / noise_var
        quadr_term = (obs @ obs - high_dim_data.T @ regr_mean) / noise_var

        # compute log determinant term
        det_noisy = self.chol_factor.logdet()
        det_term = det_noisy - log_det_P

        lh = 0.5 * (quadr_term + det_term)

        return lh
