import numpy as np
from sksparse.cholmod import cholesky
from sparse_dot_mkl import dot_product_mkl as dot_mkl

from pCN_det_free_fractional.custom_lsqr import lsqr
from pCN_nonfractional.GP_models.DeepGP import DeepGP


class LSQRSolver:
    def __init__(self, deep_gp, A, noise_std, high_dim_noise=None, tol=1e-3) -> None:
        self.deep_gp = deep_gp    # keep for copying of self
        self.gp_layer = deep_gp.top_layer.copy()
        int_alpha = 2 * int(np.ceil(deep_gp.alpha / 2))
        self.integer_alpha_gp = DeepGP(1, deep_gp.layer_n_dof, deep_gp.F, alpha=int_alpha).top_layer

        self.A = A
        self.noise_std = noise_std
        self.high_dim_noise = dot_mkl(A.T, A) / (noise_std ** 2) if high_dim_noise is None else high_dim_noise

        self.tol = tol
    
    def copy(self):
        copied_solver = LSQRSolver(self.deep_gp, self.A, self.noise_std, self.high_dim_noise, self.tol)

        # copy computed/changed attributes
        copied_solver.gp_layer = self.gp_layer.copy()
        copied_solver.int_chol = self.int_chol.copy()

        return copied_solver

    def solve(self, diag, y, is_old_diag=False, tol=None):
        n = self.A.shape[1]

        self.gp_layer.ignore_diag_vector = is_old_diag
        layer_eval = lambda x: self.gp_layer.evaluate(diag, x, fix_diag=True)
        layer_eval_T = lambda x: self.gp_layer.evaluate_T(diag, x, fix_diag=True)

        rhs = np.concatenate((np.zeros(n), y / self.noise_std))
        
        if not is_old_diag:
            self.compute_integer_chol(diag)

        if tol is None:
            tol = self.tol
        sol, info = lsqr(layer_eval, layer_eval_T, self.A, self.noise_std, rhs, atol=tol, btol=tol, N=self.solve_with_sparse_matrices)
        self.last_itn = info["itn"]
        self.gp_layer.ignore_diag_vector = False

        return sol
    
    def compute_integer_chol(self, diag):
        Q, M, _ = self.integer_alpha_gp.compute_C_inv(diag)
        QDQ_noisy = dot_mkl(Q.T, dot_mkl(M, Q)) + self.high_dim_noise
        try:
            self.int_chol.cholesky(QDQ_noisy)
        except AttributeError:
            self.int_chol = cholesky(QDQ_noisy)

    def solve_with_sparse_matrices(self, x):
        squared_sigma = (self.noise_std ** 2)
        
        high_dim_rhs = self.A.T @ x
        chol_sol = self.int_chol.solve_A(high_dim_rhs)
        sol = (x - self.A @ chol_sol / squared_sigma) / squared_sigma

        return sol
