import numpy as np
from fenics import *
from sksparse.cholmod import cholesky
from scipy.sparse import diags

from .RationalApproximation import RationalApproximation


parameters["reorder_dofs_serial"] = False
set_log_level(30)


class FESolver:
    def __init__(self, n_dof, fraction, k=3, set_up_all=True):
        if np.sqrt(n_dof) != int(np.sqrt(n_dof)):
            raise Exception("n_dof was not a square number.")
        
        self.n_dof = n_dof

        self.fraction = fraction
        self.frac_power = - (fraction - np.floor(fraction))

        self.shifted_chol_factors = {}

        if set_up_all:
            interval = self.select_interval()
            # interval = [10, 4 * (n_dof - 1) ** 2 + 150 ** 2]
            self.rational_approx = RationalApproximation(k=k, interval=interval)

            self.set_up_fenics_variables()
    
    def copy(self):
        copied_fe_solver = FESolver(self.n_dof, self.fraction, set_up_all=False)

        # reuse computed quantities
        copied_fe_solver.rational_approx = self.rational_approx

        copied_fe_solver.mesh = self.mesh
        copied_fe_solver.function_space = self.function_space
        copied_fe_solver.u = self.u
        copied_fe_solver.v = self.v

        for name in ["mass_matrix", "laplacian_matrix", "inv_mass", "sqrt_mass", "mass_solve",
                     "sqrt_mass_solve", "sqrt_mass_solve_T", "ptr_diffs", "ind_counts", "inv_idx", "chol"]:
            try:
                setattr(copied_fe_solver, name, getattr(self, name))
            except AttributeError:
                pass

        old_factors = self.shifted_chol_factors
        copied_fe_solver.shifted_chol_factors = {key: old_factors[key].copy() for key in list(old_factors.keys())}

        return copied_fe_solver
    
    def select_interval(self):
        F_minus = 50
        F_plus  = 150 ** 2

        min_eval = F_minus
        max_eval = F_plus + 4 * self.n_dof ** 2

        interval = [min_eval, max_eval]

        return interval
    
    def set_up_fenics_variables(self):
        n_edge_elements = int(np.sqrt(self.n_dof)) - 1
        self.mesh = UnitSquareMesh(n_edge_elements, n_edge_elements)
        self.function_space = FunctionSpace(self.mesh, "CG", 1)
        self.u = TrialFunction(self.function_space)
        self.v = TestFunction(self.function_space)
    
    def compute_random_rhs(self, random_sample):
        try:
            rhs = self.sqrt_mass_solve(random_sample)
        except AttributeError:
            self.compute_sqrt_mass_matrix()
            rhs = self.sqrt_mass_solve(random_sample)

        return rhs
    
    def compute_random_rhs_T(self, random_sample):
        try:
            rhs = self.sqrt_mass_solve_T(random_sample)
        except AttributeError:
            self.compute_sqrt_mass_matrix()
            rhs = self.sqrt_mass_solve_T(random_sample)

        return rhs
    
    def compute_random_rhs_inv(self, random_sample):
        try:
            rhs = self.sqrt_mass @ random_sample
        except AttributeError:
            self.compute_sqrt_mass_matrix()
            rhs = self.sqrt_mass @ random_sample

        return rhs
    
    def compute_random_rhs_inv_T(self, random_sample):
        try:
            rhs = self.sqrt_mass.T @ random_sample
        except AttributeError:
            self.compute_sqrt_mass_matrix()
            rhs = self.sqrt_mass.T @ random_sample

        return rhs
    
    def compute_mass_matrix(self):
        a = self.u * self.v * dx
        A = assemble(a, tensor=EigenMatrix())
        self.mass_matrix = A.sparray()
        mass_vector = np.array(self.mass_matrix.sum(axis=0)).flatten()

        self.inv_mass = diags(1 / mass_vector, format="csr")

        # quantities needed for fast computation of scaled matrix
        self.ptr_diffs = np.diff(self.mass_matrix.indptr)
        self.ind_counts = np.bincount(self.mass_matrix.indices)
        sort_idx = np.argsort(self.mass_matrix.indices)
        self.inv_idx = np.argsort(sort_idx)
    
    def compute_sqrt_mass_matrix(self):
        try:
            mass_chol = cholesky(self.mass_matrix.T, ordering_method='natural')
        except AttributeError:
            self.compute_mass_matrix()
            mass_chol = cholesky(self.mass_matrix.T, ordering_method='natural')
        self.sqrt_mass = mass_chol.L().T
        self.mass_solve = lambda x: mass_chol.solve_A(x)
        self.sqrt_mass_solve = lambda x: mass_chol.solve_Lt(x, use_LDLt_decomposition=False)
        self.sqrt_mass_solve_T = lambda x: mass_chol.solve_L(x, use_LDLt_decomposition=False)
    
    def compute_laplacian_matrix(self):
        s = dot(grad(self.u), grad(self.v)) * dx
        S = assemble(s, tensor=EigenMatrix())
        self.laplacian_matrix = S.sparray()
    
    def compute_operator_matrix(self, diag_vector): 
        try:
            scaled_mass = self.mass_matrix.copy()
            sorted_diag = (diag_vector ** 0.5).repeat(self.ptr_diffs) * (diag_vector ** 0.5).repeat(self.ind_counts)[self.inv_idx]
            scaled_mass.data *= sorted_diag
            op_matrix = self.laplacian_matrix + scaled_mass
        except AttributeError:
            self.compute_laplacian_matrix()
            self.compute_mass_matrix()
            scaled_mass = self.mass_matrix.copy()
            sorted_diag = (diag_vector ** 0.5).repeat(self.ptr_diffs) * (diag_vector ** 0.5).repeat(self.ind_counts)[self.inv_idx]
            scaled_mass.data *= sorted_diag
            op_matrix = self.laplacian_matrix + scaled_mass

        return op_matrix
    
    def solve_with_mass(self, y):
        try:
            x = self.mass_solve(y)
        except AttributeError:
            self.compute_sqrt_mass_matrix()
            x = self.mass_solve(y)
        
        return x
    
    def solve_with_fractional_operator(self, diag_vector, rhs_array):
        sol = rhs_array

        for i in range(int(np.floor(self.fraction))):
            integrated_rhs = self.mass_matrix @ sol
            if i == 0 and not diag_vector is None:
                sol = self.simple_solve(diag_vector, integrated_rhs)
            else:    # avoid new Cholesky factorisation
                sol = self.chol.solve_A(integrated_rhs)

        if self.frac_power != 0:
            sol = self.solve_fractional_part(diag_vector, sol)

        return sol
    
    def solve_with_fractional_operator_T(self, diag_vector, rhs_array):
        sol = rhs_array

        for i in range(int(np.floor(self.fraction))):
            if i == 0 and not diag_vector is None:
                sol = self.simple_solve(diag_vector, sol)
            else:    # avoid new Cholesky factorisation
                sol = self.chol.solve_A(sol)    # solving for A is fine because A is symmetric
            sol = self.mass_matrix @ sol

        if self.frac_power != 0:
            sol = self.solve_fractional_part_T(diag_vector, sol)

        return sol
    
    def solve_with_fractional_operator_inv(self, diag_vector, rhs_array):
        operator_matrix = self.compute_operator_matrix(diag_vector)
        sol = rhs_array

        for _ in range(int(np.floor(self.fraction))):
            sol = operator_matrix @ sol
            try:
                sol = self.mass_solve(sol)
            except AttributeError:
                self.compute_sqrt_mass_matrix()
                sol = self.mass_solve(sol)

        if self.frac_power != 0:
            sol = self.solve_fractional_part_inv(operator_matrix, sol)

        return sol
    
    def solve_with_fractional_operator_inv_T(self, diag_vector, rhs_array):
        operator_matrix = self.compute_operator_matrix(diag_vector)
        sol = rhs_array

        for _ in range(int(np.floor(self.fraction))):
            try:
                sol = self.mass_solve(sol)
            except AttributeError:
                self.compute_sqrt_mass_matrix()
                sol = self.mass_solve(sol)
            sol = operator_matrix @ sol

        if self.frac_power != 0:
            sol = self.solve_fractional_part_inv_T(operator_matrix, sol)

        return sol
    
    def solve_fractional_part(self, diag_vector, rhs_array):
        try:
            c_0, c, d = self.rational_approx.get_rational_approx_coeffs()
        except AttributeError:
            self.rational_approx.compute_rat_function(self.frac_power)
            c_0, c, d = self.rational_approx.get_rational_approx_coeffs()

        integrated_rhs = self.mass_matrix @ rhs_array

        if diag_vector is None:
            shifted_sols = np.array([self.shifted_chol_factors[shift].solve_A(integrated_rhs) for shift in d])
        else:
            operator_matrix = self.compute_operator_matrix(diag_vector)
            shifted_sols = np.array([self.shifted_solve(operator_matrix, shift, integrated_rhs) for shift in d])

        if len(rhs_array.shape) == 1:
            combined_sol = c_0 * rhs_array + c @ shifted_sols
        else:
            combined_sol = c_0 * rhs_array + np.sum(c[:, np.newaxis, np.newaxis] * shifted_sols, axis=0)

        return combined_sol
    
    def solve_fractional_part_T(self, diag_vector, rhs_array):
        try:
            c_0, c, d = self.rational_approx.get_rational_approx_coeffs()
        except AttributeError:
            self.rational_approx.compute_rat_function(self.frac_power)
            c_0, c, d = self.rational_approx.get_rational_approx_coeffs()

        if diag_vector is None:
            shifted_sols = np.array([self.shifted_chol_factors[shift].solve_A(rhs_array) for shift in d])
        else:
            operator_matrix = self.compute_operator_matrix(diag_vector)
            shifted_sols = np.array([self.shifted_solve(operator_matrix, shift, rhs_array) for shift in d])
        
        if len(rhs_array.shape) == 1:
            combined_sol = c_0 * rhs_array + self.mass_matrix @ (c @ shifted_sols)
        else:
            combined_sol = c_0 * rhs_array + self.mass_matrix @ (np.sum(c[:, np.newaxis, np.newaxis] * shifted_sols, axis=0))

        return combined_sol
    
    def solve_fractional_part_inv(self, operator_matrix, rhs_array, reuse_chol=False):
        try:
            c_0, c, d = self.rational_approx.get_rational_approx_inv_coeffs()
        except AttributeError:
            self.rational_approx.compute_rat_function(self.frac_power)
            c_0, c, d = self.rational_approx.get_rational_approx_inv_coeffs()

        integrated_rhs = self.mass_matrix @ rhs_array
        finished = False
        if reuse_chol:
            try:
                shifted_sols = np.array([self.shifted_chol_factors[shift].solve_A(integrated_rhs) for shift in d])
                finished = True
            except KeyError:
                pass
        if not finished:
            shifted_sols = np.array([self.shifted_solve(operator_matrix, shift, integrated_rhs) for shift in d])
        if len(rhs_array.shape) == 1:
            combined_sol = c_0 * rhs_array + c @ shifted_sols
        else:
            combined_sol = c_0 * rhs_array + np.sum(c[:, np.newaxis, np.newaxis] * shifted_sols, axis=0)

        return combined_sol
    
    def solve_fractional_part_inv_T(self, operator_matrix, rhs_array, reuse_chol=False):
        try:
            c_0, c, d = self.rational_approx.get_rational_approx_inv_coeffs()
        except AttributeError:
            self.rational_approx.compute_rat_function(self.frac_power)
            c_0, c, d = self.rational_approx.get_rational_approx_inv_coeffs()

        finished = False
        if reuse_chol:
            try:
                shifted_sols = np.array([self.shifted_chol_factors[shift].solve_A(rhs_array) for shift in d])
                finished = True
            except KeyError:
                pass
        if not finished:
            shifted_sols = np.array([self.shifted_solve(operator_matrix, shift, rhs_array) for shift in d])
        if len(rhs_array.shape) == 1:
            combined_sol = c @ shifted_sols
        else:
            combined_sol = np.sum(c[:, np.newaxis, np.newaxis] * shifted_sols, axis=0)
        sol = c_0 * rhs_array + self.mass_matrix @ combined_sol

        return sol
    
    def simple_solve(self, diag_vector, integrated_rhs):
        op_matrix = self.compute_operator_matrix(diag_vector)
        try:
            self.chol.cholesky_inplace(op_matrix.T)
        except AttributeError:
            self.chol = cholesky(op_matrix.T)
        
        sol = self.chol.solve_A(integrated_rhs)

        return sol
    
    def shifted_solve(self, op_matrix, shift, integrated_rhs):
        shifted_op = op_matrix - shift * self.mass_matrix
        
        try:
            self.shifted_chol_factors[shift].cholesky_inplace(shifted_op.T)
        except KeyError:
            try:
                new_factor = self.shifted_chol_factors[list(self.shifted_chol_factors.keys())[0]].copy()
                new_factor.cholesky_inplace(shifted_op.T)
            except IndexError:
                new_factor = cholesky(shifted_op.T)
            
            self.shifted_chol_factors[shift] = new_factor
        
        sol = self.shifted_chol_factors[shift].solve_A(integrated_rhs)

        return sol
