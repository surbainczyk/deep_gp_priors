import numpy as np
from fenics import *
from sksparse.cholmod import cholesky
from scipy.sparse import eye
from sparse_dot_mkl import dot_product_mkl as dot_mkl


parameters["reorder_dofs_serial"] = False
set_log_level(30)


class FESolver:
    def __init__(self, n_dof, power, k=3):
        if np.sqrt(n_dof) != int(np.sqrt(n_dof)):
            raise Exception("n_dof was not a square number.")
        
        self.n_dof = n_dof

        self.power = power

        self.set_up_fenics_variables()
    
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
    
    def compute_mass_matrix(self):
        a = self.u * self.v * dx
        A = assemble(a, tensor=EigenMatrix())
        self.mass_matrix = A.sparray()
        mass_vector = np.array(self.mass_matrix.sum(axis=0)).flatten()

        self.inv_mass = eye(self.n_dof, format="csr")
        self.inv_mass.setdiag(1 / mass_vector)

        self.log_det_mass = cholesky(self.mass_matrix.T).logdet()
    
    def compute_laplacian_matrix(self):
        s = dot(grad(self.u), grad(self.v)) * dx
        S = assemble(s, tensor=EigenMatrix())
        self.laplacian_matrix = S.sparray()
    
    def compute_operator_matrix(self, diag_vector): 
        diag_mat = eye(self.n_dof, format="csr")
        diag_mat.setdiag(diag_vector)
        try:
            op_matrix = self.laplacian_matrix + dot_mkl(diag_mat.power(0.5), dot_mkl(self.mass_matrix, diag_mat.power(0.5)))
        except AttributeError:
            self.compute_laplacian_matrix()
            self.compute_mass_matrix()
            op_matrix = self.laplacian_matrix + dot_mkl(diag_mat.power(0.5), dot_mkl(self.mass_matrix, diag_mat.power(0.5)))

        return op_matrix
    
    def solve_with_operator(self, diag_vector, rhs_array, is_base_layer=False):
        reuse_chol = is_base_layer and hasattr(self, "chol")
        if not reuse_chol:
            operator_matrix = self.compute_operator_matrix(diag_vector)
        sol = rhs_array

        for i in range(int(self.power)):
            integrated_rhs = self.mass_matrix @ sol
            if i == 0 and not reuse_chol:
                sol = self.simple_solve(operator_matrix, integrated_rhs)
            else:    # avoid new Cholesky factorisation
                sol = self.chol.solve_A(integrated_rhs)

        return sol
    
    def solve_with_operator_T(self, diag_vector, rhs_array, is_base_layer=False):
        reuse_chol = is_base_layer and hasattr(self, "chol")
        if not reuse_chol:
            operator_matrix = self.compute_operator_matrix(diag_vector)
        sol = rhs_array

        for i in range(int(self.power)):
            if i == 0 and not reuse_chol:
                sol = self.simple_solve(operator_matrix, sol)
            else:    # avoid new Cholesky factorisation
                sol = self.chol.solve_A(sol)    # solving for A is fine because A is symmetric
            sol = self.mass_matrix @ sol

        return sol
    
    def solve_with_operator_inv(self, diag_vector, rhs_array):
        operator_matrix = self.compute_operator_matrix(diag_vector)
        sol = rhs_array

        for _ in range(int(self.power)):
            sol = operator_matrix @ sol
            try:
                sol = self.mass_solve(sol)
            except AttributeError:
                self.compute_sqrt_mass_matrix()
                sol = self.mass_solve(sol)

        return sol
    
    def simple_solve(self, op_matrix, integrated_rhs):
        try:
            self.chol.cholesky_inplace(op_matrix.T)
        except AttributeError:
            self.chol = cholesky(op_matrix.T)
        
        sol = self.chol.solve_A(integrated_rhs)

        return sol
    
    def shifted_solve(self, op_matrix, shift, integrated_rhs):
        shifted_op = op_matrix - shift * self.mass_matrix
        
        try:
            self.shifted_chol.cholesky_inplace(shifted_op.T)
        except AttributeError:
            self.shifted_chol = cholesky(shifted_op.T)
        
        sol = self.shifted_chol.solve_A(integrated_rhs)

        return sol    
    def compute_polynomial_matrix(self, diag_vector):
        op_matrix = self.compute_operator_matrix(diag_vector)
        lin_op = dot_mkl(self.inv_mass, op_matrix)
            
        try:
            self.C_inv_chol.cholesky_inplace(op_matrix.T)
        except AttributeError:
            self.C_inv_chol = cholesky(op_matrix.T)
        
        lin_op_log_det = self.C_inv_chol.logdet()

        Q = eye(self.n_dof, format="csr")
        self.log_det_Q = 0
        
        int_power = int(np.floor(self.power))
        for _ in range(int_power):
            Q = dot_mkl(lin_op, Q)
            self.log_det_Q += lin_op_log_det + np.sum(np.log(self.inv_mass.data))

        return Q

