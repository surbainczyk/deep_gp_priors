import progressbar
import numpy as np
from scipy.sparse import eye
from time import time

from pCN_det_free_fractional.LSQRSolver import LSQRSolver


class pCNDetFreeSampler:
    def __init__(self, deep_gp, fix_precond=1):
        self.deep_gp = deep_gp
        self.fix_precond = fix_precond
        self.precond_use_counter = 1
        self.recomputed_precond = False

    def initialise_observations(self, observed, forward_op, noise_std):
        self.observed = observed
        self.A = forward_op
        
        self.noise_std = noise_std
        self.squared_sigma = noise_std ** 2
        self.Gamma = self.squared_sigma * eye(observed.size, format="csr")

        # preparing potential evaluation
        self.old_solver = self.prop_solver = LSQRSolver(self.deep_gp, self.A, noise_std)
        self.high_dim_data  = self.A.T @ observed

    def run_mcmc(self, its=int(1e3), burn_in=int(2e2), initial_state=None, beta=0.025, accept_rate=0.3, prior_std=1.0,
                 beta_split=int(1e2), verbose=True, breakpoint_its=None, store_iterates=False):
        self.set_up_mcmc_variables(its, store_iterates=store_iterates)
        start = time()

        old_proposal = np.zeros(self.deep_gp.n_dof) if initial_state is None else initial_state
        top_sample = np.random.randn(self.deep_gp.layer_n_dof)
        old_u, old_u_top, old_diag = self.deep_gp.evaluate(old_proposal, top_sample)
        
        self.prop_solver.compute_integer_chol(old_diag)
        self.potential_vals[0] = self.potential(old_diag)
        
        min_pot = np.inf

        self.accepted_hist = []
        self.beta_vals = []
        self.potential_its = []
        self.auxiliary_its = []
        if breakpoint_its:
            self.breakpoint_results = {}

        if verbose:
            bar, format_custom_text = self.set_up_progress_bar(its)

        for i in range(its):
            z = self.generate_auxiliary_variable(old_u_top, old_diag)
            self.auxiliary_its.append(self.old_solver.last_itn)

            # proposal step
            random_step = prior_std * np.random.randn(self.deep_gp.n_dof)
            proposal = np.sqrt(1 - beta ** 2) * old_proposal + beta * random_step
            top_sample = np.random.randn(self.deep_gp.layer_n_dof)
            u_proposal, u_top, diag_proposal = self.deep_gp.evaluate(proposal, top_sample)

            # accept/reject
            potential_val = self.potential(diag_proposal)
            self.potential_its.append(self.prop_solver.last_itn)
            aux_potential_val = self.auxiliary_potential(z, diag_proposal, old_diag)
            
            a = self.potential_vals[i] - potential_val - aux_potential_val

            unif = np.random.uniform()
            accepted = np.log(unif) < a
            self.accepted_hist.append(accepted)
            self.beta_vals.append(beta)

            # update variables
            if accepted:
                old_proposal = proposal
                old_u = u_proposal
                old_diag = diag_proposal
                old_u_top = u_top

                self.old_solver = self.prop_solver
                self.prop_solver = self.prop_solver.copy()
                self.precond_use_counter += 1
                if self.recomputed_precond:
                    self.precond_use_counter = 1
                    self.recomputed_precond = False

                if i >= burn_in and potential_val < min_pot:
                    self.u_map = u_proposal
                    self.F_map = diag_proposal

                    min_pot = potential_val
            
            self.potential_vals[i + 1] = potential_val if accepted else self.potential_vals[i]
            self.aux_potential_vals[i] = aux_potential_val if accepted else self.aux_potential_vals[i - 1]
            self.comb_potential_vals[i] = potential_val - aux_potential_val if accepted else self.comb_potential_vals[i - 1]

            if store_iterates:
                self.prop_array[:, i] = old_proposal

            if i >= burn_in:
                self.update_mcmc_statistics(old_u, old_diag)

            if (i + 1) % beta_split == 0:
                # update beta to achieve desired acceptance rate
                acc_mean = np.mean(self.accepted_hist[-beta_split:])
                beta *= 4 / 5 if acc_mean < accept_rate else 5 / 4
                beta = min(beta, 1)

                if verbose:
                    self.update_progress_bar(bar, format_custom_text, i + 1, acc_mean)
            
            if breakpoint_its and i + 1 in breakpoint_its:
                self.breakpoint_results[i + 1] = [None, self.F_mean.copy(), time() - start]
        
        if verbose:
            bar.finish()
        
        self.run_time = time() - start
        print(f"\nLast value of beta: {beta:.4e}")
        print(f"\nLast StD of diagonal vector: {np.std(old_diag):.4e}")

        self.compute_variances()

    def set_up_mcmc_variables(self, its, store_iterates=False):
        self.potential_vals = np.zeros(its + 1)
        self.aux_potential_vals = np.zeros(its)
        self.comb_potential_vals = np.zeros(its)

        self.stoch_counter = 0
        self.u_mean = self.u_map = self.u_sse = 0
        self.F_mean = self.F_map = self.F_sse = 0

        if store_iterates:
            self.prop_array = np.zeros((self.deep_gp.n_dof, its))
    
    def set_up_progress_bar(self, its):
        format_custom_text = progressbar.FormatCustomText(format='Acceptance rate: %(acc).2f', mapping=dict(acc=0.0))
        widgets = [
            ' [',
            progressbar.Timer(format= 'Elapsed time: %(elapsed)s'),
            '] ',
            progressbar.Bar('#'),' (',
            format_custom_text, ') '
            ]
        bar = progressbar.ProgressBar(max_value=its, widgets=widgets).start()

        return bar, format_custom_text

    def update_progress_bar(self, bar, format_custom_text, i, acc_mean):
        format_custom_text.update_mapping(acc=acc_mean)
        bar.update(i)
    
    def potential(self, diag):
        self.prop_solver.gp_layer = self.deep_gp.top_layer.copy()    # reuse Cholesky factors from deepGP.evaluate
        if self.precond_use_counter >= self.fix_precond:
            self.prop_solver.compute_integer_chol(diag)
            self.recomputed_precond = True
        
        sol = self.prop_solver.solve(diag, self.observed, is_old_diag=True)
        quadr_term = self.observed @ sol
        psi_val = 0.5 * quadr_term

        return psi_val
    
    def generate_auxiliary_variable(self, u_top, diag):
        noise = self.noise_std * np.random.randn(len(self.observed))
        w = self.A @ u_top + noise

        z = self.old_solver.solve(diag, w, is_old_diag=True)

        return z
    
    def auxiliary_potential(self, z, diag, old_diag):
        high_dim_z = self.A.T @ z
        z_norm_squared = z @ z

        gp = self.prop_solver.gp_layer
        C = gp.apply_C

        # compute quadratic term with w
        gp.ignore_diag_vector = True
        sol = C(diag, high_dim_z, fix_diag=True)
        quadr_term = self.squared_sigma * z_norm_squared + high_dim_z @ sol

        # compute quadratic term with w and old matrices
        old_gp = self.old_solver.gp_layer
        old_C = old_gp.apply_C

        old_gp.ignore_diag_vector = True
        sol_old = old_C(old_diag, high_dim_z, fix_diag=True)
        quadr_term_old = self.squared_sigma * z_norm_squared + high_dim_z @ sol_old
        
        # compute potential value
        psi_val = 0.5 * (quadr_term - quadr_term_old)

        return psi_val
    
    def regression(self, diag):
        gp = self.deep_gp.top_layer
        C = gp.apply_C

        sol = self.old_solver.solve(diag, self.observed, tol=1e-10)
        regr_vals = C(diag, self.A.T @ sol)

        return regr_vals
    
    def compute_top_layer_sample(self, diag):
        tol = 1e-10

        gp = self.deep_gp.top_layer
        C = gp.apply_C

        sol = self.old_solver.solve(diag, self.observed, tol=tol)
        regr_mean = C(diag, self.A.T @ sol)

        # compute covariance contribution of top layer sample
        xi_1 = np.random.randn(self.observed.size)
        sample_1 = self.A.T @ xi_1 / self.noise_std
        xi_2 = np.random.randn(self.A.shape[1])
        sample_2 = gp.evaluate_inv_T(diag, xi_2)
        
        prod_1 = C(diag, sample_1 + sample_2)
        sol_2 = self.old_solver.solve(diag, self.A @ C(diag, sample_1 + sample_2), tol=tol)
        prod_2 = C(diag, self.A.T @ sol_2)
        
        regr_cov_sample = prod_1 - prod_2

        return regr_mean + regr_cov_sample
    
    def update_mcmc_statistics(self, old_u, old_diag):
        self.stoch_counter += 1
        self.u_mean, self.u_sse = update_stochastics(old_u, self.u_mean, self.u_sse, self.stoch_counter)
        self.F_mean, self.F_sse = update_stochastics(old_diag, self.F_mean, self.F_sse, self.stoch_counter)
    
    def compute_variances(self):
        self.u_var = self.u_sse / (self.stoch_counter - 1)
        self.F_var = self.F_sse / (self.stoch_counter - 1)


def update_stochastics(new_val, mean, sse, counter):
    e = new_val - mean
    mean = mean + e / counter
    sse = sse + e * (new_val - mean)

    return mean, sse
