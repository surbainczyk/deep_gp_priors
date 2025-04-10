import progressbar
import numpy as np
from scipy.sparse import eye
from sksparse.cholmod import cholesky
from sparse_dot_mkl import dot_product_mkl as dot_mkl
from time import time

from utils import numba_sort_indices


class pCNSampler:
    def __init__(self, deep_gp):
        self.deep_gp = deep_gp

    def initialise_observations(self, observed, forward_op, noise_std):
        self.observed = observed
        self.A = forward_op
        self.A_scaled = forward_op / noise_std
        
        self.squared_sigma = noise_std ** 2
        self.Gamma = self.squared_sigma * eye(observed.size, format="csr")

        # quantities for potential evaluation
        self.y_norm_squared = observed @ observed
        self.high_dim_data  = self.A.T @ observed
        self.high_dim_noise = dot_mkl(self.A.T, self.A, cast=True) / self.squared_sigma

    def run_mcmc(self, its=int(1e3), burn_in=int(2e2), initial_state=None, beta=0.025, accept_rate=0.3, prior_std=1.0,
                 beta_split=int(1e2), verbose=True, breakpoint_its=None, store_iterates=False):
        # following the non-centred algorithm, Algorithm 1, in "How deep are deep Gaussian processes?"
        self.set_up_mcmc_variables(its, store_iterates=store_iterates)
        start = time()
        
        old_proposal = np.zeros(self.deep_gp.n_dof) if initial_state is None else initial_state
        old_u, Q, middle_mat, log_det_QDQ, old_diag = self.deep_gp.evaluate(old_proposal)
        
        self.potential_vals[0], old_regr = self.potential_and_regression(Q, log_det_QDQ, middle_mat)
        old_logdet = self.chol_factor.logdet()
        self.logdet_diffs = np.zeros(its)    # debug output
        
        min_pot = np.inf

        accepted_hist = []
        if breakpoint_its:
            self.breakpoint_results = {}

        if verbose:
            bar, format_custom_text = self.set_up_progress_bar(its)

        for i in range(its):
            # proposal step
            random_step = prior_std * np.random.randn(self.deep_gp.n_dof)
            proposal = np.sqrt(1 - beta ** 2) * old_proposal + beta * random_step
            u_proposal, Q, middle_mat, log_det_QDQ, diag_proposal = self.deep_gp.evaluate(proposal)

            # accept/reject
            potential_val, regr_vals = self.potential_and_regression(Q, log_det_QDQ, middle_mat)
            logdet = self.chol_factor.logdet()
            self.logdet_diffs[i] = abs(logdet - old_logdet)
            
            a = self.potential_vals[i] - potential_val

            unif = np.random.uniform()
            accepted = np.log(unif) < a
            accepted_hist.append(accepted)

            # update variables
            if accepted:
                old_proposal = proposal
                old_u = u_proposal
                old_diag = diag_proposal
                old_regr = regr_vals

                old_logdet = logdet

                if i >= burn_in and potential_val < min_pot:
                    self.u_map = u_proposal
                    self.F_map = diag_proposal
                    self.regr_map = regr_vals

                    min_pot = potential_val
            
            self.potential_vals[i + 1] = potential_val if accepted else self.potential_vals[i]

            if store_iterates:
                self.prop_array[:, i] = old_proposal

            if i >= burn_in:
                counter = i - burn_in + 1
                self.u_mean, self.u_sse = update_stochastics(old_u, self.u_mean, self.u_sse, counter=counter)
                self.F_mean, self.F_sse = update_stochastics(old_diag, self.F_mean, self.F_sse, counter=counter)
                self.regr_mean, self.regr_sse = update_stochastics(old_regr, self.regr_mean, self.regr_sse, counter=counter)

            if (i + 1) % beta_split == 0:
                # update beta to achieve desired acceptance rate
                acc_mean = np.mean(accepted_hist[-beta_split:])
                beta *= 4 / 5 if acc_mean < accept_rate else 5 / 4
                beta = min(beta, 1)

                if verbose:
                    self.update_progress_bar(bar, format_custom_text, i + 1, acc_mean)
            
            if breakpoint_its and i + 1 in breakpoint_its:
                self.breakpoint_results[i + 1] = [self.regr_mean.copy(), self.F_mean.copy(), time() - start]
        
        if verbose:
            bar.finish()
        
        self.u_var = self.u_sse / (counter - 1)
        self.F_var = self.F_sse / (counter - 1)
        self.regr_var = self.regr_sse / (counter - 1)
        self.last_state = old_proposal

        self.run_time = time() - start
        print(f"Final beta was beta = {beta}")

    def set_up_mcmc_variables(self, its, store_iterates=False):
        self.potential_vals = np.zeros(its + 1)
        self.u_mean = self.u_map = self.u_sse = 0
        self.F_mean = self.F_map = self.F_sse = 0
        self.regr_mean = self.regr_map = self.regr_sse = 0

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
    
    def potential_and_regression(self, Q, log_det_QDQ, middle_mat):
        # compute required matrix products
        QDQ = dot_mkl(dot_mkl(Q.T, middle_mat), Q)
        QDQ_noisy = QDQ + self.high_dim_noise

        numba_sort_indices(QDQ_noisy.indptr, QDQ_noisy.indices, QDQ_noisy.data)

        try:
            self.chol_factor.cholesky_inplace(QDQ_noisy)
        except AttributeError:
            self.chol_factor = cholesky(QDQ_noisy)

        # compute regression values and quadratic potential term
        regr_values = self.chol_factor.solve_A(self.high_dim_data) / self.squared_sigma
        quadr_term = (self.y_norm_squared - self.high_dim_data.T @ regr_values) / self.squared_sigma

        # compute log determinant term
        det_noisy = self.chol_factor.logdet()
        det_term = det_noisy - log_det_QDQ

        psi_val = 0.5 * (quadr_term + det_term)

        return psi_val, regr_values


def update_stochastics(new_val, mean, sse, counter):
    e = new_val - mean
    mean = mean + e / counter
    sse = sse + e * (new_val - mean)

    return mean, sse
