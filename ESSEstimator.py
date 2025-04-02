import numpy as np
import arviz as az


class ESSEstimator:
    def compute_lugsail_ess(self, chain, use_bm_estimator=False):
        samples, dim = chain.shape

        mu = np.mean(chain, axis=0)

        shifted_chain = chain - mu
        S = np.cov(shifted_chain, rowvar=False)

        b_by_three = int(np.floor(np.sqrt(samples) / 3))
        a = b = 3 * b_by_three

        # compute \hat{T}_b
        mean_mat = np.zeros((a, samples))
        for k in range(a):
            mean_mat[k, (k * b):((k+1) * b)] = 1 / b
        shifted_Y = mean_mat @ shifted_chain
        T_hat_b = b * np.cov(shifted_Y, rowvar=False)

        if use_bm_estimator:
            T_hat = T_hat_b
            T_eigv = np.linalg.eigvalsh(T_hat)
            T_thresh = max(abs(np.amin(T_eigv)) * 10, 1e-10)
            logdet_T_hat = np.sum(np.log(T_eigv[T_eigv > T_thresh]))    # use pseudo-determinant
        else:
            # compute \hat{T}_{b/3}
            mean_mat_bb = np.zeros((3 * a, samples))
            for k in range(3 * a):
                mean_mat_bb[k, (k * b_by_three):((k+1) * b_by_three)] = 1 / b_by_three
            shifted_Y_bb = mean_mat_bb @ shifted_chain
            T_hat_bb = b_by_three * np.cov(shifted_Y_bb, rowvar=False)

            T_hat = 2 * T_hat_b - T_hat_bb
            logdet_T_hat = np.linalg.slogdet(T_hat)[1]
        
        Sigma_hat = S + T_hat / samples

        if use_bm_estimator:
            S_eigv = np.linalg.eigvalsh(Sigma_hat)
            S_thresh = max(abs(np.amin(S_eigv)) * 10, 1e-10)
            logdet_S_hat = np.sum(np.log(S_eigv[S_eigv > S_thresh]))    # use pseudo-determinant
            
            S_eigv = np.linalg.eigvalsh(S)
            S_thresh = max(abs(np.amin(S_eigv)) * 10, 1e-10)
        else:
            logdet_S_hat = np.linalg.slogdet(Sigma_hat)[1]
        
        ess = samples * np.exp(1 / dim * (logdet_S_hat - logdet_T_hat))

        return ess
    
    def compute_lugsail_ess_multichain(self, chain_list, use_bm_estimator=False):
        samples, dim = chain_list[0].shape

        mu = np.mean([np.mean(chain, axis=0) for chain in chain_list], axis=0)

        shifted_chain_list = [chain - mu for chain in chain_list]
        S = np.mean([np.cov(shifted_chain, rowvar=False) for shifted_chain in shifted_chain_list], axis=0)

        b_by_three = int(np.floor(np.sqrt(samples) / 3))
        a = b = 3 * b_by_three

        # compute \hat{T}_b
        mean_mat = np.zeros((a, samples))
        for k in range(a):
            mean_mat[k, (k * b):((k+1) * b)] = 1 / b
        
        mean_prod = np.zeros((dim, dim))
        for shifted_chain in shifted_chain_list:
            shifted_batch = mean_mat @ shifted_chain
            mean_prod += shifted_batch.T @ shifted_batch
        T_hat_b = b / (a * len(chain_list) - 1) * mean_prod

        if use_bm_estimator:
            T_hat = T_hat_b
            T_eigv = np.linalg.eigvalsh(T_hat)
            T_thresh = max(abs(np.amin(T_eigv)) * 10, 1e-10)
            logdet_T_hat = np.sum(np.log(T_eigv[T_eigv > T_thresh]))    # use pseudo-determinant
        else:
            # compute \hat{T}_{b/3}
            mean_mat_bb = np.zeros((3 * a, samples))
            for k in range(3 * a):
                mean_mat_bb[k, (k * b_by_three):((k+1) * b_by_three)] = 1 / b_by_three
            
            mean_prod_bb = np.zeros((dim, dim))
            for shifted_chain in shifted_chain_list:
                shifted_batch_bb = mean_mat_bb @ shifted_chain
                mean_prod_bb += shifted_batch_bb.T @ shifted_batch_bb
            T_hat_bb = b_by_three / (3 * a * len(chain_list) - 1) * mean_prod_bb

            T_hat = 2 * T_hat_b - T_hat_bb
            logdet_T_hat = np.linalg.slogdet(T_hat)[1]
        
        Sigma_hat = S + T_hat / samples

        if use_bm_estimator:
            S_eigv = np.linalg.eigvalsh(Sigma_hat)
            S_thresh = max(abs(np.amin(S_eigv)) * 10, 1e-10)
            logdet_S_hat = np.sum(np.log(S_eigv[S_eigv > S_thresh]))    # use pseudo-determinant
            
            S_eigv = np.linalg.eigvalsh(S)
            S_thresh = max(abs(np.amin(S_eigv)) * 10, 1e-10)
        else:
            logdet_S_hat = np.linalg.slogdet(Sigma_hat)[1]
        
        ess = len(chain_list) * samples * np.exp(1 / dim * (logdet_S_hat - logdet_T_hat))

        return ess

    def compute_smallest_lugsail_ess(self, chain):
        samples, _ = chain.shape

        mu = np.mean(chain, axis=0)

        shifted_chain = chain - mu[np.newaxis, :]
        S = np.var(shifted_chain, axis=0, ddof=1)

        b_by_three = int(np.floor(np.sqrt(samples) / 3))
        a = b = 3 * b_by_three
        
        mean_mat_b = np.zeros((a, samples))
        for k in range(a):
            mean_mat_b[k, (k * b):((k+1) * b)] = 1 / b
        shifted_Y = mean_mat_b @ shifted_chain
        tau_b = b * np.var(shifted_Y, axis=0, ddof=1)
        
        mean_mat_bb = np.zeros((3 * a, samples))
        for k in range(3 * a):
            mean_mat_bb[k, (k * b_by_three):((k+1) * b_by_three)] = 1 / b_by_three
        shifted_Y3 = mean_mat_bb @ shifted_chain
        tau_bb = b_by_three * np.var(shifted_Y3, axis=0, ddof=1)

        tau_L = 2 * tau_b - tau_bb
        sigma_L = S + tau_L / samples

        ess = samples * sigma_L / tau_L

        smallest_ess = min(ess)
        median_ess = np.median(ess)
        
        return smallest_ess, median_ess

    def compute_smallest_lugsail_ess_multichain(self, chain_list):
        samples, dim = chain_list[0].shape

        mu = np.mean([np.mean(chain, axis=0) for chain in chain_list], axis=0)

        shifted_chain_list = [chain - mu for chain in chain_list]
        S = np.mean([np.var(shifted_chain, axis=0, ddof=1) for shifted_chain in shifted_chain_list], axis=0)

        b_by_three = int(np.floor(np.sqrt(samples) / 3))
        a = b = 3 * b_by_three
        
        mean_mat_b = np.zeros((a, samples))
        for k in range(a):
            mean_mat_b[k, (k * b):((k+1) * b)] = 1 / b
        
        sum_prod = np.zeros(dim)
        for shifted_chain in shifted_chain_list:
            shifted_batch = mean_mat_b @ shifted_chain
            sum_prod += np.sum(shifted_batch ** 2, axis=0)
        tau_b = b * (a * len(chain_list) - 1) * sum_prod
        
        mean_mat_bb = np.zeros((3 * a, samples))
        for k in range(3 * a):
            mean_mat_bb[k, (k * b_by_three):((k+1) * b_by_three)] = 1 / b_by_three
        
        sum_prod_bb = np.zeros(dim)
        for shifted_chain in shifted_chain_list:
            shifted_batch_bb = mean_mat_bb @ shifted_chain
            sum_prod_bb += np.sum(shifted_batch_bb ** 2, axis=0)
        tau_bb = b_by_three / (3 * a * len(chain_list) - 1) * sum_prod_bb

        tau_L = 2 * tau_b - tau_bb
        sigma_L = S + tau_L / samples

        ess = samples * sigma_L / tau_L

        smallest_ess = min(ess)
        median_ess = np.median(ess)
        
        return smallest_ess, median_ess

    def compute_smallest_ess(self, chain):
        dataset = az.convert_to_dataset(chain[np.newaxis, :])
        ess_data = az.ess(dataset)

        smallest_ess = min(ess_data.x.to_numpy())
        median_ess = np.median(ess_data.x.to_numpy())

        return smallest_ess, median_ess

    def compute_smallest_ess_multichain(self, chain_list):
        dataset = az.convert_to_dataset(np.array(chain_list))
        ess_data = az.ess(dataset)

        smallest_ess = min(ess_data.x.to_numpy())
        median_ess = np.median(ess_data.x.to_numpy())

        return smallest_ess, median_ess
