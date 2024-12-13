from scipy.optimize import minimize_scalar


def estimate_corr_length(gp, obs, A, noise_var, bounds=(0.01, 1.0)):
    def likelihood(rho):
        lh = gp.likelihood(rho, obs, A, noise_var)
        
        return lh

    result = minimize_scalar(likelihood, bounds=bounds)
    if result.success:
        opt_rho = result.x
    else:
        raise ValueError("minimize_scalar did not converge.")

    return opt_rho
