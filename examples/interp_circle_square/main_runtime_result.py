import sys
import numpy as np
from scipy.special import gamma

from runtime_comparison import run_ind_experiment

from pCN_nonfractional.GP_models.DeepGP import DeepGP as IntDeepGP
from pCN_nonfractional.pCNSampler import pCNSampler as IntMCMCSolver

from pCN_det_free_fractional.GP_models.DeepGP import DeepGP as RatDeepGP
from pCN_det_free_fractional.pCNDetFreeSampler import pCNDetFreeSampler as RatMCMCSolver
from examples.interp_circle_square.parameters import parameters


# Script for comparing performance for different values of alpha
plots_dir = "examples/interp_circle_square/plots_comp/"

iter_counts = [i * int(1e4) for i in range(2, 6)]
parameters["burn_in"] = int(1e4)

alpha = float(sys.argv[1])
id_no = int(sys.argv[2])

sigma = np.sqrt(gamma(alpha) * 4 * np.pi / gamma(alpha - 1))
parameters['alpha'] = alpha
parameters['sigma'] = sigma

is_fractional = np.floor(alpha / 2) != alpha / 2
DeepGPClass = RatDeepGP if is_fractional else IntDeepGP
MCMCSolverClass = RatMCMCSolver if is_fractional else IntMCMCSolver

run_ind_experiment(DeepGPClass, MCMCSolverClass, parameters, iter_counts, id_no, plots_dir)
