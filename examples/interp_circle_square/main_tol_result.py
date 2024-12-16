import sys

from tol_comparison import run_tol_experiment

from pCN_det_free_fractional.GP_models.DeepGP import DeepGP
from pCN_det_free_fractional.pCNDetFreeSampler import pCNDetFreeSampler
from examples.interp_circle_square.parameters import parameters


# Script for comparing performance for different tolerance values in LSQR solver
plots_dir = "examples/interp_circle_square/plots_tol/"

tol = float(sys.argv[1])
id_no = int(sys.argv[2])

deep_gp = DeepGP(layers=parameters["layers"], layer_n_dof=parameters["n_dof"], F=parameters["F"],
                 alpha=parameters["alpha"], base_diag=parameters["base_diag"], sigma=parameters["sigma"])
mcmc_solver = pCNDetFreeSampler(deep_gp, fix_precond=parameters["fix_precond"])

run_tol_experiment(mcmc_solver, parameters, tol, id_no, plots_dir)
