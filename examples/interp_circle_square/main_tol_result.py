import sys

from pCN_det_free_2D_rational.tol_comparison import run_tol_experiment

from pCN_det_free_2D_rational.GP_models.DeepGP import DeepGP
from pCN_det_free_2D_rational.pCNDetFreeSampler import pCNDetFreeSampler
from pCN_det_free_2D_rational.interp_circle.parameters import parameters


# Script for comparing performance for different values of alpha
plots_dir = "pCN_det_free_2D_rational/interp_circle/plots_tol/"

tol = float(sys.argv[1])
id_no = int(sys.argv[2])

deep_gp = DeepGP(layers=parameters["layers"], layer_n_dof=parameters["n_dof"], F=parameters["F"],
                 alpha=parameters["alpha"], base_diag=parameters["base_diag"], sigma=parameters["sigma"])
mcmc_solver = pCNDetFreeSampler(deep_gp, fix_precond=parameters["fix_precond"])

run_tol_experiment(mcmc_solver, parameters, tol, id_no, plots_dir)
