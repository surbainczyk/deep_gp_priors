from examples.radon_sparse.parameters import parameters
from run_experiment import run_experiment


parameters["plots_dir"] = "pCN_det_free_2D_rational/radon_sparse/plots/"
run_experiment(parameters)
