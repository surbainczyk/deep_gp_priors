from examples.radon_full.parameters import parameters
from run_experiment import run_experiment


parameters["plots_dir"] = "pCN_det_free_2D_rational/radon_full/plots/"
run_experiment(parameters)
