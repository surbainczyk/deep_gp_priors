from examples.interp_corner_slope.parameters import parameters
from run_experiment import run_experiment


parameters["plots_dir"] = "pCN_det_free_2D_rational/interp_single_step/plots/"
run_experiment(parameters)
