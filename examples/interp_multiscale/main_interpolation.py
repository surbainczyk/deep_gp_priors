from examples.interp_multiscale.parameters import parameters
from run_experiment import run_experiment


parameters["plots_dir"] = "pCN_det_free_2D_rational/interpolation/plots/"
run_experiment(parameters)
