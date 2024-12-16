from examples.interp_corner_slope.parameters import parameters
from run_experiment import run_experiment


alpha_appdx = '_a' + str(parameters['alpha']).replace('.', '_')
parameters["plots_dir"] = f"examples/interp_corner_slope/plots{alpha_appdx}/"
run_experiment(parameters)
