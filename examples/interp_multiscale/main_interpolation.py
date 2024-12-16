from examples.interp_multiscale.parameters import parameters
from run_experiment import run_experiment


alpha_appdx = '_a' + str(parameters['alpha']).replace('.', '_')
parameters["plots_dir"] = f"examples/interp_multiscale/plots{alpha_appdx}/"
run_experiment(parameters)
