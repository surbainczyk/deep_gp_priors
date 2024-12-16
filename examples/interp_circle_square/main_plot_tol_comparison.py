from tol_comparison import plot_from_tol_results


# Script for plotting tolerance comparison from saved files
plots_dir = "examples/interp_circle_square/plots_tol/"
tol_vals = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

plot_from_tol_results(tol_vals, plots_dir)
