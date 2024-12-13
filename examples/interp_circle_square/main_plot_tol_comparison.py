from pCN_det_free_2D_rational.tol_comparison import plot_from_tol_results


# Script for plotting comparison from saved files
plots_dir = "pCN_det_free_2D_rational/interp_circle/plots_tol/"
tol_vals = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]

plot_from_tol_results(tol_vals, plots_dir)
