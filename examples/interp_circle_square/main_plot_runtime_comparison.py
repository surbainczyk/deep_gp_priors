from runtime_comparison import plot_from_ind_results, print_ess_values


# Plot run-time comparison from saved files
plots_dir = "examples/interp_circle_square/plots_comp/"
alpha_vals = [1.5, 2, 2.5, 3, 3.5, 4]

iter_counts = [i * int(1e4) for i in range(2, 6)]
print_ess_values(alpha_vals, plots_dir)
plot_from_ind_results(alpha_vals, iter_counts, plots_dir)
