# Dacca figures

Each method has twelve runs: six starting vectors, each used twice. Gray shows A, which uses the package's gradient from the original starting vector. Blue shows B, which applies the same gradient after IF2. Orange shows C, which starts from the same IF2 result and uses our discounted gradient with α=0.97. All three use external Adam. For PFJAX, B and C also differ in how they compute the gradient.

| Figure | PNG | PDF |
|---|---|---|
| Likelihood distributions: half violin, box and all points | [View](dacca_native_figures_20261102T0718/likelihood_distributions.png) | [Download](dacca_native_figures_20261102T0718/likelihood_distributions.pdf) |
| Full likelihood range | [View](dacca_native_figures_20261102T0718/all_estimates.png) | [Download](dacca_native_figures_20261102T0718/all_estimates.pdf) |
| Paired C−B differences | [View](dacca_native_figures_20261102T0718/paired_C_minus_B.png) | [Download](dacca_native_figures_20261102T0718/paired_C_minus_B.pdf) |

The statistics include all scores, including those from failed fits, scores at the density floor and scores with low precision. The zoomed view changes only which points are visible. Error bars show Monte Carlo uncertainty in the estimated log likelihood, not uncertainty about the methods' performance across other starts or models.

[Results and times](optimization.md) · [Plot data](dacca_native_figures_20261102T0718/figure_data.json) · [Plot source notes](dacca_native_figures_20261102T0718/source_notes.json) · [PFJAX native/α=1/C individual values](../results/pfjax_native/native_and_extension_20261102.csv).
