# Conventions and sources

Keep correctness, ordinary PF runtime and optimization results separate. Use plain technical English; name the original function, local change, optimizer and measured quantity. Compilation, PF calls, fitting and final scoring have different units. Retain all failures and raw values.

| Meeting source, 29 October 2026 | Request |
|---|---|
| Kevin, 15:23:10–15:23:41 | Separate runtime, correctness and likelihood; time the common PF |
| Ed, 15:27:20–15:28:52 | Follow package style; add general parameter perturbations and extend the ADPF |
| Giles, 15:29:17–15:29:59 | Make changes readable as a diff |
| Ed, 15:54:22–15:55:17 | Organize and share comparison reports and code |

The exact starts, iteration counts and budgets were chosen later for this study. Kevin's three questions identify what to compare. Each claimed advantage needs its own result.

The figure layout follows DMOP's `precise_report.qmd`. We used Aaron's `compare_arms.py` as a reference for specifying inputs, summarizing fitted results and accounting for computation. [Source notes](../reports/dacca_native_figures_20261102T0718/source_notes.json) record paths, hashes and transcript references. The earlier conventions document is archived.

Keep current conclusions in the main reports. Put intermediate preparation, queue receipts and superseded plots in the archive. Frozen sources and dated numerical records are preserved byte for byte; current summaries can be regenerated from saved data.
