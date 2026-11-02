# PFJAX native A/B experiment

These are unchanged copies of the principal frozen fitting programs and protocol. `native_api.py` calls original PFJAX; `run.py` supplies external Adam and saved-result recording. Original PFJAX code was not changed. The imported Dacca model and the complete dependency tree are in the archive.

The supplement is complete. Do not execute submission or fitting programs to reproduce its tables. Extract `artifacts/experiments/dacca/pfjax_native_complete_20261102T0718.tar.gz` into a new directory. It preserves the workspace layout with `work/pfjax-native-ab-20261102` and the matching original collections. Install the recorded JAX/NumPy/Optax environment, then run `audit_comparison.py --output NEW_DIRECTORY` with the archived source and dependency directories on PYTHONPATH. This reads saved records only. `write_final_report.py` regenerates the summary and report; its output paths must be absent. See the complete report for the independent figure command.

The original IF2 populations and C scores are reused unchanged. Old α = 1 A/B results remain separate. Native basic-PF autodiff is documented as biased; no mathematical derivative certification is claimed.
