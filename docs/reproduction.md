# Reproduce the comparison

Run from the repository root. Python 3.9+ and Git suffice for saved tables and source recovery; figures need NumPy and Matplotlib. These commands do not submit jobs, fit models or run PF scoring.

A Git clone includes the current reports, figures, result tables and patches. Before running `make overview`, `make check`, `make tables` or `make prepare`, copy the separately stored archives listed in [the catalog](../archive/catalog.json) to their recorded paths. `make figures` uses the figure data included in Git.

```bash
make overview BUILD=build/review
make figures BUILD=build/review
make check BUILD=build/review
make prepare PACKAGE=pfjax BUILD=build/source
```

Use a new `BUILD` directory for each generation. The overview contains likelihoods, paired differences, fitting times and the PFJAX five-method comparison. `make check` verifies saved summaries, current and archived hashes, original source recovery and the complete package diffs. `make tables` also rebuilds earlier tables from archived inputs.

## Restore historical records

```bash
# One old report, retaining its original relative path
python3 scripts/restore_recent.py --prefix reports/pydpf_20261102 --output build/old_report

# Full original source/raw package; old and current catalog paths are accepted
python3 scripts/extract_archive.py artifacts/experiments/dacca/pydpf_review_20261102.tar.gz build/pydpf_full

# The earlier 8,527-file layout
python3 scripts/restore.py --output build/original
```

[Archive instructions](../archive/README.md) and the [catalog](../archive/catalog.json) identify files, source versions and hashes. Each complete archive contains individual scores, checkpoints, failures, environment records and programs that check the saved results. Keep its original directory layout when running those checks. The archives are stored locally and excluded from Git, so include them when transferring the project.
