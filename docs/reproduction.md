# Reproduction

Run these commands from the repository root. Recovery and table checks need Python 3.9 or later and Git, with no numerical libraries or network access.

The Makefile provides shortcuts for the commands below: `make prepare PACKAGE=cuthbert`, `make tables`, `make check` and `make check-archives`. Set `BUILD` to a fresh directory when repeating recovery or table generation. `make help` lists the commands.

## Restore a package

```bash
python3 scripts/prepare.py cuthbert --output build/cuthbert
```

Replace `cuthbert` with `pfjax`, `pydpf`, `tfp` or `particles`. The command extracts the recorded original source, applies `review/full.patch`, and compares every resulting file with the archived modified source. It refuses to overwrite an existing directory. `version.json` identifies the commit or wheel and the changed files.

PyDPF's original source is a recorded subset of the upstream repository. TFP's original source is the 0.25.0 installed wheel source. Restoring them does not create a full upstream development checkout. Full test-source archives are separately listed in the catalog.

## Rebuild result tables

```bash
python3 scripts/reproduce.py --output build/tables
```

This recomputes the two summary tables, the first Cuthbert case table and the current Cuthbert endpoint table from saved records. Every finite score, including low-precision and floor-saturated scores, contributes to its group's median. Failed fits, nonfinite scores and precision counts are reported separately. It checks all 24 new particles rows and all 968 historical rows; the 36 historical Dynamax rows remain archived and are excluded from the main summary. The dated first-case table remains unchanged. The current Cuthbert table contains all audited cases and remains separate from the earlier experiments. Allocation costs are recorded once per case in results/cuthbert/current_results.json.

The script does not recalculate PF scores. Individual PF repeats and the original scoring audits are in the archives. To repeat those audits or rebuild original figures, restore the corresponding archive and use its original instructions.

The first Cuthbert case has a separate audit of its complete checkpoints and 108 original PF scores:

```bash
python3 scripts/extract_archive.py artifacts/experiments/dacca/cuthbert_case0_20261101.tar.gz extracted/cuthbert_case0
cd extracted/cuthbert_case0
JAX_PLATFORMS=cpu JAX_ENABLE_X64=False PYTHONDONTWRITEBYTECODE=1 python audit_cuthbert_case0.py
```

This audit needs NumPy and JAX. It recalculates score summaries and random keys from the saved records without a PF call, fit or cluster connection. It has been run after extracting this archive; its output was byte-for-byte identical to the archived audit. The archive also preserves the first local audit attempt, its float32 particle-mean accumulation discrepancy, and the correction to float64 accumulation at the same tolerance.

## Restore the former directory layout

```bash
python3 scripts/restore.py --output build/original --archives
python3 build/original/scripts/verify.py
```

This restores all 8,527 files from the repository before simplification, including the old import manifest, complete source trees, frozen experiment snapshots, recorded environments, failed tests and earlier reports. File contents and modes are checked during extraction. `--archives` also copies the 50 original bulk archives: the former verification program checks their hashes even when it does not decompress them. The second command runs that program against the complete restored layout. Omit `--archives` when only inspecting source or documents.

To recover one path instead, use `--prefix`, for example:

```bash
python3 scripts/restore.py --output build/meeting --prefix docs/meeting_decisions.md
```

The archive paths in [the index](../archive/README.md) specify where to find the raw records. Extract a catalogued bulk archive with:

```bash
python3 scripts/extract_archive.py artifacts/experiments/legacy_dacca/paper_cuthbert_pypomp/reproduction.tar.gz extracted/paper
```

## Numerical reruns

The main experiment files are exact copies for review. Some launchers contain absolute workspace and cluster paths and depend on the full source tree, input files and recorded environment. Their archived versions identify what ran; they have not been rewritten during simplification. Restoring the code does not install those environments or make every launcher portable. A numerical rerun elsewhere needs a separate copy with explicit environment and path adjustments.

For example, after extracting the completed paper archive, its existing delivery program can rebuild its report from saved scores:

```bash
cd extracted/paper
python independent_audit/deliver.py --paper-root paper_experiment --scheduler-evidence independent_audit/scheduler_evidence.json --output rebuilt_delivery
```

Use that archive's dependency instructions. Simplification verifies file recovery and table regeneration; it does not claim that this scientific report command or any fitting run was rerun.

## Check preservation

```bash
python3 scripts/verify.py --archives --output build/verification.json
```

This checks the main file hashes, all archived former files, all catalogued archive checksums and compressed contents, five source reconstructions, result-table regeneration and the fixed scorer hash. It is a repository check, not a GPU correctness result.

New experiment records should be added as dated files after stopped jobs are collected and audited. Keep existing failures and snapshots. Running and queued experiment directories are outside this repository and must not be modified by repository maintenance.
