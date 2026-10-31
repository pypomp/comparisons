# Reproduction and review

## Check this repository

Run from any working directory:

```bash
python3 /path/to/comparisons/scripts/verify.py --archives
```

The command checks the imported SHA256 records, five complete patches applied to clean upstream copies, endpoint and raw-repeat counts, the fixed scorer, and all archive members. It uses Python's standard library and Git. It does not fit, evaluate or submit a cluster job. `manifests/repository_verification.json` records the check performed during assembly.

`scripts/verify.py --originals /path/to/original/workspace` additionally checks whether the originals still match the imported snapshot. This option is intended for the import audit. Live status files may legitimately change afterward; the snapshot itself remains unchanged.

## Review a package change

For each of Cuthbert, PFJAX, PyDPF, TFP and particles, compare `packages/NAME/upstream/` with `packages/NAME/candidate/`, or read `packages/NAME/review/full.patch`. The preserved review manifest identifies the upstream commit or wheel version, modified files and validation. Other patches in the review folder are subsets or earlier revisions; do not apply them after the complete patch.

PyDPF's preserved baseline is a specified subset of its repository. TFP's candidate baseline is the installed 0.25.0 wheel source, not a complete development checkout. Full upstream test-source archives are separately catalogued. These differences matter when preparing a future PR.

## Recover raw experiment records

The archive paths and SHA256 values are in [archives.json](../manifests/archives.json). For example:

```bash
python3 scripts/extract_archive.py \
  artifacts/experiments/legacy_dacca/paper_cuthbert_pypomp/reproduction.tar.gz \
  extracted/paper_dacca
```

The helper verifies the checksum and rejects path traversal, links, special files and existing extraction destinations. Archives stay unchanged. Read the extracted archive's README/REPRODUCE instructions for its audit and figure-generation commands. Numerical dependencies are specified in its original environment/source records.

For the completed paper archive, the preserved delivery program supports:

```bash
cd extracted/paper_dacca
python independent_audit/deliver.py \
  --paper-root paper_experiment \
  --scheduler-evidence independent_audit/scheduler_evidence.json \
  --output rebuilt_delivery
```

This rebuild needs the dependencies stated by that archive. It was not rerun during repository assembly. The original verified delivery and all raw PF values are included unchanged.

Some historical scripts contain absolute workspace and cluster paths. Their source has deliberately not been rewritten, because their hashes identify what ran. The new integrity/extraction helpers are portable; a numerical rerun requires reconstructing the documented environment and explicitly adapting launch paths in a separate revision. No claim is made that all copied historical runners execute unmodified from this repository.

## Add later results

Create a new dated snapshot after collecting a stopped task and checking source hashes, terminal files, scheduler exit/cost, protocol, endpoints and raw scores. Preserve the previous snapshots. Update the package status and archive catalog only with observed evidence. Running or pending frozen sources are never edited here or on the cluster. A source change requires a new revision and its own validation.

Bulk archives are local, untracked files. Any later distribution must include the listed archives or provide a deliberate storage arrangement; copying only Git history omits them. No remote publication is configured.
