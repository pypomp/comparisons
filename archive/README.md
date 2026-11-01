# Archived files

All 8,527 tracked files from commit `3d810f4bca1220d2ff26f15e3fb618477d12128d` are stored in `artifacts/history/comparisons_20261031_before_simplification.tar.gz`. This includes original and modified sources, failed checks, raw records copied into Git, earlier reports and internal notes. No original record was deleted from this archive or edited to change a result.

The 50 original source and raw-data archives remain at their original paths. [The catalog](catalog.json) lists those archives, the full pre-simplification copy and later additions, with sizes and SHA256 values. [The history record](history.json) identifies the former commit; `files.json.gz` lists the size, mode and SHA256 of each of its 8,527 files. `copies.json` maps unchanged files kept in the main tree to their former paths.

The [1 November validation records](../reports/validation_20261101.md) add the completed Cuthbert v2 diagnosis and 515 TFP CPU test-file pairs in `artifacts/validation/20261101_completed_checks.tar.gz`. Earlier snapshots remain unchanged.

## Recover or inspect

```bash
python3 scripts/restore.py --output build/original
python3 scripts/verify.py --archives
```

The first command restores the previous layout. To restore only one file or directory, add `--prefix` with a path from the table below. Existing destinations are never overwritten.

| Material | Path inside the restored layout |
|---|---|
| Original and modified package sources, tests, environments and failures | `packages/NAME/` |
| Exact original-PF call review and source locations | `reports/native_pf_audit_20261031/` |
| Meeting notes with speakers, transcript times and checked excerpts | `docs/meeting_decisions.md`, `docs/meeting_review_20261031.json` |
| Earlier 968 endpoint rows, figures and reports | `reports/legacy_dacca/` |
| Earlier source and score audits | `experiments/legacy_dacca/` |
| New interface programs, inputs, job records and recovered results | `experiments/public_interfaces/snapshot_20261031/` |
| Cuthbert GPU failures and other package failures | `experiments/public_interfaces/snapshot_20261031/recovery_20261031/` |
| Completed subset of TFP upstream-test file pairs | `experiments/public_interfaces/tfp_upstream_tests_snapshot/` |
| Original file origins and hashes | `manifests/import.json` |

The separate bulk archives contain full checkpoints and raw PF repeats. Their original audits and reproduction instructions are retained. Use `scripts/extract_archive.py` as described in [Reproduction](../docs/reproduction.md).

Dynamax's 36 old records are retained here. They are excluded from current main tables because the earlier experiment supplied its own PF.

These archives are local files ignored by Git. A transfer must include the catalogued files as well as the repository. Git history also remains intact; this cleanup reduces the main file tree, not historical Git storage.
