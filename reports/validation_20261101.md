# Validation update: 1 November 2026

This update contains completed tests, not new optimization results. All failures and original records are in the dated archive listed in [the catalog](../archive/catalog.json).

## Cuthbert GPU diagnosis

Job 65144329 ran on A40 node gl1516 for 801 seconds and exited `FAILED 1:0`. It completed the planned diagnosis of the previous v2 implementation. The following three tests failed in both the unchanged original and modified code:

| Test | Reported maximum absolute discrepancy on both sides |
|---|---:|
| `test_gaussian_taper_log_likelihood_gradient` | 0.41011396 |
| `test_smoother[0-3-1-25]` | 0.00000217069444 |
| `test_smoother[0-3-2-25]` | 0.00000202640667 |

These are the same failed assertions at the original tolerances. The full numeric arrays are not claimed to be bitwise identical. The diagnosis establishes that the three failures also occur in the original code in this environment; it does not establish their underlying numerical cause.

Both versions passed all 64 PF regression tests. The modified v2 code passed 31 interface tests, its J=32 full-length Dacca integration check, the J=5000 cost probe and the small A/B/C driver check. These are v2 records. They do not validate the later v3 change to the differentiation path, or provide formal optimization results.

Source verification covered 757 remote files; 50 transferred files and 45 terminal-manifest entries were checked. The failed tests were read again from their XML records. [Audit](../results/validation_20261101/cuthbert_audit.json).

At the scheduler query on 1 November, v3 job 65144479 was still pending for priority. Its multinomial, 12-start, J=5000, memory and runtime checks remain required before formal A/B/C submission. No tolerance was changed and no failed optimization was repeated.

## TFP CPU upstream tests

All 515 configured test files were attempted in original/modified pairs. The runner has stopped. Independent review checked 1,030 process records, 1,010 XML files and 2,601 source-file hashes. Each side had the same per-file exit status, state, reported test counts and failed-test identities:

| Per-side outcome | Files |
|---|---:|
| Normal exit, code 0 | 370 |
| Test or collection failure, code 1 | 127 |
| No tests collected, code 5 | 8 |
| 180-second timeout | 9 |
| Abort | 1 |

There were no additional failed test identities in the modified version. This is not a passing full suite: common failures, no-test exits, timeouts and aborts remain unresolved. XML-emitting processes reported 26,754 tests, including 1,006 failures, 54 errors and 1,614 skips per side. These counts exclude processes that did not produce XML.

The original `status.json` retains `finished=false` because the runner did not update that field at exit. Completion is established by `COMPLETE.json`, all 515 pair records, removal of `RUNNING.json` and the absence of the runner PID. The stale field is preserved. [Audit](../results/validation_20261101/tfp_audit.json).

These CPU tests do not resolve TFP's GPU gradient-equality failure. Its new A/B/C experiment remains unrun.

## Recheck the records

Extract `artifacts/validation/20261101_completed_checks.tar.gz` with `scripts/extract_archive.py`. From the extracted directory, run `python3 audit_completed_checks_20261101.py` to repeat the source, XML and record checks. This reads the existing results and writes derived audit JSON; it does not run tests, PFs or optimizers. The archive's file manifest was created after completion, not by the original runners.
