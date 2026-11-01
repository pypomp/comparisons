# Cuthbert A/B/C: first four cases

Four of twelve planned cases have completed and been audited: 12 of 36 endpoints, covering the first two starting parameter vectors with two repeats each. These cases followed the original order and were not selected by their scores. Cases 1, 2 and 3 ended with Slurm exit code `0:0`; each used 2,048 seconds on one A40 in gl1516.

## Independent likelihoods

A uses the original start and 400 Adam updates with alpha=1. B uses 175 public IF2 passes followed by 400 Adam updates with alpha=1. C uses the same IF2 result as B, with alpha=0.97 and 400 Adam updates. All four IF2 runs and all twelve Adam runs completed their planned updates.

| Case | Start | Repeat | A log mean likelihood | B log mean likelihood | C log mean likelihood | Allocation GPU seconds |
|---|---|---|---:|---:|---:|---:|
| 0 | 0 | 0 | -3749.346585 | -3747.740628 | -3744.844912 | 1882 |
| 1 | 0 | 1 | -3749.191836 | -3745.203526 | -3744.010797 | 2048 |
| 2 | 1 | 0 | -3796.621696 | -3746.262568 | -3744.227977 | 2048 |
| 3 | 1 | 1 | -3780.159897 | -3750.942197 | -3744.408473 | 2048 |

Every endpoint was evaluated by the fixed original PyPOMP PF with 5,000 particles and 36 independent repeats. All 432 original PF values were included in the recalculation; all twelve scores meet MCSE <= 1 and likelihood ESS >= 10. There are no failed fits, nonfinite scores or density-floor outcomes among these twelve records. The [endpoint table](../results/cuthbert/endpoints.csv) includes each MCSE, ESS and fit count.

C has the highest likelihood in each of the four observed cases, followed by B and A. This is an interim result for two of the six planned starting vectors. It does not establish the result for the unfinished cases or a general ordering of the optimizers. A and B have different amounts of computation; B/C have the same initialization, zero initial Adam state, learning rates and training keys.

## Fitting time

| Case | A synchronous calls, seconds | B including IF2, seconds | C including IF2, seconds |
|---|---:|---:|---:|
| 0 | 557.541 | 629.296 | 629.779 |
| 1 | 556.823 | 627.838 | 628.185 |
| 2 | 554.260 | 625.255 | 625.292 |
| 3 | 556.667 | 627.595 | 627.709 |

These method times exclude compilation, Python checks, file operations and evaluation. Each B/C entry includes its full IF2 initialization cost; physical total cost counts the shared IF2 once per case. Compilation for alpha=1 is shared by A/B and attributed to the first arm executed. The recorded execution orders for cases 1, 2 and 3 are B/C/A, C/A/B and A/B/C respectively.

Cases 1-3 ran concurrently on three separate one-GPU allocations. Their overlapping wall-clock interval was 2,048 seconds; their allocation cost was 6,144 GPU seconds. Including case 0, the completed fitting/evaluation allocations used 8,026 GPU seconds. This is a resource total, not elapsed waiting time.

The v3 check (879 seconds), repeated-gradient diagnosis (321 seconds), completed allocations (8,026 seconds) and eight remaining 57-minute limits total 36,586 seconds, below the 42,480-second formal allowance. The earlier 2,677 validation seconds remain in their separate original validation budget. The 05:56 UTC scheduler query showed cases 4-6 running and cases 7-11 waiting for the existing three-task concurrency limit. No jobs were resubmitted.

## Checks and reproduction

For each newly completed case, collection verified 110 source files, 571 referenced source files, 40 transferred files and 30 terminal files. The saved-data audit checked the full IF2 swarm and history, Adam counters and traces, final parameters, natural-scale conversion, rotated arm order, B/C pairing, fixed score-arm codes, raw PF statistics and actual allocation costs. All recorded failed-test history is preserved. These outcome audits do not certify mathematical gradient correctness.

The [new archive](../artifacts/experiments/dacca/cuthbert_cases1_3_20261101T0556.tar.gz) contains 827 files, including complete source, checkpoints, raw scores, collection records and audits for cases 1-3. Its SHA256 is `09b5bdc711b1d89bee7fa5c712565920e20eed8b7d0a92cff6bb75ee68f06119`. All members were verified. Extracting the archive and rerunning each saved-data audit produced byte-for-byte identical audit files, without PF calls or fitting. Case 0 remains in its [earlier report and archive](cuthbert_case0_20261101.md).

After extracting the new archive, with NumPy and JAX installed:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=False PYTHONDONTWRITEBYTECODE=1 python audit_cuthbert_formal_case.py --case 1 --output checked_case1.json
```

Use cases 2 and 3 and new output filenames to repeat their audits. Compare outputs with the corresponding `cuthbert_formal_v4_collected/caseNN/audit.json`. The original receipts predate scientific review; the separate audit files record that completed review. [Current result records](../results/cuthbert/current_results.json).
