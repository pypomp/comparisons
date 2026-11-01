# Cuthbert A/B/C: seven completed cases

Seven of twelve planned cases have completed and been audited, giving 21 of 36 endpoints. Starts 0, 1 and 2 have both repeats; start 3 has its first repeat. The cases followed the original order and were not selected by score. Cases 4-6 each completed in 2,000 seconds on a one-A40 allocation in gl1516.

## Independent likelihoods

A starts directly with 400 Adam updates at alpha=1. B uses 175 passes of the added public IF2 implementation, then 400 Adam updates at alpha=1. C uses the same IF2 endpoint as B and 400 Adam updates at alpha=0.97. All seven IF2 runs completed 175 passes and all 21 Adam runs completed 400 updates. B/C have the same initial parameters, zero initial Adam state, learning rates and training keys. A/B have different total amounts of computation.

| Case | Start | Repeat | A log mean likelihood | B log mean likelihood | C log mean likelihood | Allocation GPU seconds |
|---|---|---|---:|---:|---:|---:|
| 0 | 0 | 0 | -3749.346585 | -3747.740628 | -3744.844912 | 1882 |
| 1 | 0 | 1 | -3749.191836 | -3745.203526 | -3744.010797 | 2048 |
| 2 | 1 | 0 | -3796.621696 | -3746.262568 | -3744.227977 | 2048 |
| 3 | 1 | 1 | -3780.159897 | -3750.942197 | -3744.408473 | 2048 |
| 4 | 2 | 0 | -3833.311910 | -3749.740281 | -3746.065480 | 2000 |
| 5 | 2 | 1 | -3835.318563 | -3747.374447 | -3744.307549 | 2000 |
| 6 | 3 | 0 | -4502.184005* | -3746.218160 | -3744.594199 | 2000 |

*Case 6, A has a low-precision score. It reached the prescribed maximum of 288 PF repeats, with MCSE=0.719573 and likelihood ESS=1.925079. The ESS is below the required 10, although MCSE is below 1. This score and every raw repeat remain in the table and archive. The fit completed its 400 updates; the precision failure is an evaluation result, not an interrupted fit. It will not be rescored beyond the protocol limit.

All other endpoints use 36 PF repeats and meet both precision requirements. The fixed original PyPOMP PF uses 5,000 particles for every repeat. All 1,008 saved PF values were included in the independent recalculation of log(mean likelihood), MCSE and likelihood ESS. There are no failed fits, nonfinite scores or density-floor outcomes among these 21 endpoints. The [endpoint table](../results/cuthbert/endpoints.csv) records the precision flag, MCSE, ESS and iteration count separately.

C has the highest estimated likelihood in each observed case, followed by B and A. The low-precision A score in case 6 is flagged above. Five cases are unfinished; these partial outcomes do not establish a general optimizer ordering.

## Fitting time

| Case | A synchronous calls, seconds | B including IF2, seconds | C including IF2, seconds |
|---|---:|---:|---:|
| 0 | 557.541 | 629.296 | 629.779 |
| 1 | 556.823 | 627.838 | 628.185 |
| 2 | 554.260 | 625.255 | 625.292 |
| 3 | 556.667 | 627.595 | 627.709 |
| 4 | 554.021 | 625.155 | 624.926 |
| 5 | 556.668 | 627.716 | 627.676 |
| 6 | 554.785 | 625.585 | 625.604 |

These times cover synchronous computation calls and exclude compilation, Python checks, file operations and independent evaluation. Both B and C include the full IF2 initialization in their method cost; physical allocation cost counts the shared IF2 once. A/B share alpha=1 compilation, charged to the first executed arm. Cases 4, 5 and 6 followed B/C/A, C/A/B and A/B/C execution order.

The three new allocations each used 2,000 seconds and overlapped in wall time; their total resource cost is 6,000 GPU seconds. Completed fitting/evaluation allocations now total 14,026 GPU seconds. The jobs crossed the daylight-saving clock change: their recorded local end times precede their local start times. Cost uses Slurm ElapsedRaw=2,000 seconds for each job, rather than subtracting those ambiguous local timestamps.

The v3 check (879 seconds), repeat diagnosis (321 seconds), completed allocations (14,026 seconds), and five remaining 57-minute limits sum to 32,326 seconds, within the 42,480-second allowance. Earlier validation used a separate 2,677 seconds. At 06:27 UTC, the existing array had cases 7 and 8 running and cases 9-11 pending with JobArrayTaskLimit. No task was resubmitted and no fit or score was repeated.

## Checks and reproduction

Each new collection verified 110 frozen source files, 571 referenced source files, 40 transferred files and 30 terminal files. Saved-data audits checked full IF2 swarms and histories, Adam counters and traces, terminal parameters, natural-scale conversion, rotated execution order, B/C pairing, score-arm codes, adaptive evaluation stages, random keys and actual allocation costs. The 576 new raw PF values were all retained and recalculated. Historical failed tests remain unchanged; these outcome audits do not certify mathematical gradient correctness.

The [cases 4-6 archive](../artifacts/experiments/dacca/cuthbert_cases4_6_20261101T0626.tar.gz) contains 827 files: full source, checkpoints, raw scores, collection records and audit programs. SHA256: `47348dbbe2b6f2915fcf2a60b44b5decf674933eab039b976971a98a7fbf4d56`. Every member was verified, and each audit was reproduced byte-for-byte after extraction, without PF calls or fitting. [Cases 0-3](cuthbert_cases0_3_20261101.md) and their original archives remain unchanged.

After extraction, with NumPy and JAX installed:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=False PYTHONDONTWRITEBYTECODE=1 python audit_cuthbert_formal_case.py --case 4 --output checked_case4.json
```

Repeat for cases 5 and 6 with new output filenames. Compare with `cuthbert_formal_v4_collected/caseNN/audit.json`. The audit programs read saved records and make no PF calls. [Current case records and costs](../results/cuthbert/current_results.json).
