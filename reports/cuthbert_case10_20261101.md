# Cuthbert A/B/C: case 10

Case 10 (start 5, repeat 0) completed on one A40 in gl1529, job 65160224_10, with Slurm exit 0:0 and 2,053 allocated GPU seconds. All 175 public IF2 passes and each of the three 400-update Adam runs completed. The last planned case, repeat 1 of the same starting vector, was still running at 07:27 UTC.

| Method | Independent log mean likelihood | MCSE | Likelihood ESS | PF repeats | Method fit seconds |
|---|---:|---:|---:|---:|---:|
| A: direct alpha=1 | -3768.950269 | 0.155780 | 19.466148 | 36 | 554.844 |
| B: IF2, then alpha=1 | -3748.262771 | 0.143607 | 20.908290 | 36 | 626.066 |
| C: same IF2, then alpha=0.97 | -3745.913990 | 0.128941 | 22.757371 | 36 | 626.122 |

The fixed original PyPOMP PF used J=5,000 for every independent score repeat. All three scores meet MCSE <= 1 and likelihood ESS >= 10. All 108 raw values were recalculated; no failed fit, nonfinite score or density-floor outcome occurred in this case. B/C use identical initial parameters, zero initial Adam state, learning rates and training keys. The execution order was B/C/A. Method times cover synchronous computation, exclude compilation, Python/file operations and scoring, and include the full IF2 initialization for B/C. Physical allocation cost counts the shared IF2 and shared alpha=1 compilation once. A/B have different total computation.

## Current totals

There are 33 audited endpoints from eleven cases and 1,944 saved PF values. Thirty-one fits completed 400 Adam updates; the two A failures in cases 8 and 9 remain at 43 and 60 accepted updates. A scores in cases 6, 8 and 9 remain low precision at R=288; the other thirty scores pass. No nonfinite or density-floor score has occurred. C's estimated likelihood exceeds B's in the eleven observed pairs. The last case is required before reporting the planned comparison as complete. [Earlier failures and outcomes](cuthbert_cases0_9_20261101.md) remain unchanged; [current endpoint records](../results/cuthbert/endpoints.csv) include failure and precision fields separately.

Completed fitting/evaluation allocations used 21,181 GPU seconds. Including the 879-second v3 check, 321-second repeated-gradient diagnosis and the remaining 3,420-second allocation limit gives 25,801 seconds, within 42,480 seconds. Earlier validation's 2,677 seconds remain separately recorded. Resource totals are not wall-clock waiting time, and the shorter times of failed A runs are not successful-fit latencies.

## Audit and archive

Collection verified 110 source files, 571 referenced files, 40 transferred files and 30 terminal files. The audit checked IF2 state and history, Adam counters, last parameters, B/C pairing, execution order, independent score keys and actual costs.

The [case 10 archive](../artifacts/experiments/dacca/cuthbert_cases10_10_20261101T0726.tar.gz) contains 741 files, including original source, checkpoints, raw scores and audit programs. SHA256: `754d92a1ffdb308417d99545a75c4324b3e2f6ba4b2d958fdfa1e96af77de494`. Every member was verified, and the extracted case audit reproduces byte-for-byte without PF calls or fitting. Reproduce it after extraction, with NumPy and JAX installed:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=False PYTHONDONTWRITEBYTECODE=1 python audit_cuthbert_formal_case.py --case 10 --output checked_case10.json
```

A separate all-case audit has been prepared for the final review. Its missing-case check correctly refuses to write a final report while case 11 is absent. Its second scalar calculation, using Python double precision and math.fsum, agrees on all 1,944 saved values currently available. This is preparation, not a completed 36-endpoint audit. After all twelve collections are available, it will recheck every case, verify all 36 identities and inspected cross-case score/training keys, retain failures and low precision, and sum actual costs. It does not certify mathematical gradients or inspect unsaved rejected proposals. Final figures and the complete review package are still pending.
