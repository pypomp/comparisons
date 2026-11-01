# Cuthbert A/B/C: ten audited cases

Ten of twelve planned cases have finished, covering the first five starting vectors with two repeats each. There are 30 retained endpoints: 28 fits completed 400 Adam updates and two direct-A fits stopped after a nonfinite update check. All ten shared IF2 runs completed 175 passes. The final starting vector and its two repeats are unfinished. These cases followed the original order and were not selected by their scores.

## Independent likelihoods and failed fits

A starts directly with 400 planned Adam updates at alpha=1. B uses 175 passes of the added public IF2 implementation followed by 400 Adam updates at alpha=1. C uses the same IF2 endpoint with alpha=0.97 and 400 Adam updates. B/C share their initial parameters, zero initial Adam state, learning rates and training keys. A/B have different total amounts of computation.

| Case | Start | Repeat | A log mean likelihood | B log mean likelihood | C log mean likelihood | A accepted updates | Allocation GPU seconds |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | -3749.346585 | -3747.740628 | -3744.844912 | 400 | 1882 |
| 1 | 0 | 1 | -3749.191836 | -3745.203526 | -3744.010797 | 400 | 2048 |
| 2 | 1 | 0 | -3796.621696 | -3746.262568 | -3744.227977 | 400 | 2048 |
| 3 | 1 | 1 | -3780.159897 | -3750.942197 | -3744.408473 | 400 | 2048 |
| 4 | 2 | 0 | -3833.311910 | -3749.740281 | -3746.065480 | 400 | 2000 |
| 5 | 2 | 1 | -3835.318563 | -3747.374447 | -3744.307549 | 400 | 2000 |
| 6 | 3 | 0 | -4502.184005* | -3746.218160 | -3744.594199 | 400 | 2000 |
| 7 | 3 | 1 | -4250.237730 | -3748.064527 | -3745.709941 | 400 | 2043 |
| 8 | 4 | 0 | -4577.584559* | -3745.106671 | -3744.175548 | 43 | 1592 |
| 9 | 4 | 1 | -4713.211742* | -3746.358664 | -3744.393958 | 60 | 1467 |

An asterisk marks a low-precision score. Cases 8 and 9 A are failed fits evaluated at their last valid parameters; they are not complete 400-update results.

| A record | Fit outcome | PF repeats | MCSE | Likelihood ESS | Score precision |
|---|---|---:|---:|---:|---|
| Case 6 | 400 updates completed | 288 | 0.719573 | 1.925079 | Below ESS requirement |
| Case 8 | 43 updates accepted; next proposal rejected | 288 | 0.508959 | 3.822461 | Below ESS requirement |
| Case 9 | 60 updates accepted; next proposal rejected | 288 | 0.922216 | 1.175085 | Below ESS requirement |

The two failed fits are the two repeats of start 4. Their saved checkpoints have 44 and 61 attempted updates respectively and preserve the last accepted parameter and Adam state. The recorded reason is a nonfinite loss, gradient, estimation-scale or natural-scale parameter, or Adam state at zero-based update 43 or 60. The rejection check does not record which of these values was nonfinite; the retained record cannot isolate that cause. No failed fit was retried and no checkpoint was selected by its likelihood. B and C completed all 400 updates in both cases.

All 30 endpoints have independent scores from the fixed original PyPOMP PF at J=5,000. Twenty-seven scores use R=36 and meet MCSE <= 1 and likelihood ESS >= 10. The three A scores above reached the fixed R=288 limit with MCSE below 1 but ESS below 10; all remain flagged as low precision, with no further scoring. All 1,836 original PF values were retained and used to recalculate log(mean likelihood), MCSE and ESS. No nonfinite or density-floor score occurred. [Endpoint records](../results/cuthbert/endpoints.csv) include fit failure, accepted updates, failure reason and score precision separately.

C's estimated likelihood exceeds B's in all ten observed pairs. This is an interim comparison for five starting vectors. A has eight complete fits and two retained failed fits, so its failure rate and shorter failed-run times must remain visible. The unfinished cases are still required for the planned comparison.

## Fitting time and allocation cost

| Case | A synchronous calls, seconds | B including IF2, seconds | C including IF2, seconds |
|---|---:|---:|---:|
| 0 | 557.541 | 629.296 | 629.779 |
| 1 | 556.823 | 627.838 | 628.185 |
| 2 | 554.260 | 625.255 | 625.292 |
| 3 | 556.667 | 627.595 | 627.709 |
| 4 | 554.021 | 625.155 | 624.926 |
| 5 | 556.668 | 627.716 | 627.676 |
| 6 | 554.785 | 625.585 | 625.604 |
| 7 | 554.404 | 625.642 | 625.427 |
| 8 | 61.260 (failed) | 627.277 | 627.495 |
| 9 | 85.182 (failed) | 628.301 | 628.238 |

Method times exclude compilation, Python checks, file operations and independent evaluation. Each B/C method cost includes the full IF2 initialization; physical cost counts the shared IF2 once. A/B share alpha=1 compilation, attributed to the first executed arm. Case 8 A's 61.260 seconds and case 9 A's 85.182 seconds are the cost of failed runs with 43 and 60 accepted updates, including their rejected attempts. They are not successful 400-update latencies or evidence of acceleration.

New allocations used 2,043 seconds for case 7 on gl1516, 1,592 seconds for case 8 on gl1527 and 1,467 seconds for case 9 on gl1527, each with one A40. Each allocation ended with Slurm exit 0:0, despite the two retained fit failures. These three resource costs sum to 5,102 GPU seconds; overlapping wall time is not summed as elapsed waiting time. All ten completed fitting/evaluation allocations used 19,128 GPU seconds. Earlier jobs crossing the daylight-saving change use Slurm ElapsedRaw for cost, preserving their original local timestamps.

The v3 check (879 seconds), repeat diagnosis (321 seconds), completed allocations (19,128 seconds), and two remaining 57-minute limits sum to 27,168 seconds, within the original 42,480-second allowance. Earlier validation's 2,677 seconds remain separately recorded. The 06:57 UTC snapshot shows case 10 running and case 11 pending with JobArrayTaskLimit. No jobs were resubmitted.

## Checks and reproduction

For each new case, collection verified 110 frozen source files, 571 referenced files, 40 transferred files and 30 terminal files. The saved-data audits checked full IF2 state, initial parameters, Adam counters and traces, last accepted parameters, recorded failures, rotated order, B/C pairing, scoring keys, adaptive score stages and actual costs. The 828 new raw PF values were all recalculated. This audit preserves the earlier failed validation tests and does not certify mathematical gradient correctness.

The [cases 7-9 archive](../artifacts/experiments/dacca/cuthbert_cases7_9_20261101T0656.tar.gz) contains 827 files including full source, checkpoints, raw scores, failure records, scheduler costs and audit programs. SHA256: `3673262126002459bff37ab1a36ce267670398ef1b83a3d7e11e9c187219d99a`. Every member was verified, and all three audits were reproduced byte-for-byte after extraction without fitting or PF calls. The [cases 0-6 report](cuthbert_cases0_6_20261101.md) and all earlier archives remain unchanged.

After extraction, with NumPy and JAX installed:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=False PYTHONDONTWRITEBYTECODE=1 python audit_cuthbert_formal_case.py --case 7 --output checked_case7.json
```

Repeat for cases 8 and 9 with new output filenames. Compare the outputs with `cuthbert_formal_v4_collected/caseNN/audit.json`. [Current case records and costs](../results/cuthbert/current_results.json).
