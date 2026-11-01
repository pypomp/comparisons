# Cuthbert: first completed A/B/C case

Case 0 completed all three methods on one NVIDIA A40 (gl1524). Slurm job `65158659_0` ended with exit code `0:0` after 1,882 seconds (31 minutes 22 seconds). It is one of twelve planned cases, with 3 of 36 endpoints now audited.

## Independent likelihood evaluation

All methods used Dacca's 600 months, 20 Gaussian Euler steps per month and 5,000 particles. A started from the original parameter vector. B and C started from the same result of 175 public IF2 passes. Each completed 400 Adam updates.

| Method | Initialization and gradient | Log mean likelihood | MCSE | Likelihood ESS | PF repeats |
|---|---|---:|---:|---:|---:|
| A | Original start; α=1 | −3749.346585 | 0.182492 | 16.623484 | 36 |
| B | IF2 result; α=1 | −3747.740628 | 0.174701 | 17.406310 | 36 |
| C | Same IF2 result; α=0.97 | −3744.844912 | 0.119964 | 23.941005 | 36 |

These are evaluations by the fixed original PyPOMP PF with independent training and evaluation streams. All 108 original PF values were used to recalculate the log mean likelihood, MCSE and ESS. All three meet the recorded precision rule. There are no failed fits, nonfinite scores or density-floor scores in this case. C is higher than B by 2.896 log-likelihood units in this case; the twelve-case comparison is unfinished.

The [audit](../results/cuthbert/case0_20261101.json) verifies the full 5,000-particle IF2 parameter swarm, 175 accepted IF2 passes, 400 Adam updates per method and the final saved parameters. B/C initial parameters, zero Adam state, learning rates and training keys match. Final parameters are the last saved iterates; they were not selected by the best training loss. The inspected training and evaluation keys have no collisions; this check does not enumerate every internal random-key descendant.

## Computation

| Quantity | Seconds |
|---|---:|
| Complete Slurm allocation | 1882.000 |
| Fitting subprocess, including startup and file operations | 1798.676 |
| Fitting program | 1783.221 |
| Synchronized fitting calls, shared IF2 counted once | 1744.593 |
| Fitting compilation, shared α=1 counted once | 18.335 |
| Evaluation subprocess, including startup | 75.483 |
| Evaluation program's internal timer | 10.269 |

These measurements are nested and must not be added together. The shared IF2 calls took 72.023 seconds. A's synchronous fitting calls took 557.541 seconds. B and C, each charged the full IF2 prefix for a method-level comparison, took 629.296 and 629.779 seconds. These method times exclude compilation, host checks and file operations. A/B share α=1 compilation; attributing that compilation to its first use does not make B intrinsically cheaper. A and B perform different total work. Ordinary PF timing is reported [separately](timing.md).

## Remaining cases and records

After reviewing the completed case, only cases 1–11 were submitted as array `65160224`, with at most three simultaneous A40 allocations and the same 57-minute limit per case. No case was repeated. The v3 package, fitting program, scorer, starts and scientific settings are unchanged. [Actual submission](../results/cuthbert/submission_cases1_11_20261101.json).

At 04:34 UTC on 1 November 2026, all eleven tasks were pending with `PartitionDown`; `scontrol` reported `spgpu State=DOWN` and no start estimate was available. This is a scheduler condition, not a failed fit. [Recorded queries](../results/cuthbert/scheduler_cases1_11_20261101.json).

Including the v3 check (879 seconds), repeat diagnosis (321 seconds), completed case (1,882 seconds) and eleven remaining 57-minute limits gives 40,702 GPU seconds, below the original 42,480-second allowance. The earlier 2,677 seconds of validation belong to the separate original validation budget and remain recorded.

Source verification covered 110 fitting/evaluation files and 571 referenced validation/package files. Collection verified 40 transferred files and 30 terminal files. The complete [archive](../artifacts/experiments/dacca/cuthbert_case0_20261101.tar.gz) contains 744 files, including checkpoints, raw scores, source, failed local audit history and submission records. Every member was checked; extracting it and rerunning the saved-data audit produced an identical audit. [Reproduction commands](../docs/reproduction.md).

This result proceeds under the recorded acceptance of forward-likelihood agreement. It does not change the earlier GPU gradient failures into passes or certify the mathematical gradient. The [original-repeat diagnosis](cuthbert_repeats_20261101.md) documents why the old elementwise rule alone was insufficient to stop this empirical comparison.
