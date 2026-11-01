# Results and tests

The original repository snapshot is from 31 October 2026. A [1 November validation update](validation_20261101.md) adds the completed Cuthbert v2 GPU diagnosis and all 515 TFP CPU test-file pairs. A [further diagnostic update](gpu_diagnosis_20261101.md) records new CPU controls and submitted GPU checks for PFJAX, PyDPF and TFP. These three checks and Cuthbert v3 remained pending at 00:30 UTC on 1 November. This page is not a live job monitor. Earlier job records and failures remain [archived](../archive/README.md).

The [Dacca report](dacca_comparison.md) describes the model and experiment design. The [source review](../archive/README.md) identifies the package functions used. Dynamax is excluded from the current comparison; its earlier 36 estimates remain in the historical reports and figures.

## Correctness

| Package | Tests and comparisons | Unfinished work |
|---|---|---|
| Cuthbert | Earlier local coverage: 1548 unique tests eventually passed after dependency repairs. Current v3: 32 CPU interface tests, full T=600 checks with both resamplers at J=32, and 12 original and modified ADPF comparisons passed. GPU diagnosis reproduced all three previous failures in unchanged original code; both sides passed 64 PF tests | Revised GPU check 65144479 remains pending. Full GPU correctness is not established |
| PFJAX | 72 configured CPU tests and 72 configured GPU tests passed, with upstream exclusions retained | A Dacca gradient coordinate failed its tolerance. Repeated-call and checkpointing diagnosis 65151851 is pending; no formal optimization result |
| PyDPF | Native PF likelihood increments matched in nine GPU pairs; repaired test run passed 33/34 | A float32 multinomial exact-gradient test failed. Original-repeatability diagnosis 65151857 is pending |
| TFP | The separate PF subset passed 120 tests with 23 skipped on both sides after a test-harness name-binding repair. All 515 full-suite file pairs have now been attempted; no additional failed test identities in the candidate | GPU alpha=1 exact-gradient check failed; diagnosis 65151858 is pending. The full CPU suite did not pass |
| particles | Nine CPU PF pairs matched likelihood/state/weights/RNG; public IF2 tests and collected-result audits passed | The pinned upstream has no pytest suite; this is not a claim that an upstream test suite passed |

The original TFP snapshot of 324 pairs remains archived. The completed run has 515 pairs: 370 normal exits, 127 test/collection failures, eight no-test exits, nine timeouts and one abort per side. Original/candidate outcomes and failed-test identities match in every pair. This does not imply that all tests passed. [Completed-run details](validation_20261101.md).

The Cuthbert v3 change preserves the original alpha=1 differentiation path by defaulting to `checkpoint=False`. A recorded CPU multinomial diagnosis showed that adding checkpointing to the original ADPF reproduces the prior discrepancy. Disabling it passed the recorded checks without changing tolerances. This local diagnosis still requires the pending v3 GPU checks. The three earlier failed tests have now also failed in unchanged upstream on A40; their underlying numerical cause remains undetermined.

## Native particle-filter runtime

These are measured ordinary PF calls for the original Dacca model, T=600 and J=5000. Values are medians of seven timed calls, with warmup/compilation separate. The GPU calls have batch size 1 on A40 nodes. The CPU result is not part of a GPU speed ranking.

| Original PF | Hardware | Seconds per PF call |
|---|---|---:|
| Cuthbert | A40 | 0.393526 |
| PFJAX | A40 | 0.220416 |
| PyDPF | A40 | 8.57610 |
| TFP | A40 | 6.278312 |
| particles | CPU | 5.496709 |

These are implementation-specific timings under the recorded environments. They are not optimizer running times or evidence that a method reaches a better likelihood sooner. PyDPF's candidate PF median was 8.56534 seconds and particles' candidate median was 5.526815 seconds; these observations do not establish a general speed improvement.

## Optimization

The [earlier completed comparison](../archive/README.md) contains 968 fitted parameter vectors and 37,584 individual PF evaluations. It includes the two-package paper experiment and six other packages, with their different scopes explicitly retained. Its final reports, full-range figures, optional 1.5-IQR display figures, old failed fits, low-precision scores and density-floor outcomes are preserved. PFJAX's corrected evaluation_v2 alone supplies its final scores; the superseded scores remain as history.

The new experiment has a different status:

| Package | New endpoints | State |
|---|---:|---|
| Cuthbert | 0/36 | v3 formal comparison prepared; GPU checks pending; not submitted |
| PFJAX | 0/36 | Dependent formal job cancelled before starting after GPU validation failure |
| PyDPF | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| TFP | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| particles | 24/24 | Fits and independent scores audited; final new figures and reproducibility archive still pending |

For new particles, the median independent log likelihood is −4169.070668 for A (1800-second external Nelder-Mead) and −3784.099127 for B (900-second public IF2, then Nelder-Mead to the same total budget). B is higher in all twelve observed pairs. Precision passes are 9/12 for A and 12/12 for B; A tasks 0, 2 and 12 reached R=288 and remain low precision. There were no failed fits or density-floor scores. These findings apply to the recorded starts and budgets, not universal optimizer superiority.

The new particles fits used 43,647 CPU seconds, plus 155 CPU seconds for the earlier checks. Independent A40 scoring used 96 seconds and is counted separately. [Audit](../results/particles/audit.json) · [24 endpoint records](../results/particles/endpoints.csv).

Cuthbert is the next experiment. It can start after the original-code comparisons, GPU correctness tests and J=5000 memory and runtime checks pass. The prepared submission program checks these results before submitting the 12 A/B/C jobs.
