# Correctness

These checks compare the original package code with our changes. Results below include the records collected through 1 November 2026. Pending jobs are listed in the dated [status](status.md).

| Package | Tests and comparisons | Unfinished work |
|---|---|---|
| Cuthbert | Earlier local coverage: 1548 unique tests eventually passed after dependency repairs. Current v3: 32 CPU interface tests, full T=600 checks with both resamplers at J=32, and 12 original and modified ADPF comparisons passed. GPU diagnosis reproduced all three previous failures in unchanged original code; both sides passed 64 PF tests | Revised GPU check 65144479 remains pending. Full GPU correctness is not established |
| PFJAX | 72 configured CPU tests and 72 configured GPU tests passed, with upstream exclusions retained | A Dacca gradient coordinate failed its tolerance. Repeated-call and checkpointing diagnosis 65151851 is pending; no formal optimization result |
| PyDPF | Native PF likelihood increments matched in nine GPU pairs; repaired test run passed 33/34 | A float32 multinomial exact-gradient test failed. Original-repeatability diagnosis 65151857 is pending |
| TFP | The separate PF subset passed 120 tests with 23 skipped on both sides after a test-harness name-binding repair. All 515 full-suite file pairs have now been attempted; no additional failed test identities in the candidate | GPU alpha=1 exact-gradient check failed; diagnosis 65151858 is pending. The full CPU suite did not pass |
| particles | Nine CPU PF pairs matched likelihood/state/weights/RNG; public IF2 tests and collected-result audits passed | The pinned upstream has no pytest suite; this is not a claim that an upstream test suite passed |

The original TFP snapshot of 324 pairs remains archived. The completed run has 515 pairs: 370 normal exits, 127 test/collection failures, eight no-test exits, nine timeouts and one abort per side. Original/candidate outcomes and failed-test identities match in every pair. This does not imply that all tests passed. [Completed-run details](validation_20261101.md).

The Cuthbert v3 change preserves the original alpha=1 differentiation path by defaulting to `checkpoint=False`. A recorded CPU multinomial diagnosis showed that adding checkpointing to the original ADPF reproduces the prior discrepancy. Disabling it passed the recorded checks without changing tolerances. This local diagnosis still requires the pending v3 GPU checks. The three earlier failed tests have now also failed in unchanged upstream on A40; their underlying numerical cause remains undetermined.
