# Correctness

These checks compare the original package code with our changes. Results below include the records collected through 1 November 2026. Pending jobs are listed in the dated [status](status.md).

The [comparison with existing Kevin/Aaron code](reference_tests_20261101.md) reports the actual test formulas and recalculation: 23/24 saved pairs satisfy the CuTe formula, with one J5000 exception. These are retrospective diagnostics, not Dacca-specific acceptance thresholds.

The [review of the gradient tests](gradient_tests_20261101.md) distinguishes an implementation discrepancy from GPU repeatability and from mathematical correctness. The original Dacca thresholds were not calibrated against repeated GPU calculations. Their failures remain recorded; they alone do not establish an implementation defect. The user accepts forward-PF agreement for proceeding with the empirical comparison while derivative agreement is disclosed separately.

| Package | Tests and comparisons | Unfinished work |
|---|---|---|
| Cuthbert | Earlier local coverage: 1548 unique tests eventually passed after dependency repairs. v3 GPU job 65144479 passed 32 interface tests, both T=600/J=32 resampling checks, and the small A/B/C driver check | At the twelve planned starts, gradient comparisons failed in 1/12 pairs at J=32 and 8/12 at J=5000. PF regression stopped after 61/64 passes at its time limit; the earlier-failure subset also timed out. See [v3 results](cuthbert_v3_20261101.md) |
| PFJAX | 72 configured CPU tests and 72 configured GPU tests passed, with upstream exclusions retained. Diagnosis 65151851 completed; forward histories agree | Repeated gradients vary within both implementations. Cross-repeat pairs passed 27/27 with checkpoint and 21/27 without it at J=8. Full-scale validation remains unfinished. [Diagnosis](pfjax_gpu_20261101.md) |
| PyDPF | Native PF likelihood increments matched in nine GPU pairs; repaired test run passed 33/34. Diagnosis 65151857 is collected and audited; all 200 cross-repeat gradient pairs agree exactly in deterministic mode | Default GPU mode also varies within the original code on the small fixture. Full Dacca validation remains separate. [Diagnosis](pydpf_gpu_20261101.md) |
| TFP | The separate PF subset passed 120 tests with 23 skipped on both sides after a test-harness name-binding repair. All 515 full-suite file pairs have now been attempted; no additional failed test identities in the candidate | GPU alpha=1 exact-gradient check failed; diagnosis 65151858 is running. The full CPU suite did not pass |
| particles | Nine CPU PF pairs matched likelihood/state/weights/RNG; public IF2 tests and collected-result audits passed | The pinned upstream has no pytest suite; this is not a claim that an upstream test suite passed |

The original TFP snapshot of 324 pairs remains archived. The completed run has 515 pairs: 370 normal exits, 127 test/collection failures, eight no-test exits, nine timeouts and one abort per side. Original/candidate outcomes and failed-test identities match in every pair. This does not imply that all tests passed. [Completed-run details](validation_20261101.md).

The Cuthbert v3 change calls the unchanged original alpha=1 resampling correction and defaults to `checkpoint=False`. Disabling checkpointing passed the recorded CPU checks, but did not make all GPU gradients agree at the planned Dacca starts. The three earlier failed tests also failed in unchanged upstream in a separate A40 diagnosis; that result does not explain the new gradient differences. A new diagnostic control compares the original and modified code repeatedly, including a wrapper that uses the full original `cuthbert.filter` time loop. This control has not replaced production code.
