# Experiment status

The [saved scheduler query](../results/current_jobs.json) is from 03:22 UTC on 1 November 2026. Cuthbert v3, PFJAX and PyDPF have ended and been reviewed; TFP is running; the existing Cuthbert gradient diagnosis and the first full A/B/C case are pending. This page records that query; it does not poll the cluster.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | 65144479 failed; collected and reviewed. Gradient diagnosis 65156615 pending | 0/36; first full case 0 submitted as 65158659; remaining 11 cases await actual cost review |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 running on gl1524 | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

Cuthbert case 0 is the first complete A/B/C fit of the planned twelve, submitted once as 65158659 with a 57-minute A40 allocation. It has no fitted endpoints yet. The v3 package, fitting program, scorer and scientific iteration counts are unchanged. The [comparison with existing Kevin/Aaron code](reference_tests_20261101.md) records the actual test arithmetic, its limits and the explicit decision to proceed on forward-likelihood agreement. Gradients remain a separate diagnostic. All 12 allocations plus the existing checks fit within 11.8 GPU hours; remaining cases follow after the first case's actual cost is reviewed. [Submission](../results/cuthbert/submission_v4.json) · [Acceptance and cost](../results/cuthbert/acceptance_v4.json).

All current diagnostic jobs request an A40, four CPUs and 48 GiB host memory. The [earlier resource adjustment](queue_20261101.md) preserved job identities and frozen sources. Numerical failures, timeouts and all costs remain recorded.

The reports address separate questions:

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
