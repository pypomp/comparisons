# Experiment status

The [saved scheduler query](../results/current_jobs.json) is from 02:57 UTC on 1 November 2026. Cuthbert v3, PFJAX and PyDPF have ended and been reviewed; TFP and the new Cuthbert gradient diagnosis are pending for priority. This page records that query; it does not poll the cluster.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | 65144479 failed; collected and reviewed. Gradient diagnosis 65156615 pending | 0/36; forward agreement accepted; budget and revised submission plan remain unresolved |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 pending | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

Cuthbert A/B/C has not started. The [gradient-test review](gradient_tests_20261101.md) accepts forward PF agreement for the empirical comparison and retains numerical derivative differences separately. The uncalibrated gradient threshold alone is not a reason to stop; the conservative runtime estimate still exceeds the existing submission limit. The new seven-minute diagnosis compares repeated calculations and the original time loop without modifying production code. [Cuthbert results](cuthbert_v3_20261101.md) · [PFJAX results](pfjax_gpu_20261101.md) · [PyDPF results](pydpf_gpu_20261101.md).

All current diagnostic jobs request an A40, four CPUs and 48 GiB host memory. The [earlier resource adjustment](queue_20261101.md) preserved job identities and frozen sources. Numerical failures, timeouts and all costs remain recorded.

The reports address separate questions:

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
