# Experiment status

The [saved scheduler query](../results/current_jobs.json) is from 02:48 UTC on 1 November 2026. Cuthbert v3 and PFJAX have ended and been reviewed. PyDPF is running; TFP and the new Cuthbert gradient diagnosis are pending for priority. This page records that query; it does not poll the cluster.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | 65144479 failed; collected and reviewed. Gradient diagnosis 65156615 pending | 0/36; gradient, incomplete regression and runtime requirements prevent A/B/C submission |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 running, gl1516 | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 pending | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

Cuthbert A/B/C has not started: its J=5000 gradient comparisons failed in 8/12 pairs and its conservative runtime estimate exceeds the existing submission limit. The new seven-minute diagnosis compares repeated calculations and the original time loop without modifying production code. [Cuthbert results](cuthbert_v3_20261101.md) · [PFJAX results](pfjax_gpu_20261101.md).

All current diagnostic jobs request an A40, four CPUs and 48 GiB host memory. The [earlier resource adjustment](queue_20261101.md) preserved job identities and frozen sources. Numerical failures, timeouts and all costs remain recorded.

The reports address separate questions:

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
