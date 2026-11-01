# Experiment status

The [saved scheduler query](../results/current_jobs.json) is from 02:28 UTC on 1 November 2026. Cuthbert v3 was running on gl1516; the other three GPU checks were pending for priority, without estimated start times. The four existing jobs now request 48 GiB of host memory. See the [queue inspection and adjustment](queue_20261101.md). This page records that query; it does not poll the cluster.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | 65144479 running, gl1516 | A/B/C prepared; 0/36 estimates; not submitted |
| PFJAX | 65151851 pending | 0/36; earlier dependent jobs cancelled before starting |
| PyDPF | 65151857 pending | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 pending | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

Cuthbert A/B/C can start once the v3 GPU comparisons, J=5000 memory check and runtime check pass. PFJAX, PyDPF and TFP have unresolved gradient comparisons. The diagnostic jobs record repeated calls to the original and modified code to help locate the differences.

The reports address separate questions:

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
