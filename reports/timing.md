# Particle-filter runtime

These are measured ordinary PF calls for the original Dacca model, T=600 and J=5000. Values are medians of seven timed calls, with warmup/compilation separate. The GPU calls have batch size 1 on A40 nodes. The CPU result is not part of a GPU speed ranking.

| Original PF | Hardware | Seconds per PF call |
|---|---|---:|
| Cuthbert | A40 | 0.393526 |
| PFJAX | A40 | 0.220416 |
| PyDPF | A40 | 8.57610 |
| TFP | A40 | 6.278312 |
| particles | CPU | 5.496709 |

These are implementation-specific timings under the recorded environments. They are not optimizer running times or evidence that a method reaches a better likelihood sooner. PyDPF's candidate PF median was 8.56534 seconds and particles' candidate median was 5.526815 seconds; these observations do not establish a general speed improvement.

The [Dacca protocol](../experiments/dacca/README.md) describes the model. Optimization costs are reported separately in [Optimization](optimization.md). Full timing records and environments are retained in the [archive](../archive/README.md).
