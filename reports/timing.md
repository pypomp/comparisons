# Runtime

Kevin proposed timing the common PF (29 October, 15:23:18). We report ordinary PF calls, gradient calculations and full optimization separately, with the units and measurement dates for each.

## Original ordinary-PF measurements

These calls use the original Dacca model, T=600 and J=5000. PyPOMP uses its native model; the other packages use study-written model callbacks. Values are medians of seven timed calls after warmup; GPU batch size is one on A40. Compilation is separate. The CPU row is not part of a GPU ranking.

| Package PF and model | Hardware | Seconds per ordinary PF |
|---|---|---:|
| PyPOMP (native model) | A40 | **0.216745** |
| Cuthbert | A40 | 0.393526 |
| PFJAX | A40 | 0.220416 |
| PyDPF | A40 | 8.576100 |
| TFP | A40 | 6.278312 |
| particles | CPU | 5.496709 |

These times measure the complete filter and model implementation used in each row. They do not isolate library overhead or establish the fastest implementation possible in each package. The [archived records](../docs/reproduction.md#restore-historical-records) contain the original code review and individual timings. The [PyPOMP supplement](pypomp_pf_timing_A40_20261102.md) used native systematic resampling, whereas Cuthbert/PFJAX used multinomial resampling. Its compilation took 1.851 seconds and is excluded from the table. The allocations differ; the earlier failed attempt is retained in the linked report.

PyDPF timings include our original Dacca callback. [Later callback optimizations](pydpf_backward_fusion_20261102.md) were not used in the completed fits.


## Details

[Earlier PyPOMP timings at different settings](pypomp_pf_timing_20261102.md) and [PyDPF callback measurements](pydpf_backward_fusion_20261102.md) are archived. They do not replace the original fitting results.

[Fitting times](optimization.md#fitting-time) are reported separately. Total GPU use counts the shared IF2 run once. A batch time covers all fits in that batch. The supervised maze study is listed under [Active work](status.md).
