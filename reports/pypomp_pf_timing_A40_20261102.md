# PyPOMP ordinary-PF timing on A40

The median of seven warmed calls was **0.216744510 seconds per PF** (range 0.216410400–0.221013231). Each call ran one ordinary PF with T=600, J=5000, float32 and twenty Euler steps per month. Compilation took **1.850996366 seconds**, reported separately. All nine calls, including two warmups, returned finite likelihoods.

| Root seed | Seconds per PF | Log likelihood |
|---|---:|---:|
| 952000 | 0.216443900 | -3749.232910 |
| 952001 | 0.216733931 | -3748.837158 |
| 952002 | 0.221013231 | -3749.223877 |
| 952003 | 0.217205612 | -3749.224609 |
| 952004 | 0.216410400 | -3749.086914 |
| 952005 | 0.216744510 | -3747.928711 |
| 952006 | 0.220259208 | -3749.196533 |

The filter and native Dacca model are unchanged PyPOMP code. The timer includes typed-key creation, the executable call and synchronization; host scalar conversion and file writing follow it. Parameter transformation and the root-key split remain inside the executable. The two warmups took 0.221059844 and 0.218766239 seconds.

The [first attempt](pypomp_pf_timing_A40_failed_20261102.md) failed before any timed call. This user-authorized correction changes the benchmark's two root-key creation sites from legacy `PRNGKey` to scalar typed `random.key(..., impl="threefry2x32")`, with the same seeds. Before resubmission, the full T600/J5000 path passed CPU compilation and one finite PF check. The old key failure was reproduced, and key data agreed for all planned seeds. The CPU check is separate from this GPU measurement.

Job 65263775 completed with exit 0:0 on gl1500/NVIDIA A40, using 56 Slurm allocation seconds within its 300-second cap. The earlier failed attempt used 56 seconds, giving 112 GPU allocation seconds across both attempts. Compilation, warmups and timed calls are already included in those allocations.

PyPOMP uses its native model and **systematic resampling**; the earlier Cuthbert/PFJAX measurements use study-written model callbacks and multinomial resampling. These are different allocations and unpaired random draws. The table compares recorded implementations, without isolating library overhead or claiming an optimization advantage. The earlier RTX PRO 6000/J8192/batch 32 timing remains a different-setting result.

The saved audit verified 111 frozen source files, 118 transferred files and 2 terminal records, including 99 unchanged original PyPOMP files, the unique launch, all nine seeds and the seven-value median. The [complete archive](../artifacts/experiments/dacca/pypomp_pf_timing_A40_keyfix_20261102.tar.gz) retains source, inputs, raw results, logs, CPU checks, the correction diff and submission records. Its extracted audit matches byte for byte; no PF was rerun during that verification. The first audit incorrectly required CPU and GPU parameter transforms to be byte-identical. The corrected audit checks identical estimation inputs and exact agreement with the earlier A40 parameter record; the two CPU/GPU rounding differences and the original audit error are retained.

[Structured data](../results/pypomp/pf_timing_A40_20261102.json) · [Runtime table](timing.md) · [Earlier different-setting timings](pypomp_pf_timing_20261102.md).

中文：修正key格式并先通过完整CPU检查后，单次A40补测完成。普通PF中位耗时0.216745秒，编译1.851秒单列；原失败记录保留。
