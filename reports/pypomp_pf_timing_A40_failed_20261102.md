# PyPOMP A40 timing: failed attempt

Job **65241013 failed before any warmup or timed PF call**. No median is available. Slurm records exit `1:0` and 56 allocation seconds on A40 node `gl1529`; the limit was 300 seconds. The 56 seconds are job resource use, not PF runtime.

The saved traceback ends during JAX tracing with `invalid PRNG key data: expected key_data.ndim >= 1; got ndim=0`. The timing script supplies legacy `jax.random.PRNGKey` values. In the retained native model interpolation code, a non-scalar key is indexed with `keys[0]`, leaving a scalar integer that `jax.random.split` rejects. This identifies a key-format mismatch in this attempted integration. No corrected execution was run, and this failure says nothing about PyPOMP's PF speed.

The requested setting remains T=600, J=5000, float32, twenty Euler steps per month and one PF per call, with two warmups and seven timed calls. The saved record contains zero completed calls and no completed compilation time. Input construction had passed, but it had not exercised this traced filter path.

The intended call uses the unchanged native Dacca model and systematic resampling. Earlier Cuthbert/PFJAX timings use study callbacks and multinomial resampling, in different allocations. The earlier RTX PRO 6000/J8192/batch-32 report remains separate and cannot fill this missing row. No fitting, IF2, gradients or endpoint scoring was run.

The partial audit verified 107 frozen source files (including 99 unchanged PyPOMP files), 114 transferred files, both terminal result records, the launch claim, scheduler state and failure logs. This is verification of a failed attempt, not a successful timing audit. Full source, inputs, logs, submission and audit are in the [failure archive](../artifacts/experiments/dacca/pypomp_pf_timing_A40_failed_20261102.tar.gz); its extracted saved-data audit reproduces the original audit exactly. No automatic retry is authorized for this attempt.

[Saved audit and protocol](../results/pypomp/pf_timing_A40_failed_20261102.json) · [Runtime table](timing.md) · [Earlier measurements](pypomp_pf_timing_20261102.md).

中文：本次计时脚本与原生模型路径的随机数 key 格式不兼容，追踪阶段失败。尚无预热、计时或中位耗时；56 秒只是作业占用时间。失败记录已完整保存，未自动重跑。
