# PyDPF: full Dacca validation submitted

Job **65169994** runs the new `pydpf_check_v4` on one A40, with 4 CPUs, 48 GiB host memory and a 26-minute limit. It was pending for Priority at the saved 09:39 UTC query. No GPU result from this check has been reviewed.

The check uses T=600, J=5000 and the six planned starting points, with two slots per start in the original twelve-row batch. For two validation seeds it evaluates the original `ParticleFilter`, original `StopGradientDPF`, and extended `StopGradientDPF` at alpha=1 and .97. All use systematic resampling, float32, the existing interval checkpointing, and default nondeterministic execution. Full monthly likelihood factors, summed likelihoods, gradients, parameters and final random-generator state are saved. Likelihood and RNG comparisons remain exact; the old uncalibrated gradient equality requirement is not used to stop the empirical comparison.

Three public IF2 passes retain the full B=1/J=5000 parameter swarm and history. Four isolated Adam proposals check finite estimation parameters, natural parameters and optimizer state. These are validation calls, not a formal optimization trajectory. The original native-PF timing benchmark is not repeated.

A CPU preflight at T=12/J=8/B=12 passed all six batch forward/RNG comparisons, eight finite calls, four finite proposals and three IF2 passes. Its saved arrays were independently audited. This short preflight is not full Dacca GPU evidence, and its timings are not GPU cost forecasts.

Before sbatch, the earlier v3 deployment stopped because an unused historical driver reference was absent at the expected remote path. It used no GPU time. v4 preserves that attempt, copies the historical driver and protocol into the new source directory, and verifies 84 unchanged runtime reference files. Neither revision changed frozen package code.

Earlier validation used 234 + 1411 + 306 = **1951 GPU seconds**. Adding the new allocation's maximum 1560 seconds gives **3511 of the original 3600-second validation budget**. The separate 30-GPU-hour formal budget is unused. The new result must establish full-scale likelihood agreement, finite values, memory use and actual rates before formal submission. All failures and timeouts will be retained.

[Submission and CPU audit](../results/pydpf/full_Dacca_submission_20261101.json). Full source, records and recovery commands are in the dated archive listed in [the archive catalog](../archive/catalog.json).
