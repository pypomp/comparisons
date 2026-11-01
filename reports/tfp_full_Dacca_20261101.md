# TFP: full-Dacca checks and revised time allocation

Job 65170859 ended FAILED (1:0) after 701 A40 GPU seconds. Its complete saved numerical checks passed; its original time split did not. The failed job and original cost result are retained.

The check used the original Dacca T600/J5000/B12 settings, six starts with two replicate slots, the original production execution settings and the same differentiated execution for original and public filters. All 72 matched endpoint pairs had exactly equal monthly likelihood increments. All nine filter/gradient calls were finite, as were three complete public IF2 passes and four actual independent Keras Adam proposals. Peak reported GPU memory was 6,479,740,160 bytes, below 85% of the device's 48,305,799,168 bytes.

Original self-repeat gradients varied by up to 9.537e−7. The old componentwise comparisons remain diagnostic; these observations do not certify mathematical derivatives. The discounted gradient is intended to differ from alpha1. This check does not repeat the ordinary-PF timing benchmark or perform a formal fit.

| Phase | Unchanged conservative forecast (seconds) | Old allocation | New allocation |
|---|---:|---:|---:|
| Shared IF2 prefix | 17,543.879 | 28,800 | 21,600 |
| A gradient | 14,529.744 | 14,400 | 16,800 |
| B gradient | 14,529.744 | 14,400 | 16,800 |
| C gradient | 14,625.764 | 14,400 | 16,800 |
| Total allocation | | 72,000 | 72,000 |

The forecast retains the original 10% IF2 and 20% gradient multipliers, graph/warmup costs and fixed reserves. Batch time is counted once. Only the allocation split changed; particle counts, 175 IF2 passes, 400 Adam updates, starts, keys and production execution settings did not. Validation used 2,336 of its separate 3,600 GPU-second ceiling.

The new `tfp_formal_v3` revision submitted only the shared prefix as job 65171217, with A40/4 CPUs/48 GiB and a six-hour limit. All 3,090 runtime source files match the completed check. The driver adds rejected IF2 cloud/likelihood/natural-parameter records and removes the obsolete repeated gradient comparison; [the diff](../experiments/dacca/tfp/driver_changes_v3.diff) shows both changes. A/B/C have not been submitted. The new runtime guard accepts only the prefix; actual prefix states and cost will be audited before a later gradient revision that also reviews complete rejected-Adam proposal recording.

The initial SSH attempt was denied by the local sandbox before deployment or sbatch. With the same existing authentication, reconciliation confirmed no remote directory and no matching queued or historical job. Source verification and a CPU import then completed before one real sbatch call. No filtering ran on the login node. Both attempt records are preserved.

[Collected arrays, audit and actual submission](../results/tfp/full_Dacca_20261101.json) · [Archive catalog](../archive/catalog.json). The source and archive checks establish provenance and reproducibility of saved records, not universal correctness or a completed optimization result.
