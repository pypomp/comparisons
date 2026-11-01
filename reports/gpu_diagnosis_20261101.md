# PFJAX, PyDPF and TFP GPU checks

The three earlier GPU failures remain unresolved. This update records additional CPU controls and three submitted GPU diagnostic jobs. It contains no new optimization results. The scheduler query at 00:30 UTC on 1 November 2026 showed all three jobs pending for priority; Cuthbert v3 job 65144479 was also pending.

## What failed

| Package | Recorded GPU failure | Next comparison |
|---|---|---|
| PFJAX | One Dacca gradient coordinate differed by 0.000608444, exceeding the existing `rtol=2e-5, atol=2e-4` check | Repeat the historical corrected callback and the public correction. Save monthly states, weights and ancestors. Compare both with and without gradient checkpointing |
| PyDPF | One of six gradient entries in the float32 multinomial test differed by 2.38419e-7; exact equality was required | Repeat unchanged upstream, candidate default and explicit alpha=1 five times each. Compare forward values, gradients and final RNG state, with default and deterministic Torch settings |
| TFP | Forward values matched exactly, but 22 of 23 alpha=1 gradient entries differed; maximum difference was 1.41561e-7 | Repeat the same original graph, rebuild it, and compare against the public interface. Test default execution, deterministic TensorFlow operations and execution without gradient checkpointing |

PFJAX's two compared correction implementations both call `pfjax.particle_filter`. The correction itself is added code, including at alpha=1. PyDPF compares against its original `StopGradientDPF`. TFP compares against its original `particle_filter` and SMC kernel. These controls distinguish repeatability of the original implementation from differences introduced by our changes.

The existing failures and tolerances are retained. A deterministic setting or a checkpointing change in this diagnostic does not automatically become the configuration for a formal experiment.

## Completed CPU controls

- PFJAX: T=600, J=8, alpha=0, 0.97 and 1. With and without checkpointing, both implementations produced exactly matching values and gradients. Repeated calls matched. The printed JAX expressions for each pair were identical.
- PyDPF: the five-observation, 24-particle test model passed exact comparisons of values, gradients and RNG state across both resamplers and float32/float64, with five repeats of each implementation.
- TFP: T=600, J=8, alpha=1. The original and public interfaces, repeated calls and separately rebuilt graphs matched exactly.

These are CPU observations. They do not establish GPU correctness. In particular, the new TFP GPU check uses J=5000, as in the failed check. The [local check record](../results/validation_20261101/other_packages_local_review.json) records versions and scope.

## Submitted GPU diagnostics

| Package | Job | GPU | Maximum allocation |
|---|---|---|---|
| PFJAX | 65151851 | One A40 | 20 minutes |
| PyDPF | 65151857 | One A40 | 20 minutes |
| TFP | 65151858 | One A40 | 20 minutes |

PyDPF and TFP use the remaining validation budgets: prior actual costs plus the full diagnostic allocations are at most 2,845 and 1,632 GPU seconds, respectively, within their 3,600-second limits. PFJAX reserves 20 minutes from its unused 11.8-hour formal budget. Future formal allocations must account for this reservation and the measured diagnostic cost.

The programs verify the referenced frozen sources before and after execution, save raw arrays and comparisons, and write terminal file hashes. A successful process exit means that the diagnostic ran; it does not mean all equality checks passed. Only stopped jobs may be collected, one package at a time. Native PF timing and formal fitting are not repeated.

## Code and records

The [archive catalog](../archive/catalog.json) lists `artifacts/validation/gpu_diagnostics_20261101_setup.tar.gz`. It contains the diagnostic programs, protocols, submission receipts, all referenced package sources, CPU arrays, printed JAX expressions and SHA256 records. Earlier failed GPU results remain in the prior archives.

To extract the setup, use:

```sh
python3 scripts/extract_archive.py artifacts/validation/gpu_diagnostics_20261101_setup.tar.gz build/gpu_diagnostics
```

The archive's `README.md` describes its paths and how to rerun a local control. GPU launch programs contain the original cluster paths and should be reviewed before reuse. The current submitted jobs must not be submitted again.

No package implementation was changed in this update. SHA256 checks confirmed that all 1,442 protected Cuthbert source and check files remained unchanged. Cuthbert v3 and its prepared A/B/C comparison retain their existing code and validation requirements.
