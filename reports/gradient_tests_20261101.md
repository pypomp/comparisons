# Review of the gradient tests

The Dacca test compared one call to the original Cuthbert ADPF with one call to our alpha=1 interface at each parameter/key pair. It used `numpy.allclose` with `rtol=1e-5, atol=1e-4`. These numbers were local regression settings. They were not taken from a paper or calibrated against repeated GPU calculations. The test did not measure the original implementation's repeatability before treating disagreement as a reason to stop the optimization comparison.

The calculation is a useful regression comparison, but its interpretation was too strong. It checks agreement between two implementations of a gradient estimator. It is not a comparison with the exact Dacca likelihood derivative, and a failed comparison alone does not show that the new estimator is wrong. The historical job state, tolerances and failed assertions remain unchanged.

## What the references support

- [Ścibior and Wood (2021), sections 2–3](https://arxiv.org/html/2106.10314v2) derive derivative corrections that leave the PF forward pass unchanged. This supports inspecting the backward computation separately. It does not prescribe our Dacca tolerances or a rule that every fixed-key GPU gradient must agree to those tolerances.
- [JAX's `stop_gradient` documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html) explicitly gives equal-valued functions with different automatic derivatives. Finite differences of an unchanged forward PF are therefore not a general reference for the deliberately corrected derivative. Our failed Dacca test did not use finite differences.
- [JAX's numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs) explains that compilation can change floating-point results. Its [indexed-update documentation](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html) also permits nondeterministic ordering of conflicting updates. These are reasons to measure repeatability; they do not establish the operation responsible for this Cuthbert result.
- [NumPy's `allclose` documentation](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) defines an elementwise numerical comparison and asks users to choose a suitable tolerance. It does not turn the chosen tolerance into a scientific correctness criterion.

## Recomputed scale of the Cuthbert differences

No PF or optimization was rerun. All 24 saved pairs were reread, their original results reproduced, and descriptive differences computed in float64 from the saved float32 gradients. Relative vector difference means `norm(public - original) / norm(original)`.

| Quantity | J=32, twelve pairs | J=5000, twelve pairs |
|---|---:|---:|
| Forward log likelihoods exactly equal | 12/12 | 12/12 |
| Original elementwise test passed | 11/12 | 4/12 |
| Largest relative vector difference | 0.0000764% | 0.0184561% |
| Smallest gradient-direction cosine | 0.9999999999998155 | 0.9999999999731588 |
| Gradient components with different signs | 0/276 | 0/276 |
| Largest individual component relative difference | 0.0100491% | 0.243049% |

These are descriptive measures, not replacement pass thresholds. A vector norm can hide smaller coordinates. Agreement of these directions does not establish identical Adam updates with accumulated moments or identical outcomes after 400 updates. Nevertheless, reporting only “8/12 failed” and the largest absolute difference of 3.4453125 omitted essential scale information. [Recomputed measures](../results/cuthbert/gradient_scales.json).

The separate 32-test interface suite includes a five-observation float64 model with an explicitly written recurrence and fixed recorded random draws/ancestry. It passed on the GPU. That is additional local implementation evidence; it is neither a complete Dacca derivative proof nor an independent reference for the marginal-likelihood gradient.

## Consequence for the experiment

The user accepts forward-likelihood agreement as sufficient to proceed with the empirical A/B/C comparison while derivative agreement is disclosed separately. We therefore accept the observed PF forward agreement and retain the gradient differences as an unresolved numerical diagnostic. The uncalibrated elementwise threshold alone will no longer be treated as evidence of an implementation defect or as a sufficient reason to stop all optimization work. This is an explicit change in how the checks are used, not a relabeling of the original failures or a claim of mathematically certified gradients.

The already submitted repeatability job remains useful; it does not change queued production source. The independent time-budget issue still needs a feasible submission plan. The previous submit program also depends on a failed job, so it cannot simply be rerun. Any replacement submission protocol must document this decision and preserve the scientific iteration counts, all previous evidence and total authorized allocation.

The same review recovered the completed [PyDPF repeatability diagnosis](pydpf_gpu_20261101.md). Full records and the recomputation programs are archived under `artifacts/validation/gradient_test_review_20261101.tar.gz`, with its SHA256 in the [catalog](../archive/catalog.json).
