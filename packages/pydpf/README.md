# PyDPF

Local revision 1, based on the original version recorded in [source details](version.json). The changes have not been submitted upstream.

## Original code and IF2

The new `IteratedFiltering` wraps the model's prior, transition and observation callbacks so each particle contains both parameters and state. It calls the original `ParticleFilter` and `SIS.forward`; those classes are unchanged. The outer IF2 passes retain the full parameter population and apply the perturbation and cooling schedules.

## Derivatives and optimization

The gradient stage uses the original `StopGradientDPF` derivative correction, extended with an `alpha` argument. Parameter updates use external `torch.optim.Adam`.

The copied original code includes the specified package subtree, root README and license, and a native tutorial. It is a subset of the upstream repository.

## Dacca callback performance

The later [forward and backward optimization](../../reports/pydpf_backward_fusion_20261102.md) changes our Dacca transition callbacks and leaves the package's filter unchanged. We compared individual call times and first derivatives with the old and new callbacks. The completed A/B/C fits were not rerun, so their final estimates and fitting times are unchanged. [Runtime](../../reports/timing.md) gives the units and limitations of the measurements.

## Validation and reproduction

[Changed files](changes/) · [Complete diff](review/full.patch) · [Version](version.json) · [Validation programs](validation/)

From the repository root, restore the original code and apply the diff:

```bash
python3 scripts/prepare.py pydpf --output build/pydpf
```

The command checks every resulting file against the saved modified source. Original source trees, environments, test outputs and failures remain in the [archive](../../archive/README.md). [Correctness](../../reports/correctness.md) lists the tests and unresolved failures. [PF runtime](../../reports/timing.md) and [optimization](../../reports/optimization.md) report measurements separately. [Reproduction instructions](../../docs/reproduction.md) distinguish restoring files from rerunning numerical experiments.
