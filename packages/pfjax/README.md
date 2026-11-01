# PFJAX

Local revision 2, based on the original version recorded in [source details](version.json). The changes have not been submitted upstream.

## Original code and IF2

The new `iterated_filter` calls the original `particle_filter`. We added an `initial_particles` argument so later IF2 passes can start from the previous parameter population. Model callbacks apply parameter perturbations; the original resampler selects state and parameter ancestors together.

## Derivatives and optimization

The new `particle_filter_mop` adds the resampling derivative correction, including at α=1. This is our addition, not PFJAX's original optimal-transport or score estimator. The experiment uses external Optax Adam.

The earlier PFJAX experiment used an IF2 filtering loop we wrote ourselves. The new code calls the package's full PF. The earlier results therefore need to be read with their own source code.

## Validation and reproduction

[Changed files](changes/) · [Complete diff](review/full.patch) · [Version](version.json) · [Validation programs](validation/)

From the repository root, restore the original code and apply the diff:

```bash
python3 scripts/prepare.py pfjax --output build/pfjax
```

The command checks every resulting file against the saved modified source. Original source trees, environments, test outputs and failures remain in the [archive](../../archive/README.md). [Correctness](../../reports/correctness.md) lists the tests and unresolved failures. [PF runtime](../../reports/timing.md) and [optimization](../../reports/optimization.md) report measurements separately. [Reproduction instructions](../../docs/reproduction.md) distinguish restoring files from rerunning numerical experiments.
