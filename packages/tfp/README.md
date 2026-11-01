# TensorFlow Probability

Local revision 1, based on the original version recorded in [source details](version.json). The changes have not been submitted upstream.

The new `iterated_filter` wraps the parameterized model and calls the original `sequential_monte_carlo` loop. State and parameter particles are resampled together.

For gradient optimization, we added `gradient_discount` to the existing `particle_filter`, `sequential_monte_carlo` and `SequentialMonteCarlo` calls. At α=1 the code retains the original derivative expression. Parameter updates use external Keras Adam.

The original source here is from the TFP 0.25.0 wheel. It is not a full development checkout.


## Inspect and restore

[Changed files](changes/) · [Complete diff](review/full.patch) · [Version](version.json) · [Validation programs](validation/)

From the repository root, restore the original code and apply the diff:

```bash
python3 scripts/prepare.py tfp --output build/tfp
```

The command checks every resulting file against the saved modified source. Original source trees, environments, test outputs and failures remain in the [archive](../../archive/README.md). [Results and tests](../../reports/status.md) describes the included evidence. [Reproduction instructions](../../docs/reproduction.md) distinguish restoring files from rerunning numerical experiments.
