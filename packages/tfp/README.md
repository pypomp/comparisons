# TensorFlow Probability

Local revision 1, based on the original version recorded in [source details](../../manifests/packages.json). The changes have not been submitted upstream.

The new `iterated_filter` wraps the parameterized model and calls the original `sequential_monte_carlo` loop. State and parameter particles are resampled together.

For gradient optimization, we added `gradient_discount` to the existing `particle_filter`, `sequential_monte_carlo` and `SequentialMonteCarlo` calls. At α=1 the code retains the original derivative expression. Parameter updates use external Keras Adam.

The original source here is from the TFP 0.25.0 wheel. It is not a full development checkout.

## Code and tests

- [Original source](upstream/)
- [Modified source](candidate/)
- [Complete diff](review/full.patch)
- [Function descriptions](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Source versions and file hashes](review/review_manifest.json)
- [Test programs](validation/) and [test and source records](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/tfp/)
- [New experiment and tests](../../experiments/public_interfaces/README.md)

## Included results

The GPU α=1 comparison failed an exact-gradient check. New optimization jobs were cancelled before starting. Both versions passed 120 official PF tests, with 23 skipped, after a test-harness name-binding repair. Original common failures and the incomplete full test run are preserved separately.

[Results and tests](../../reports/status.md) gives the dates, measured PF times and remaining work. Earlier experiments used their own saved drivers. Their results do not validate this code revision.

The [archive list](../../manifests/archives.json) identifies the original source and raw-result archives under `artifacts/`. From the repository root, `python3 scripts/verify.py` checks that the diff applied to the copied original source produces the modified source exactly. Numerical tests are recorded separately.
