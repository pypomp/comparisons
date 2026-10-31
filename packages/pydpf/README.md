# PyDPF

Local revision 1, based on the original version recorded in [source details](../../manifests/packages.json). The changes have not been submitted upstream.

The new `IteratedFiltering` wraps the model's prior, transition and observation callbacks so each particle contains both parameters and state. It calls the original `ParticleFilter` and `SIS.forward`; those classes are unchanged. The outer IF2 passes retain the full parameter population and apply the perturbation and cooling schedules.

The gradient stage uses the original `StopGradientDPF` derivative correction, extended with an `alpha` argument. Parameter updates use external `torch.optim.Adam`.

The copied original code includes the specified package subtree, root README and license, and a native tutorial. It is a subset of the upstream repository.

## Code and tests

- [Original source](upstream/)
- [Modified source](candidate/)
- [Complete diff](review/full.patch)
- [Function descriptions](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Source versions and file hashes](review/review_manifest.json)
- [Test programs](validation/) and [test and source records](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/pydpf/)
- [New experiment and tests](../../experiments/public_interfaces/README.md)

## Included results

Nine paired GPU runs matched all 600 PF likelihood increments. A later test run passed 33 of 34 tests; a float32 multinomial exact-equality check failed. New optimization jobs were cancelled before starting.

[Results and tests](../../reports/status.md) gives the dates, measured PF times and remaining work. Earlier experiments used their own saved drivers. Their results do not validate this code revision.

The [archive list](../../manifests/archives.json) identifies the original source and raw-result archives under `artifacts/`. From the repository root, `python3 scripts/verify.py` checks that the diff applied to the copied original source produces the modified source exactly. Numerical tests are recorded separately.
