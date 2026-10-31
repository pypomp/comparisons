# PFJAX

Local revision 2, based on the original version recorded in [source details](../../manifests/packages.json). The changes have not been submitted upstream.

The new `iterated_filter` calls the original `particle_filter`. We added an `initial_particles` argument so later IF2 passes can start from the previous parameter population. Model callbacks apply parameter perturbations; the original resampler selects state and parameter ancestors together.

The new `particle_filter_mop` adds the resampling derivative correction, including at α=1. This is our addition, not PFJAX's original optimal-transport or score estimator. The experiment uses external Optax Adam.

The earlier PFJAX experiment used an IF2 filtering loop we wrote ourselves. The new code calls the package's full PF. The earlier results therefore need to be read with their own source code.

## Code and tests

- [Original source](upstream/)
- [Modified source](candidate/)
- [Complete diff](review/full.patch)
- [Function descriptions](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Source versions and file hashes](review/review_manifest.json)
- [Test programs](validation/) and [test and source records](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/pfjax/)
- [New experiment and tests](../../experiments/public_interfaces/README.md)

## Included results

The configured CPU tests passed. The GPU run passed 72 configured tests, but one Dacca gradient coordinate failed its original tolerance. The failure is unresolved, and the new optimization job was cancelled before fitting.

[Results and tests](../../reports/status.md) gives the dates, measured PF times and remaining work. Earlier experiments used their own saved drivers. Their results do not validate this code revision.

The [archive list](../../manifests/archives.json) identifies the original source and raw-result archives under `artifacts/`. From the repository root, `python3 scripts/verify.py` checks that the diff applied to the copied original source produces the modified source exactly. Numerical tests are recorded separately.
