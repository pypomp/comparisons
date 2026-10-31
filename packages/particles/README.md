# particles

Local revision 1, based on the original version recorded in [source details](../../manifests/packages.json). The changes have not been submitted upstream.

The new `iterated_filtering` and `_PerturbedBootstrap` give joint state and parameter particles to the original `SMC.run` loop and `Bootstrap` model calls. They add parameter perturbations, cooling and retention of the parameter population between IF2 passes. Existing Python source files are unchanged.

The experiment uses external SciPy Nelder-Mead. Method A runs it for 1,800 seconds. Method B uses IF2 for the first 900 seconds and then Nelder-Mead for the remaining time. This is not a differentiable-PF experiment.

## Code and tests

- [Original source](upstream/)
- [Modified source](candidate/)
- [Complete diff](review/full.patch)
- [Function descriptions](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Source versions and file hashes](review/review_manifest.json)
- [Test programs](validation/) and [test and source records](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/particles/)
- [New experiment and tests](../../experiments/public_interfaces/README.md)

## Included results

Nine paired CPU PF runs matched likelihoods, states, weights and random-number state. The added IF2 tests passed. All 24 new fits and independent scores have been checked; 21 meet the precision requirement. Three A scores remain below it at 288 repeats. Final figures and the reproducibility archive are pending.

[Results and tests](../../reports/status.md) gives the dates, measured PF times and remaining work. Earlier experiments used their own saved drivers. Their results do not validate this code revision.

The [archive list](../../manifests/archives.json) identifies the original source and raw-result archives under `artifacts/`. From the repository root, `python3 scripts/verify.py` checks that the diff applied to the copied original source produces the modified source exactly. Numerical tests are recorded separately.
