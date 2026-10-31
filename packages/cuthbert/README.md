# Cuthbert

Local revision 3, based on the original version recorded in [source details](../../manifests/packages.json). The changes have not been submitted upstream.

The new IF2 and DMOP functions call `particle_filter.build_filter` and use the original `filter_prepare` and `filter_combine` updates. Our `scan_final` function loops over observation times and returns the final filter state. It does not call the top-level `cuthbert.filter`.

IF2 adds a parameter vector to each state particle, applies random-walk perturbations, and selects state and parameter ancestors together. It retains the parameter population between IF2 passes and reduces perturbations according to the cooling schedule. At α=1, the gradient calculation calls the unchanged original `stop_gradient_decorator`. At α<1, it uses the discounted decorator we added.

The planned A/B/C experiment calls `iterated_filtering.fit` and `dmop.log_likelihood`, with external Optax `scale_by_adam` for parameter updates. It does not call the combined `ifad.fit` function.

## Code and tests

- [Original source](upstream/)
- [Modified source](candidate/)
- [Complete diff](review/full.patch)
- [Function descriptions](review/README_EN.txt)
- [Function locations](review/function_locations.json)
- [Source versions and file hashes](review/review_manifest.json)
- [Test programs](validation/) and [test and source records](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/paper_cuthbert_pypomp/)
- [New experiment and tests](../../experiments/public_interfaces/README.md)

## Included results

The revised CPU checks passed. The earlier full GPU test run timed out with three failures. Jobs 65144329 and 65144479, which compare the original code and test the revision, were pending at the included scheduler query on 31 October 2026, 22:09:39 UTC. No v3 A/B/C optimization results are included.

[Results and tests](../../reports/status.md) gives the dates, measured PF times and remaining work. Earlier experiments used their own saved drivers. Their results do not validate this code revision.

The [archive list](../../manifests/archives.json) identifies the original source and raw-result archives under `artifacts/`. From the repository root, `python3 scripts/verify.py` checks that the diff applied to the copied original source produces the modified source exactly. Numerical tests are recorded separately.
