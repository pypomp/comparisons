# Cuthbert

Local revision 3, based on the original version recorded in [source details](version.json). The changes have not been submitted upstream.

All 36 final estimates in the [completed A/B/C experiment](../../reports/cuthbert_20261101/report.md) have been checked. The earlier v3 gradient comparisons failed or timed out. Repeated evaluations of the unchanged original code also failed the earlier gradient criterion. The optimization experiment proceeded after agreement in forward likelihoods was accepted. See [correctness](../../reports/correctness.md).

## Original code and IF2

The new IF2 and DMOP functions call `particle_filter.build_filter` and use the original `filter_prepare` and `filter_combine` updates. Our `scan_final` function loops over observation times and returns the final filter state. It does not call the top-level `cuthbert.filter`.

## Our changes

IF2 adds a parameter vector to each state particle, applies random-walk perturbations, and selects state and parameter ancestors together. It retains the parameter population between IF2 passes and reduces perturbations according to the cooling schedule. At α=1, the gradient calculation calls the unchanged original `stop_gradient_decorator`. At α<1, it uses the discounted decorator we added.

## Experiment calls

The completed A/B/C experiment calls `iterated_filtering.fit` and `dmop.log_likelihood`, with external Optax `scale_by_adam` for parameter updates. It does not call the combined `ifad.fit` function.

In the [distribution plots](../../reports/dacca_native_figures_20261102T0718/report.md), B is labelled IF2 + native stop-gradient PF because the alpha=1 branch calls the original `stop_gradient_decorator`. We added IF2, the loop over observations and the calling interface. Adam is external.

## Validation and reproduction

[Changed files](changes/) · [Complete diff](review/full.patch) · [Version](version.json) · [Validation programs](validation/)

From the repository root, restore the original code and apply the diff:

```bash
python3 scripts/prepare.py cuthbert --output build/cuthbert
```

The command checks every resulting file against the saved modified source. Original source trees, environments, test outputs and failures remain in the [archive](../../archive/README.md). [Correctness](../../reports/correctness.md) lists the tests and unresolved failures. [PF runtime](../../reports/timing.md) and [optimization](../../reports/optimization.md) report measurements separately. [Reproduction instructions](../../docs/reproduction.md) distinguish restoring files from rerunning numerical experiments.
