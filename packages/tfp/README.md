# TensorFlow Probability

Public interface v1. This is a local candidate change, not a claim that the original package already provides these methods or that upstream has accepted them.

The added iterated_filter uses the original sequential_monte_carlo time loop with parameterized distributions and joint state/parameter particles. The gradient_discount argument is added along particle_filter, sequential_monte_carlo and SequentialMonteCarlo, with default alpha=1 retaining the original expression. Keras Adam is external. The candidate baseline consists of TFP 0.25.0 wheel source files.

- [Original source](upstream/)
- [Candidate source](candidate/)
- [Complete patch](review/full.patch)
- [Exact calls and explanation](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Original review manifest](review/review_manifest.json)
- [Validation programs](validation/) and [preserved review results](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/tfp/)
- [New experiment and GPU/CPU evidence](../../experiments/public_interfaces/README.md)

Current evidence in this repository snapshot: The GPU alpha=1 comparison failed an exact-gradient check. New formal optimization was cancelled before starting. Official PF test-file comparisons and a partial full-upstream test run are preserved, including common failures and skips; they are not an all-tests-passed claim.

The earlier completed experiment used its own frozen driver. Its final scores do not validate this newer interface. Read the archive's own source/protocol when interpreting those earlier results. Reproduction archives and code-review ZIPs are physically preserved under `artifacts/`; the [archive catalog](../../manifests/archives.json) gives their paths and hashes.

The patch is checked against the source copied here by `python3 scripts/verify.py` from the repository root. Numerical validation and native PF timing are different checks; see [the status report](../../reports/status.md).
