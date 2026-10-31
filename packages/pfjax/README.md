# PFJAX

Public interface v2. This is a local candidate change, not a claim that the original package already provides these methods or that upstream has accepted them.

The added iterated_filter calls the full original particle_filter through an initial_particles argument added to its initializer. Model callbacks add parameter perturbations; native resampling jointly selects states and parameters. particle_filter_mop adds the gradient correction, including alpha=1. This correction is not the original OT or score estimator. The example and formal driver use external Optax Adam.

- [Original source](upstream/)
- [Candidate source](candidate/)
- [Complete patch](review/full.patch)
- [Exact calls and explanation](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Original review manifest](review/review_manifest.json)
- [Validation programs](validation/) and [preserved review results](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/pfjax/)
- [New experiment and GPU/CPU evidence](../../experiments/public_interfaces/README.md)

Current evidence in this repository snapshot: CPU configured tests passed. The recovered GPU run passed 72 configured tests but a Dacca gradient coordinate failed its original tolerance. The failure remains unresolved; the new formal job was cancelled before fitting.

The earlier completed experiment used its own frozen driver. Its final scores do not validate this newer interface. Read the archive's own source/protocol when interpreting those earlier results. Reproduction archives and code-review ZIPs are physically preserved under `artifacts/`; the [archive catalog](../../manifests/archives.json) gives their paths and hashes.

The patch is checked against the source copied here by `python3 scripts/verify.py` from the repository root. Numerical validation and native PF timing are different checks; see [the status report](../../reports/status.md).
