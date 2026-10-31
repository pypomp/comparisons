# particles

Public interface v1. This is a local candidate change, not a claim that the original package already provides these methods or that upstream has accepted them.

The added iterated_filtering and _PerturbedBootstrap provide joint state/parameter particles to the original SMC.run loop and Bootstrap model calls. They add perturbation, cooling and full parameter-swarm propagation. Existing Python source files remain unchanged. The optimizer used by the Dacca experiment is external SciPy Nelder–Mead.

- [Original source](upstream/)
- [Candidate source](candidate/)
- [Complete patch](review/full.patch)
- [Exact calls and explanation](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Original review manifest](review/review_manifest.json)
- [Validation programs](validation/) and [preserved review results](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/particles/)
- [New experiment and GPU/CPU evidence](../../experiments/public_interfaces/README.md)

Current evidence in this repository snapshot: The CPU forward PF matched likelihood, states, weights and RNG in nine pairs. Public IF2 checks passed. The new comparison has 24 audited endpoints with 21 precision passes; three low-precision A endpoints are retained. Final new-comparison figures and release packaging remain pending.

The earlier completed experiment used its own frozen driver. Its final scores do not validate this newer interface. Read the archive's own source/protocol when interpreting those earlier results. Reproduction archives and code-review ZIPs are physically preserved under `artifacts/`; the [archive catalog](../../manifests/archives.json) gives their paths and hashes.

The patch is checked against the source copied here by `python3 scripts/verify.py` from the repository root. Numerical validation and native PF timing are different checks; see [the status report](../../reports/status.md).
