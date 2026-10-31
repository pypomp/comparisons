# Cuthbert

Public interface v3. This is a local candidate change, not a claim that the original package already provides these methods or that upstream has accepted them.

The added IF2 interface uses particle_filter.build_filter and the original PF updates and resamplers. Parameter and state particles share ancestor selection. Full parameter swarms, perturbations, cooling and outer iterations are added. Alpha=1 calls the unchanged original stop_gradient_decorator; alpha<1 uses the added discounted decorator. The experiment driver calls iterated_filtering.fit and dmop.log_likelihood with external Optax scale_by_adam, rather than complete ifad.fit.

- [Original source](upstream/)
- [Candidate source](candidate/)
- [Complete patch](review/full.patch)
- [Exact calls and explanation](review/README_EN.txt)
- [Function locations](review/function_locations.json)
- [Original review manifest](review/review_manifest.json)
- [Validation programs](validation/) and [preserved review results](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/paper_cuthbert_pypomp/)
- [New experiment and GPU/CPU evidence](../../experiments/public_interfaces/README.md)

Current evidence in this repository snapshot: The revised CPU checks passed. The full earlier GPU suite timed out with three failures. Original/candidate GPU diagnosis 65144329 and revised gate 65144479 were pending at 2026-10-31 22:09:39 UTC. No v3 A/B/C optimization result exists in this snapshot.

The earlier completed experiment used its own frozen driver. Its final scores do not validate this newer interface. Read the archive's own source/protocol when interpreting those earlier results. Reproduction archives and code-review ZIPs are physically preserved under `artifacts/`; the [archive catalog](../../manifests/archives.json) gives their paths and hashes.

The patch is checked against the source copied here by `python3 scripts/verify.py` from the repository root. Numerical validation and native PF timing are different checks; see [the status report](../../reports/status.md).
