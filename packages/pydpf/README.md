# PyDPF

Public interface v1. This is a local candidate change, not a claim that the original package already provides these methods or that upstream has accepted them.

IteratedFiltering wraps model prior, transition and observation callbacks so each particle carries parameters and state, then calls the original ParticleFilter and SIS.forward. The added outer passes preserve the full parameter swarm. The original StopGradientDPF correction is reused and an alpha parameter is added to discount its derivative. Adam is external torch.optim.Adam.

- [Original source](upstream/)
- [Candidate source](candidate/)
- [Complete patch](review/full.patch)
- [Exact calls and explanation](review/call_chain.txt)
- [Function locations](review/function_locations.json)
- [Original review manifest](review/review_manifest.json)
- [Validation programs](validation/) and [preserved review results](review/)
- [Earlier completed experiment](../../experiments/legacy_dacca/pydpf/)
- [New experiment and GPU/CPU evidence](../../experiments/public_interfaces/README.md)

Current evidence in this repository snapshot: GPU forward PF comparisons matched all 600 likelihood increments in nine pairs. The repaired gate passed 33 of 34 tests; a multinomial float32 exact-equality check failed. Formal optimization was cancelled before starting. This baseline contains a reviewed repository subset, not every upstream file.

The earlier completed experiment used its own frozen driver. Its final scores do not validate this newer interface. Read the archive's own source/protocol when interpreting those earlier results. Reproduction archives and code-review ZIPs are physically preserved under `artifacts/`; the [archive catalog](../../manifests/archives.json) gives their paths and hashes.

The patch is checked against the source copied here by `python3 scripts/verify.py` from the repository root. Numerical validation and native PF timing are different checks; see [the status report](../../reports/status.md).
