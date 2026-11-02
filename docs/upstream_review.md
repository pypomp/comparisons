# Installing the additions in separate repositories

The five local revisions can be installed and their added functions called without the comparisons directory. Each includes a complete patch, tests, documentation and a small CPU example. No remote fork, commit, push or PR was created.

| Package | Local checks | Patch and installation instructions |
|---|---|---|
| Cuthbert | 1,540 tests passed; added files pass Ruff; type check passed with one existing warning; strict documentation build passed | [Cuthbert](../packages/cuthbert/upstream_20261102/README.md) |
| PFJAX | 72 tests passed; new documentation page rendered and included in navigation | [PFJAX](../packages/pfjax/upstream_20261102/README.md) |
| PyDPF | 34 tests passed; strict documentation build passed | [PyDPF](../packages/pydpf/upstream_20261102/README.md) |
| particles | 34 tests passed; new documentation page rendered | [particles](../packages/particles/upstream_20261102/README.md) |
| TFP | 7 added tests passed; BUILD declarations and both substrate rewrites checked | [TFP](../packages/tfp/upstream_20261102/README.md) |

All wheels built and installed in separate directories. Each example then ran outside its source tree, with imports checked against the installed wheel. Dependencies came from existing local Python environments; this is not a fresh dependency-resolution test or the full upstream CI matrix.

PFJAX, PyDPF and TFP tests no longer import a neighboring `upstream/` tree. Small reference arrays from the unmodified package are included with source hashes and dependency versions. Independent IF2 and gradient recursions remain in the tests; tolerances were not loosened. The exact reference comparisons were tested on the recorded CPU environments, not every supported backend/version.

TFP's tests and example now live in its package tree. Its new IF2 module has a TensorFlow-only BUILD target and is omitted from the NumPy and JAX substrates; the upstream rewrite program confirmed both omissions. Bazel is not installed here, so these checks do not establish a successful Bazel build. The tested base remains v0.25.0.

PFJAX's strict full-site build still reports warnings in existing API documentation. particles' strict full-site build still reports existing documentation issues and a missing optional `sklearn` import. The added pages' reported warnings were fixed. These full-site checks remain incomplete; their original logs are retained.

The numerical library files are byte-identical to the experimental revisions. This work changes test portability, build declarations, documentation and example formatting; it does not establish new Dacca results or GPU behavior. Historical patches under `changes/` and `review/` remain available. Apply each new **full** patch to its recorded upstream base, not on top of the old patch.

This follows [Ed's request](conventions.md) to add general parameter perturbations and ADPF extensions in each package's style (29 October, 15:27:20–15:28:52). Dacca data, optimization drivers and results stay in comparisons. Author review and final CI checks are still needed before submission.

[Full source, patches and check logs](../artifacts/validation/package_standalone_revisions_20261102.tar.gz) · [Recorded checks](../results/package_standalone_20261102.json) · [Archive catalog](../archive/catalog.json)
