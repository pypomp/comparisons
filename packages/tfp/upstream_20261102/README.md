# tfp: standalone installation revision

Base: `v0.25.0`. Apply `full.patch` from that upstream root with `git apply`. Then follow the added `PARAMETER_ESTIMATION.md` for installation, the small example and tests.

The full patch includes the earlier general estimation additions and this revision's portable tests and build/documentation changes. Do not apply it on top of the old experiment patch. Numerical library files are byte-identical to the experiment revision.

[Current checks and remaining CI work](../../../docs/upstream_review.md). Old experiment sources under `../changes/` and `../review/` remain unchanged. No remote fork or PR has been created.
