# pfjax: standalone installation revision

Base: `97652aa1bdff73a92c0286549b010e99cc6f7264`. Apply `full.patch` from that upstream root with `git apply`. Then follow the added `PARAMETER_ESTIMATION.md` for installation, the small example and tests.

The full patch includes the earlier general estimation additions and this revision's portable tests and build/documentation changes. Do not apply it on top of the old experiment patch. Numerical library files are byte-identical to the experiment revision.

[Current checks and remaining CI work](../../../docs/upstream_review.md). Old experiment sources under `../changes/` and `../review/` remain unchanged. No remote fork or PR has been created.
