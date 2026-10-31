# Preserved history

Dynamax was removed from the main comparison on 31 October 2026 because its earlier experiment used our own PF with the package's optimizer interface. Its [code and result links](../../packages/dynamax/README.md) remain at their original imported locations for provenance only. The unmodified earlier combined reports and figures still include those historical results; future main comparisons exclude them.

This directory and its corresponding `artifacts/experiments/history/` contain earlier Cuthbert probes and earlier public-interface revisions. They explain implementation changes and failed checks. They are not additional replications of the current experiment.

Do not merge earlier CPU/J=1024 or common-warm-start probes with the later paper J=5000 experiment or the new A/B/C comparison. Each archive retains its original protocol and scope.

PFJAX's superseded independent-score release is kept separately under `experiments/legacy_dacca/pfjax/superseded_scores/` and the corresponding archive directory. The corrected evaluation_v2 results alone enter the final legacy comparison; old and new scores are never selected by which is larger.
