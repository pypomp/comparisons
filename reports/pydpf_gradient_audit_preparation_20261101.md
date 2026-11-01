# PyDPF gradient audit preparation

The 17:27 UTC scheduler query shows PyDPF A/B/C jobs 65177901, 65177902 and 65177903 pending for priority. TFP A/B/C jobs 65177155, 65177156 and 65177157 remain running. No new job was submitted, no running result directory was collected, and neither package has an audited gradient endpoint.

`audit_pydpf_gradient_v3.py` is now prepared outside the frozen experiment. It combines the saved-state core with checks of the actual collection, all source and terminal files, launch records, the completed physical IF2 results, original starts or warm parameters, full final Adam states, fixed-slot update traces, last valid parameters, complete rejected proposals, per-iteration seeds, independent PF records and actual allocation cost. The production wrapper requires `COMPLETED 0:0`, complete fitting and scoring records, and one launch and scoring attempt. Interrupted allocations require a separate partial audit; missing records do not become successful results.

The score check reads all twelve records and every raw value using the fixed original PyPOMP source, the frozen scorer, root 94781000 and fixed A/B/C codes. It checks parameter order and hashes, typed endpoint/chunk/repeat keys, adaptive stopping, MCSE, likelihood ESS, nonfinite values and floor flags. The final record must retain the values in its last partial checkpoint. Floor saturation is counted separately from numerical precision.

The optional all-phase audit repeats the three saved phase audits, checks B/C initial parameters, empty Adam hashes, paired seeds and recorded generator-state hash prefixes, then counts the physical warm allocation once. It checks 812 planned distinct Torch seeds for the warm/A/shared-B-C domains and separately checks uniqueness of JAX scoring keys. Torch and JAX have different generators; this does not claim to compare their internal random draws or use a common key space. A/B workloads remain different.

Preparation controls are saved data and explicitly synthetic metadata, not model evaluations:

| Control | Result |
|---|---|
| Existing six small CPU gradient runs, twelve rows, three rejected proposals | Previous saved-state core passes; nine corrupted copies rejected |
| Twelve synthetic score records, 900 values and 937 keys | Nine corruptions rejected; nine precision flags include one floor result, so eight are nonfloor precise; low-ESS, nonfinite and unscorable records retained |
| Synthetic allocation, terminal, launch and cost records | Fifteen corruptions rejected; missing real collection refuses output |
| Synthetic three-method pairing and total-cost records | Seven corruptions rejected, including different B/C generator hashes and duplicate scoring keys |

The wrapper has not yet audited a real stopped gradient phase. Its actual collection schema must still be checked when results arrive. Successful intermediate proposals are not all saved; the audit neither reconstructs missing states nor certifies mathematical gradients. The natural-parameter transform check uses the existing float32 audit rule (relative tolerance 2e-6, absolute tolerance 2e-7), not a relaxed PF or gradient acceptance threshold.

The first archive's member hashes passed, but its extracted missing-input control failed because a temporary import path hid the Torch dependency before checking for the absent collection. That archive, log and original scripts are retained as a failed preparation attempt. The auditor now checks for the collection before importing the Torch core. The corrected archive reproduces all saved core, score, metadata and aggregate JSONs byte for byte. No PF, IF2, optimizer step, refit or rescore was executed. All 7,395 protected frozen files remain unchanged.

See the [structured preparation and archive records](../results/pydpf/gradient_audit_preparation_20261101.json), [completed warm audit and actual submissions](pydpf_warm_20261101.md), and [archive catalog](../archive/catalog.json).
