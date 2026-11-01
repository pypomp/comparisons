# TFP gradient submission preparation

At 12:57 UTC on 1 November 2026, PyDPF shared IF2 job 65170880 was running on gl1503 (7,392 elapsed seconds) and TFP prefix job 65171217 was running on gl1526 (3,091 seconds). Neither prefix was collected, approved for subsequent gradient phases or rerun. PFJAX, Cuthbert and particles remain complete.

The separate local TFP preparation now verifies the recorded source, collected prefix, scientific audit, terminal hashes, full zero-Adam-state hashes, original scientific settings and fixed independent scorer before an A/B/C attempt. It reads the existing physical prefix from the earlier revision. The prepared driver cannot dispatch an IF2 phase. An exclusive per-method directory claim permits one attempt even if two callers arrive simultaneously. A claimed but failed startup is retained; it is not automatically retried.

The [driver diff](../experiments/dacca/tfp/gradient_submission_preparation_20261101.diff) changes the prefix location and startup checks. The Adam function's syntax tree is otherwise identical to the earlier tested proposal-recording preparation. The package/API source is exactly the prior recording version; all other package, model and input files must match the frozen prefix revision. The original live sources were not edited.

Synthetic file controls reject changed prefix, zero-Adam, checkpoint, completion, terminal and source records; a wrong source revision; path traversal; missing real collection or release approval; IF2 dispatch; and duplicate attempts. Two concurrent callers obtain one claim. These tests contain no PF, IF2 or fitting calls. They do not test a successful production approval, because the actual prefix has not finished or been audited.

The prepared directory has no frozen manifest, actual prefix reference, release approval or jobs record. Its batch file is a template, not a deployed launcher. After the original prefix stops, collect it once and audit all twelve rows, retained failures, parameter clouds, initialization and costs. Only then can the existing authorization be used to create the new gradient revision and submit the unrun A/B/C phases. The original prefix-only guard remains unchanged. Its 6-hour prefix plus three 16,800-second gradient reservations still total the original 20 GPU hours.

All preparation files, synthetic records, prior proposal tests and original prefix source are archived. Every archive member was verified; extracted guard-control results and the earlier saved-driver audit reproduce byte for byte. All 4,289 monitored frozen files remain unchanged. These are integrity and local preparation checks, not new GPU results or mathematical derivative certification.

Archive `artifacts/validation/tfp_gradient_guard_preparation_20261101T1256.tar.gz` contains 6,341 files; SHA256 `b893dc5da4ea10f1348ee733c1bd0a389bd0f5688cbfc596313bb17a8fb2154d`. [Structured preparation record](../results/tfp/gradient_submission_preparation_20261101.json).
