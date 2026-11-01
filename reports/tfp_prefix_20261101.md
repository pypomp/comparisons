# TFP: completed IF2 prefix and gradient submissions

The shared IF2 job 65171217 completed on A40 node gl1526 with exit code 0:0 and 12,747 elapsed GPU seconds. All twelve fits completed 175 passes, for 2,100 accepted passes and no failed prefix rows. These are initialization results; no A/B/C endpoint or independent score is yet available.

The saved-data audit verified 3,144 frozen source files, 60 transferred files and 56 terminal files. It checked every original start, final 5,000-particle parameter cloud, retained last state, 175-pass call record, final mean, finite natural parameters and all 95 canonical zero-Adam arrays. The full final clouds are retained; intermediate full clouds are not. Successful-call root/member pairs were reconstructed from frozen source and iteration records, rather than separately recorded for every internal random draw.

Synchronized IF2 calls used 12,561.817 seconds; the recorded prefix stage wall time was 12,662.022 seconds. Both are parts of the 12,747-second Slurm allocation and must not be added to it. The physical prefix runs once; method B and method C will each include its full cost in their logical method totals.

| Gradient phase | Job | Update plan | Allocation limit |
|---|---|---|---:|
| A | 65177155 | Original starts, alpha1, 400 Adam updates, B12 | 16,800 s |
| B | 65177156 | Audited IF2 starts, alpha1, 400 Adam updates, B12 | 16,800 s |
| C | 65177157 | Same IF2 starts, alpha0.97, 400 Adam updates, B12 | 16,800 s |

All three previously unrun phases were submitted once in `tfp_formal_v4`. Their original T600/J5000 model, six starts and two replicates, learning rates, training keys, production execution settings and fixed original PyPOMP scorer are unchanged. B/C independently begin with the same parameters and canonical zero Adam state. Their score keys remain independent. The [driver diff](../experiments/dacca/tfp/gradient_v4_driver_changes.diff) and [proposal-recording diff](../experiments/dacca/tfp/gradient_v4_proposal_recording.diff) show the changes: full rejected proposals are saved, and the driver reads the existing physical prefix from the older revision. It cannot dispatch another IF2 phase.

The new frozen source manifest is `b997485eb85e73c56f2270541d684238d593370802ed6ddaedc3d1c47ec12b07` (3106 files). Startup checks require the actual prefix audit, source and terminal hashes, zero-state hashes, original settings, fixed scorer and internal review. All three checks passed against the real collected prefix in an isolated local tree and on the remote host before submission; no launch claim or fit was made during those checks. CPU imports do not evaluate a PF. Exclusive per-method claims prevent duplicate starts, including retries after a claimed startup failure. Whole-call exceptions without a returned proposal retain the exception and last state; missing fields are not invented.

Actual prefix cost plus the three full gradient reservations is 12,747 + 50,400 = 63,147 GPU seconds, within the original 72,000-second formal budget. The separate validation history used 2,336 seconds. The earlier failed cost split and all other historical failures retain their original status; this audit does not certify mathematical derivatives.

The [structured record](../results/tfp/prefix_and_gradients_20261101.json) includes actual job identifiers and the dated scheduler snapshot. The [prefix audit](../results/tfp/prefix_audit_20261101.json) is reproduced byte for byte after extracting the archive, as is the earlier saved-proposal audit. All three production startup checks also pass from the restored records. No fit or independent PF was rerun, and no running gradient directory was collected.

Archive `artifacts/experiments/dacca/tfp_prefix_and_gradient_submission_20261101T1556.tar.gz` contains 9,492 files, SHA256 `0c6c04036f3240f0f5708a61876b385654557c8a832863b7765c7f145c179f54`. [Archive catalog](../archive/catalog.json).
