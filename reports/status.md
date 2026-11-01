# Experiment status

The [saved scheduler query](../results/current_jobs.json) was captured at 20:37 UTC on 1 November 2026. This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 36/36 endpoints audited; two A fits failed and three A scores are low precision; final report, figures and reproduction archive prepared |
| PFJAX | [Matched execution controls: 324/324 likelihood pairs exactly equal](pfjax_execution_controls_20261101.md); original forward-only/differentiated values also differ | [36/36 endpoints and final review delivered](pfjax_20261101/report.md); 35 complete fits, one failed A fit with density-floor score; all other 35 scores meet precision |
| PyDPF | [48/48 full-Dacca forward pairs exactly equal](pydpf_native_DPF_20261101.md); original failed checks retained | 0/36; [twelve complete IF2 fits audited; A/B/C submitted once](pydpf_warm_20261101.md), using the same physical warm results |
| TFP | [72/72 full-Dacca forward pairs exactly equal](tfp_full_Dacca_20261101.md); finite values and memory pass, original time split fails | [36/36 endpoints and final review delivered](tfp_20261101/report.md); 35 complete fits, one failed A fit and floor score; 35 other scores meet precision |
| particles | CPU checks and independent GPU evaluation complete | [24/24 results, figures and reproduction archive delivered](particles_20261101/report.md); 21 scores precise, three low-ESS A scores retained |

[The complete Cuthbert report](cuthbert_20261101/report.md) contains all 36 retained endpoints and 2,052 original PF values. Two A fits in cases 8 and 9 failed at 43 and 60 accepted updates; those two scores and case 6 A remain low precision at R=288. The other 33 scores pass. C's estimated likelihood exceeds B's in all twelve pairs, with a median paired difference of 2.351684. The formal allocations used 23,034 GPU seconds. The complete source, pairing, random-key, raw-score and cost audit has passed; old validation failures remain failures. PyDPF remains unfinished; all three existing gradient jobs are running in the recorded snapshot. TFP is complete with its failed fit retained. PFJAX has its own completed report, figures and verified reproduction archive. The particles report, figures and full reproduction archive are complete.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).

The [TFP saved-state and score audit preparation](tfp_gradient_audit_preparation_20261101.md) passes local saved-record controls. All three actual phases and the complete paired audit have now passed; A's numerical fit failure remains explicitly recorded. No model or optimizer was run by these preparation checks.

The [PyDPF phase, score and aggregate audit preparation](pydpf_gradient_audit_preparation_20261101.md) is now available. Existing jobs remain queued or running; no new gradient result is claimed.

The [TFP three-method audit preparation](tfp_all_methods_audit_preparation_20261101.md) now checks complete metadata, paired starts and streams, retained failures, independent scores and total cost using saved and synthetic controls. The [actual three-method audit](../results/tfp/final_audit_20261101.json) is complete. It retains 36 endpoints, 1,332 raw PF values, one failed fit and its floor score.

The separate [PyDPF maze pilot](pydpf_maze_20261101/report.md) has a verified time-budget limitation: job 65185523 failed at its original cost check after 1,475 GPU seconds. Ten finite full-length GPU controls give eight exact matched forward pairs; the forecast of 34,374.759 seconds exceeds the 330.428 seconds remaining. No model was trained and no held-out RMSE exists. The report, cost figure and full archive retain this outcome; no retry or larger allocation was made. Existing PyDPF Dacca A/B/C jobs 65177901/65177902/65177903 remain running in the 20:37 snapshot (8,464/4,995/3,797 seconds respectively), with their separate original budget.
