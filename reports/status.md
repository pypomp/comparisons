# Experiment status

The [saved scheduler query](../results/current_jobs.json) was captured at 13:27 UTC on 1 November 2026. This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 36/36 endpoints audited; two A fits failed and three A scores are low precision; final report, figures and reproduction archive prepared |
| PFJAX | [Matched execution controls: 324/324 likelihood pairs exactly equal](pfjax_execution_controls_20261101.md); original forward-only/differentiated values also differ | [36/36 endpoints and final review delivered](pfjax_20261101/report.md); 35 complete fits, one failed A fit with density-floor score; all other 35 scores meet precision |
| PyDPF | [48/48 full-Dacca forward pairs exactly equal](pydpf_native_DPF_20261101.md); original failed checks retained | 0/36; shared IF2 job 65170880 running on gl1503; no gradient phase submitted |
| TFP | [72/72 full-Dacca forward pairs exactly equal](tfp_full_Dacca_20261101.md); finite values and memory pass, original time split fails | 0/36; shared prefix job 65171217 submitted under revised limits within the same 20 GPU hours, running on gl1526 |
| particles | CPU checks and independent GPU evaluation complete | [24/24 results, figures and reproduction archive delivered](particles_20261101/report.md); 21 scores precise, three low-ESS A scores retained |

[The complete Cuthbert report](cuthbert_20261101/report.md) contains all 36 retained endpoints and 2,052 original PF values. Two A fits in cases 8 and 9 failed at 43 and 60 accepted updates; those two scores and case 6 A remain low precision at R=288. The other 33 scores pass. C's estimated likelihood exceeds B's in all twelve pairs, with a median paired difference of 2.351684. The formal allocations used 23,034 GPU seconds. The complete source, pairing, random-key, raw-score and cost audit has passed; old validation failures remain failures. PyDPF and TFP remain unfinished. PFJAX has its own completed report, figures and verified reproduction archive. The particles report, figures and full reproduction archive are complete.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
