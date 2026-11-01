# Experiment status

The [saved scheduler queries](../results/current_jobs.json) include the 08:26 UTC query on 1 November 2026 and the separate PFJAX post-submission query. This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 36/36 endpoints audited; two A fits failed and three A scores are low precision; final report, figures and reproduction archive prepared |
| PFJAX | [Full Dacca check 65168298 submitted](pfjax_check_v3_20261101.md); running at its saved query | 0/36; awaiting full-scale output and cost review |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree on the tested linear model | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 failed after three subprocess timeouts; partial arrays reviewed | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

[The complete Cuthbert report](cuthbert_20261101/report.md) contains all 36 retained endpoints and 2,052 original PF values. Two A fits in cases 8 and 9 failed at 43 and 60 accepted updates; those two scores and case 6 A remain low precision at R=288. The other 33 scores pass. C's estimated likelihood exceeds B's in all twelve pairs, with a median paired difference of 2.351684. The formal allocations used 23,034 GPU seconds. The complete source, pairing, random-key, raw-score and cost audit has passed; old validation failures remain failures. PFJAX, PyDPF, TFP and the final particles figures remain separate unfinished work.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
