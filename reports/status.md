# Experiment status

The [saved scheduler queries](../results/current_jobs.json) are from 06:57 UTC on 1 November 2026. This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 30/36 endpoints audited; two A fits failed and three A scores are low precision; case 10 running, case 11 pending |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree on the tested linear model | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 failed after three subprocess timeouts; partial arrays reviewed | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

The [ten audited Cuthbert cases](cuthbert_cases0_9_20261101.md) contain 30 retained endpoints and 1,836 original PF values. All ten IF2 runs and all B/C fits completed their updates. Direct A failed after 43 and 60 accepted updates in cases 8 and 9; their last valid parameters and full costs are retained. Those two A scores and case 6 A reached R=288 with ESS below 10. The other 27 scores pass. C's estimated likelihood exceeds B's in all ten observed pairs, covering five starting vectors; the sixth vector remains unfinished. Completed allocations used 19,128 GPU seconds. The saved query shows case 10 running and case 11 pending with JobArrayTaskLimit. No fit or score was repeated.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
