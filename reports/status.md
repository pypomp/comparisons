# Experiment status

The [saved scheduler queries](../results/current_jobs.json) are from 04:57 UTC on 1 November 2026 (the detailed partition query is from 04:34 UTC). This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 3/36 endpoints audited; cases 1–11 submitted as 65160224 and pending because spgpu is DOWN |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree on the tested linear model | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 failed after three subprocess timeouts; partial arrays reviewed | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

The [first Cuthbert A/B/C case](cuthbert_case0_20261101.md) completed in 31 minutes 22 seconds. All three methods completed 400 Adam updates, with 175 shared IF2 passes for B/C. All three independent scores meet the precision rule; 108 original PF values, complete optimizer states, source hashes and costs have been reviewed. The remaining eleven cases retain the 57-minute limit and unchanged source. Their allocation limits plus observed costs remain within 11.8 GPU hours. The scheduler reports `PartitionDown` and no estimated start time; no task has been resubmitted to bypass it.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
