# Experiment status

The [saved scheduler queries](../results/current_jobs.json) are from 05:56 UTC on 1 November 2026. This page is a recorded snapshot, not a live cluster view.

| Package | GPU check | New optimization results |
|---|---|---|
| Cuthbert v3 | Earlier failed checks retained; the repeat diagnosis shows that the old gradient rule also rejects original self-repeats | 12/36 endpoints audited; cases 4–6 running as 65160224; cases 7–11 waiting for the concurrency limit |
| PFJAX | 65151851 completed and reviewed; repeated gradients vary | 0/36; full-scale validation still required |
| PyDPF | 65151857 completed and reviewed; deterministic controls agree on the tested linear model | 0/36; earlier dependent jobs cancelled before starting |
| TFP | 65151858 failed after three subprocess timeouts; partial arrays reviewed | 0/36; earlier dependent jobs cancelled before starting |
| particles | CPU checks and independent GPU evaluation complete | 24/24 estimates audited; 21 scores meet the precision requirement; final figures pending |

The [first four Cuthbert A/B/C cases](cuthbert_cases0_3_20261101.md) have been audited. All twelve methods completed 400 Adam updates and all four shared IF2 runs completed 175 passes. All twelve independent scores satisfy the precision rule; 432 original PF values, full states, source hashes and costs have been reviewed. C has the highest likelihood in each observed case, followed by B and A; only the first two starting vectors have finished. Completed allocations used 8,026 GPU seconds across four jobs. The remaining eight cases retain their original settings and fit within the budget. The scheduler shows cases 4-6 running and cases 7-11 waiting for the three-task concurrency limit.

The [TFP diagnosis](tfp_gpu_20261101.md) retains all three timeouts. Its deterministic control has 12/12 available cross pairs with identical likelihood increments and gradients; without checkpointing, repeated gradients vary even within unchanged upstream.

The [Cuthbert repeat diagnosis](cuthbert_repeats_20261101.md) retains 54 finite evaluations with identical corresponding likelihoods. At J=5000, seven of nine unchanged-original self pairs fail the old gradient rule. The empirical A/B/C comparison proceeds under the user's acceptance of forward-likelihood agreement; derivative correctness remains a separate question. The [reference-test review](reference_tests_20261101.md) records the actual Kevin/Aaron code examined and its limits.

| Question | Report |
|---|---|
| Do the changes preserve the required filter and derivative calculations? | [Correctness](correctness.md) |
| How long does one ordinary PF call take? | [PF runtime](timing.md) |
| Does IF2 initialization or gradient discounting improve the fitted likelihood? | [Optimization](optimization.md) |

Completed checks: [Cuthbert v2 and TFP CPU tests](validation_20261101.md). Diagnostic design and CPU controls: [PFJAX, PyDPF and TFP](gpu_diagnosis_20261101.md). Earlier results, failures and full test logs remain in the [archive](../archive/README.md).
