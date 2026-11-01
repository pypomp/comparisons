# Likelihood optimization

The [Dacca protocol](../experiments/dacca/README.md) defines A, B and C, their initialization, update counts and independent evaluation.

The [earlier completed comparison](../archive/README.md) contains 968 fitted parameter vectors and 37,584 individual PF evaluations. It includes the two-package paper experiment and six other packages, with their different scopes explicitly retained. Its final reports, full-range figures, optional 1.5-IQR display figures, old failed fits, low-precision scores and density-floor outcomes are preserved. PFJAX's corrected evaluation_v2 alone supplies its final scores; the superseded scores remain as history.

The new experiment has a different status:

| Package | New endpoints | State |
|---|---:|---|
| Cuthbert | 33/36 | Eleven cases audited; two A fits failed, three A scores low precision; last case running |
| PFJAX | 0/36 | Dependent formal job cancelled before starting after GPU validation failure |
| PyDPF | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| TFP | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| particles | 24/24 | Fits and independent scores audited; final new figures and reproducibility archive still pending |

For new particles, the median independent log likelihood is −4169.070668 for A (1800-second external Nelder-Mead) and −3784.099127 for B (900-second public IF2, then Nelder-Mead to the same total budget). B is higher in all twelve observed pairs. Precision passes are 9/12 for A and 12/12 for B; A tasks 0, 2 and 12 reached R=288 and remain low precision. There were no failed fits or density-floor scores. These findings apply to the recorded starts and budgets, not universal optimizer superiority.

The new particles fits used 43,647 CPU seconds, plus 155 CPU seconds for the earlier checks. Independent A40 scoring used 96 seconds and is counted separately. [Audit](../results/particles/audit.json) · [24 endpoint records](../results/particles/endpoints.csv).

[Case 10](cuthbert_case10_20261101.md) completed all three fits. Its independent A/B/C log mean likelihoods are -3768.950269, -3748.262771 and -3745.913990; all use R=36 and pass the precision rule. This brings the count to 33 endpoints and 1,944 original PF values. The two A failures and three low-precision A scores remain in the [earlier records](cuthbert_cases0_9_20261101.md). C's estimated likelihood exceeds B's in all eleven observed pairs; the final case is unfinished.

Completed allocations used 21,181 GPU seconds. The v3 check, repeated-gradient diagnosis, completed jobs and last allocation limit total 25,801 seconds, within 42,480 seconds. Failure cost remains distinct from successful 400-update latency. [Endpoint scores, failures and times](../results/cuthbert/endpoints.csv) and [case-level costs](../results/cuthbert/current_results.json) are separate. Final figures and the full review package await the remaining case and complete audit.

Cuthbert's forward-PF agreement was accepted for proceeding with this empirical experiment. The [test-method review](reference_tests_20261101.md) and [original self-repeat diagnosis](cuthbert_repeats_20261101.md) explain why the old elementwise gradient threshold alone could not distinguish an extension error. This decision does not certify derivative correctness or erase the [earlier v3 failures](cuthbert_v3_20261101.md).
