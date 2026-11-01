# Likelihood optimization

The [Dacca protocol](../experiments/dacca/README.md) defines A, B and C, their initialization, update counts and independent evaluation.

The [earlier completed comparison](../archive/README.md) contains 968 fitted parameter vectors and 37,584 individual PF evaluations. It includes the two-package paper experiment and six other packages, with their different scopes explicitly retained. Its final reports, full-range figures, optional 1.5-IQR display figures, old failed fits, low-precision scores and density-floor outcomes are preserved. PFJAX's corrected evaluation_v2 alone supplies its final scores; the superseded scores remain as history.

The new experiment has a different status:

| Package | New endpoints | State |
|---|---:|---|
| Cuthbert | 36/36 | Complete report and audit; two A fits failed, three A scores low precision |
| PFJAX | 3/36 | [First complete case audited](pfjax_case0_20261101.md); all three fits and scores pass; remaining eleven cases submitted once as 65170500 |
| PyDPF | 0/36 | [Startup 65170442 failed before formal fitting](pydpf_startup_and_tfp_20261101.md); new warm job 65170880 submitted; gradients not submitted |
| TFP | 0/36 | [Full-Dacca check 65170859](pydpf_startup_and_tfp_20261101.md) running; no new formal job |
| particles | 24/24 | Fits and independent scores audited; final new figures and reproducibility archive still pending |

For new particles, the median independent log likelihood is −4169.070668 for A (1800-second external Nelder-Mead) and −3784.099127 for B (900-second public IF2, then Nelder-Mead to the same total budget). B is higher in all twelve observed pairs. Precision passes are 9/12 for A and 12/12 for B; A tasks 0, 2 and 12 reached R=288 and remain low precision. There were no failed fits or density-floor scores. These findings apply to the recorded starts and budgets, not universal optimizer superiority.

The new particles fits used 43,647 CPU seconds, plus 155 CPU seconds for the earlier checks. Independent A40 scoring used 96 seconds and is counted separately. [Audit](../results/particles/audit.json) · [24 endpoint records](../results/particles/endpoints.csv).

[The complete Cuthbert report](cuthbert_20261101/report.md) includes all 36 retained endpoints and 2,052 original PF values. The all-endpoint median log mean likelihoods are −3814.966803 (A), −3747.557538 (B) and −3744.501336 (C). C exceeds B in all twelve paired cases, with median difference 2.351684. Two A fits stopped early and three A scores remain low precision; all B/C fits completed and their scores passed. These results apply to the recorded six starts and two replicates per start.

Median synchronized fitting-kernel times among complete fits are 555.756 seconds for A (10 fits), 627.436 for B and 627.586 for C (12 each). B/C include their IF2 prefix, while A does not; these are not equal total workloads. Failed A times are retained separately. The twelve allocations used 23,034 GPU seconds; the v3 check and repeat diagnosis bring formal-budget usage to 24,234 of 42,480 seconds. [All endpoint scores, failures and times](../results/cuthbert/endpoints.csv) and [complete audit](../results/cuthbert/final_audit_20261101.json) remain separate from the ordinary-PF timing.

Cuthbert's forward-PF agreement was accepted for proceeding with this empirical experiment. The [test-method review](reference_tests_20261101.md) and [original self-repeat diagnosis](cuthbert_repeats_20261101.md) explain why the old elementwise gradient threshold alone could not distinguish an extension error. This decision does not certify derivative correctness or erase the [earlier v3 failures](cuthbert_v3_20261101.md).
