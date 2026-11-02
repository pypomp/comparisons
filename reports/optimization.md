# Likelihood optimization

The [Dacca protocol](../experiments/dacca/README.md) defines A, B and C, their initialization, update counts and independent evaluation.

The [earlier completed comparison](../archive/README.md) contains 968 fitted parameter vectors and 37,584 individual PF evaluations. It includes the two-package paper experiment and six other packages, with their different scopes explicitly retained. Its final reports, full-range figures, optional 1.5-IQR display figures, old failed fits, low-precision scores and density-floor outcomes are preserved. PFJAX's corrected evaluation_v2 alone supplies its final scores; the superseded scores remain as history.

The new experiment has a different status:

| Package | New endpoints | State |
|---|---:|---|
| Cuthbert | 36/36 | Complete report and audit; two A fits failed, three A scores low precision |
| PFJAX | 36/36 | [Final report, figures and verified reproduction archive](pfjax_20261101/report.md); one failed A fit with floor score, 35 other scores meet precision; B is worse than A in case1 |
| PyDPF | 24/36 | [A/B completed and audited](pydpf_AB_20261102.md); all 24 fits complete, one low-ESS A score and one B floor score retained; C pending |
| TFP | 36/36 | [Final report, figures and verified reproduction archive](tfp_20261101/report.md); 35 complete fits, one failed A fit with floor score; 35 other scores meet precision |
| particles | 24/24 | [Final report, figures and full reproduction archive](particles_20261101/report.md) complete; three low-ESS scores retained |

For new particles, the median independent log likelihood is −4169.070668 for A (1800-second external Nelder-Mead) and −3784.099127 for B (900-second public IF2, then Nelder-Mead to the same total budget). B is higher in all twelve observed pairs. Precision passes are 9/12 for A and 12/12 for B; A tasks 0, 2 and 12 reached R=288 and remain low precision. There were no failed fits or density-floor scores. These findings apply to the recorded starts and budgets, not universal optimizer superiority.

The new particles fits used 43,647 CPU seconds, plus 155 CPU seconds for the earlier checks. Independent A40 scoring used 96 seconds and is counted separately. [Audit](../results/particles/audit.json) · [24 endpoint records](../results/particles/endpoints.csv).

[The complete Cuthbert report](cuthbert_20261101/report.md) includes all 36 retained endpoints and 2,052 original PF values. The all-endpoint median log mean likelihoods are −3814.966803 (A), −3747.557538 (B) and −3744.501336 (C). C exceeds B in all twelve paired cases, with median difference 2.351684. Two A fits stopped early and three A scores remain low precision; all B/C fits completed and their scores passed. These results apply to the recorded six starts and two replicates per start.

Median synchronized fitting-kernel times among complete fits are 555.756 seconds for A (10 fits), 627.436 for B and 627.586 for C (12 each). B/C include their IF2 prefix, while A does not; these are not equal total workloads. Failed A times are retained separately. The twelve allocations used 23,034 GPU seconds; the v3 check and repeat diagnosis bring formal-budget usage to 24,234 of 42,480 seconds. [All endpoint scores, failures and times](../results/cuthbert/endpoints.csv) and [complete audit](../results/cuthbert/final_audit_20261101.json) remain separate from the ordinary-PF timing.

Cuthbert's forward-PF agreement was accepted for proceeding with this empirical experiment. The [test-method review](reference_tests_20261101.md) and [original self-repeat diagnosis](cuthbert_repeats_20261101.md) explain why the old elementwise gradient threshold alone could not distinguish an extension error. This decision does not certify derivative correctness or erase the [earlier v3 failures](cuthbert_v3_20261101.md).

The [complete PFJAX report](pfjax_20261101/report.md) retains all 36 endpoints and 1,548 original PF values. A/B/C all-endpoint median log mean likelihoods are −3793.495698, −3748.631853 and −3745.089672. C exceeds B in all twelve observed pairs (median difference 3.762021); B exceeds A in eleven. A case8 failed after 116 updates and scored at the density floor, which is retained separately from the 35 nonfloor scores meeting precision. Complete-fit median fitting-kernel times are 529.630 seconds for A (11 fits), 569.591 for B and 569.732 for C (12 each). B/C include the full IF2 prefix. Formal allocations used 20,481 GPU seconds; with 841 seconds of checks charged to this experiment, usage is 21,322 of 42,480 seconds.

The [complete TFP report](tfp_20261101/report.md) retains 36 endpoints and 1,332 original PF values. A/B/C median log mean likelihoods over all twelve endpoints are −3809.127311, −3746.926753 and −3744.294332. C exceeds B in all twelve observed pairs, with median difference 2.391187. A case6 failed after 64 accepted updates: its loss was finite and 23 gradients were NaN; its independent score is at the density floor. The remaining 35 fits completed and their scores meet precision. These are results for six starts and two repetitions, not universal superiority.

TFP fitting uses entire B12 batches. Gradient kernel seconds are 10,348.495 / 9,229.294 / 10,429.009 for A/B/C; B/C logical costs including the shared IF2 prefix are 21,791.111 / 22,990.826 seconds. They are not per-endpoint times and are not multiplied by twelve. Formal allocations cost 43,479 GPU seconds within 72,000; the prior 2,336 validation seconds are separate. The physical prefix is counted once, and the failed A member is not successful acceleration.
