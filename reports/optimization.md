# Likelihood optimization

The [Dacca protocol](../experiments/dacca/README.md) defines A, B and C, their initialization, update counts and independent evaluation.

The [earlier completed comparison](../archive/README.md) contains 968 fitted parameter vectors and 37,584 individual PF evaluations. It includes the two-package paper experiment and six other packages, with their different scopes explicitly retained. Its final reports, full-range figures, optional 1.5-IQR display figures, old failed fits, low-precision scores and density-floor outcomes are preserved. PFJAX's corrected evaluation_v2 alone supplies its final scores; the superseded scores remain as history.

The new experiment has a different status:

| Package | New endpoints | State |
|---|---:|---|
| Cuthbert | 0/36 | First complete A/B/C case submitted as 65158659; remaining 11 cases follow after actual cost review. Package and fitting code remain v3 |
| PFJAX | 0/36 | Dependent formal job cancelled before starting after GPU validation failure |
| PyDPF | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| TFP | 0/36 | Dependent formal jobs cancelled before starting after GPU validation failure |
| particles | 24/24 | Fits and independent scores audited; final new figures and reproducibility archive still pending |

For new particles, the median independent log likelihood is −4169.070668 for A (1800-second external Nelder-Mead) and −3784.099127 for B (900-second public IF2, then Nelder-Mead to the same total budget). B is higher in all twelve observed pairs. Precision passes are 9/12 for A and 12/12 for B; A tasks 0, 2 and 12 reached R=288 and remain low precision. There were no failed fits or density-floor scores. These findings apply to the recorded starts and budgets, not universal optimizer superiority.

The new particles fits used 43,647 CPU seconds, plus 155 CPU seconds for the earlier checks. Independent A40 scoring used 96 seconds and is counted separately. [Audit](../results/particles/audit.json) · [24 endpoint records](../results/particles/endpoints.csv).

Cuthbert's forward-PF agreement is accepted for proceeding with the empirical experiment; the [gradient-test review](gradient_tests_20261101.md) explains why its uncalibrated elementwise threshold is not a sufficient standalone reason to stop. Derivative correctness is not being certified. In v3, measured IF2 and gradient calls gave a conservative estimate of 3419.47 seconds per case, above the existing 3120-second submission limit. This is an estimate, not an observed complete optimization time. The revised submission protocol allocates 57minutes per case within the existing total budget and starts only case0 first. The forecast is unchanged; its former additional 240-second reserve is removed explicitly. All scientific iterations, original failures and the v3 fitting source are retained. The completed repeated-gradient diagnosis used 321 seconds, leaving a combined reservation of 42,240 seconds for the v3 check, diagnosis and twelve 57-minute cases, within the 42,480-second allowance. See the [test-method review](reference_tests_20261101.md), [acceptance and cost record](../results/cuthbert/acceptance_v4.json), and [actual submission](../results/cuthbert/submission_v4.json). [v3 results and failures](cuthbert_v3_20261101.md).
