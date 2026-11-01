# Repeated Cuthbert GPU gradients

Job 65156615 completed on an A40 at gl1524 in 321 Slurm seconds. All 54 planned evaluations were saved and are finite. Every corresponding forward log likelihood is identical. Gradients vary between repeated calls, including repeated calls to the unchanged original ADPF. At J=5000, seven of nine original-versus-original pairs fail the earlier componentwise rule (`rtol=1e-5`, `atol=1e-4`).

The diagnosis uses Dacca with 600 months, 20 Euler steps per month, cases 0, 4 and 8, J=32 and J=5000, and three repeats per function. Each case uses the same parameters and key across the three functions:

- The original `particle_filter.build_filter`, original stop-gradient resampling correction and full `cuthbert.filter` time loop.
- The public v3 `dmop.log_likelihood` with alpha=1 and checkpoint disabled.
- A diagnostic wrapper with the public parameter/input conversion and alpha=1 correction, but the full original `cuthbert.filter` time loop.

The third function tests whether restoring the original time loop removes the observed difference. It does not remove repeated-gradient variation. It is a diagnostic control; the production source was not changed.

| J | Case | Original self pairs satisfying old rule | Original / public v3 pairs satisfying old rule | Maximum original self difference | Maximum original / public difference |
|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 3/3 | 9/9 | 0.0009765625 | 0.00146484375 |
| 32 | 4 | 3/3 | 3/9 | 0.00048828125 | 0.0009765625 |
| 32 | 8 | 3/3 | 9/9 | 0.00146484375 | 0.00146484375 |
| 5000 | 0 | 2/3 | 5/9 | 0.021484375 | 0.03076171875 |
| 5000 | 4 | 0/3 | 0/9 | 3.767578125 | 3.63671875 |
| 5000 | 8 | 0/3 | 0/9 | 2.73828125 | 3.1455078125 |

The differences are absolute differences between unnormalized log-likelihood gradients, not likelihood differences or relative errors. Pair counts reuse the three evaluations per function and are not independent replicates. The audit recomputed all 48 comparisons saved by the job, then examined all 162 within-function and original/control cross pairs. Every pair has exactly equal forward values; none has bitwise-identical gradients. [Full summary](../results/cuthbert/repeated_gradients_20261101.json).

This directly demonstrates that the earlier gradient threshold rejects repeated calculations by the original implementation. A failure under that rule cannot, on its own, establish a defect in the added interface. The source of GPU variation has not been isolated, and this diagnosis does not prove the mathematical correctness of every derivative. Its seed is different from the earlier twelve-start validation, so it supplements those records rather than replacing them.

The empirical A/B/C comparison continues under the user's recorded acceptance of forward-likelihood agreement. No tolerance or old test outcome was rewritten. Case 0 was already submitted as 65158659; the other eleven cases still await its measured cost and endpoint audit. This diagnosis does not authorize a duplicate case or any refit.

The source audit verified 17 diagnostic files and 571 referenced files; transfer and terminal checks covered 19 and 13 files respectively. The diagnostic program reports 108.01 seconds internally; the full Slurm cost is 321 seconds, which includes work outside that timer. Cuthbert's revised budget reservation is now 879 seconds for v3 checks + 321 for this diagnosis + 12 × 3420 for formal cases = 42,240 seconds, within 42,480 seconds (11.8 GPU hours). The separate earlier validation cost remains 2677 seconds.

Raw gradients, printed JAX expressions, logs, all failed comparisons, source records and the audit program are archived in `artifacts/validation/cuthbert_repeats_20261101.tar.gz`, listed with its SHA256 in the [catalog](../archive/catalog.json). Extract into a new directory and run `audit_cuthbert_gradient_diagnosis.py` with NumPy installed to reproduce the audit without a GPU. The referenced v3 sources remain in the earlier archived source snapshots.
