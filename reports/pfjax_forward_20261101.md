# PFJAX: full Dacca forward comparison

The full-scale check **65168298 failed** its existing forward-likelihood rule in four of twelve case identities. It ran on A40 node `gl1527` for 194 Slurm seconds. The complete result was collected once, with nine source files, 394 referenced files, 35 transfer hashes and 29 terminal hashes verified. [The saved audit and values](../results/pfjax/forward_checks_20261101.json) retain the failure. No PFJAX formal A/B/C fit has been submitted.

## Likelihood values

These are validation calls at the six original starts, each with two keys, using T=600, 20 Euler steps per month and J=5000. They are not fitted endpoints. The original PF was called with `jit` for likelihood only. The historical corrected wrapper and the new public interface were called with `jit(value_and_grad(...))`. This difference in evaluation context is the subject of the next diagnostic.

| Case | Original PF, likelihood only | Public PF, value returned with gradient | Public minus original | Existing rule |
|---:|---:|---:|---:|---|
| 0 | -6978.252441 | -6966.536133 | +11.716309 | Fail |
| 1 | -6942.188477 | -6923.374023 | +18.814453 | Fail |
| 2 | -5883.137207 | -5883.137207 | +0.000000 | Pass |
| 3 | -5883.790039 | -5883.790039 | +0.000000 | Pass |
| 4 | -10074.389648 | -10074.389648 | +0.000000 | Pass |
| 5 | -9506.627930 | -9506.627930 | +0.000000 | Pass |
| 6 | -7989.441406 | -7989.441406 | +0.000000 | Pass |
| 7 | -8148.718262 | -8148.717773 | +0.000488 | Pass |
| 8 | -9237.376953 | -9237.375977 | +0.000977 | Pass |
| 9 | -9086.968750 | -9086.968750 | +0.000000 | Pass |
| 10 | -7920.050781 | -7895.887695 | +24.163086 | Fail |
| 11 | -7896.213867 | -8008.529297 | -112.315430 | Fail |

The existing rule is absolute difference at most 0.002, with zero relative tolerance. It has not been changed. Both public alpha values, 1 and 0.97, return exactly the historical corrected value in all twelve cases. Across the two public alpha values, 16/24 comparisons with the original likelihood-only call pass. The largest absolute difference is **112.315430**, which cannot be dismissed as a last-digit difference. Agreement with the historical wrapper does not establish agreement with the original forward calculation.

All recorded values and gradients are finite. The earlier componentwise alpha-1 gradient rule passes in 9/12 pairs with the historical wrapper; that rule remains diagnostic rather than the reason for withholding formal fitting. The historical correction and both public alpha paths are our added code, not an original PFJAX ADPF.

## Public-call cost and memory

All five public IF2 calls retained a finite 5,000 by 23 parameter population. All eight Adam proposals had finite loss, gradients, estimation and natural parameters, and moments. The slowest synchronized IF2 call took 0.233664 seconds; the slowest Adam proposal took 1.330333 seconds. The existing conservative forecast for 175 IF2 passes and three sets of 400 Adam updates is **2,810.537 seconds**, below the 3,120-second planning threshold for a 56-minute allocation. The largest compiler memory estimate was 284,374,804 bytes, below 85% of the reported device limit. These are check-call measurements, not formal optimization times or a repeated native-PF benchmark.

## New diagnostic

Job **65168970** was submitted once with a 12-minute A40 limit. It was pending for priority at the saved post-submission query. It compares the unmodified original PF and public alpha 0, 1 and 0.97, each evaluated both as likelihood only and with `value_and_grad`. Cases 0 and 11 represent the two starts with failures; case 2 is a passing control. Every path has three repeats. A second wrapper disables checkpointing for diagnosis; the production source and checkpoint setting remain unchanged.

This will test whether differentiation changes the original PF's forward result before attributing the discrepancy to the extension. The cause is currently unresolved. The CPU preflight used eight particles, retained all 144 evaluations, and reproduced 1,656 forward comparisons exactly. That is not a GPU result. Gradients at different alpha values are expected to differ and are not counted as equivalence tests.

The original PFJAX formal allowance is 42,480 GPU seconds. The prior diagnosis used 293 seconds and this check used 194. With 720 seconds reserved for the new diagnostic and twelve later cases at 3,360 seconds, the upper reservation is 41,527 seconds. No iteration count, particle count or start was reduced.

## Records and reproduction

The [dated archive](../artifacts/validation/pfjax_forward_checks_20261101T0856.tar.gz) contains all 666 files: unchanged source, full stopped-check arrays and states, failed attempts, the scientific audit, and the new diagnostic source and CPU records. All members were verified; extracting it and rerunning the saved-data GPU audit and CPU comparison analysis reproduced their JSON files byte for byte. These audits make no PF call. The archive SHA256 is `15886a4866e19def899c2d7dada5eaa3522b2ef7b91c04b9647e6c7d68f706a9`. No result from the pending diagnostic is included. Earlier archives and failures remain unchanged.
