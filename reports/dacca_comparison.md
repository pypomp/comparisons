# Particle filtering and likelihood optimization for Dacca

This report compares package implementations of particle filtering and examines whether IF2 initialization and gradient discounting improve likelihood optimization within a package. It accompanies local changes to Cuthbert, PFJAX, PyDPF, TensorFlow Probability and particles. PyPOMP supplies the independent PF used to evaluate fitted parameters. Earlier pypfilt results are also retained.

The new experiments are incomplete. This report describes the code, settings and results present in the repository as assembled on 31 October 2026. [Results and tests](status.md) gives the dates and numerical details. The [meeting notes](../docs/meeting_decisions.md) explain the questions that motivated the work.

## Model and evaluation

The Dacca cholera model has 600 monthly observations. We simulate each month with 20 Euler steps and Gaussian process noise. The new experiments use 5,000 particles and six common starting parameter vectors, each repeated twice.

The implementation updates 23 coordinates: 18 process or observation coordinates and five initial-value coordinates that are normalized. The latter representation is redundant, so 23 is not a count of independently identifiable parameters. A comparison with the 15-parameter description used elsewhere requires reconciling the model parameterization and which quantities are fixed.

Every fitted parameter vector is evaluated with the same fixed, original PyPOMP PF, using random streams separate from training. The reported log likelihood is the logarithm of the mean likelihood across repeated PF runs. Evaluation starts with 36 repeats and increases to 72, 144 or 288 when needed. The required precision is MCSE at most 1 and likelihood ESS at least 10. Raw repeats, failed fits, scores below these thresholds and scores at the observation-density floor are retained. A small MCSE at that floor does not indicate a good fit.

## Package code and our changes

The [function table](../docs/package_calls.md) identifies the original package calls, added code and external optimizers. The [source review](native_pf_audit_20261031/report.md) links those descriptions to source lines.

Cuthbert uses its original PF update functions and its original α=1 ADPF correction. We added IF2, gradient discounting and a loop over observation times. The other four new implementations call the original full PF or SMC loop. PFJAX's α=1 resampling derivative correction is an addition we wrote. It is therefore not a PFJAX-provided ADPF baseline. Adam comes from Optax, PyTorch or Keras; particles and pypfilt use external SciPy Nelder-Mead.

Each modified package has an original source directory, a modified source directory, a complete diff and test records. These are local changes awaiting review by package authors. Writing Dacca model callbacks is separate from changing a particle filter.

## Correctness

We compare original and modified filters, test the added functions, and record failures from both versions. Cuthbert's revised CPU checks passed. Its earlier GPU test run timed out with three failures; the subsequent original-code comparisons and revised GPU tests had not finished in the included scheduler records. PFJAX, PyDPF and TFP have unresolved GPU gradient or equality checks. The [test table](status.md) gives the exact status, including incomplete test coverage.

Matching fitted likelihoods are useful as a check, but do not replace tests of filtering, perturbations and derivatives. Earlier completed experiments also cannot validate a later code revision.

## Ordinary PF runtime

For the original Dacca model at T=600 and J=5000, seven timed PF calls gave median times of 0.393526 seconds for Cuthbert, 0.220416 for PFJAX, 8.57610 for PyDPF and 6.278312 for TFP on A40 GPUs, with one filter per call. The particles CPU median was 5.496709 seconds. Compilation and warmup were recorded separately. Environments and measurement details are linked from [the runtime table](status.md).

These measurements describe PF calls. Optimization also includes gradients, parameter updates and, for some methods, IF2 initialization. Its total runtime must be reported separately. The CPU measurement is not part of a GPU speed ranking.

## Optimization experiments

The four GPU packages use the following design. The exact settings were chosen after the meeting.

| Method | Starting parameters | Gradient optimization |
|---|---|---|
| A | Original starting vector | α=1, Adam, 400 updates |
| B | Result of 175 IF2 passes | α=1, Adam, 400 updates |
| C | Same IF2 result as B | α=0.97, Adam, 400 updates |

B and C share the starting vector, zero initial Adam state, learning rates and training random stream. Comparing them examines discounting after the same initialization. A and B differ in their use of IF2 and total computation. IF2 details differ between packages; this is a within-package comparison, not a claim that all implementations execute identical algorithms. The shared IF2 run is counted once in the total computation and included in each method's reported cost when methods are described separately.

There are no new GPU optimization results in the included records. For particles, A uses 1,800 seconds of external Nelder-Mead; B uses IF2 for the first 900 seconds and Nelder-Mead for the remainder. Median independent log likelihoods are −4169.070668 for A and −3784.099127 for B. These medians use all 12 estimates per method, including the three imprecise A scores. B is higher in all 12 observed pairs. Nine A scores and all 12 B scores meet the precision requirement. A tasks 0, 2 and 12 remain below it at 288 repeats.

The particles runs used 43,647 CPU seconds for fitting and 155 CPU seconds for earlier checks. Independent GPU scoring used another 96 seconds. [Parameter estimates](../experiments/public_interfaces/snapshot_20261031/recovery_20261031/particles/endpoints.csv) and [the result checks](../experiments/public_interfaces/snapshot_20261031/recovery_20261031/particles/audit.json) retain every case. Final figures and the reproducibility archive for this new comparison are pending.

[Earlier Dacca experiments](legacy_dacca/report_CN.txt) are complete and use their own saved code and protocols. Their figures and records remain available for reference. They are separate from the new experiments described here. Dynamax's earlier self-written PF comparison is retained only in that history.

## Code and reproduction

The [reproduction instructions](../docs/reproduction.md) describe how to inspect diffs, verify files and extract raw results. Numerical reruns need the recorded environments; some saved drivers still contain their original workspace paths. Large raw archives are stored locally outside Git. Before sharing the repository, those archives and the remaining environment and path instructions must accompany the report.
