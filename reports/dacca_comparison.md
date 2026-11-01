# Particle filtering and likelihood optimization for Dacca

We compare particle-filter implementations and examine whether IF2 initialization and gradient discounting improve likelihood optimization within each package. The code extends Cuthbert, PFJAX, PyDPF, TensorFlow Probability and particles. PyPOMP supplies the fixed PF used for independent evaluation.

The new GPU optimization experiments are unfinished. [Experiment status](status.md) records their progress. Results are reported separately as [correctness](correctness.md), [ordinary PF runtime](timing.md) and [likelihood optimization](optimization.md).

## Model

The Dacca cholera model has 600 monthly observations. Each month is simulated with 20 Euler steps and Gaussian process noise. The new experiments use 5,000 particles and six common starting parameter vectors, each repeated twice.

The implementation updates 23 coordinates: 18 process or observation coordinates and five initial-value coordinates that are normalized. The latter representation is redundant, so 23 is not a count of independently identifiable parameters. Comparing this with the 15-parameter description used elsewhere requires reconciling the parameterization and fixed quantities.

## Package changes

The [function table](../docs/package_calls.md) identifies the original package calls, added code and external optimizers. Each [package directory](../packages/) contains the changed files and a complete diff against the recorded original version.

Cuthbert uses its original PF updates and its original α=1 derivative correction. We added IF2, gradient discounting and a loop over observation times. The other four new implementations call the original full PF or SMC loop. PFJAX's α=1 resampling derivative correction is our addition. Adam comes from Optax, PyTorch or Keras; the particles experiment uses external SciPy Nelder-Mead. Dacca model callbacks are separate from the changes to filtering algorithms.

## Optimization design

The four GPU packages use the following design:

| Method | Starting parameters | Gradient optimization |
|---|---|---|
| A | Original starting vector | α=1, Adam, 400 updates |
| B | Result of 175 IF2 passes | α=1, Adam, 400 updates |
| C | Same IF2 result as B | α=0.97, Adam, 400 updates |

B and C share starting parameters, zero initial Adam state, learning rates and training random streams. Their comparison examines discounting after the same initialization. A and B differ in their use of IF2 and total computation. The IF2 implementations also differ between packages, so the optimization comparisons are made within each package.

The shared IF2 run is counted once in total computation. Each method's separately reported cost includes the full initialization it needs. Compilation, fitting and scoring times are reported separately. Time for a batch is counted once.

particles uses a CPU experiment: A runs external Nelder-Mead for 1,800 seconds; B uses IF2 for the first 900 seconds and Nelder-Mead for the remainder. Independent GPU evaluation is charged separately. The [protocols](../experiments/dacca/README.md) give the settings and programs.

## Evaluation

Every fitted parameter vector is evaluated by the fixed original PyPOMP PF, with random streams separate from training. The reported log likelihood is the logarithm of the mean likelihood across PF runs. Evaluation starts with 36 repeats and increases to 72, 144 or 288 when needed. The precision requirement is MCSE at most 1 and likelihood ESS at least 10.

Raw repeats, failed fits, low-precision scores and observation-density-floor scores are retained. A small MCSE at the density floor does not indicate a good fit. Training losses and perturbed IF2 likelihoods are not used as final scores.

## Records and reproduction

[Earlier completed experiments](../archive/README.md) retain their own source, protocols, figures and failures. They include PyPOMP and pypfilt comparisons. Dynamax's self-written PF experiment is historical only. Earlier results do not validate the new interfaces described here.

[Reproduction](../docs/reproduction.md) gives commands to restore source, apply diffs and rebuild tables. The repository also records the [examples and meeting decisions](../docs/conventions.md) used to organize the code and reports. The changes remain local and have not been submitted upstream.
