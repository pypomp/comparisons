# Original functions, additions and optimizers

"Original" means the package version recorded for this study. The table separates its functions from our model callbacks, IF2 additions and external optimizers. An experiment can call the original filter while using an optimization interface that we wrote.

The [upstream contribution review](upstream_review.md) checks patch application, installation, examples and the remaining changes needed for independent forks.

| Package and current A/B gradient | Original calls | Local additions | External optimizer |
|---|---|---|---|
| [Cuthbert](../packages/cuthbert/README.md): native stop-gradient correction | `particle_filter.build_filter`, `filter_prepare`, `filter_combine`, resampling and `stop_gradient_decorator` at α=1 | Parameter/state augmentation, perturbation/cooling and IF2; `scan_final` observation loop; `dmop.log_likelihood` and α<1 discounting | Optax `scale_by_adam`; the driver does not call the complete `ifad.fit` interface |
| [PFJAX native supplement](../experiments/dacca/pfjax_native_20261102/README.md): basic-PF direct autodiff | Unchanged full `particle_filter`, `resample_multinomial`, JAX autodiff | Dacca callbacks and external training driver; B reuses the earlier IF2 endpoint | Optax Adam; basic-PF gradient is biased, not the package's optimal-transport estimator |
| [PyDPF](../packages/pydpf/README.md): `StopGradientDPF` | `ParticleFilter`, `SIS.forward`, `FilteringModel`, resampling/likelihood and original stop-gradient correction | `IteratedFiltering`, state/parameter wrappers, perturbation/cooling and full swarm retention; an `alpha` argument | `torch.optim.Adam`; not an upstream MLE interface |
| [TFP](../packages/tfp/README.md): differentiable `particle_filter` | `sequential_monte_carlo`, `SequentialMonteCarlo.one_step`, `WeightedParticles`, resampling and `particle_filter(unbiased_gradients=True)` | `iterated_filter` parameter/model wrappers and `gradient_discount`; α=1 retains the original derivative expression | Keras Adam |
| [particles](../packages/particles/README.md): no gradient arm | `SMC.run`, `Bootstrap`, resampling and `logLt` | `iterated_filtering`, `_PerturbedBootstrap`, joint state/parameter resampling and perturbation/cooling; existing Python files unchanged | SciPy Nelder-Mead; B adds IF2 |

Each gradient method has twelve runs. C starts from the same parameters as B and uses our discounted gradient with α=0.97. We added IF2 to each package. See the [design](../reports/dacca_comparison.md) and [final results](../reports/optimization.md) for the settings and limitations.

## Earlier PFJAX extension

The [PFJAX package diff](../packages/pfjax/review/full.patch) adds `initial_particles` for IF2 and `particle_filter_mop` for resampling derivative correction, including α=1. That extension's original A/B results are not native PFJAX A/B. They remain in [their report](../reports/pfjax_20261101/report.md) and the [five-method comparison](../reports/optimization.md#pfjax-native-baseline-and-the-earlier-extension).

Native B and C differ in both gradient calculation and α. Comparing our α=1 B with C isolates the discount within our added correction. We report native A/B separately from A/B using our correction.

## Model callbacks and later performance work

We wrote the Dacca model callbacks separately from the filter changes. The [later PyDPF forward/backward optimization](../reports/pydpf_backward_fusion_20261102.md) changes these callbacks and leaves the package's filter unchanged. We compared individual calls with the old and new callbacks, without rerunning the completed fits. The maze study uses a different model and objective.

Each package README links the version, changed files, complete diff and test programs. [Experiment source locations](../experiments/dacca/README.md) identify the code copies retained here and the archives containing the full code for each completed experiment. The changes are local and have not been accepted upstream. [Correctness](../reports/correctness.md) describes the limitations and failed checks.

## Evaluation and earlier experiments

Final endpoint evaluation uses the unchanged original [PyPOMP scorer](../experiments/dacca/score.py), separate from training. It computes log(mean likelihood), retains all raw values and applies the recorded MCSE/ESS checks. Perturbed IF2 likelihoods are not final scores.

Earlier PyPOMP experiments used its `functional.mif` / `functional.train` and Adam. Earlier pypfilt experiments used `pypfilt.fit` for filtering with external Nelder-Mead. That interface did not provide native MLE, IF2 or IFAD. These experiments and the earlier Dynamax records remain in [the archive](../archive/README.md). Their results do not validate the package functions added later.
