# How each package is used

Source review, 31 October 2026. Cuthbert uses its original PF update functions through a loop we added. PFJAX, PyDPF, TFP and particles call the full original PF or SMC loop in their new interfaces. The table below identifies the calls and our changes.

Earlier PFJAX IF2 used a filtering loop we wrote ourselves. Its new interface uses the package PF. These versions must be distinguished when interpreting results. [Chinese notes](report_CN.md) are also available.

## Current implementations

| Package | Call chain | Work done by the original package | Code we added or changed |
|---|---|---|---|
| Cuthbert v3 | New `iterated_filtering.fit` / `dmop.log_likelihood` → original `particle_filter.build_filter` → added `scan_final` → original `filter_prepare` / `filter_combine` | Per-step resampling, state propagation, observation weights and likelihood updates; original `stop_gradient_decorator` at α=1 | Joint state and parameter storage, perturbations, cooling, outer IF2 iterations and final parameter population; a `jax.lax.scan` time loop retaining only the final state; gradient discounting for α<1 |
| PFJAX v2 | New `iterated_filter_step` / `particle_filter_mop` → original `particle_filter` | Complete PF time loop, original `filter_step` and original multinomial resampling | An `initial_particles` argument; parameterized model wrappers, perturbations and outer IF2 iterations; a resampling gradient correction, including at α=1 |
| PyDPF v1 | New `IteratedFiltering` → original `ParticleFilter` → original `SIS.forward`; the gradient stage calls the extended `StopGradientDPF` | Complete time loop, resampling, weights and likelihood; the original `StopGradientDPF` already provides the α=1 gradient treatment | Parameterized model wrappers, perturbations, cooling and repeated filtering; an `alpha` argument in `StopGradientDPF` / `StopGradientResampler` |
| TFP v1 | New `iterated_filter_step` → original `sequential_monte_carlo`; the gradient stage uses the extended original `particle_filter` → SMC | Complete filtering loop, `SequentialMonteCarlo.one_step`, original resampling and existing unbiased-gradient treatment | Joint state and parameter propagation, perturbations and repeated filtering; propagation of `gradient_discount` and modification of weight derivatives; random streams assigned by start and replicate |
| particles v1 | New `if2_step` → original `SMC.run`; model wrappers call original `Bootstrap.M0/M/logG` | Complete time loop, resampling, weight normalization and accumulation of `logLt` | Parameters carried with states, perturbations, cooling and outer IF2 iterations; external SciPy Nelder-Mead in the experiments |

The added functions are in local candidate packages. This does not mean they were present upstream or have been accepted by package authors. Dacca state simulation and observation densities are supplied through model callbacks. Writing these callbacks is distinct from writing the PF itself.

## Details by package

Cuthbert. The original `cuthbert/smc/particle_filter.py`, multinomial and systematic resampling files are unchanged byte for byte. The original `stop_gradient_decorator` function is also unchanged. The new interfaces do not directly call the top-level `cuthbert.filter`: `scan_final` traverses time and calls the original update functions. The implementation adds IF2 and gradient discounting to those original update functions. The A/B/C experiment driver calls the new `iterated_filtering.fit`, `dmop.log_likelihood` and external Optax; it does not call the complete `ifad.fit` interface.

PFJAX. The `month` function and `jax.lax.scan(month, ...)` in the earlier `formal_v1/bridge.py:if2` were locally written, reusing only the original resampling function. That implementation does not satisfy the criterion of reusing the complete PF loop, and its earlier results must be interpreted accordingly. The new interface calls the full original `particle_filter`; its nested `filter_step` is unchanged byte for byte, while the file adds an initial-particle argument. The new driver takes the Dacca model class from the earlier bridge but does not call `bridge.if2` or `bridge.objective`. In both the earlier and new gradient stages, the α=1 correction was added locally. It must not be called PFJAX's native ADPF or native optimal-transport method.

PyDPF. The `SIS` and `ParticleFilter` classes are unchanged byte for byte. The new driver calls the added `IteratedFiltering` through `api_adapter.warm_if2` and calls `ParticleFilter` / `StopGradientDPF` through `api_adapter.factors`. The presence of earlier experiment functions in a historical model file does not mean the new driver calls them. Parameter optimization uses external `torch.optim.Adam`.

TFP. IF2 calls the complete lower-level SMC loop, rather than the top-level `particle_filter`; the gradient stage calls the latter. The wrapper assigning random streams to starts and replicates still calls the original `tfp.experimental.mcmc.resample_systematic`. It does not implement its own ancestor-sampling algorithm. `gradient_discount` modifies the existing gradient-treatment branch; the original resampling file is unchanged. Parameter optimization uses external Keras Adam.

particles. The original `core.py`, `state_space_models.py` and `resampling.py` are unchanged byte for byte. The added IF2 wrappers call the original `SMC` and `Bootstrap`. Nelder-Mead comes from SciPy and must not be called a native particles MLE algorithm. This experiment does not compare differentiable PFs.

## Reference and earlier CPU comparison

PyPOMP supplies the reference implementation. The repository's `functional/pfilter.py`, `mif.py`, `train.py` and `mop.py` were compared with fixed commit `17f8798e7d944977335c81b8525353fa90e14be9`; all four files match exactly. The earlier paper experiment uses the package's IF2 and training functions, and formal endpoint evaluation uses its frozen original PF.

pypfilt. The earlier experiment calls `pypfilt.fit`, which calls the original `pfilter.run`. We supplied Dacca model callbacks, likelihood-increment recording and external SciPy Nelder-Mead. We did not replace its PF time loop or provide a new IF2 or IFAD interface for this package. The original `fit`, PF and resampling sources were copied from the existing reproduction archive for inspection, with archive-member hashes recorded.

## Files checked

- [Source locations](source_locations.md) link to the inspected functions and line numbers in both the added interfaces and experiment drivers.
- [Source-check results](audit.json) records 15 exact-text comparisons of original code, 25 call locations and file SHA256 values.
- [Additional source records](evidence_sources.json) identify the earlier PFJAX sources, TFP and particles drivers, pypfilt archive members and PyPOMP commit.
- Run `python3 scripts/audit_native_pf.py` to repeat the source checks. Run `python3 scripts/verify.py` to verify that the complete patches reproduce the candidate sources.

These checks establish which source functions are used and which are unchanged. GPU validation failures remain unresolved, as recorded in [results and tests](../status.md). Initial perturbations, first-interval treatment and cooling differ between packages; the implementations do not execute identical IF2 algorithms across packages.
