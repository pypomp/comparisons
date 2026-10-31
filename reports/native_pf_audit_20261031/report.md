# Native particle filter reuse: source audit

We re-examined the candidate sources, corresponding original sources, complete patches and experiment drivers. **All five new interfaces extend package PF code, but at different levels.** Cuthbert reuses the original per-step update functions. The other four reuse the complete filtering loop. We found no other new interface with the same arrangement as the excluded Dynamax experiment, where the Dacca PF was locally written and the package mainly supplied an optimization interface.

Two distinctions must remain explicit: earlier PFJAX IF2 used a local PF time loop, and current Cuthbert still uses an added time loop. Neither should be described as calling the complete original PF driver. [Chinese supplementary notes](report_CN.md) are preserved.

## Current implementations

| Package | Call chain | Work done by the original package | Local additions or modifications |
|---|---|---|---|
| Cuthbert v3 | New `iterated_filtering.fit` / `dmop.log_likelihood` → original `particle_filter.build_filter` → new `scan_final` → original `filter_prepare` / `filter_combine` | Per-step resampling, state propagation, observation weights and likelihood updates; original `stop_gradient_decorator` at α=1 | Joint state and parameter storage, perturbations, cooling, outer IF2 iterations and terminal parameter population; a `jax.lax.scan` time loop retaining only the final state; gradient discounting for α<1 |
| PFJAX v2 | New `iterated_filter_step` / `particle_filter_mop` → original `particle_filter` | Complete PF time loop, original `filter_step` and original multinomial resampling | An `initial_particles` argument; parameterized model wrappers, perturbations and outer IF2 iterations; a resampling gradient correction, including at α=1 |
| PyDPF v1 | New `IteratedFiltering` → original `ParticleFilter` → original `SIS.forward`; the gradient stage calls the extended `StopGradientDPF` | Complete time loop, resampling, weights and likelihood; the original `StopGradientDPF` already provides the α=1 gradient treatment | Parameterized model wrappers, perturbations, cooling and repeated filtering; an `alpha` argument in `StopGradientDPF` / `StopGradientResampler` |
| TFP v1 | New `iterated_filter_step` → original `sequential_monte_carlo`; the gradient stage uses the extended original `particle_filter` → SMC | Complete filtering loop, `SequentialMonteCarlo.one_step`, original resampling and existing unbiased-gradient treatment | Joint state and parameter propagation, perturbations and repeated filtering; propagation of `gradient_discount` and modification of weight derivatives; random streams assigned by start and replicate |
| particles v1 | New `if2_step` → original `SMC.run`; model wrappers call original `Bootstrap.M0/M/logG` | Complete time loop, resampling, weight normalization and accumulation of `logLt` | Parameters carried with states, perturbations, cooling and outer IF2 iterations; external SciPy Nelder–Mead in the experiments |

The added functions are in local candidate packages. This does not mean they were present upstream or have been accepted by package authors. Dacca state simulation and observation densities are supplied through model callbacks. Writing these callbacks is distinct from writing the PF itself.

## Implementation boundaries

**Cuthbert.** The original `cuthbert/smc/particle_filter.py`, multinomial and systematic resampling files are unchanged byte for byte. The original `stop_gradient_decorator` function is also unchanged. The new interfaces do not directly call the top-level `cuthbert.filter`: `scan_final` traverses time and calls the original update functions. The appropriate description is therefore “IF2 and gradient discounting built on Cuthbert's original PF update functions.” The formal A/B/C driver calls the new `iterated_filtering.fit`, `dmop.log_likelihood` and external Optax; it does not call the complete `ifad.fit` interface.

**PFJAX.** The `month` function and `jax.lax.scan(month, ...)` in the earlier `formal_v1/bridge.py:if2` were locally written, reusing only the original resampling function. That implementation does not satisfy the criterion of reusing the complete PF loop, and its earlier results must be interpreted accordingly. The new interface calls the full original `particle_filter`; its nested `filter_step` is unchanged byte for byte, while the file adds an initial-particle argument. The new driver takes the Dacca model class from the earlier bridge but does not call `bridge.if2` or `bridge.objective`. In both the earlier and new gradient stages, the α=1 correction was added locally. It must not be called PFJAX's native ADPF or native optimal-transport method.

**PyDPF.** The `SIS` and `ParticleFilter` classes are unchanged byte for byte. The new driver calls the added `IteratedFiltering` through `api_adapter.warm_if2` and calls `ParticleFilter` / `StopGradientDPF` through `api_adapter.factors`. The presence of earlier experiment functions in a historical model file does not mean the new driver calls them. Parameter optimization uses external `torch.optim.Adam`.

**TFP.** IF2 calls the complete lower-level SMC loop, rather than the top-level `particle_filter`; the gradient stage calls the latter. The wrapper assigning random streams to starts and replicates still calls the original `tfp.experimental.mcmc.resample_systematic`. It does not implement its own ancestor-sampling algorithm. `gradient_discount` modifies the existing gradient-treatment branch; the original resampling file is unchanged. Parameter optimization uses external Keras Adam.

**particles.** The original `core.py`, `state_space_models.py` and `resampling.py` are unchanged byte for byte. The added IF2 wrappers call the original `SMC` and `Bootstrap`. Nelder–Mead comes from SciPy and must not be called a native particles MLE algorithm. This experiment does not compare differentiable PFs.

## Reference and earlier CPU comparison

**PyPOMP** supplies the reference implementation. The repository's `functional/pfilter.py`, `mif.py`, `train.py` and `mop.py` were compared with fixed commit `17f8798e7d944977335c81b8525353fa90e14be9`; all four files match exactly. The earlier paper experiment uses the package's IF2 and training functions, and formal endpoint evaluation uses its frozen original PF.

**pypfilt.** The earlier experiment calls `pypfilt.fit`, which calls the original `pfilter.run`. We supplied Dacca model callbacks, likelihood-increment recording and external SciPy Nelder–Mead. We did not replace its PF time loop or provide a new IF2 or IFAD interface for this package. The original `fit`, PF and resampling sources were copied from the existing reproduction archive for inspection, with archive-member hashes recorded.

## Evidence and limitations

- [Source locations](source_locations.md) link to the inspected functions and line numbers in both the added interfaces and experiment drivers.
- [Machine-readable audit](audit.json) records 15 exact-text comparisons of original code, 25 call locations and file SHA256 values.
- [Supplementary source provenance](evidence_sources.json) identifies the earlier PFJAX sources, TFP and particles drivers, pypfilt archive members and PyPOMP commit.
- Run `python3 scripts/audit_native_pf.py` to repeat the source checks. Run `python3 scripts/verify.py` to verify that the complete patches reproduce the candidate sources.

This is a source and call-chain audit. No GPU tests, fits or scores were rerun. It does not resolve the existing GPU validation failures or establish optimization outcomes for the new interfaces. Correctness, native PF runtime and optimization outcomes remain separate. Initial perturbations, first-interval treatment and cooling also differ between packages, so these implementations must not be described as exactly the same algorithm across packages.
