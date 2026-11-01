# Iterated filtering

`fit` adds Gaussian parameter perturbations and repeated passes through the
data to the existing bootstrap PF. A model factory returns the same initial,
propagation and observation callbacks accepted by `particle_filter.build_filter`.
The factory applies any parameter transformations; it does not implement IF2.

The module wraps those callbacks so that each particle contains both the model
state and its parameters. It calls the original PF update with these augmented
particles. State and parameters are resampled together. The complete terminal
parameter swarm, rather than just its mean, is passed to the next iteration.
Initial and within-pass perturbation scales can be specified separately.

The implementation uses the original `filter_prepare`, `filter_combine` and
systematic or multinomial resampler. Its time loop stores only the terminal
filter state. The model callbacks do not need their own perturbation logic.
This interface currently supports unconditional bootstrap resampling and a
geometric cooling factor between passes; other schedules are not implied.

::: cuthbert.smc.iterated_filtering
