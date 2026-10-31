# IFAD

`fit` composes `iterated_filtering.fit` and `dmop.fit`, then evaluates the final
parameters with independent original bootstrap PF draws. All three calls use
the same model factory. IF2 supplies the starting parameters for a newly
initialized optimizer in the DMOP stage; the stages have separate random keys.

This module does not implement another particle filter. It delegates fitting
to the two modules above and scoring to `particle_filter.build_filter` with an
ordinary resampler. Parameter transformations stay in the model factory;
perturbation scales and optimizer settings are supplied through the API.

::: cuthbert.smc.ifad
