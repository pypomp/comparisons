# Sequential Monte Carlo Methods

- [Particle filter](particle_filter.md) - flexible particle filter for general Feynman-Kac models.
- [Marginal particle filter](marginal_particle_filter.md) - $O(N^2)$ variant of the particle filter.
- [Backward sampler](backward_sampler.md) - flexible backward smoothing using approaches in [`cuthbertlib.smc.smoothing`](../../api_cuthbertlib/smc.md).
- [Discounted MOP](dmop.md) - surrogate gradients and optimization for smooth bootstrap models.
- [Iterated filtering](iterated_filtering.md) - parameter perturbations and cooling across repeated bootstrap filters.
- [IFAD](ifad.md) - IF2 warm start, DMOP refinement and independent endpoint evaluation.

The core atomic functions can be found in [`cuthbertlib.smc`](../../api_cuthbertlib/smc.md).

## Feynman-Kac Models

--8<-- "cuthbert/smc/README.md:feynman_kac"
