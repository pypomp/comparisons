# Parameter estimation with IFAD

IFAD uses iterated filtering (IF2) to find a starting point, then automatic
differentiation to refine it. This example fits a continuous linear Gaussian
model using discounted MOP (DMOP) derivatives. An exact Kalman likelihood gives
an independent check of the fitted endpoint.

For the method, see [Tan et al., Automatic Differentiation for Inference on
Partially Observed Markov Process Models](https://arxiv.org/abs/2407.03085v2).
The IF2 stage follows [Ionides et al. (2015)](https://doi.org/10.1073/pnas.1410597112).

The implementation supports bootstrap particle filters with resampling at every
observation. The model samplers must be pathwise differentiable for the DMOP
stage. Adaptive resampling, guided proposals and discrete transition-score
corrections are outside this example's scope.

Install the `examples` extra to obtain Optax. The library itself accepts an
Optax-compatible optimizer without requiring Optax as a core dependency.

## Model and parameters

We use a stationary AR(1) latent process with known persistence and observation
noise, and estimate its mean and process standard deviation. The estimation
scale is `(mean, log_process_sd)`; constraints are applied inside the model
factory. Its three callbacks use the existing particle-filter signatures.

```{.python #ifad-imports}
"""Use the existing PF callbacks for ADPF, IF2 and IFAD."""

import jax
import jax.numpy as jnp
import optax

from cuthbert import filter as run_filter
from cuthbert.smc import dmop, ifad, particle_filter
from cuthbertlib.resampling import autodiff, systematic

jax.config.update("jax_enable_x64", True)
RHO, OBS_SD = 0.65, 0.5


def model(parameters):
    """Construct a reparameterized bootstrap model."""
    mean, log_q = parameters
    q = jnp.exp(log_q)

    def init_sample(key):
        return mean + q / jnp.sqrt(1 - RHO**2) * jax.random.normal(key)

    def propagate_sample(key, state, model_inputs):
        return mean + RHO * (state - mean) + q * jax.random.normal(key)

    def log_potential(state_prev, state, model_inputs):
        z = (model_inputs - state) / OBS_SD
        return -0.5 * (z**2 + jnp.log(2 * jnp.pi * OBS_SD**2))

    return init_sample, propagate_sample, log_potential
```

The data, IF2, DMOP and endpoint scoring each use separate random streams.

```{.python #ifad-data}
def simulate(key, parameters, length):
    """Generate observations from the same model."""
    init, propagate, _ = model(parameters)
    init_key, steps_key = jax.random.split(key)

    def step(x, k):
        process_key, observation_key = jax.random.split(k)
        x = propagate(process_key, x, None)
        return x, x + OBS_SD * jax.random.normal(observation_key)

    return jax.lax.scan(
        step, init(init_key), jax.random.split(steps_key, length)
    )[1]


data_key, fit_key = jax.random.split(jax.random.key(2026))
truth = jnp.array([1.25, jnp.log(0.45)])
observations = simulate(data_key, truth, 60)
initial = jnp.array([-1.5, jnp.log(1.2)])
```

## Using the existing particle filter

The same model factory can be passed to the existing particle filter. No
parameter perturbations or gradient corrections belong in these model callbacks.
The original ADPF is obtained by wrapping the resampling function with the
existing `stop_gradient_decorator`:

```{.python #ifad-existing-pf}
def original_log_likelihood(parameters, key, resampler):
    """Evaluate the original PF using the same model callbacks."""
    init_sample, propagate_sample, log_potential = model(parameters)
    obj = particle_filter.build_filter(
        init_sample, propagate_sample, log_potential, 64, resampler
    )
    init_key, filter_key = jax.random.split(key)
    states = run_filter(
        obj, observations, obj.init_prepare(key=init_key), key=filter_key
    )
    return states.log_normalizing_constant[-1]


comparison_key = jax.random.key(19)
original_adpf = autodiff.stop_gradient_decorator(systematic.resampling)
adpf_value, adpf_gradient = jax.jit(
    jax.value_and_grad(
        lambda p: original_log_likelihood(p, comparison_key, original_adpf)
    )
)(initial)
dmop_value, dmop_gradient = jax.jit(
    jax.value_and_grad(
        lambda p: dmop.log_likelihood(
            model, p, observations, key=comparison_key, n_particles=64, alpha=1.0
        )
    )
)(initial)
assert bool(jnp.allclose(adpf_value, dmop_value, rtol=1e-10, atol=1e-10))
assert bool(jnp.allclose(adpf_gradient, dmop_gradient, rtol=1e-10, atol=1e-10))
```

The IF2 module wraps these callbacks to carry and perturb parameters alongside
the state. The DMOP module passes a discounted resampling decorator to the
same `particle_filter.build_filter`. Both use the original PF update functions.
Their internal time loop returns only the final state and can rematerialize
steps for differentiation; `run_filter` above returns the full state history.
The optimizer is supplied separately: this example uses Optax Adam.

## IF2 followed by DMOP

Each IF2 particle contains both a state and parameters. They share ancestry.
Gaussian parameter perturbations shrink geometrically between passes. The
terminal weighted parameter swarm is resampled before starting the next pass.
For an initial-value parameter, use a nonzero `initial_perturbation_scale` and
zero `perturbation_scale` in that coordinate.

DMOP uses an inherited log-weight derivative correction discounted by `alpha`.
The forward calculation is the usual bootstrap PF likelihood for every alpha.
At `alpha=1`, the wrapper calls cuthbert's existing stop-gradient
resampler. For `alpha<1`, it is a discounted surrogate derivative, so finite
differences of the forward PF are not a valid gradient reference. At `alpha=0`,
pathwise state derivatives remain, but inherited weight derivatives vanish.

```{.python #ifad-fit}
@jax.jit
def fit(parameters, key):
    """Run both fitting stages and independent endpoint scoring."""
    return ifad.fit(
        model,
        parameters,
        observations,
        key=key,
        n_particles=256,
        n_if2_iterations=20,
        n_dmop_iterations=60,
        perturbation_scale=jnp.array([0.12, 0.08]),
        cooling_rate=0.9,
        alpha=0.97,
        optimizer=optax.adam(0.03),
        max_gradient_norm=50.0,
        evaluation_particles=2048,
        evaluation_replicates=8,
    )


result = jax.block_until_ready(fit(initial, fit_key))
assert bool(result.success), "Inspect stage success flags and traces on failure"
print("Initial parameters:", initial)
print("IF2 endpoint:", result.warm_start.parameters)
print("IFAD endpoint:", result.parameters)
print("Independent PF log-likelihoods:", result.evaluation_log_likelihoods)
print("Log of the mean PF likelihood:", float(result.log_likelihood))
```

The IF2 likelihood trace belongs to perturbed models; it is not an endpoint
score for the original model. DMOP's objective is negative log-likelihood,
with its final entry evaluated under a new key. IFAD's endpoint score uses
independent, unmodified bootstrap PFs and averages likelihoods on the natural
scale before taking the logarithm. The raw log-likelihood repetitions are
also returned.

`success` reports finite execution, not optimizer convergence. An invalid
update freezes the previous parameters and skips later updates. IF2 failure
skips refinement and scoring. Inspect `step_accepted` and
`completed_iterations` to identify where a stage stopped.

## Independent exact check

This scalar Kalman recursion uses the same stationary initialization and data
convention as the simulator. It evaluates all three endpoints without using
particle randomness or the fitted objective trace.

```{.python #ifad-check}
def exact_log_likelihood(parameters):
    """Evaluate the original model by a scalar Kalman recursion."""
    mean, log_q = parameters
    q2 = jnp.exp(2 * log_q)

    def step(carry, y):
        m, variance, loglik = carry
        predicted = mean + RHO * (m - mean)
        variance = RHO**2 * variance + q2
        innovation_variance = variance + OBS_SD**2
        error = y - predicted
        loglik -= 0.5 * (
            jnp.log(2 * jnp.pi * innovation_variance)
            + error**2 / innovation_variance
        )
        gain = variance / innovation_variance
        return (
            predicted + gain * error,
            (1 - gain) * variance,
            loglik,
        ), None

    return jax.lax.scan(
        step, (mean, q2 / (1 - RHO**2), jnp.array(0.0)), observations
    )[0][2]


exact_scores = jax.vmap(exact_log_likelihood)(
    jnp.stack([initial, result.warm_start.parameters, result.parameters])
)
print("Exact log-likelihoods [initial, IF2, IFAD]:", exact_scores)
assert bool(jnp.all(jnp.isfinite(exact_scores)))
# A regression check for this fixed seed and deliberately poor start only.
assert float(exact_scores[-1]) > float(exact_scores[0])
```

## Key Takeaways

- The existing PF and resampling kernels support both stages through a model
  factory and joint state/parameter particles.
- Parameters and traces are on the estimation scale. Optimizer masks apply to
  DMOP; IF2 coordinates are controlled by their perturbation scales.
- Independent endpoint evaluation is essential. This fixed-seed illustration
  establishes a working pipeline, not an optimization or speed advantage over
  other methods.

## Next Steps

- Read the [DMOP](../api_cuthbert/smc/dmop.md),
  [IF2](../api_cuthbert/smc/iterated_filtering.md) and
  [IFAD](../api_cuthbert/smc/ifad.md) API documentation.
- Compare against the existing [differentiable PF example](diff_pf_resampling.md).
- For performance comparisons, repeat multiple starts under matched total
  computation, count both fitting stages, and use independent endpoint scores.

<!--- entangled-tangle-block
```{.python file=examples_scripts/parameter_estimation_ifad.py}
<<ifad-imports>>
<<ifad-data>>
<<ifad-existing-pf>>
<<ifad-fit>>
<<ifad-check>>
```
-->
