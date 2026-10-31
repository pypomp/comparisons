# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#examples_scripts/parameter_estimation_ifad.py>>[init]
# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#ifad-imports>>[init]
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
# ~/~ end
# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#ifad-data>>[init]
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
# ~/~ end
# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#ifad-existing-pf>>[init]
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
# ~/~ end
# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#ifad-fit>>[init]
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
# ~/~ end
# ~/~ begin <<docs/examples/parameter_estimation_ifad.md#ifad-check>>[init]
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
# ~/~ end
# ~/~ end
