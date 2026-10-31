"""Discounted MOP gradients and optimization for smooth bootstrap models.

All parameters and traces are on the estimation scale. Constraints belong in
the model factory. Every observation resamples; adaptive/guided filters and
discrete process-score corrections are deliberately outside this API.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cuthbert.smc import _estimation as util
from cuthbert.smc import particle_filter
from cuthbert.smc._estimation import ModelFactory, Optimizer
from cuthbertlib.resampling import discounted_stop_gradient_decorator


class GradientFitResult(NamedTuple):
    """Optimization output; objectives align with the parameter history.

    ``objectives[m]`` is a negative PF log-likelihood at ``parameter_history[m]``
    using a fresh key. The last entry is a separate final evaluation. On a failed
    update, parameters freeze at the last valid value; remaining objectives are
    NaN and ``success`` is false. Failure is never turned into a zero gradient.
    """

    parameters: Any
    parameter_history: Any
    objectives: Any
    gradient_norms: Any
    step_accepted: Any
    completed_iterations: Any
    success: Any


def log_likelihood(
    model_factory: ModelFactory,
    parameters: Any,
    model_inputs: Any,
    *,
    key: Any,
    n_particles: int,
    alpha: float = 0.97,
    resampling: str = "systematic",
    checkpoint: bool = False,
) -> jax.Array:
    """Return the bootstrap PF log-likelihood with a DMOP surrogate derivative.

    Changing alpha leaves forward PF values unchanged. For alpha < 1 its
    derivative is discounted; finite differences of the unchanged forward PF
    are NOT a reference for that surrogate derivative.

    Args:
        model_factory: Estimation-scale parameters to bootstrap callbacks.
        parameters: A floating-point parameter PyTree.
        model_inputs: Observation inputs with a common leading time axis.
        key: Random key, independent across optimization/evaluation calls.
        n_particles: Positive static particle count.
        alpha: Static discount in [0, 1].
        resampling: 'systematic' or 'multinomial', performed at every time.
        checkpoint: Rematerialize filter steps during differentiation. Disabled
            by default to retain the original ADPF differentiation path.
            Enabling this can change floating-point gradients.

    Returns:
        Scalar log-likelihood with the chosen surrogate gradient.
    """
    util.positive_int("n_particles", n_particles)
    parameters = util.parameters_tree(parameters)
    inputs = util.inputs_tree(model_inputs)
    sample0, propagate, potential = model_factory(parameters)
    sample_ancestors = discounted_stop_gradient_decorator(
        util.resampler(resampling), alpha
    )
    obj = particle_filter.build_filter(
        sample0, propagate, potential, n_particles, sample_ancestors
    )
    init_key, filter_key = jax.random.split(key)
    final = util.scan_final(
        obj, inputs, obj.init_prepare(key=init_key), filter_key, checkpoint=checkpoint
    )
    return final.log_normalizing_constant


def fit(
    model_factory: ModelFactory,
    initial_parameters: Any,
    model_inputs: Any,
    *,
    key: Any,
    n_particles: int,
    n_iterations: int,
    optimizer: Optimizer,
    alpha: float = 0.97,
    resampling: str = "systematic",
    trainable_mask: Any = True,
    max_gradient_norm: float | None = None,
    checkpoint: bool = False,
) -> GradientFitResult:
    """Minimize negative log-likelihood with DMOP and an Optax-style optimizer.

    Args:
        model_factory: Estimation-scale parameters to bootstrap callbacks.
        initial_parameters: Floating-point PyTree, also defining trace structure.
        model_inputs: Time-indexed observation inputs.
        key: Root key; every iteration and the final evaluation get distinct keys.
        n_particles: Positive static particle count.
        n_iterations: Nonnegative static number of gradient updates.
        optimizer: Object with init(parameters) and update(gradients, state,
            parameters) methods. Updates are added to parameters, as in Optax.
        alpha: Static DMOP discount in [0, 1].
        resampling: Unconditional 'systematic' or 'multinomial'.
        trainable_mask: Boolean scalar or parameter-shaped PyTree. False entries
            freeze parameters even if an optimizer applies momentum/weight decay.
        max_gradient_norm: Optional positive global gradient clipping threshold.
        checkpoint: Rematerialize each filter step in reverse mode. Disabled
            by default; enabling it can change floating-point gradients.

    Returns:
        Traces, last valid parameters and an explicit success flag.
    """
    util.positive_int("n_iterations", n_iterations, allow_zero=True)
    util.positive_int("n_particles", n_particles)
    util.scalar_range("alpha", alpha, 0, 1)
    util.resampler(resampling)
    if max_gradient_norm is not None:
        util.scalar_range("max_gradient_norm", max_gradient_norm, 1e-30, float("inf"))
    initial = util.parameters_tree(initial_parameters)
    inputs = util.inputs_tree(model_inputs)
    mask = util.broadcast_tree(
        trainable_mask, initial, name="trainable_mask", boolean=True
    )
    keys = jax.random.split(key, n_iterations + 1)

    def objective(parameters, k):
        return -log_likelihood(
            model_factory,
            parameters,
            inputs,
            key=k,
            n_particles=n_particles,
            alpha=alpha,
            resampling=resampling,
            checkpoint=checkpoint,
        )

    value_dtype = jax.eval_shape(objective, initial, keys[0]).dtype

    def step(carry, k):
        parameters, opt_state, active = carry

        def advance(_):
            value, gradient = jax.value_and_grad(objective)(parameters, k)
            gradient = jax.tree.map(lambda g, m: jnp.where(m, g, 0), gradient, mask)
            finite = jnp.isfinite(value) & util.all_finite(gradient)
            norm = jnp.asarray(
                jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(gradient))),
                dtype=value_dtype,
            )
            finite = finite & jnp.isfinite(norm)
            if max_gradient_norm is not None:
                factor = jnp.minimum(1.0, max_gradient_norm / jnp.maximum(norm, 1e-30))
                gradient = jax.tree.map(lambda g: factor.astype(g.dtype) * g, gradient)
            updates, next_opt = optimizer.update(gradient, opt_state, parameters)
            proposal = jax.tree.map(
                lambda p, u, m: p + jnp.where(m, u, 0), parameters, updates, mask
            )
            accepted = finite & util.all_finite(proposal) & util.all_finite(next_opt)
            result = util.select_tree(accepted, proposal, parameters)
            next_opt = util.select_tree(accepted, next_opt, opt_state)
            return (result, next_opt, accepted), (result, value, norm, accepted)

        def skip(_):
            nan = jnp.asarray(jnp.nan, dtype=value_dtype)
            return carry, (parameters, nan, nan, jnp.array(False))

        return jax.lax.cond(active, advance, skip, operand=None)

    (parameters, _, active), (history, values, norms, accepted) = jax.lax.scan(
        step, (initial, optimizer.init(initial), util.all_finite(initial)), keys[:-1]
    )
    final_value = jax.lax.cond(
        active,
        lambda _: objective(parameters, keys[-1]),
        lambda _: jnp.asarray(jnp.nan, dtype=values.dtype),
        operand=None,
    )
    history = jax.tree.map(lambda p, h: jnp.concatenate((p[None], h)), initial, history)
    return GradientFitResult(
        parameters,
        history,
        jnp.concatenate((values, final_value[None])),
        norms,
        accepted,
        jnp.sum(accepted),
        active & jnp.isfinite(final_value),
    )
