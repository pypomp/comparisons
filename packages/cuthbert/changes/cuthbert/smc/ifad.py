"""IF2 warm start followed by DMOP optimization and independent PF scoring."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cuthbert.smc import _estimation as util
from cuthbert.smc import dmop, iterated_filtering, particle_filter
from cuthbert.smc._estimation import ModelFactory, Optimizer


class IFADResult(NamedTuple):
    """Two-stage fit with independent, unperturbed endpoint PF evaluations.

    ``log_likelihood`` is log(mean(exp(evaluation_log_likelihoods))). Individual
    repetitions are retained for Monte Carlo diagnostics. No uncertainty or
    optimizer-superiority claim is implied by success, which means finite runs.
    """

    parameters: Any
    warm_start: iterated_filtering.IteratedFilteringResult
    refinement: dmop.GradientFitResult
    evaluation_log_likelihoods: Any
    log_likelihood: Any
    success: Any


def fit(
    model_factory: ModelFactory,
    initial_parameters: Any,
    model_inputs: Any,
    *,
    key: Any,
    n_particles: int,
    n_if2_iterations: int,
    n_dmop_iterations: int,
    perturbation_scale: Any,
    optimizer: Optimizer,
    initial_perturbation_scale: Any = None,
    cooling_rate: float = 0.9,
    alpha: float = 0.97,
    resampling: str = "systematic",
    trainable_mask: Any = True,
    max_gradient_norm: float | None = None,
    evaluation_particles: int | None = None,
    evaluation_replicates: int = 8,
    checkpoint: bool = False,
) -> IFADResult:
    """Run a complete IFAD pipeline for a smooth bootstrap model.

    Args:
        model_factory: Estimation-scale parameters to bootstrap callbacks.
        initial_parameters: Initial floating-point parameter PyTree.
        model_inputs: Time-indexed observation inputs.
        key: Root key, split into IF2, DMOP and independent evaluation streams.
        n_particles: Particle count for both fitting stages.
        n_if2_iterations: Nonnegative number of IF2 passes.
        n_dmop_iterations: Nonnegative number of DMOP updates.
        perturbation_scale: IF2 process perturbation standard deviations.
        optimizer: Optax-compatible optimizer for DMOP (minimizing negative logL).
        initial_perturbation_scale: IF2 initial noise; defaults to process scales.
        cooling_rate: IF2 geometric cooling multiplier in (0, 1].
        alpha: DMOP discount in [0, 1].
        resampling: Unconditional 'systematic' or 'multinomial'.
        trainable_mask: Coordinates updated in the DMOP stage. IF2 coordinates
            are controlled separately by their perturbation scales.
        max_gradient_norm: Optional DMOP global gradient clipping threshold.
        evaluation_particles: PF scoring particle count, defaulting to fitting.
        evaluation_replicates: Positive number of independent scoring PFs.
        checkpoint: Rematerialize DMOP filter steps in reverse mode. Disabled
            by default; enabling it can change floating-point gradients.

    Returns:
        Both stages' traces and independently evaluated final parameters.
        A failed IF2 stage skips refinement and evaluation and reports failure.
    """
    util.positive_int("evaluation_replicates", evaluation_replicates)
    n_eval = n_particles if evaluation_particles is None else evaluation_particles
    util.positive_int("evaluation_particles", n_eval)
    inputs = util.inputs_tree(model_inputs)
    warm_key, gradient_key, evaluation_key = jax.random.split(key, 3)
    warm = iterated_filtering.fit(
        model_factory,
        initial_parameters,
        inputs,
        key=warm_key,
        n_particles=n_particles,
        n_iterations=n_if2_iterations,
        perturbation_scale=perturbation_scale,
        initial_perturbation_scale=initial_perturbation_scale,
        cooling_rate=cooling_rate,
        resampling=resampling,
    )

    def refine(parameters):
        return dmop.fit(
            model_factory,
            parameters,
            inputs,
            key=gradient_key,
            n_particles=n_particles,
            n_iterations=n_dmop_iterations,
            optimizer=optimizer,
            alpha=alpha,
            resampling=resampling,
            trainable_mask=trainable_mask,
            max_gradient_norm=max_gradient_norm,
            checkpoint=checkpoint,
        )

    refinement_shape = jax.eval_shape(refine, warm.parameters)

    def skip_refinement(parameters):
        history = jax.tree.map(
            lambda p: jnp.broadcast_to(p, (n_dmop_iterations + 1,) + p.shape),
            parameters,
        )
        return dmop.GradientFitResult(
            parameters,
            history,
            jnp.full(
                (n_dmop_iterations + 1,), jnp.nan, refinement_shape.objectives.dtype
            ),
            jnp.full(
                (n_dmop_iterations,), jnp.nan, refinement_shape.gradient_norms.dtype
            ),
            jnp.zeros(n_dmop_iterations, dtype=bool),
            jnp.asarray(0, dtype=refinement_shape.completed_iterations.dtype),
            jnp.array(False),
        )

    refinement = jax.lax.cond(warm.success, refine, skip_refinement, warm.parameters)

    def evaluate(k):
        obj = particle_filter.build_filter(
            *model_factory(refinement.parameters), n_eval, util.resampler(resampling)
        )
        init_key, filter_key = jax.random.split(k)
        return util.scan_final(
            obj, inputs, obj.init_prepare(key=init_key), filter_key, checkpoint=False
        ).log_normalizing_constant

    evaluation_keys = jax.random.split(evaluation_key, evaluation_replicates)
    evaluation_dtype = jax.eval_shape(evaluate, evaluation_keys[0]).dtype
    logliks = jax.lax.cond(
        warm.success & refinement.success,
        lambda _: jax.vmap(evaluate)(evaluation_keys),
        lambda _: jnp.full((evaluation_replicates,), jnp.nan, dtype=evaluation_dtype),
        operand=None,
    )
    loglik = jax.scipy.special.logsumexp(logliks) - jnp.log(evaluation_replicates)
    return IFADResult(
        refinement.parameters,
        warm,
        refinement,
        logliks,
        loglik,
        warm.success & refinement.success & util.all_finite(logliks),
    )
