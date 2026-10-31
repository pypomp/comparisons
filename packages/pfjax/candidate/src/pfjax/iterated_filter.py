"""Iterated filtering using PFJAX's particle_filter and model callbacks."""

import math
import numbers

import jax
import jax.numpy as jnp
from jax import random

from . import utils
from .particle_filter import particle_filter
from .particle_resamplers import resample_multinomial


def _settings(model, y_meas, parameter_particles, cooling_rate, cooling_interval):
    if callable(getattr(model, "pf_aux", None)):
        raise ValueError("iterated filtering does not support pf_aux")
    if parameter_particles.ndim != 2 or min(parameter_particles.shape) < 1:
        raise ValueError(
            "parameter_particles must have shape (n_particles, n_parameters)"
        )
    if not jnp.issubdtype(parameter_particles.dtype, jnp.floating):
        raise TypeError("parameters must have floating dtype")
    leaves = jax.tree.leaves(y_meas)
    if not leaves or any(y.ndim == 0 for y in leaves):
        raise ValueError("observations need a leading time axis")
    n_obs = leaves[0].shape[0]
    if n_obs < 1 or any(y.shape[0] != n_obs for y in leaves):
        raise ValueError("observation leaves need the same nonempty time axis")
    for value, name in [
        (cooling_rate, "cooling_rate"),
        (cooling_interval, "cooling_interval"),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError(name + " must be a fixed finite real number")
    if not 0 < cooling_rate <= 1 or cooling_interval <= 0:
        raise ValueError("require 0 < cooling_rate <= 1 and cooling_interval > 0")
    return n_obs


def iterated_filter_step(
    model,
    key,
    y_meas,
    parameter_particles,
    perturbation_scale,
    *,
    initial_scale=0.0,
    iteration=0,
    cooling_rate=0.9,
    cooling_interval=1.0,
    within_pass_cooling=False,
):
    """Run one IF2 pass, retaining the entire terminal parameter population.

    ``model`` supplies the ordinary ``pf_init`` and ``pf_step`` methods, each
    taking a single parameter vector. State and observations may be PyTrees.
    No transition density is required. Auxiliary filters are not supported.

    Parameters have shape ``(J, P)``. Scales are nonnegative scalars or vectors
    of length P in estimation coordinates; the model applies any transforms.
    Before pf_init, initial_scale is applied once. Before each later pf_step,
    perturbation_scale is applied. Use zero process scale for initial-value
    parameters. The cooling factor is rate**((iteration + fraction)/interval),
    with fraction=t/n_obs when within_pass_cooling is True, otherwise zero.
    t=0 denotes initialization, so no transition precedes observation zero
    unless the model's own pf_init implements one.

    Calls the original PF loop with augmented (state, parameters, time) leaves.
    Multinomial resampling jointly selects them; a final resampling converts
    the last weighted population into equal-weight parameters for the next
    pass. Returns parameter_particles, their mean theta, perturbed loglik and
    valid. The loglik is not a fixed-parameter evaluation score. valid checks
    settings and finite terminal values, not statistical convergence. Under
    JIT invalid scales/nonfinite values produce valid=False; no retry occurs.
    """
    parameters = jnp.asarray(parameter_particles)
    n_obs = _settings(model, y_meas, parameters, cooling_rate, cooling_interval)
    n_particles, n_parameters = parameters.shape
    scale = jnp.broadcast_to(
        jnp.asarray(perturbation_scale, parameters.dtype), (n_parameters,)
    )
    initial_scale = jnp.broadcast_to(
        jnp.asarray(initial_scale, parameters.dtype), (n_parameters,)
    )
    settings_valid = (
        jnp.all(jnp.isfinite(scale) & (scale >= 0))
        & jnp.all(jnp.isfinite(initial_scale) & (initial_scale >= 0))
        & jnp.isfinite(iteration)
        & (iteration >= 0)
    )
    init_key, filter_key, final_key = random.split(key, 3)
    parameters = parameters + cooling_rate ** (
        iteration / cooling_interval
    ) * initial_scale * random.normal(
        init_key, parameters.shape, dtype=parameters.dtype
    )
    # Exactly the keys particle_filter would allocate to initialization.
    init_keys = random.split(filter_key, n_particles + 1)[1:]
    states, logw = jax.vmap(model.pf_init, in_axes=(0, None, 0))(
        init_keys, utils.tree_subset(y_meas, 0), parameters
    )

    class PerturbedModel:
        def pf_step(self, key, x_prev, y_curr, theta):
            perturb_key, state_key = random.split(key)
            t = x_prev["time"] + 1
            # Dividing the int32 clock produces float32 even when the
            # parameter population is float64. Preserve its dtype for cooling.
            fraction = (
                t.astype(parameters.dtype) / n_obs if within_pass_cooling else 0.0
            )
            factor = cooling_rate ** ((iteration + fraction) / cooling_interval)
            p = x_prev["parameters"] + factor * scale * random.normal(
                perturb_key, (n_parameters,), dtype=parameters.dtype
            )
            x, w = model.pf_step(state_key, x_prev["state"], y_curr, p)
            return {"state": x, "parameters": p, "time": t}, w

    initial = {
        "state": states,
        "parameters": parameters,
        "time": jnp.zeros(n_particles, dtype=jnp.int32),
    }
    result = particle_filter(
        PerturbedModel(),
        filter_key,
        y_meas,
        None,
        n_particles,
        initial_particles=(initial, logw),
    )
    # Only the parameter population is needed after the last observation.
    final = resample_multinomial(
        final_key, result["x_particles"]["parameters"], result["logw"]
    )
    parameters = final["x_particles"]
    return {
        "parameter_particles": parameters,
        "theta": jnp.mean(parameters, axis=0),
        "loglik": result["loglik"],
        "valid": settings_valid
        & jnp.all(jnp.isfinite(parameters))
        & jnp.isfinite(result["loglik"]),
    }


def iterated_filter(
    model,
    key,
    y_meas,
    parameter_particles,
    perturbation_scale,
    n_iter,
    *,
    initial_scale=0.0,
    cooling_rate=0.9,
    cooling_interval=1.0,
    within_pass_cooling=False,
    iteration_offset=0,
):
    """Repeat IF2 passes without replacing the population by its mean.

    Pass i uses fold_in(key, iteration_offset+i). With the returned whole
    population, the same root key and the next iteration_offset reproduce a
    split run. No optimizer or checkpoint writer is included. n_iter and the
    cooling settings are static; parameters/scales may be JIT inputs. All
    passes execute even if valid=False: callers must inspect the flags.
    """
    if (
        isinstance(n_iter, bool)
        or not isinstance(n_iter, numbers.Integral)
        or n_iter < 0
    ):
        raise ValueError("n_iter must be a fixed nonnegative integer")
    if (
        isinstance(iteration_offset, bool)
        or not isinstance(iteration_offset, numbers.Integral)
        or iteration_offset < 0
    ):
        raise ValueError("iteration_offset must be a fixed nonnegative integer")
    parameters = jnp.asarray(parameter_particles)
    _settings(model, y_meas, parameters, cooling_rate, cooling_interval)

    def one(parameters, i):
        result = iterated_filter_step(
            model,
            random.fold_in(key, i),
            y_meas,
            parameters,
            perturbation_scale,
            initial_scale=initial_scale,
            iteration=i,
            cooling_rate=cooling_rate,
            cooling_interval=cooling_interval,
            within_pass_cooling=within_pass_cooling,
        )
        return result["parameter_particles"], {
            k: result[k] for k in ["theta", "loglik", "valid"]
        }

    parameters, trace = jax.lax.scan(
        one, parameters, jnp.arange(n_iter) + iteration_offset
    )
    return {
        "parameter_particles": parameters,
        "theta": jnp.mean(parameters, axis=0),
        "history": trace,
    }
