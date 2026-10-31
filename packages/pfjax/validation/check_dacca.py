"""Small-particle full-series integration, not a new formal Dacca comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pfjax as pf
from jax import random
from jax.scipy.special import logsumexp

import bridge
import dacca_model as m


def reference_if2(model, key, data, particles, iteration):
    # Independent explicit time loop, same newly documented key assignment.
    j, p = particles.shape
    ik, key, fk = random.split(key, 3)
    particles = particles + 0.5 ** (iteration / 50) * m.INITIAL_SCALE * random.normal(
        ik, particles.shape
    )
    keys = random.split(key, j + 1)
    key = keys[0]
    init = jax.jit(jax.vmap(model.pf_init, (0, None, 0)))
    states, w = init(keys[1:], jax.tree.map(lambda a: a[0], data), particles)
    ll = logsumexp(w)

    @jax.jit
    def month(key, states, particles, w, observation, t):
        key, rk = random.split(key)
        ancestors = random.choice(rk, jnp.arange(j), (j,), p=jax.nn.softmax(w))
        particles, states = particles[ancestors], states[ancestors]
        keys = random.split(key, j + 1)
        split = jax.vmap(lambda k: random.split(k))(keys[1:])
        noises = jax.vmap(lambda k: random.normal(k, (p,)))(split[:, 0])
        particles += 0.5 ** ((iteration + t / 600) / 50) * m.PROCESS_SCALE * noises
        states, w = jax.vmap(model.pf_step, (0, 0, None, 0))(
            split[:, 1], states, observation, particles
        )
        return keys[0], states, particles, w

    for t in range(1, 600):
        key, states, particles, w = month(
            key, states, particles, w, jax.tree.map(lambda a: a[t], data), t
        )
        ll = ll + logsumexp(w)
    ancestors = random.choice(fk, jnp.arange(j), (j,), p=jax.nn.softmax(w))
    return particles[ancestors], ll - 600 * jnp.log(j)


def main(output):
    started = time.perf_counter()
    model, data = bridge.Dacca(), m.inputs(600)
    p = jnp.asarray(m.PUBLISHED, dtype=jnp.float32)
    population = jnp.broadcast_to(p, (8, 23))
    records = {
        "T": 600,
        "Euler_steps": 20,
        "J": 8,
        "free_parameters": 23,
        "devices": [str(x) for x in jax.devices()],
        "float32": True,
        "historical_IF2_trajectory_replayed": False,
        "passed": False,
        "copied_source_hashes": {
            n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()
            for n in ["bridge.py", "dacca_model.py", "input/frozen.npz"]
        },
    }
    key = random.PRNGKey(20261030)
    reference = population
    passes = []
    for i in range(2):
        ik = random.fold_in(key, i)
        new = jax.jit(
            lambda q: pf.iterated_filter_step(
                model,
                ik,
                data,
                q,
                m.PROCESS_SCALE,
                initial_scale=m.INITIAL_SCALE,
                iteration=i,
                cooling_rate=0.5,
                cooling_interval=50.0,
                within_pass_cooling=True,
            )
        )(population)
        reference, ll = reference_if2(model, ik, data, reference, i)
        jax.block_until_ready(new)
        error = float(jnp.max(jnp.abs(new["parameter_particles"] - reference)))
        ll_error = float(jnp.abs(new["loglik"] - ll))
        assert bool(new["valid"])
        np.testing.assert_allclose(
            new["parameter_particles"], reference, rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(new["loglik"], ll, rtol=2e-6, atol=2e-3)
        passes.append(
            {
                "pass": i,
                "max_parameter_difference_from_explicit_loop": error,
                "perturbed_loglik_difference": ll_error,
                "valid": bool(new["valid"]),
            }
        )
        population = new["parameter_particles"]
    records["IF2"] = passes
    comparisons = []
    for alpha in [0.0, 0.97, 1.0]:
        native_fn = lambda q: pf.particle_filter(
            model, random.PRNGKey(120), data, q, 8
        )["loglik"]
        new_fn = lambda q: pf.particle_filter_mop(
            model, random.PRNGKey(120), data, q, 8, alpha=alpha
        )["loglik"]
        old_fn = lambda q: pf.particle_filter(
            bridge.CorrectedDacca(),
            random.PRNGKey(120),
            data,
            q,
            8,
            resampler=bridge.corrected_resampler(alpha),
        )["loglik"]
        native_value = jax.jit(native_fn)(p)
        value, gradient = jax.jit(jax.value_and_grad(new_fn))(p)
        old_value, old_gradient = jax.jit(jax.value_and_grad(old_fn))(p)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_allclose(value, native_value, rtol=0, atol=0.002)
        np.testing.assert_allclose(value, old_value, rtol=0, atol=0.002)
        np.testing.assert_allclose(gradient, old_gradient, rtol=2e-5, atol=2e-4)
        comparisons.append(
            {
                "alpha": alpha,
                "loglik": float(value),
                "forward_difference": float(value - native_value),
                "max_gradient_difference_from_frozen_wrapper": float(
                    jnp.max(jnp.abs(gradient - old_gradient))
                ),
            }
        )
    records["gradient_comparisons"] = comparisons
    theta = jnp.mean(population, axis=0)
    optimizer = optax.adam(1e-6)
    grad_fn = jax.jit(
        jax.value_and_grad(
            lambda q: -pf.particle_filter_mop(
                model, random.PRNGKey(210), data, q, 8, alpha=0.97
            )["loglik"]
            / 600
        )
    )
    loss, gradient = grad_fn(theta)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    updates, state = optimizer.update(gradient, optimizer.init(theta), theta)
    theta = optax.apply_updates(theta, updates)
    raw = jax.jit(
        jax.vmap(lambda k: pf.particle_filter(model, k, data, theta, 8)["loglik"])
    )(random.split(random.PRNGKey(310), 2))
    assert bool(jnp.all(jnp.isfinite(raw)))
    records.update(
        passed=True,
        Adam_updates=1,
        optimizer="external Optax Adam",
        raw_independent_PF_scores=np.asarray(raw).tolist(),
        score_precision_claimed=False,
        new_formal_comparison=False,
        elapsed_seconds_including_compilation=time.perf_counter() - started,
    )
    Path(output).write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
