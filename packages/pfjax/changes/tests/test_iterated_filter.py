"""Checks for IF2 composition, native behavior and the weight derivative."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy.special import logsumexp

import pfjax as pf


class Model:
    _bootstrap = True

    def __init__(self, nonlinear=False):
        self.nonlinear = nonlinear

    def pf_init(self, key, y_init, theta):
        x = theta[0] + 0.3 * random.normal(key, (), dtype=theta.dtype)
        return {"x": x, "sum": x}, -0.5 * (y_init - x) ** 2

    def pf_step(self, key, x_prev, y_curr, theta):
        a = 0.6 if theta.size == 1 else jax.nn.sigmoid(theta[1])
        z = jnp.sin(x_prev["x"]) if self.nonlinear else x_prev["x"]
        x = a * z + theta[0] + 0.2 * random.normal(key, (), dtype=theta.dtype)
        return {"x": x, "sum": x_prev["sum"] + x}, -0.5 * (y_curr - x) ** 2


def close_tree(a, b, exact=False):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        if exact:
            np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-6)


def original_filter():
    path = Path(__file__).resolve().parents[2] / "upstream/src/pfjax/particle_filter.py"
    spec = importlib.util.spec_from_file_location("pristine_pf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.particle_filter


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_native_unchanged(history, compiled):
    model, key = Model(), random.PRNGKey(8)
    y, p = jnp.array([0.1, 1.0, -0.2]), jnp.array([0.2])
    old = lambda q: original_filter()(model, key, y, q, 7, history=history)
    new = lambda q: pf.particle_filter(model, key, y, q, 7, history=history)
    if compiled:
        old, new = jax.jit(old), jax.jit(new)
    close_tree(old(p), new(p), exact=True)
    close_tree(
        jax.grad(lambda q: old(q)["loglik"])(p),
        jax.grad(lambda q: new(q)["loglik"])(p),
        exact=True,
    )


@pytest.mark.parametrize("n_obs", [1, 4])
@pytest.mark.parametrize("compiled", [False, True])
def test_initial_population(n_obs, compiled):
    model, key, p = Model(), random.PRNGKey(3), jnp.array([0.2])
    y = jnp.arange(n_obs) / 5

    def supplied(p):
        keys = random.split(key, 6)[1:]
        initial = jax.vmap(model.pf_init, (0, None, None))(keys, y[0], p)
        return pf.particle_filter(
            model, key, y, p, 5, history=True, initial_particles=initial
        )

    ordinary = lambda p: pf.particle_filter(model, key, y, p, 5, history=True)
    if compiled:
        supplied, ordinary = jax.jit(supplied), jax.jit(ordinary)
    close_tree(supplied(p), ordinary(p), exact=True)


def manual_if2(model, key, y, particles, scale, initial_scale, iteration, within):
    # Independent Python time loop: no call to IF2 or particle_filter.
    j, p = particles.shape
    ik, key, fk = random.split(key, 3)
    particles = particles + 0.8 ** (iteration / 3) * initial_scale * random.normal(
        ik, particles.shape, dtype=particles.dtype
    )
    keys = random.split(key, j + 1)
    key = keys[0]
    x, w = jax.vmap(model.pf_init, (0, None, 0))(keys[1:], y[0], particles)
    ll = logsumexp(w)
    for t in range(1, len(y)):
        key, rk = random.split(key)
        # Independently select joint ancestors without the resampler helper.
        ancestors = random.choice(rk, jnp.arange(j), (j,), p=jax.nn.softmax(w))
        particles = particles[ancestors]
        x = jax.tree.map(lambda a: a[ancestors], x)
        keys = random.split(key, j + 1)
        key = keys[0]
        perturb_keys, state_keys = jax.vmap(lambda k: random.split(k))(
            keys[1:]
        ).swapaxes(0, 1)
        fraction = t / len(y) if within else 0.0
        particles += (
            0.8 ** ((iteration + fraction) / 3)
            * scale
            * jax.vmap(lambda k: random.normal(k, (p,), dtype=particles.dtype))(
                perturb_keys
            )
        )
        x, w = jax.vmap(model.pf_step, (0, 0, None, 0))(state_keys, x, y[t], particles)
        ll += logsumexp(w)
    ancestors = random.choice(fk, jnp.arange(j), (j,), p=jax.nn.softmax(w))
    return particles[ancestors], ll - len(y) * jnp.log(j)


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("within", [False, True])
@pytest.mark.parametrize("n_obs", [1, 5])
def test_if2_manual(dimension, within, n_obs):
    model = Model(dimension == 3)
    key, y = random.PRNGKey(20), jnp.linspace(-0.3, 0.9, n_obs)
    particles = jnp.arange(6 * dimension, dtype=float).reshape(6, dimension) / 100
    scale = jnp.full(dimension, 0.02).at[-1].set(0.0)
    init_scale = jnp.full(dimension, 0.03)
    reference = particles
    for i in range(2):
        ik = random.fold_in(key, i)
        result = jax.jit(
            lambda p: pf.iterated_filter_step(
                model,
                ik,
                y,
                p,
                scale,
                initial_scale=init_scale,
                iteration=i,
                cooling_rate=0.8,
                cooling_interval=3.0,
                within_pass_cooling=within,
            )
        )(particles)
        reference, ll = manual_if2(
            model, ik, y, reference, scale, init_scale, i, within
        )
        close_tree(result["parameter_particles"], reference)
        close_tree(result["loglik"], ll)
        assert result["valid"]
        particles = result["parameter_particles"]


def test_full_population_continuation_and_dynamic_scales():
    m, key = Model(True), random.PRNGKey(37)
    y, p = jnp.array([0.4, -0.2, 0.6]), jnp.arange(21, dtype=float).reshape(7, 3) / 100
    f = jax.jit(
        lambda p, s, n, offset: pf.iterated_filter(
            m, key, y, p, s, n, initial_scale=0.0, iteration_offset=offset
        ),
        static_argnums=(2, 3),
    )
    scale = jnp.array([0.03, 0.0, 0.0])
    full = f(p, scale, 3, 0)
    first = f(p, scale, 1, 0)
    split = f(first["parameter_particles"], scale, 2, 1)
    close_tree(full["parameter_particles"], split["parameter_particles"], exact=True)
    close_tree(full["history"]["loglik"][1:], split["history"]["loglik"], exact=True)
    assert np.all(full["history"]["valid"])
    # A frozen coordinate may be selected, but never changed by perturbation.
    assert set(np.asarray(full["parameter_particles"][:, 2])).issubset(
        set(np.asarray(p[:, 2]))
    )
    zero = f(p, scale, 0, 0)
    close_tree(zero["parameter_particles"], p, exact=True)
    assert zero["history"]["theta"].shape == (0, 3)
    assert not np.any(f(p, jnp.array([-0.1, 0.0, 0.0]), 2, 0)["history"]["valid"])


@pytest.mark.skipif(not jax.config.x64_enabled, reason="requires float64")
def test_within_pass_cooling_preserves_float64_precision():
    # Seven observations give fractions not representable in binary. An int32
    # time / n_obs must not reduce a float64 parameter's cooling to float32.
    model, key = Model(), random.PRNGKey(42)
    y = jnp.linspace(-0.3, 0.9, 7, dtype=jnp.float64)
    population = jnp.arange(12, dtype=jnp.float64).reshape(6, 2) / 100
    scale = jnp.array([0.02, 0.03], dtype=jnp.float64)
    start = jnp.zeros(2, dtype=jnp.float64)
    result = jax.jit(
        lambda q: pf.iterated_filter_step(
            model,
            key,
            y,
            q,
            scale,
            initial_scale=start,
            iteration=2,
            cooling_rate=0.8,
            cooling_interval=3.0,
            within_pass_cooling=True,
        )
    )(population)
    expected, ll = manual_if2(model, key, y, population, scale, start, 2, True)
    assert result["parameter_particles"].dtype == jnp.float64
    np.testing.assert_allclose(
        result["parameter_particles"], expected, rtol=0, atol=5e-14
    )
    np.testing.assert_allclose(result["loglik"], ll, rtol=0, atol=5e-13)


def manual_gradient(model, key, y, theta, j, alpha):
    # Propagate the state tangent and weight derivative as numerical arrays.
    # No stop_gradient correction function or differentiated PF loop is used.
    keys = random.split(key, j + 1)
    key = keys[0]
    init = lambda p: jax.vmap(model.pf_init, (0, None, None))(keys[1:], y[0], p)
    x, w = init(theta)
    dx, dw = jax.jacfwd(init)(theta)
    total = jnp.sum(jax.nn.softmax(w)[:, None] * dw, axis=0)
    for t in range(1, len(y)):
        key, rk = random.split(key)
        a = random.choice(rk, jnp.arange(j), (j,), p=jax.nn.softmax(w))
        correction = alpha * (dw[a] - jnp.mean(dw[a], axis=0))
        x, dx = jax.tree.map(lambda z: z[a], x), jax.tree.map(lambda z: z[a], dx)
        keys = random.split(key, j + 1)
        key = keys[0]
        f = lambda x, p: jax.vmap(model.pf_step, (0, 0, None, None))(
            keys[1:], x, y[t], p
        )

        def tangent(v, dstate):
            return jax.jvp(f, (x, theta), (dstate, v))[1]

        new_dx, dw = jax.vmap(tangent, in_axes=(0, -1), out_axes=-1)(
            jnp.eye(theta.size), dx
        )
        x, w = f(x, theta)
        dx, dw = new_dx, dw + correction
        total += jnp.sum(jax.nn.softmax(w)[:, None] * dw, axis=0)
    return total


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("dimension", [1, 3])
def test_mop_gradient_and_forward(alpha, dimension):
    m, k, y, p = (
        Model(dimension == 3),
        random.PRNGKey(13),
        jnp.array([0.7, -0.3, 0.1, 0.4]),
        jnp.arange(dimension, dtype=float) / 10,
    )
    fn = lambda p: pf.particle_filter_mop(m, k, y, p, 9, alpha=alpha, history=True)
    result = jax.jit(fn)(p)
    native = jax.jit(lambda p: pf.particle_filter(m, k, y, p, 9, history=True))(p)
    close_tree(result, native)
    derivative = jax.jit(jax.grad(lambda p: fn(p)["loglik"]))(p)
    expected = manual_gradient(m, k, y, p, 9, alpha)
    close_tree(derivative, expected)
    if alpha == 0.0:
        assert np.linalg.norm(derivative) > 0.01  # state derivatives remain
        close_tree(
            derivative,
            jax.grad(lambda p: pf.particle_filter(m, k, y, p, 9)["loglik"])(p),
        )


def test_invalid_configuration_and_nonfinite():
    m, k, y, p = Model(), random.PRNGKey(1), jnp.ones(3), jnp.ones((4, 1))
    for alpha in [True, -1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            pf.particle_filter_mop(m, k, y, p[0], 4, alpha=alpha)
    m.pf_aux = lambda *args: 0.0
    with pytest.raises(ValueError):
        pf.iterated_filter_step(m, k, y, p, 0.1)
    with pytest.raises(ValueError):
        pf.particle_filter_mop(m, k, y, p[0], 4)
    del m.pf_aux
    with pytest.raises(ValueError):
        pf.particle_filter(
            m, k, y, p[0], 4, score=True, initial_particles=(p, jnp.zeros(4))
        )
    with pytest.raises(ValueError):
        pf.particle_filter(m, k, y, p[0], 4, initial_particles=(p, jnp.zeros(3)))
    for rate in [0.0, 1.1, float("inf")]:
        with pytest.raises(ValueError):
            pf.iterated_filter_step(m, k, y, p, 0.1, cooling_rate=rate)

    class Bad(Model):
        def pf_step(self, key, x_prev, y_curr, theta):
            return x_prev, jnp.asarray(-jnp.inf)

    bad = jax.jit(lambda p: pf.iterated_filter_step(Bad(), k, y, p, 0.1))(p)
    assert not bad["valid"]


def test_observation_pytree():
    class DictModel(Model):
        def pf_init(self, key, y_init, theta):
            return super().pf_init(key, y_init["y"], theta)

        def pf_step(self, key, x_prev, y_curr, theta):
            return super().pf_step(key, x_prev, y_curr["y"], theta)

    k, y, p = random.PRNGKey(4), jnp.ones(3), jnp.zeros((5, 1))
    a = pf.iterated_filter_step(Model(), k, y, p, 0.02)
    b = pf.iterated_filter_step(DictModel(), k, {"y": y}, p, 0.02)
    close_tree(a, b, exact=True)
