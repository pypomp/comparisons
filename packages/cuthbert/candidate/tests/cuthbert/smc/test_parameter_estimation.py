"""Independent recursion and end-to-end checks for bootstrap parameter fitting."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cuthbert import filter as run_filter
from cuthbert.smc import dmop, ifad, iterated_filtering
from cuthbert.smc import particle_filter as pf
from cuthbertlib.resampling import autodiff, multinomial, systematic


class Optimizer(NamedTuple):
    init: object
    update: object


def sgd(rate):
    return Optimizer(
        lambda _: (),
        lambda g, s, p: (jax.tree.map(lambda x: -rate * x, g), s),
    )


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def model(theta):
    phi, log_q, initial_mean = theta

    def init(k):
        return initial_mean + jax.random.normal(k, (), dtype=theta.dtype)

    def propagate(k, x, y):
        return jnp.tanh(phi) * x + jnp.exp(log_q) * jax.random.normal(
            k, (), dtype=theta.dtype
        )

    def potential(old, x, y):
        return -0.5 * ((y - x) / 0.7) ** 2 - jnp.log(0.7 * jnp.sqrt(2 * jnp.pi))

    return init, propagate, potential


Y = [0.3, -0.5, 0.7, 1.0, 0.2]


def test_public_alpha_one_uses_original_adpf(monkeypatch):
    """The undiscounted comparison must execute the original package decorator."""
    original = autodiff.stop_gradient_decorator
    calls = []

    def record_original(resampler):
        calls.append(resampler)
        return original(resampler)

    monkeypatch.setattr(autodiff, "stop_gradient_decorator", record_original)
    theta = jnp.array([0.4, -0.7, 0.2])
    value, gradient = jax.value_and_grad(
        lambda p: dmop.log_likelihood(
            model,
            p,
            jnp.array(Y),
            key=jax.random.key(13),
            n_particles=32,
            alpha=1.0,
            resampling="systematic",
        )
    )(theta)
    assert calls == [systematic.resampling]
    assert np.isfinite(value) and np.isfinite(gradient).all()


def stock_run(theta, ys, key, resampler, n=32):
    obj = pf.build_filter(*model(theta), n, resampler)
    ik, fk = jax.random.split(key)
    return run_filter(obj, ys, obj.init_prepare(key=ik), key=fk)


@pytest.mark.parametrize("method", ["systematic", "multinomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 1.0])
def test_dmop_matches_explicit_recursion(method, alpha):
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(13)
    n, t_count = 32, len(Y)
    base = systematic.resampling if method == "systematic" else multinomial.resampling
    stock = stock_run(theta, ys, key, base, n)
    wrapped = autodiff.discounted_stop_gradient_decorator(base, alpha)
    with_correction = stock_run(theta, ys, key, wrapped, n)
    np.testing.assert_array_equal(stock.particles, with_correction.particles)
    np.testing.assert_array_equal(
        stock.ancestor_indices, with_correction.ancestor_indices
    )
    np.testing.assert_allclose(
        stock.log_normalizing_constant,
        with_correction.log_normalizing_constant,
        rtol=1e-14,
        atol=1e-14,
    )

    # Explicit propagate/weight/resample MOP recurrence with recorded randomness.
    # cuthbert also resamples BEFORE its first propagation; retain that ancestry.
    ik, fk = jax.random.split(key)
    initial_noise = jax.vmap(lambda k: jax.random.normal(k))(jax.random.split(ik, n))
    prep = jax.random.split(fk, t_count)
    step_keys = jnp.concatenate([ik[None], prep[:-1]])
    noise = jax.vmap(
        lambda k: jax.vmap(lambda s: jax.random.normal(s))(
            jax.random.split(k, n + 1)[1:]
        )
    )(step_keys)
    ancestry = stock.ancestor_indices[1:]

    def reference(th):
        x = (th[2] + initial_noise)[ancestry[0]]
        score, ll = jnp.zeros(n), jnp.array(0.0)
        for t in range(t_count):
            prior = alpha * score
            x = jnp.tanh(th[0]) * x + jnp.exp(th[1]) * noise[t]
            g = model(th)[2](x, x, ys[t])
            ll += jax.scipy.special.logsumexp(prior + g) - jax.scipy.special.logsumexp(
                prior
            )
            if t + 1 < t_count:
                idx = ancestry[t + 1]
                x = x[idx]
                score = (prior + g - jax.lax.stop_gradient(g))[idx]
        return ll

    fn = partial(
        dmop.log_likelihood,
        model,
        model_inputs=ys,
        key=key,
        n_particles=n,
        alpha=alpha,
        resampling=method,
    )
    value, grad = jax.jit(jax.value_and_grad(fn))(theta)
    rv, rg = jax.value_and_grad(reference)(theta)
    np.testing.assert_allclose(value, rv, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad, rg, rtol=1e-12, atol=1e-12)
    if alpha == 0:
        expected = jax.grad(
            lambda p: stock_run(p, ys, key, base, n).log_normalizing_constant[-1]
        )(theta)
        np.testing.assert_allclose(grad, expected, rtol=1e-12, atol=1e-12)
    if alpha == 1:
        expected = jax.grad(
            lambda p: stock_run(
                p, ys, key, autodiff.stop_gradient_decorator(base), n
            ).log_normalizing_constant[-1]
        )(theta)
        np.testing.assert_allclose(grad, expected, rtol=1e-12, atol=1e-12)


def manual_if2(theta, ys, key, n, iterations, scale, init_scale, cooling):
    """Plain Python-loop IF2 reference including terminal resampling."""
    swarm = jnp.broadcast_to(theta, (n, theta.size))
    likelihoods, means = [], [theta]
    for m, mk in enumerate(jax.random.split(key, iterations)):
        nk, ik, fk, tk = jax.random.split(mk, 4)
        rate = cooling**m
        swarm = swarm + rate * init_scale * jax.random.normal(
            jax.random.split(nk, 1)[0], swarm.shape
        )
        x = jax.vmap(lambda p, k: model(p)[0](k))(swarm, jax.random.split(ik, n))
        weights, ll = jnp.zeros(n), jnp.array(0.0)
        step_keys = jnp.concatenate([ik[None], jax.random.split(fk, len(ys))[:-1]])
        for y, sk in zip(ys, step_keys, strict=True):
            keys = jax.random.split(sk, n + 1)
            idx, _, _ = systematic.resampling(keys[0], weights, x, n)
            x, swarm = x[idx], swarm[idx]
            split = jax.vmap(jax.random.split)(keys[1:])
            noise_keys = jax.vmap(lambda k: jax.random.split(k, 1)[0])(split[:, 0])
            eps = jax.vmap(lambda k: jax.random.normal(k, theta.shape))(noise_keys)
            swarm = swarm + rate * scale * eps
            old = x
            x = jax.vmap(lambda k, s, p: model(p)[1](k, s, y))(split[:, 1], x, swarm)
            weights = jax.vmap(lambda a, b, p: model(p)[2](a, b, y))(old, x, swarm)
            ll += jax.scipy.special.logsumexp(weights) - jnp.log(n)
        idx, _, _ = systematic.resampling(tk, weights, x, n)
        swarm = swarm[idx]
        means.append(jnp.mean(swarm, axis=0))
        likelihoods.append(ll)
    return swarm, jnp.stack(means), jnp.stack(likelihoods)


@pytest.mark.parametrize("scale", [jnp.zeros(3), jnp.array([0.1, 0.04, 0.0])])
@pytest.mark.parametrize("initial_scale", [[0.0, 0.0, 0.0], [0.02, 0.0, 0.3]])
def test_if2_cooling_and_initial_parameters_against_reference(scale, initial_scale):
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(5)
    init_scale = jnp.array(initial_scale)
    expected, means, logliks = manual_if2(theta, ys, key, 16, 3, scale, init_scale, 0.6)
    result = iterated_filtering.fit(
        model,
        theta,
        ys,
        key=key,
        n_particles=16,
        n_iterations=3,
        perturbation_scale=scale,
        initial_perturbation_scale=init_scale,
        cooling_rate=0.6,
    )
    assert result.success and result.completed_iterations == 3
    np.testing.assert_allclose(result.parameter_swarm, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(result.parameter_history, means, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        result.perturbed_log_likelihoods, logliks, rtol=2e-6, atol=1e-7
    )


def tree_model(parameters):
    return model(jnp.concatenate([parameters["dynamic"], parameters["initial"][None]]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pytree_jit_vmap_masks_and_checkpoint(dtype):
    theta = {"dynamic": jnp.array([0.4, -0.7], dtype), "initial": jnp.array(0.2, dtype)}
    ys = jnp.array(Y, dtype)
    key = jax.random.key(8)

    def fn(p, k):
        return dmop.fit(
            tree_model,
            p,
            ys,
            key=k,
            n_particles=16,
            n_iterations=2,
            optimizer=sgd(0.01),
            trainable_mask={"dynamic": True, "initial": False},
            max_gradient_norm=2.0,
        )

    result = jax.jit(fn)(theta, key)
    assert result.success
    np.testing.assert_array_equal(
        result.parameter_history["initial"], 0.2 * np.ones(3, dtype=np.dtype(dtype))
    )
    batch = jax.jit(jax.vmap(fn, in_axes=(None, 0)))(theta, jax.random.split(key, 2))
    assert np.asarray(batch.success).all()
    for checkpoint in (False, True):
        value, gradient = jax.value_and_grad(
            lambda p: dmop.log_likelihood(
                tree_model, p, ys, key=key, n_particles=16, checkpoint=checkpoint
            )
        )(theta)
        if not checkpoint:
            expected_value, expected_gradient = value, gradient
        else:
            np.testing.assert_allclose(value, expected_value, rtol=1e-6)
            for a, b in zip(
                jax.tree.leaves(gradient),
                jax.tree.leaves(expected_gradient),
                strict=True,
            ):
                np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    warm = jax.jit(
        lambda p, k: iterated_filtering.fit(
            tree_model,
            p,
            ys,
            key=k,
            n_particles=16,
            n_iterations=2,
            perturbation_scale={"dynamic": 0.01, "initial": 0.0},
            initial_perturbation_scale={"dynamic": 0.0, "initial": 0.05},
        )
    )(theta, key)
    assert warm.success
    assert warm.parameters["dynamic"].dtype == dtype


def test_objective_trace_alignment_and_final_evaluation():
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(16)
    result = dmop.fit(
        model, theta, ys, key=key, n_particles=16, n_iterations=3, optimizer=sgd(0.01)
    )
    keys = jax.random.split(key, 4)
    for i in range(4):
        objective = -dmop.log_likelihood(
            model, result.parameter_history[i], ys, key=keys[i], n_particles=16
        )
        np.testing.assert_allclose(result.objectives[i], objective, rtol=1e-12)


def test_ifad_composition_and_independent_scoring():
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(42)
    result = jax.jit(
        lambda p, k: ifad.fit(
            model,
            p,
            ys,
            key=k,
            n_particles=16,
            n_if2_iterations=2,
            n_dmop_iterations=2,
            perturbation_scale=0.05,
            optimizer=sgd(0.01),
            evaluation_particles=24,
            evaluation_replicates=3,
        )
    )(theta, key)
    assert result.success
    warm_key, gradient_key, evaluation_key = jax.random.split(key, 3)
    warm = iterated_filtering.fit(
        model,
        theta,
        ys,
        key=warm_key,
        n_particles=16,
        n_iterations=2,
        perturbation_scale=0.05,
    )
    refine = dmop.fit(
        model,
        warm.parameters,
        ys,
        key=gradient_key,
        n_particles=16,
        n_iterations=2,
        optimizer=sgd(0.01),
    )
    np.testing.assert_allclose(result.parameters, refine.parameters, atol=1e-12)
    expected = jnp.stack(
        [
            stock_run(
                result.parameters, ys, k, systematic.resampling, 24
            ).log_normalizing_constant[-1]
            for k in jax.random.split(evaluation_key, 3)
        ]
    )
    np.testing.assert_allclose(result.evaluation_log_likelihoods, expected, atol=1e-12)
    np.testing.assert_allclose(
        result.log_likelihood,
        jax.scipy.special.logsumexp(expected) - jnp.log(3),
        atol=1e-12,
    )


def test_failures_do_not_silently_update_or_proceed():
    theta, ys = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y)
    bad_opt = Optimizer(lambda _: (), lambda g, s, p: (jnp.full_like(p, jnp.nan), s))
    result = dmop.fit(
        model,
        theta,
        ys,
        key=jax.random.key(1),
        n_particles=8,
        n_iterations=3,
        optimizer=bad_opt,
    )
    assert not result.success and result.completed_iterations == 0
    np.testing.assert_array_equal(result.parameters, theta)
    assert np.isnan(result.objectives[1:]).all()

    def impossible(p):
        init, propagate, _ = model(p)
        return init, propagate, lambda old, x, y: jnp.array(-jnp.inf)

    failed = ifad.fit(
        impossible,
        theta,
        jnp.array([0.0]),
        key=jax.random.key(2),
        n_particles=8,
        n_if2_iterations=2,
        n_dmop_iterations=3,
        perturbation_scale=0.1,
        optimizer=sgd(0.01),
    )
    assert not failed.success and not failed.warm_start.success
    assert failed.refinement.completed_iterations == 0
    np.testing.assert_array_equal(failed.parameters, theta)
    assert np.isnan(failed.evaluation_log_likelihoods).all()


def test_zero_iterations_and_frozen_update_even_with_weight_decay():
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(2)
    warm = iterated_filtering.fit(
        model, theta, ys, key=key, n_particles=8, n_iterations=0, perturbation_scale=0.1
    )
    np.testing.assert_array_equal(warm.parameters, theta)
    assert warm.parameter_history.shape == (1, 3) and warm.success
    result = dmop.fit(
        model, theta, ys, key=key, n_particles=8, n_iterations=0, optimizer=sgd(0.01)
    )
    assert result.parameter_history.shape == (1, 3) and result.success
    decay = Optimizer(lambda _: (), lambda g, s, p: (-0.1 * p, s))
    frozen = dmop.fit(
        model,
        theta,
        ys,
        key=key,
        n_particles=8,
        n_iterations=2,
        optimizer=decay,
        trainable_mask=False,
    )
    np.testing.assert_array_equal(frozen.parameters, theta)


@pytest.mark.parametrize("valid", [True, False])
def test_ifad_accepts_captured_scale_arrays_under_jit(valid):
    # A normal user call constructs scales before jitting a fitting function.
    # The captured array can be concrete while operations on it are traced.
    theta, ys = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y)
    scale = jnp.array([0.01 if valid else -0.01, 0.02, 0.0])
    initial_scale = jnp.array([0.0, 0.0, 0.03])

    @jax.jit
    def run(parameters):
        return ifad.fit(
            model,
            parameters,
            ys,
            key=jax.random.key(73),
            n_particles=8,
            n_if2_iterations=1,
            n_dmop_iterations=1,
            perturbation_scale=scale,
            initial_perturbation_scale=initial_scale,
            optimizer=sgd(0.001),
            evaluation_replicates=2,
        )

    result = run(theta)
    assert bool(result.success) == valid
    if valid:
        assert result.warm_start.completed_iterations == 1
        assert result.refinement.completed_iterations == 1
        assert np.isfinite(result.evaluation_log_likelihoods).all()
    else:
        assert result.warm_start.completed_iterations == 0
        assert result.refinement.completed_iterations == 0
        np.testing.assert_array_equal(result.parameters, theta)
        assert np.isnan(result.evaluation_log_likelihoods).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1},
        {"alpha": float("nan")},
        {"resampling": "adaptive"},
        {"n_particles": 0},
    ],
)
def test_invalid_dmop_configuration(kwargs):
    options = dict(key=jax.random.key(0), n_particles=8)
    options.update(kwargs)
    with pytest.raises(ValueError):
        dmop.log_likelihood(model, jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), **options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cooling_rate": 0},
        {"perturbation_scale": -1},
        {"initial_perturbation_scale": [1.0, 2.0]},
        {"n_iterations": -1},
    ],
)
def test_invalid_if2_configuration(kwargs):
    options = dict(
        key=jax.random.key(0), n_particles=8, n_iterations=2, perturbation_scale=0.1
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        iterated_filtering.fit(
            model, jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), **options
        )


def test_clipping_direction_and_invalid_mask():
    def gaussian_mean(p):
        return (
            lambda k: jnp.array(0.0),
            lambda k, x, y: x,
            lambda old, x, y: -0.5 * (y - p) ** 2,
        )

    options = dict(
        key=jax.random.key(0), n_particles=4, n_iterations=1, optimizer=sgd(0.1)
    )
    result = dmop.fit(
        gaussian_mean,
        jnp.array(0.0),
        jnp.array([10.0]),
        max_gradient_norm=2.0,
        **options,
    )
    np.testing.assert_allclose(result.parameters, 0.2)
    np.testing.assert_allclose(result.gradient_norms, [10.0])
    with pytest.raises(ValueError, match="boolean"):
        dmop.fit(
            gaussian_mean,
            jnp.array(0.0),
            jnp.array([10.0]),
            trainable_mask=1.0,
            **options,
        )


def test_dynamic_invalid_scale_and_ifad_refinement_failure():
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(0)
    run = jax.jit(
        lambda scale: iterated_filtering.fit(
            model,
            theta,
            ys,
            key=key,
            n_particles=8,
            n_iterations=2,
            perturbation_scale=scale,
        )
    )
    failed = run(jnp.array(-1.0))
    assert not failed.success and failed.completed_iterations == 0
    np.testing.assert_array_equal(failed.parameters, theta)
    bad = Optimizer(lambda _: (), lambda g, s, p: (jnp.full_like(p, jnp.nan), s))
    result = ifad.fit(
        model,
        theta,
        ys,
        key=key,
        n_particles=8,
        n_if2_iterations=1,
        n_dmop_iterations=2,
        perturbation_scale=0.01,
        optimizer=bad,
    )
    assert (
        result.warm_start.success
        and not result.refinement.success
        and not result.success
    )
    np.testing.assert_array_equal(result.parameters, result.warm_start.parameters)
    assert np.isnan(result.evaluation_log_likelihoods).all()


def test_batched_ifad_with_multinomial_and_pytree():
    theta = {"dynamic": jnp.array([0.4, -0.7]), "initial": jnp.array(0.2)}

    # Adapt input structure while retaining the same parameter PyTree.
    def input_tree_model(p):
        init, propagate, potential = tree_model(p)
        return (
            init,
            lambda k, x, inp: propagate(k, x, inp["observations"]),
            lambda old, x, inp: potential(old, x, inp["observations"]),
        )

    def run(k):
        return ifad.fit(
            input_tree_model,
            theta,
            {"observations": jnp.array(Y)},
            key=k,
            n_particles=8,
            n_if2_iterations=1,
            n_dmop_iterations=1,
            perturbation_scale=0.01,
            optimizer=sgd(0.01),
            resampling="multinomial",
            evaluation_replicates=2,
        )

    result = jax.jit(jax.vmap(run))(jax.random.split(jax.random.key(1), 2))
    assert np.asarray(result.success).all()
    assert result.parameters["dynamic"].shape == (2, 2)
