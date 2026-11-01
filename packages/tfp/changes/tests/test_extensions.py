"""Targeted numerical checks against pristine code and explicit recursions."""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from if2_and_discount import model_functions, always_resample

REL = "tensorflow_probability/python/experimental/mcmc"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, ROOT.parent / "upstream" / REL / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_smc = load("original_smc", "sequential_monte_carlo_kernel.py")
original_pf = load("original_pf", "particle_filter.py")
# Only this isolated pristine module's dependency is bound to the pristine
# kernel. No installed/candidate module or global resampling function changes.
original_pf.smc_kernel = original_smc
smc = importlib.import_module(
    "tensorflow_probability.python.experimental.mcmc.sequential_monte_carlo_kernel"
)
pf = importlib.import_module(
    "tensorflow_probability.python.experimental.mcmc.particle_filter"
)
wr = importlib.import_module(
    "tensorflow_probability.python.experimental.mcmc.weighted_resampling"
)


def manual_pass(
    cloud,
    y,
    prior,
    transition,
    observation,
    process,
    initial,
    index,
    rate,
    interval,
    within,
    resampler,
    key,
):
    """No IF2 or SMC calls: explicit normalization, index selection and time loop."""
    k = tf.random.experimental.stateless_split(key, 2)
    params = cloud + tf.random.stateless_normal(
        tf.shape(cloud), k[0], dtype=cloud.dtype
    ) * initial * rate ** (index / interval)
    state = prior(params).sample(seed=tf.random.experimental.stateless_fold_in(k[0], 1))
    weight = tf.fill(
        tf.shape(params)[:-1], -tf.math.log(tf.cast(tf.shape(cloud)[0], cloud.dtype))
    ) + observation(0, state, params).log_prob(y[0])
    total = tf.zeros(tf.shape(cloud)[1:-1], cloud.dtype)
    seed = k[1]
    for t in range(y.shape[0]):
        one, seed = samplers.split_seed(seed)
        proposal_key, resample_key = samplers.split_seed(one)
        if t:
            step_keys = tf.random.experimental.stateless_split(proposal_key, 2)
            exponent = (
                index + ((t - 1) / (y.shape[0] - 1) if within else 0.0)
            ) / interval
            params = (
                params
                + tf.random.stateless_normal(
                    tf.shape(params), step_keys[0], dtype=cloud.dtype
                )
                * process
                * rate**exponent
            )
            state = transition(t - 1, state, params).sample(seed=step_keys[1])
            weight = weight + observation(t, state, params).log_prob(y[t])
        norm = tf.nn.log_softmax(weight, axis=0)
        total = total + weight[0] - norm[0]
        pair, _, weight = wr.resample(
            {"x": state, "p": params}, weight, resampler, seed=resample_key
        )
        state, params = pair["x"], pair["p"]
    return params, total


class ExtensionsTest(unittest.TestCase):
    def check_same(self, a, b, atol=0):
        for x, y in zip(tf.nest.flatten(a), tf.nest.flatten(b)):
            np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=0, atol=atol)

    def test_default_and_one_match_original(self):
        for dtype in (tf.float32, tf.float64):
            for resampler in (wr.resample_systematic, wr.resample_independent):
                for graph in (False, True):
                    for criterion in (always_resample, smc.ess_below_threshold, None):
                        with self.subTest(
                            dtype=dtype.name,
                            resampler=resampler.__name__,
                            graph=graph,
                            criterion=str(criterion),
                        ):
                            prior, trans, obs = model_functions()
                            p = tf.constant([[0.2, 0.1], [0.4, -0.1]], dtype)
                            y = tf.constant(
                                [[0.1, 0.2], [0.2, 0.5], [0.4, 0.3], [0.6, 0.1]], dtype
                            )

                            def run(function, kw):
                                def compute(p):
                                    with tf.GradientTape() as tape:
                                        tape.watch(p)
                                        result = function(
                                            y,
                                            prior(p),
                                            lambda t, x: trans(t, x, p),
                                            lambda t, x: obs(t, x, p),
                                            7,
                                            resample_fn=resampler,
                                            resample_criterion_fn=criterion,
                                            seed=tf.constant([301, 7]),
                                            **kw,
                                        )
                                        ll = tf.reduce_sum(result[-1])
                                    return result, tape.gradient(ll, p)

                                return (
                                    tf.function(compute, autograph=False)
                                    if graph
                                    else compute
                                )(p)

                            baseline = run(original_pf.particle_filter, {})
                            self.check_same(run(pf.particle_filter, {}), baseline)
                            self.check_same(
                                run(pf.particle_filter, {"gradient_discount": 1.0}),
                                baseline,
                            )

    def test_discount_against_manual_gradient_recursion(self):
        for alpha in (0.0, 0.5, 0.97, 1.0):
            for adaptive in (False, True):
                with self.subTest(alpha=alpha, adaptive=adaptive):

                    def run(manual):
                        theta = tf.constant(0.3, tf.float64)
                        decisions = []
                        with tf.GradientTape() as tape:
                            tape.watch(theta)
                            x = (
                                tf.linspace(
                                    tf.cast(-1.0, theta.dtype),
                                    tf.cast(1.0, theta.dtype),
                                    5,
                                )
                                + 0.2 * theta
                            )
                            weight = -0.5 * (x - theta) ** 2 - tf.math.log(
                                tf.cast(5, theta.dtype)
                            )
                            state = smc.WeightedParticles(x, weight)

                            def proposal(step, s, seed):
                                del seed
                                z = 0.6 * s.particles + 0.1 * theta
                                return smc.WeightedParticles(
                                    z,
                                    s.log_weights
                                    - 0.5
                                    * (z - tf.cast(step, theta.dtype) / 10.0) ** 2
                                    * (1.0 + theta),
                                )

                            def adaptive_criterion(s, particles_dim=0):
                                del particles_dim
                                decision = tf.reduce_mean(s.particles) > 0.065
                                decisions.append(bool(decision))
                                return decision

                            criterion = adaptive_criterion if adaptive else always_resample
                            kernel = smc.SequentialMonteCarlo(
                                proposal,
                                resample_criterion_fn=criterion,
                                gradient_discount=alpha,
                            )
                            result = kernel.bootstrap_results(state)
                            total = tf.zeros([], theta.dtype)
                            for i in range(4):
                                key = tf.constant([407, i])
                                if manual:
                                    if i:
                                        state = proposal(i - 1, state, None)
                                    norm = tf.nn.log_softmax(state.log_weights, axis=0)
                                    total += state.log_weights[0] - norm[0]
                                    _, rs = samplers.split_seed(
                                        samplers.sanitize_seed(key)
                                    )
                                    z, idx, uniform = wr.resample(
                                        state.particles,
                                        tf.stop_gradient(state.log_weights),
                                        wr.resample_systematic,
                                        seed=rs,
                                    )
                                    selected = tf.gather(norm, idx)
                                    corrected = uniform + alpha * (
                                        selected - tf.stop_gradient(selected)
                                    )
                                    choose = criterion(state)
                                    state = smc.WeightedParticles(
                                        tf.where(choose, z, state.particles),
                                        tf.where(choose, corrected, norm),
                                    )
                                else:
                                    state, result = kernel.one_step(
                                        state, result, seed=key
                                    )
                                    total += result.incremental_log_marginal_likelihood
                        if adaptive:
                            self.assertIn(True, decisions)
                            self.assertIn(False, decisions)
                        return total, tape.gradient(total, theta)

                    actual, expected = run(False), run(True)
                    self.check_same(actual, expected, atol=1e-12)
                    self.assertGreater(abs(float(actual[1])), 1e-5)

    def test_if2_against_explicit_time_loop(self):
        for dimension, nonlinear in [(1, False), (3, True)]:
            for within in (False, True):
                for resampler in (wr.resample_systematic, wr.resample_independent):
                    with self.subTest(
                        dimension=dimension, within=within, sampler=resampler.__name__
                    ):
                        p = tf.constant([[0.2, 0.1, 2.0], [0.4, -0.1, 3.0]], tf.float64)
                        cloud = tf.broadcast_to(p[None], [7, 2, 3])
                        y = tf.constant(
                            [[0.1, 0.2], [0.2, 0.5], [0.4, 0.3], [0.6, 0.1]], tf.float64
                        )
                        prior, transition, obs = model_functions(dimension, nonlinear)
                        process = tf.constant([0.05, 0.03, 0.0], tf.float64)
                        initial = tf.constant([0.1, 0.0, 0.0], tf.float64)
                        for i in range(2):
                            key = tf.constant([800, i])
                            kwargs = dict(
                                initial_scale=initial,
                                iteration=float(i),
                                cooling_rate=0.7,
                                cooling_interval=3.0,
                                within_pass_cooling=within,
                                resample_fn=resampler,
                                seed=key,
                            )
                            actual = tfp.experimental.mcmc.iterated_filter_step(
                                cloud, y, prior, transition, obs, process, **kwargs
                            )
                            expected = manual_pass(
                                cloud,
                                y,
                                prior,
                                transition,
                                obs,
                                process,
                                initial,
                                float(i),
                                0.7,
                                3.0,
                                within,
                                resampler,
                                key,
                            )
                            self.check_same(actual, expected, atol=2e-14)
                            cloud = actual[0]
                        self.check_same(
                            cloud[..., 2], tf.broadcast_to(p[None, :, 2], [7, 2])
                        )

    def test_multi_pass_graph_and_frozen_coordinates(self):
        p = tf.constant([[0.2, 0.1, 2.0], [0.4, -0.1, 3.0]], tf.float32)
        y = tf.zeros([4, 2])
        prior, trans, obs = model_functions()
        root = tf.constant([900, 1])

        def run(p):
            return tfp.experimental.mcmc.iterated_filter(
                p,
                y,
                prior,
                trans,
                obs,
                7,
                3,
                [0.05, 0.03, 0.0],
                initial_scale=[0.1, 0.0, 0.0],
                seed=root,
            )

        result = tf.function(run, autograph=False)(p)
        cloud = tf.broadcast_to(p[None], [7, 2, 3])
        history = [p]
        ll = []
        for i in range(3):
            cloud, score = tfp.experimental.mcmc.iterated_filter_step(
                cloud,
                y,
                prior,
                trans,
                obs,
                [0.05, 0.03, 0.0],
                initial_scale=[0.1, 0.0, 0.0],
                iteration=float(i),
                seed=tf.random.experimental.stateless_fold_in(root, i),
            )
            history.append(tf.reduce_mean(cloud, 0))
            ll.append(score)
        # Eager-vs-graph kernels can differ by float32 rounding.
        self.check_same(result.parameter_particles, cloud, atol=2e-6)
        self.check_same(result.parameter_history, tf.stack(history), atol=2e-6)
        self.check_same(result.perturbed_log_likelihoods, tf.stack(ll), atol=2e-6)

    def test_input_and_failure_checks(self):
        for alpha in (-0.1, 1.1, float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                smc.SequentialMonteCarlo(lambda *a: None, gradient_discount=alpha)
        with self.assertRaises(ValueError):
            smc.SequentialMonteCarlo(
                lambda *a: None, gradient_discount=0.9, unbiased_gradients=False
            )
        with self.assertRaises(ValueError):
            smc.SequentialMonteCarlo(
                lambda *a: None, gradient_discount=0.9, resample_criterion_fn=None
            )
        prior, trans, obs = model_functions()
        cloud = tf.ones([4, 2, 2])
        y = tf.zeros([3, 2])
        for scale in (-0.1, float("nan")):
            with (
                self.subTest(scale=scale),
                self.assertRaises(tf.errors.InvalidArgumentError),
            ):
                tfp.experimental.mcmc.iterated_filter_step(
                    cloud, y, prior, trans, obs, scale, seed=[2, 1]
                )
        with self.assertRaises(ValueError):
            tfp.experimental.mcmc.particle_filter(
                y,
                prior(cloud[0]),
                lambda t, x: trans(t, x, cloud[0]),
                lambda t, x: obs(t, x, cloud[0]),
                4,
                proposal_fn=lambda *a: None,
                gradient_discount=0.9,
            )
        with self.assertRaises(tf.errors.InvalidArgumentError):
            tfp.experimental.mcmc.iterated_filter_step(
                cloud,
                tf.fill([3, 2], float("nan")),
                prior,
                trans,
                obs,
                0.1,
                seed=[2, 1],
            )

    def test_zero_passes(self):
        prior, trans, obs = model_functions()
        p = tf.constant([[0.1234567, 0.3], [0.7, 0.1]])
        result = tfp.experimental.mcmc.iterated_filter(
            p, tf.zeros([3, 2]), prior, trans, obs, 7, 0, 0.1, seed=[3, 1]
        )
        self.check_same(result.parameters, p)
        self.assertEqual(tuple(result.perturbed_log_likelihoods.shape), (0, 2))

    def test_nested_state_and_no_batch_dimension(self):
        tfd = tfp.distributions
        prior = lambda p: tfd.JointDistributionNamed(
            {
                "a": tfd.Normal(p[..., 0], tf.cast(0.1, p.dtype)),
                "b": tfd.Normal(p[..., 1], tf.cast(0.2, p.dtype)),
            }
        )
        transition = lambda t, x, p: tfd.JointDistributionNamed(
            {
                "a": tfd.Normal(0.6 * x["a"] + p[..., 0], tf.cast(0.2, p.dtype)),
                "b": tfd.Normal(0.4 * x["b"] + p[..., 1], tf.cast(0.1, p.dtype)),
            }
        )
        observation = lambda t, x, p: tfd.Normal(x["a"] + x["b"], tf.cast(0.7, p.dtype))
        cloud = tf.broadcast_to(tf.constant([[0.2, 0.1]], tf.float64), [5, 2])
        y = tf.constant([0.3, 0.1, -0.2], tf.float64)
        process = tf.constant([0.03, 0.02], tf.float64)
        start = tf.constant([0.1, 0.0], tf.float64)
        seed = tf.constant([701, 3])
        original_resample = wr.resample
        actual = tfp.experimental.mcmc.iterated_filter_step(
            cloud,
            y,
            prior,
            transition,
            observation,
            process,
            initial_scale=start,
            cooling_rate=0.7,
            seed=seed,
        )
        expected = manual_pass(
            cloud,
            y,
            prior,
            transition,
            observation,
            process,
            start,
            0.0,
            0.7,
            1.0,
            False,
            wr.resample_systematic,
            seed,
        )
        self.check_same(actual, expected, atol=2e-14)
        self.assertIs(wr.resample, original_resample)


if __name__ == "__main__":
    unittest.main(verbosity=2)
