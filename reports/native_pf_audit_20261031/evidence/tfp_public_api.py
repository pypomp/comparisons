"""Dacca adapter using the new public interfaces, without monkeypatching TFP."""
import importlib.util
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import model as m
import keyed

ROOT = Path(__file__).resolve().parent
tfd = tfp.distributions


def pristine_pf():
    base = ROOT / 'upstream/tensorflow_probability/python/experimental/mcmc'
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, base / filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    kernel = load('pristine_tfp_smc', 'sequential_monte_carlo_kernel.py')
    pf = load('pristine_tfp_pf', 'particle_filter.py')
    pf.smc_kernel = kernel
    return pf.particle_filter


def pf_kernel(model, J, alpha=1., pristine=False, differentiate=False):
    function = pristine_pf() if pristine else tfp.experimental.mcmc.particle_filter
    @tf.function(autograph=False)
    def run(e, root, members):
        with tf.GradientTape() as tape:
            tape.watch(e)
            theta = m.natural(e)
            observations = tf.broadcast_to(model.y[:, None], [model.T + 1, tf.shape(e)[0]])
            kwargs = {} if pristine else {'gradient_discount': alpha}
            inc = function(
                observations,
                tfd.Independent(tfd.Deterministic(model.initial(theta)), 1),
                lambda t, x: keyed.Transition(model, x, theta, t, members),
                lambda t, x: m.Observation(model, x, theta, t), J,
                resample_fn=keyed.resampler(members),
                resample_criterion_fn=m.always_resample, unbiased_gradients=True,
                trace_fn=lambda s, r: r.incremental_log_marginal_likelihood,
                seed=root, **kwargs)
            loss = -tf.reduce_sum(inc, 0) / model.T
            total = tf.reduce_sum(loss)
        if differentiate:
            return inc, tape.gradient(total, e)
        return inc
    return run


def if2_kernel(model, J):
    init = tf.constant([.16 if n in m.IVPS else 0. for n in m.FREE], tf.float32)
    process = tf.constant([0. if n in m.IVPS else .02 for n in m.FREE], tf.float32)
    @tf.function(autograph=False)
    def run(cloud, index, root, members):
        observations = tf.broadcast_to(model.y[:, None], [model.T + 1, tf.shape(cloud)[1]])
        return tfp.experimental.mcmc.iterated_filter_step(
            cloud, observations,
            lambda p: tfd.Independent(tfd.Deterministic(model.initial(m.natural(p))), 1),
            lambda t, x, p: keyed.Transition(model, x, m.natural(p), t, members),
            lambda t, x, p: m.Observation(model, x, m.natural(p), t),
            process, initial_scale=init, iteration=index, cooling_rate=.5,
            cooling_interval=50., within_pass_cooling=True,
            normal_sample_fn=lambda p, k: keyed.keyed_normal(p, k, members, 23),
            resample_fn=keyed.resampler(members), seed=root)
    return run


def gradient_kernel(model, J, alpha=1.):
    kernel = pf_kernel(model, J, alpha=alpha, differentiate=True)
    @tf.function(autograph=False)
    def run(e, root, members):
        inc, grad = kernel(e, root, members)
        return -tf.reduce_sum(inc, 0) / model.T, grad
    return run


class Adam:
    """Native Keras coordinate-specific Adam, preserving per-row failures."""
    def __init__(self, model, e, J, alpha, active=None, kernel=None):
        self.B = int(e.shape[0])
        self.coords = [tf.Variable(e[:, i]) for i in range(23)]
        self.optimizers = [tf.keras.optimizers.Adam(.001, beta_1=.9, beta_2=.999,
                           epsilon=1e-8) for _ in range(23)]
        for opt, var in zip(self.optimizers, self.coords):
            opt.build([var])
        self.active = tf.Variable(tf.ones([self.B], tf.bool) if active is None else active, trainable=False)
        self.accepted = tf.Variable(tf.zeros([self.B], tf.int32), trainable=False)
        self.kernel = kernel or gradient_kernel(model, J, alpha)
        @tf.function(autograph=False)
        def step(root, members, rates):
            previous = [tf.identity(x) for x in self.coords]
            old_states = [[tf.identity(v) for v in opt.variables] for opt in self.optimizers]
            loss, g = self.kernel(tf.stack(previous, -1), root, members)
            valid = tf.math.is_finite(loss) & tf.reduce_all(tf.math.is_finite(g), 1)
            active = self.active & valid
            clean = tf.where(active[:, None], g, tf.zeros_like(g))
            for i, (opt, var) in enumerate(zip(self.optimizers, self.coords)):
                opt.learning_rate.assign(rates[i])
                opt.apply_gradients([(clean[:, i], var)])
            proposal = tf.stack(self.coords, -1)
            valid_point = tf.reduce_all(tf.math.is_finite(proposal), 1) & tf.reduce_all(tf.math.is_finite(m.natural(proposal)), 1)
            for opt in self.optimizers:
                for v in opt.variables:
                    if v.shape.rank == 1:
                        valid_point &= tf.math.is_finite(v)
            active &= valid_point
            for i, (opt, var) in enumerate(zip(self.optimizers, self.coords)):
                var.assign(tf.where(active, var, previous[i]))
                for v, old in zip(opt.variables, old_states[i]):
                    if v.shape.rank == 1:
                        v.assign(tf.where(active, v, old))
            self.active.assign(active)
            self.accepted.assign_add(tf.cast(active, tf.int32))
            return tf.stack(self.coords, -1), loss, g, active
        self.step = step

    def parameters(self):
        return tf.stack(self.coords, -1).numpy()

    def snapshot(self):
        state = {'parameters': self.parameters(), 'active': self.active.numpy(),
                 'accepted': self.accepted.numpy()}
        for i, opt in enumerate(self.optimizers):
            for j, var in enumerate(opt.variables):
                state[f'opt_{i}_{j}'] = var.numpy()
        return state

    def restore(self, state):
        for i, var in enumerate(self.coords):
            var.assign(state['parameters'][:, i])
        for i, opt in enumerate(self.optimizers):
            for j, var in enumerate(opt.variables):
                var.assign(state[f'opt_{i}_{j}'])
        self.active.assign(state['active'])
        self.accepted.assign(state['accepted'])
