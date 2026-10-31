"""Locate forward differences without changing the frozen comparison code."""

from pathlib import Path
import json
import os
import tensorflow as tf
import tensorflow_probability as tfp
import model as m
import keyed

tf.config.threading.set_intra_op_parallelism_threads(1)
if os.environ.get("TFP_DIAGNOSTIC_NO_ARITHMETIC") == "1":
    tf.config.optimizer.set_experimental_options(
        {"arithmetic_optimization": False, "remapping": False}
    )
initial = tf.constant(m.to_est(m.DATA["theta_published"]), tf.float32)
if initial.shape.rank == 1:
    initial = initial[None]
members = tf.constant([[0, 0]], tf.int32)
J = 8
init_scale = tf.constant([0.16 if n in m.IVPS else 0.0 for n in m.FREE])
process = tf.constant([0.0 if n in m.IVPS else 0.02 for n in m.FREE])
rows = []
for T in [1, 3, 600]:
    model = keyed.Dacca(T=T, recompute=True)

    class SameShapeTransition(keyed.Transition):
        def _sample_n(self, n, seed=None):
            tf.debugging.assert_equal(n, 1)
            noise = keyed.keyed_normal(self.state, seed, self.members, 20)
            value = self.model._interval(
                self.state,
                self.theta,
                noise,
                tf.gather(self.model.cov, self.t),
                tf.gather(self.model.dt, self.t),
            )
            return value[None]

    prior = lambda p: tfp.distributions.Independent(
        tfp.distributions.Deterministic(model.initial(m.natural(p))), 1
    )
    observation = lambda t, x, p: m.Observation(model, x, m.natural(p), t)
    cloud = tf.broadcast_to(initial[None], [J, 1, 23])
    key = tf.constant([20261101, 0])
    idx = tf.constant(0.0)
    old = keyed.if2_step(model, J, 0.5, False, return_cloud=True)(
        cloud, idx, key, members
    )
    for cls in [keyed.Transition, SameShapeTransition]:

        @tf.function(autograph=False)
        def new(cloud, idx, key):
            return tfp.experimental.mcmc.iterated_filter_step(
                cloud,
                tf.broadcast_to(model.y[:, None], [T + 1, 1]),
                prior,
                lambda t, x, p: cls(model, x, m.natural(p), t, members),
                observation,
                process,
                initial_scale=init_scale,
                iteration=idx,
                cooling_rate=0.5,
                cooling_interval=50.0,
                within_pass_cooling=True,
                normal_sample_fn=lambda p, k: keyed.keyed_normal(p, k, members, 23),
                resample_fn=keyed.resampler(members),
                seed=key,
            )

        actual = new(cloud, idx, key)
        row = {
            "T": T,
            "distribution": cls.__name__,
            "max_parameter_diff": float(tf.reduce_max(tf.abs(actual[0] - old[0]))),
            "max_likelihood_diff": float(tf.reduce_max(tf.abs(actual[1] - old[1]))),
            "old_ll": old[1].numpy().tolist(),
            "new_ll": actual[1].numpy().tolist(),
        }
        rows.append(row)
        print(row, flush=True)
Path(__file__).with_name(
    "if2_diagnostic_no_arithmetic.json"
    if os.environ.get("TFP_DIAGNOSTIC_NO_ARITHMETIC") == "1"
    else "if2_diagnostic.json"
).write_text(json.dumps(rows, indent=2) + "\n")
