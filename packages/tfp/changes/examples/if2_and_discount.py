"""Execute the proposed IF2 API, native TFP PF and external Keras Adam."""

import json
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def model_functions(dimension=1, nonlinear=False):
    def prior(parameters):
        loc = tf.broadcast_to(
            parameters[..., :1], tf.concat([tf.shape(parameters)[:-1], [dimension]], 0)
        )
        return tfd.Independent(tfd.Normal(loc, tf.cast(0.2, parameters.dtype)), 1)

    def transition(step, state, parameters):
        del step
        previous = tf.sin(state) if nonlinear else state
        loc = 0.6 * previous + parameters[..., 1:2]
        return tfd.Independent(tfd.Normal(loc, tf.cast(0.3, state.dtype)), 1)

    def observation(step, state, parameters):
        del step, parameters
        return tfd.Normal(tf.reduce_sum(state, -1), tf.cast(0.7, state.dtype))

    return prior, transition, observation


def always_resample(state, particles_dim=0):
    return tf.ones_like(
        tf.reduce_sum(state.log_weights, particles_dim, keepdims=True), tf.bool
    )


def main():
    initial = tf.constant([[0.2, 0.1], [0.4, -0.1]], tf.float32)
    y = tf.constant([[0.1, 0.2], [0.2, 0.5], [0.4, 0.3], [0.6, 0.1]], tf.float32)
    prior, transition, observation = model_functions()
    warm = tfp.experimental.mcmc.iterated_filter(
        initial,
        y,
        prior,
        transition,
        observation,
        16,
        3,
        [0.05, 0.05],
        initial_scale=[0.1, 0.0],
        seed=tf.constant([20261029, 1]),
    )
    parameters = tf.Variable(warm.parameters)
    optimizer = tf.keras.optimizers.Adam(0.005)

    @tf.function(autograph=False)
    def loglik(parameters, seed):
        increments = tfp.experimental.mcmc.particle_filter(
            y,
            prior(parameters),
            lambda t, x: transition(t, x, parameters),
            lambda t, x: observation(t, x, parameters),
            16,
            resample_criterion_fn=always_resample,
            gradient_discount=0.97,
            trace_fn=lambda s, r: r.incremental_log_marginal_likelihood,
            seed=seed,
        )
        return tf.reduce_sum(increments, 0)

    for i in range(3):
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(loglik(parameters, tf.constant([20261030, i]))) / 4.0
        gradient = tape.gradient(loss, parameters)
        tf.debugging.assert_all_finite(gradient, "nonfinite example gradient")
        optimizer.apply_gradients([(gradient, parameters)])
    tf.debugging.assert_all_finite(parameters, "nonfinite example endpoint")
    raw = tf.stack([loglik(parameters, tf.constant([20261031, i])) for i in range(4)])
    tf.debugging.assert_all_finite(raw, "nonfinite example scores")
    print(
        json.dumps(
            {
                "completed": True,
                "IF2_passes": 3,
                "Keras_Adam_updates": 3,
                "independent_score_repetitions": 4,
                "logmeanlik": (tf.reduce_logsumexp(raw, 0) - tf.math.log(4.0))
                .numpy()
                .tolist(),
                "MCSE_assessed": False,
                "package_path": tfp.__file__,
            }
        )
    )


if __name__ == "__main__":
    main()
