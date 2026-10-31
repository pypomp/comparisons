"""Original Dacca integration checks of the candidate interfaces, on CPU."""

import hashlib
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import model as m
import keyed

tfd = tfp.distributions
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT.parents[1] / "outputs/competitor_estimation/tfp_implementation_v1"


def load_original(name, filename):
    p = ROOT / "upstream/tensorflow_probability/python/experimental/mcmc" / filename
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-graph-rewrites", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.disable_graph_rewrites:
        tf.config.optimizer.set_experimental_options(
            {"arithmetic_optimization": False, "remapping": False}
        )
    mode = "controlled_graph" if args.disable_graph_rewrites else "default_graph"
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    started = time.perf_counter()
    model = keyed.Dacca(T=600, recompute=True)
    initial = tf.constant(m.to_est(m.DATA["theta_published"]), tf.float32)
    if initial.shape.rank == 1:
        initial = initial[None]
    J = 8
    members = tf.constant([[0, 0]], tf.int32)
    init_scale = tf.constant([0.16 if n in m.IVPS else 0.0 for n in m.FREE])
    process = tf.constant([0.0 if n in m.IVPS else 0.02 for n in m.FREE])
    observations = tf.broadcast_to(model.y[:, None], [601, 1])
    prior = lambda p: tfd.Independent(tfd.Deterministic(model.initial(m.natural(p))), 1)
    transition = lambda t, x, p: keyed.Transition(model, x, m.natural(p), t, members)
    observation = lambda t, x, p: m.Observation(model, x, m.natural(p), t)
    old_if2 = keyed.if2_step(model, J, 0.5, False, return_cloud=True)

    @tf.function(autograph=False)
    def new_if2(cloud, index, key):
        return tfp.experimental.mcmc.iterated_filter_step(
            cloud,
            observations,
            prior,
            transition,
            observation,
            process,
            initial_scale=init_scale,
            iteration=index,
            cooling_rate=0.5,
            cooling_interval=50.0,
            within_pass_cooling=True,
            normal_sample_fn=lambda p, k: keyed.keyed_normal(p, k, members, 23),
            resample_fn=keyed.resampler(members),
            seed=key,
        )

    cloud = tf.broadcast_to(initial[None], [J, 1, 23])
    old_cloud = cloud
    comparisons = []
    for i in range(2):
        key = tf.constant([20261101, i])
        idx = tf.constant(float(i))
        expected, old_ll = old_if2(old_cloud, idx, key, members)
        actual, new_ll = new_if2(cloud, idx, key)
        comparisons.append(
            {
                "pass": i,
                "max_parameter_difference": float(
                    tf.reduce_max(tf.abs(actual - expected))
                ),
                "old_loglik": old_ll.numpy().tolist(),
                "new_loglik": new_ll.numpy().tolist(),
                "exact_cloud": bool(tf.reduce_all(actual == expected)),
                "exact_loglik": bool(tf.reduce_all(old_ll == new_ll)),
            }
        )
        tf.debugging.assert_all_finite(actual, "nonfinite IF2 cloud")
        tf.debugging.assert_all_finite(new_ll, "nonfinite IF2 likelihood")
        if args.disable_graph_rewrites:
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
            np.testing.assert_array_equal(new_ll.numpy(), old_ll.numpy())
        cloud, old_cloud = actual, expected
        print("IF2 pass", i, comparisons[-1], flush=True)
    pristine_smc = load_original(
        "dacca_original_smc", "sequential_monte_carlo_kernel.py"
    )
    pristine_pf = load_original("dacca_original_pf", "particle_filter.py")
    pristine_pf.smc_kernel = pristine_smc

    def kernel(function, discount=None, normalize_loss=False):
        @tf.function(autograph=False)
        def run(p, key):
            with tf.GradientTape() as tape:
                tape.watch(p)
                theta = m.natural(p)
                kwargs = {} if discount is None else {"gradient_discount": discount}
                inc = function(
                    observations,
                    tfd.Independent(tfd.Deterministic(model.initial(theta)), 1),
                    lambda t, x: keyed.Transition(model, x, theta, t, members),
                    lambda t, x: m.Observation(model, x, theta, t),
                    J,
                    resample_fn=keyed.resampler(members),
                    resample_criterion_fn=m.always_resample,
                    trace_fn=lambda s, r: r.incremental_log_marginal_likelihood,
                    seed=key,
                    **kwargs,
                )
                # Compare the same differentiated objective, including where
                # its scaling occurs. Dividing an already computed gradient
                # is algebraically equivalent but not the same float32 graph.
                total = (tf.reduce_sum(-tf.reduce_sum(inc, axis=0) / model.T)
                         if normalize_loss else tf.reduce_sum(inc))
            return inc, tape.gradient(total, p)

        return run

    key = tf.constant([20261102, 0])
    baseline = kernel(pristine_pf.particle_filter)(initial, key)
    gradients = {}
    for alpha in [0.0, 0.97, 1.0]:
        inc, g = kernel(tfp.experimental.mcmc.particle_filter, alpha)(initial, key)
        np.testing.assert_array_equal(inc.numpy(), baseline[0].numpy())
        tf.debugging.assert_all_finite(g, "nonfinite gradient")
        if alpha == 1:
            np.testing.assert_array_equal(g.numpy(), baseline[1].numpy())
        gradients[str(alpha)] = {
            "forward_exact": True,
            "gradient_finite": True,
            "max_gradient_difference_from_alpha1": float(
                tf.reduce_max(tf.abs(g - baseline[1]))
            ),
        }
        if alpha == 0.97:
            discounted = kernel(tfp.experimental.mcmc.particle_filter, alpha)
        print("alpha", alpha, "checked", flush=True)
    # Compare the previous temporary-wrapper mechanism at one fixed point.
    old_gradient = keyed.gradient(model, J, "discounted", 0.97, False)
    loss, old_g = old_gradient(initial, key, members)
    inc, new_g = discounted(initial, key)
    old_diff = float(tf.reduce_max(tf.abs(old_g + new_g / 600.0)))
    np.testing.assert_allclose(
        old_g.numpy(), (-new_g / 600.0).numpy(), rtol=2e-5, atol=1e-5
    )
    _, matching_g = kernel(tfp.experimental.mcmc.particle_filter, 0.97,
                           normalize_loss=True)(initial, key)
    matched_diff = float(tf.reduce_max(tf.abs(old_g - matching_g)))
    np.testing.assert_array_equal(old_g.numpy(), matching_g.numpy())
    parameters = tf.Variable(tf.reduce_mean(cloud, 0))
    optimizer = tf.keras.optimizers.Adam(1e-6)
    before = parameters.numpy().copy()
    # The GradientTape is inside the PF function; use its returned gradient.
    train_inc, train_g = discounted(parameters.read_value(), tf.constant([20261103, 0]))
    tf.debugging.assert_all_finite(train_g, "nonfinite continuation gradient")
    optimizer.apply_gradients([(-train_g / 600.0, parameters)])
    tf.debugging.assert_all_finite(parameters, "nonfinite continuation point")
    # Score with the unchanged original PF and independent keys, with no tape.
    score_function = kernel(pristine_pf.particle_filter)
    raw = []
    for i in range(2):
        inc, _ = score_function(parameters.read_value(), tf.constant([20261104, i]))
        raw.append(float(tf.reduce_sum(inc)))
    assert np.isfinite(raw).all()
    result = {
        "passed": True,
        "purpose": "API integration; not a new estimation comparison",
        "graph_mode": mode,
        "optimizer_options": tf.config.optimizer.get_experimental_options(),
        "IF2_bitwise_identity_to_frozen": all(
            x["exact_cloud"] and x["exact_loglik"] for x in comparisons
        ),
        "T": 600,
        "Euler_steps_per_month": 20,
        "free_parameters": 23,
        "particles": J,
        "device": "CPU",
        "dtype": "float32",
        "IF2_passes": 2,
        "Adam_updates": 1,
        "learning_rate": 1e-6,
        "independent_score_repetitions": 2,
        "MCSE_assessed": False,
        "if2_frozen_formal_comparison": comparisons,
        "gradient_checks": gradients,
        "old_wrapper_loss": loss.numpy().tolist(),
        "old_wrapper_loss_gradient_max_difference": old_diff,
        "same_scaled_objective_gradient_max_difference": matched_diff,
        "same_scaled_objective_gradient_bitwise_equal": True,
        "historical_trajectory_identity_claimed": False,
        "Adam_max_parameter_change": float(np.max(np.abs(parameters.numpy() - before))),
        "score_raw": raw,
        "elapsed_seconds_including_graph_builds": time.perf_counter() - started,
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "candidate_path": tfp.__file__,
        "frozen_copy_hashes": {
            n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
            for n in ["model.py", "keyed.py", "input/frozen.npz"]
        },
    }
    (args.output or (OUT / ("dacca_check_" + mode + ".json"))).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
