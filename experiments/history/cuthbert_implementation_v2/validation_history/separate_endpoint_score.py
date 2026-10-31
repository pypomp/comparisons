"""Check public estimation APIs with frozen Dacca callbacks; not a benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import dacca_model as model
from cuthbert import filter as run_filter
from cuthbert.smc import dmop, ifad, iterated_filtering
from cuthbert.smc import particle_filter as pf
from cuthbertlib.resampling import autodiff, systematic


def main():
    """Compare native filtering, public IF2 and composed public IFAD calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    parameters = jnp.asarray(model.PUBLISHED, dtype=jnp.float32)
    observations = model.inputs(600)
    particles = 32
    root = jax.random.key(20261030)
    pf_key, if2_key, ifad_key = jax.random.split(root, 3)
    checks = {}
    # Fixed before running. This checks finite-precision implementation agreement,
    # not Monte Carlo accuracy or a fitted endpoint's scientific quality.
    rtol, atol = 1e-5, 1e-4

    def ordinary(p, key, resampler):
        obj = pf.build_filter(*model.factory(p), particles, resampler)
        ik, fk = jax.random.split(key)
        return run_filter(
            obj, observations, obj.init_prepare(key=ik), key=fk
        ).log_normalizing_constant[-1]

    native_value = jax.jit(lambda p: ordinary(p, pf_key, systematic.resampling))(
        parameters
    )
    corrected = autodiff.stop_gradient_decorator(systematic.resampling)
    native_adpf_value, native_adpf_gradient = jax.jit(
        jax.value_and_grad(lambda p: ordinary(p, pf_key, corrected))
    )(parameters)
    assert np.isfinite(native_adpf_gradient).all()
    for alpha in (0.0, 0.97, 1.0):
        value, gradient = jax.jit(
            jax.value_and_grad(
                lambda p: dmop.log_likelihood(
                    model.factory,
                    p,
                    observations,
                    key=pf_key,
                    n_particles=particles,
                    alpha=alpha,
                )
            )
        )(parameters)
        np.testing.assert_allclose(value, native_value, rtol=rtol, atol=atol)
        assert np.isfinite(gradient).all()
        entry = {"forward_matches_original_pf": True, "finite_gradient": True}
        if alpha == 1:
            np.testing.assert_allclose(value, native_adpf_value, rtol=rtol, atol=atol)
            np.testing.assert_allclose(
                gradient, native_adpf_gradient, rtol=rtol, atol=atol
            )
            entry["gradient_matches_original_adpf"] = True
            entry["max_gradient_absolute_difference"] = float(
                jnp.max(jnp.abs(gradient - native_adpf_gradient))
            )
        checks[f"alpha_{alpha}"] = entry

    no_perturbation = jax.jit(
        lambda p: iterated_filtering.fit(
            model.factory,
            p,
            observations,
            key=if2_key,
            n_particles=particles,
            n_iterations=1,
            perturbation_scale=0.0,
        )
    )(parameters)
    pass_key = jax.random.split(if2_key, 1)[0]
    _, init_key, filter_key, _ = jax.random.split(pass_key, 4)

    def fixed_parameter_filter(p):
        init, propagate, potential = model.factory(p)

        def process_with_if2_key(key, state, data):
            # IF2 consumes separate parameter-noise and process subkeys even at
            # zero perturbation. Couple the process draws, not just the root key.
            return propagate(jax.random.split(key)[1], state, data)

        obj = pf.build_filter(
            init, process_with_if2_key, potential, particles, systematic.resampling
        )
        return run_filter(
            obj, observations, obj.init_prepare(key=init_key), key=filter_key
        ).log_normalizing_constant[-1]

    expected = jax.jit(fixed_parameter_filter)(parameters)
    assert bool(no_perturbation.success)
    np.testing.assert_allclose(
        no_perturbation.perturbed_log_likelihoods[0], expected, rtol=rtol, atol=atol
    )
    np.testing.assert_array_equal(
        no_perturbation.parameter_swarm,
        np.broadcast_to(np.asarray(parameters), (particles, len(parameters))),
    )
    checks["zero_perturbation_if2_matches_original_pf"] = True

    optimizer = optax.adam(1e-5)
    options = dict(
        n_particles=particles,
        perturbation_scale=jnp.asarray(model.PROCESS_SCALE) * 0.01,
        initial_perturbation_scale=jnp.asarray(model.INITIAL_SCALE) * 0.01,
    )
    result = jax.jit(
        lambda p: ifad.fit(
            model.factory,
            p,
            observations,
            key=ifad_key,
            n_if2_iterations=1,
            n_dmop_iterations=1,
            alpha=0.97,
            optimizer=optimizer,
            max_gradient_norm=100.0,
            evaluation_particles=particles,
            evaluation_replicates=2,
            **options,
        )
    )(parameters)
    assert bool(result.success)
    assert int(result.warm_start.completed_iterations) == 1
    assert int(result.refinement.completed_iterations) == 1
    warm_key, gradient_key, evaluation_key = jax.random.split(ifad_key, 3)
    warm = jax.jit(
        lambda p: iterated_filtering.fit(
            model.factory, p, observations, key=warm_key, n_iterations=1, **options
        )
    )(parameters)
    refine = jax.jit(
        lambda p: dmop.fit(
            model.factory,
            p,
            observations,
            key=gradient_key,
            n_particles=particles,
            n_iterations=1,
            optimizer=optimizer,
            alpha=0.97,
            max_gradient_norm=100.0,
        )
    )(warm.parameters)
    np.testing.assert_allclose(result.parameters, refine.parameters, rtol=rtol, atol=atol)
    scores = jax.jit(
        jax.vmap(lambda k: ordinary(refine.parameters, k, systematic.resampling))
    )(jax.random.split(evaluation_key, 2))
    np.testing.assert_allclose(
        result.evaluation_log_likelihoods, scores, rtol=rtol, atol=atol
    )
    checks["public_ifad_matches_separate_public_calls"] = True
    checks["public_ifad_scoring_matches_original_pf"] = True
    files = [Path(model.__file__), Path(model.__file__).parent / "input/frozen.npz"]
    report = {
        "passed": True,
        "purpose": "Full-length Dacca public-API integration check only; not an estimation comparison.",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "modules": {m.__name__: m.__file__ for m in (pf, dmop, ifad, iterated_filtering)},
        "observations": 600,
        "euler_steps_per_observation": 20,
        "free_parameters": len(parameters),
        "particles": particles,
        "if2_iterations": 1,
        "gradient_updates": 1,
        "score_replicates": 2,
        "score_precision_assessed": False,
        "rtol": rtol,
        "atol": atol,
        "checks": checks,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
