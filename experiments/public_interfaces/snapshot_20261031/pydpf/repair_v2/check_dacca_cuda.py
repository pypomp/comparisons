"""Check frozen Dacca equations with the proposed PyDPF interfaces on CUDA."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import pydpf
import dacca_model as frozen
import api_adapter


def factory(model, parameters):
    """Regular PyDPF model adapter supporting shared or per-particle parameters."""
    theta = frozen.natural(parameters)

    class Prior(pydpf.Module):
        def sample(self, batch_size, n_particles, **data):
            state = model.initial(theta)
            if state.ndim == 2:
                state = state[:, None, :].expand(batch_size, n_particles, -1)
            return model.interval(state, theta, 0)

    class Dynamic(pydpf.Module):
        def sample(self, prev_state, t, **data):
            return model.interval(prev_state, theta, t)

    class Observation(pydpf.Module):
        def fitness(self, state, observation, **data):
            return model.observation(state, theta, observation)

    return pydpf.FilteringModel(
        prior_model=Prior(), dynamic_model=Dynamic(), observation_model=Observation()
    )


def score(model, parameters, particles, seed, alpha=1.0):
    model.generator.manual_seed(seed)
    pf = pydpf.StopGradientDPF(
        SSM=factory(model, parameters),
        resampling_generator=model.generator,
        alpha=alpha,
    )
    pf.update()
    factors = pf(
        n_particles=particles,
        time_extent=model.T - 1,
        aggregation_function=pydpf.LogLikelihoodFactors(),
        observation=model.y,
    )
    assert factors.shape[0] == model.T
    return factors.sum(0).reshape(len(parameters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU required; CPU fallback prohibited"
    torch.set_num_threads(1)
    started = time.perf_counter()
    particles, iterations = 8, 2
    # Published parameters are frozen in the same file as the model/data inputs.
    initial = torch.tensor(frozen.to_est(frozen.DATA["theta_published"]), device="cuda:0")
    model = frozen.Dacca(device="cuda:0", T=600, checkpoint_intervals=True)
    original = frozen.Dacca(device="cuda:0", T=600, checkpoint_intervals=True)
    start_seed = 123001
    original_history, original_ll = frozen.if2(
        original, initial, particles, iterations, cooling=0.5, seed=start_seed
    )
    model.generator.manual_seed(start_seed)
    warm = pydpf.IteratedFiltering(
        lambda p: factory(model, p),
        generator=model.generator,
        perturbation_scale=initial.new_tensor(
            [0 if n in frozen.IVPS else 0.02 for n in frozen.FREE]
        ),
        initial_scale=initial.new_tensor(
            [0.16 if n in frozen.IVPS else 0 for n in frozen.FREE]
        ),
        cooling_rate=0.5,
        cooling_interval=50,
        within_pass_cooling=True,
    )(initial, model.y, n_particles=particles, n_iterations=iterations)
    assert warm.success and warm.completed_iterations == iterations
    torch.testing.assert_close(warm.parameter_history, original_history, rtol=0, atol=0)
    torch.testing.assert_close(
        warm.perturbed_log_likelihoods, original_ll, rtol=0, atol=0
    )
    assert torch.equal(model.generator.get_state(), original.generator.get_state())

    # Use a fixed published point for gradient agreement, not a selected optimum.
    parameters = initial.clone().requires_grad_()
    baseline = original.likelihood(parameters, particles, 123002, method="native")
    baseline_gradient = torch.autograd.grad(baseline.sum(), parameters)[0]
    assert torch.isfinite(baseline_gradient).all()
    pristine_value = api_adapter.factors(model, parameters, particles, 123002, alpha=1.0, package=api_adapter.original_package()).sum(0).reshape(len(parameters))
    pristine_gradient = torch.autograd.grad(pristine_value.sum(), parameters)[0]
    torch.testing.assert_close(pristine_value, baseline, rtol=0, atol=0)
    torch.testing.assert_close(pristine_gradient, baseline_gradient, rtol=0, atol=0)
    checks = {}
    for alpha in [0.0, 0.97, 1.0]:
        value = score(model, parameters, particles, 123002, alpha)
        gradient = torch.autograd.grad(value.sum(), parameters)[0]
        torch.testing.assert_close(value, baseline, rtol=0, atol=0)
        assert torch.isfinite(gradient).all()
        entry = {"forward_matches_native": True, "gradient_finite": True}
        if alpha == 1:
            torch.testing.assert_close(gradient, baseline_gradient, rtol=0, atol=0)
            entry["gradient_matches_native_exactly"] = True
        checks[str(alpha)] = entry

    # The old experimental decorator and the new native extension use different
    # floating-point ordering for the zero-valued correction. Retain diagnostics.
    old_value = original.likelihood(
        parameters, particles, 123002, method="discounted", alpha=0.97
    )
    old_gradient = torch.autograd.grad(old_value.sum(), parameters)[0]
    new_value = score(model, parameters, particles, 123002, 0.97)
    new_gradient = torch.autograd.grad(new_value.sum(), parameters)[0]

    train_parameters = torch.nn.Parameter(warm.parameters.clone())
    optimizer = torch.optim.Adam([train_parameters], lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    loss = -score(model, train_parameters, particles, 123003, 0.97).sum() / model.T
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(train_parameters.grad).all()
    optimizer.step()
    assert torch.isfinite(train_parameters).all()
    with torch.no_grad():
        raw = torch.stack(
            [score(model, train_parameters, particles, k) for k in [123004, 123005]]
        )
    assert torch.isfinite(raw).all()
    source = Path(__file__).parent
    report = {
        "passed": True,
        "purpose": "API integration and numerical agreement; not an estimation comparison",
        "torch_version": torch.__version__,
        "device": "cuda:0",
        "dtype": "float32",
        "package_path": pydpf.__file__,
        "T": 600,
        "Euler_steps_per_month": 20,
        "free_parameters": len(frozen.FREE),
        "particles": particles,
        "IF2_passes": iterations,
        "Adam_updates": 1,
        "evaluation_repetitions": 2,
        "score_precision_assessed": False,
        "gradient_clipping": False,
        "Adam_learning_rate": 1e-6,
        "IF2_history_and_likelihood_match_frozen_reference_exactly": True,
        "IF2_generator_state_matches_frozen_reference": True,
        "alpha_checks": checks,
        "pristine_upstream_Dacca_forward_and_gradient_exact": True,
        "historical_discounted_decorator_comparison": {
            "old_log_likelihood": old_value.detach().tolist(),
            "new_log_likelihood": new_value.detach().tolist(),
            "max_gradient_absolute_difference": float(
                (old_gradient - new_gradient).abs().max()
            ),
            "same_random_seed_and_parameters": True,
            "not_a_claim_of_identical_historical_training_trajectories": True,
        },
        "evaluation_raw": raw.tolist(),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ["dacca_model.py", "input/frozen.npz"]
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
