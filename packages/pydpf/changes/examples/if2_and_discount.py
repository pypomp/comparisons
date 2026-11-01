"""Use one model with the original ADPF, IF2, and discounted ADPF interfaces."""

import json

import torch
import pydpf


def model_factory(
    parameters, generator, *, dimension=1, nonlinear=False, package=pydpf
):
    """Build ordinary PyDPF model components; no parameter perturbations here.

    Parameter rows may be shared across particles (B, P) or specific to each
    particle (B, J, P). The latter is required for the IF2 extension.
    """
    theta = parameters[:, None, :] if parameters.ndim == 2 else parameters
    coefficient = torch.tanh(theta[..., :1])
    location = theta[..., 1 : dimension + 1]
    scale = 0.1 * theta[..., -1:].exp()

    def noise(shape):
        return torch.randn(
            shape, generator=generator, device=theta.device, dtype=theta.dtype
        )

    class Prior(package.Module):
        def sample(self, batch_size, n_particles, **data):
            return location + 0.3 * noise((batch_size, n_particles, dimension))

    class Dynamic(package.Module):
        def sample(self, prev_state, **data):
            state = torch.sin(prev_state) if nonlinear else prev_state
            return coefficient * state + location + scale * noise(prev_state.shape)

    class Observation(package.Module):
        def fitness(self, state, observation, **data):
            residual = (observation[:, :1] - state.sum(dim=-1)) / 0.6
            return -0.5 * residual.square() - torch.log(
                theta.new_tensor(0.6 * (2 * torch.pi) ** 0.5)
            )

    return package.FilteringModel(
        prior_model=Prior(), dynamic_model=Dynamic(), observation_model=Observation()
    )


def log_likelihood(parameters, observation, generator, *, alpha=1.0):
    """Use PyDPF's existing filter/likelihood outputs and the added alpha option."""
    pf = pydpf.StopGradientDPF(
        SSM=model_factory(parameters, generator),
        resampling_generator=generator,
        alpha=alpha,
    )
    pf.update()  # Native PyDPF cache maintenance, as in its training tutorial.
    factors = pf(
        n_particles=64,
        time_extent=len(observation) - 1,
        aggregation_function=pydpf.LogLikelihoodFactors(),
        observation=observation,
    )
    if factors.shape[0] != len(observation):
        raise RuntimeError("Incomplete filtering trajectory")
    return factors.sum(dim=0).reshape(parameters.shape[0])


def main():
    """Demonstrate composition, not an optimization-method comparison."""
    observation = torch.linspace(-0.4, 0.8, 12).reshape(12, 1, 1)
    initial = torch.tensor([[0.2, 0.0, -0.2]])
    fit_generator = torch.Generator().manual_seed(20261029)
    warm = pydpf.IteratedFiltering(
        lambda theta: model_factory(theta, fit_generator),
        generator=fit_generator,
        perturbation_scale=torch.tensor([0.02, 0.02, 0.0]),
        initial_scale=torch.tensor([0.0, 0.0, 0.03]),
    )(initial, observation, n_particles=64, n_iterations=3)
    if not warm.success:
        raise RuntimeError(warm.failure_reason)

    # This is an external Torch optimizer, following PyDPF's tutorial.
    parameters = torch.nn.Parameter(warm.parameters.clone())
    optimizer = torch.optim.Adam([parameters], lr=0.002)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = -log_likelihood(parameters, observation, fit_generator, alpha=0.97).sum()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(parameters.grad).all():
            raise RuntimeError("Nonfinite loss or gradient")
        optimizer.step()
        if not torch.isfinite(parameters).all():
            raise RuntimeError("Nonfinite parameter update")

    evaluation_generator = torch.Generator().manual_seed(20261030)
    with torch.no_grad():
        raw = torch.stack(
            [
                log_likelihood(parameters, observation, evaluation_generator)
                for _ in range(4)
            ]
        )
        score = torch.logsumexp(raw, dim=0) - torch.log(raw.new_tensor(len(raw)))
    print(
        json.dumps(
            {
                "example_completed": True,
                "if2_passes": warm.completed_iterations,
                "adam_updates": 3,
                "score_replicates": 4,
                "log_mean_likelihood": score.tolist(),
                "score_precision_assessed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
