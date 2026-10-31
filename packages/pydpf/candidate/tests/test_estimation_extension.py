"""Compare with pinned upstream code and explicit particle recursions."""

import importlib.util
from math import log
from pathlib import Path
import sys

import pytest
import torch
import pydpf
from pydpf.utils import normalise, batched_select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from if2_and_discount import model_factory

BASE = Path(__file__).resolve().parents[2] / "upstream/pydpf/pydpf"
spec = importlib.util.spec_from_file_location(
    "_pydpf_original", BASE / "__init__.py", submodule_search_locations=[str(BASE)]
)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
torch.set_num_threads(1)


def data(dimension=1, dtype=torch.float32):
    theta = torch.zeros((2, dimension + 2), dtype=dtype)
    theta[:, 0] = torch.tensor([0.2, -0.3])
    theta[:, -1] = -0.4
    obs = torch.linspace(-0.5, 1.0, 10, dtype=dtype).reshape(5, 2, 1)
    return theta, obs


def native(
    theta,
    obs,
    generator,
    *,
    package=pydpf,
    alpha=None,
    multinomial=False,
    dimension=1,
    nonlinear=False,
):
    kw = {} if alpha is None else {"alpha": alpha}
    pf = package.StopGradientDPF(
        SSM=model_factory(
            theta, generator, package=package, dimension=dimension, nonlinear=nonlinear
        ),
        resampling_generator=generator,
        multinomial=multinomial,
        **kw,
    )
    return pf(
        n_particles=24,
        time_extent=len(obs) - 1,
        aggregation_function=package.LogLikelihoodFactors(),
        observation=obs,
    )


@pytest.mark.parametrize("multinomial", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_and_alpha_one_match_pristine_upstream(multinomial, dtype):
    theta, obs = data(dtype=dtype)
    theta.requires_grad_()
    records = []
    for package, alpha in [(original, None), (pydpf, None), (pydpf, 1.0)]:
        g = torch.Generator().manual_seed(123)
        value = native(
            theta, obs, g, package=package, alpha=alpha, multinomial=multinomial
        )
        gradient = torch.autograd.grad(value.sum(), theta)[0]
        records.append((value.detach(), gradient, g.get_state()))
    for value in records[1:]:
        for a, b in zip(records[0], value):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def explicit_gradient(theta, obs, generator, alpha, multinomial):
    """Direct bootstrap recursion with the inherited-weight derivative written out."""
    ssm = model_factory(theta, generator)
    state = ssm.prior_model.sample(batch_size=len(theta), n_particles=24)
    weights, factor = normalise(ssm.observation_model.fitness(state, obs[0]))
    factors = [(factor - log(24)).float()]
    sampler = (
        pydpf.MultinomialResampler(generator)
        if multinomial
        else pydpf.SystematicResampler(generator)
    )
    for t in range(1, len(obs)):
        ancestors, uniform = sampler(state, weights)
        inherited = batched_select(weights, sampler.cache["sampled_indices"])
        # Native forward arithmetic, with the specified derivative attached.
        primal = (uniform + inherited - inherited.detach()).detach()
        corrected = primal + alpha * (inherited - inherited.detach())
        state = ssm.dynamic_model.sample(ancestors, t=t)
        weights, factor = normalise(
            corrected + ssm.observation_model.fitness(state, obs[t])
        )
        factors.append(factor.float())
    return torch.stack(factors)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("multinomial", [False, True])
def test_discounted_forward_and_explicit_gradient(alpha, multinomial):
    theta, obs = data(dtype=torch.float64)
    theta.requires_grad_()
    value = native(
        theta,
        obs,
        torch.Generator().manual_seed(31),
        alpha=alpha,
        multinomial=multinomial,
    )
    expected = explicit_gradient(
        theta, obs, torch.Generator().manual_seed(31), alpha, multinomial
    )
    reference = native(
        theta,
        obs,
        torch.Generator().manual_seed(31),
        package=original,
        multinomial=multinomial,
    )
    torch.testing.assert_close(value, reference, rtol=0, atol=0)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(value.sum(), theta)[0]
    expected_grad = torch.autograd.grad(expected.sum(), theta)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-10)
    assert actual_grad.abs().sum() > 0  # alpha=0 still has state-path derivatives.


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("need_weight", [False, True])
def test_resampler_default_branches_and_cache(grad_enabled, need_weight):
    values = []
    for package in [original, pydpf]:
        resampler = package.StopGradientResampler(
            package.SystematicResampler(torch.Generator().manual_seed(3))
        )
        resampler._need_weight_output = need_weight
        state = torch.arange(10.0).reshape(2, 5, 1).requires_grad_()
        weight = (
            torch.randn((2, 5), generator=torch.Generator().manual_seed(4))
            .log_softmax(1)
            .requires_grad_()
        )
        with torch.set_grad_enabled(grad_enabled):
            result = resampler(state, weight)
        values.append((result, resampler.cache))
    torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0)
    torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)


def explicit_if2(
    initial,
    obs,
    generator,
    scale,
    initial_scale,
    *,
    dimension,
    nonlinear,
    passes,
    rate,
    interval,
    within,
    multinomial,
):
    """Independent time loop: no ParticleFilter or IteratedFiltering call."""
    swarm = initial[:, None, :].expand(2, 24, -1).clone()
    histories, likelihoods = [initial], []
    sampler = (
        pydpf.MultinomialResampler(generator)
        if multinomial
        else pydpf.SystematicResampler(generator)
    )

    def noise():
        return torch.randn(swarm.shape, generator=generator, dtype=swarm.dtype)

    for m in range(passes):
        parameters = swarm + noise() * initial_scale * rate ** (m / interval)
        parameters = parameters + noise() * scale * rate ** (m / interval)
        ssm = model_factory(
            parameters, generator, dimension=dimension, nonlinear=nonlinear
        )
        state = ssm.prior_model.sample(2, 24)
        weights, ll = normalise(ssm.observation_model.fitness(state, obs[0]))
        factors = [(ll - log(24)).float()]
        for t in range(1, len(obs)):
            joined, uniform = sampler(torch.cat((state, parameters), -1), weights)
            parameters = joined[..., dimension:] + noise() * scale * rate ** (
                (m + (t / len(obs) if within else 0)) / interval
            )
            ssm = model_factory(
                parameters, generator, dimension=dimension, nonlinear=nonlinear
            )
            state = ssm.dynamic_model.sample(joined[..., :dimension], t=t)
            weights, ll = normalise(
                uniform + ssm.observation_model.fitness(state, obs[t])
            )
            factors.append(ll.float())
        terminal, _ = sampler(torch.cat((state, parameters), -1), weights)
        swarm = terminal[..., dimension:]
        histories.append(swarm.mean(1))
        likelihoods.append(torch.stack(factors).sum(0).reshape(2))
    return swarm, torch.stack(histories), torch.stack(likelihoods)


@pytest.mark.parametrize("dimension,nonlinear", [(1, False), (3, True)])
@pytest.mark.parametrize("multinomial", [False, True])
@pytest.mark.parametrize("within", [False, True])
def test_if2_full_swarm_cooling_and_model_dimensions(
    dimension, nonlinear, multinomial, within
):
    theta, obs = data(dimension)
    scale = torch.full((dimension + 2,), 0.03)
    scale[-1] = 0
    initial_scale = torch.zeros_like(scale)
    initial_scale[-1] = 0.04
    generator = torch.Generator().manual_seed(1234)
    result = pydpf.IteratedFiltering(
        lambda p: model_factory(p, generator, dimension=dimension, nonlinear=nonlinear),
        generator=generator,
        perturbation_scale=scale,
        initial_scale=initial_scale,
        cooling_rate=0.5,
        cooling_interval=2,
        within_pass_cooling=within,
        multinomial=multinomial,
    )(theta, obs, n_particles=24, n_iterations=3)
    expected_generator = torch.Generator().manual_seed(1234)
    expected = explicit_if2(
        theta,
        obs,
        expected_generator,
        scale,
        initial_scale,
        dimension=dimension,
        nonlinear=nonlinear,
        passes=3,
        rate=0.5,
        interval=2,
        within=within,
        multinomial=multinomial,
    )
    assert result.success and result.completed_iterations == 3
    for a, b in zip(
        (
            result.parameter_swarm,
            result.parameter_history,
            result.perturbed_log_likelihoods,
        ),
        expected,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_zero_iterations_and_frozen_coordinates():
    theta, obs = data()
    generator = torch.Generator().manual_seed(3)
    obj = pydpf.IteratedFiltering(
        lambda p: model_factory(p, generator), generator=generator, perturbation_scale=0
    )
    state = generator.get_state().clone()
    result = obj(theta, obs, n_particles=24, n_iterations=0)
    assert result.success and result.completed_iterations == 0
    assert torch.equal(generator.get_state(), state)
    assert result.perturbed_log_likelihoods.shape == (0, 2)
    result = obj(theta, obs, n_particles=24, n_iterations=2)
    torch.testing.assert_close(
        result.parameter_swarm, theta[:, None].expand(2, 24, 3), rtol=0, atol=0
    )


def test_failed_pass_retains_preceding_swarm():
    theta, obs = data()
    generator = torch.Generator().manual_seed(3)

    def factory(p):
        ssm = model_factory(p, generator)
        sample = ssm.dynamic_model.sample
        ssm.dynamic_model.sample = lambda prev_state, **kw: sample(
            prev_state, **kw
        ) * float("nan")
        return ssm

    result = pydpf.IteratedFiltering(
        factory, generator=generator, perturbation_scale=0.02
    )(theta, obs, n_particles=24, n_iterations=3)
    assert not result.success and result.completed_iterations == 0
    assert result.failure_reason
    torch.testing.assert_close(
        result.parameter_swarm, theta[:, None].expand(2, 24, 3), rtol=0, atol=0
    )


@pytest.mark.parametrize("alpha", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_alpha(alpha):
    theta, _ = data()
    g = torch.Generator()
    with pytest.raises(ValueError):
        pydpf.StopGradientDPF(SSM=model_factory(theta, g), alpha=alpha)


@pytest.mark.parametrize("scale", [-0.1, float("nan"), [0.1, 0.2]])
def test_invalid_perturbation_scales(scale):
    theta, obs = data()
    g = torch.Generator()
    with pytest.raises(ValueError):
        pydpf.IteratedFiltering(
            lambda p: model_factory(p, g), generator=g, perturbation_scale=scale
        )(theta, obs, 24, 1)


def test_discount_and_if2_reject_guided_proposals():
    theta, obs = data()
    g = torch.Generator()

    def factory(p):
        ssm = model_factory(p, g)
        ssm.proposal_model = ssm.dynamic_model
        return ssm

    with pytest.raises(ValueError):
        pydpf.StopGradientDPF(SSM=factory(theta), alpha=0.97)
    with pytest.raises(ValueError):
        pydpf.IteratedFiltering(factory, generator=g, perturbation_scale=0.01)(
            theta, obs, 24, 1
        )
    with pytest.raises(ValueError):
        pydpf.StopGradientDPF(
            SSM=model_factory(theta, g), alpha=0.97, use_REINFORCE_for_proposal=True
        )
