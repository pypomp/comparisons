"""IF2 parameter perturbations using the existing bootstrap ParticleFilter."""

from math import isfinite
from typing import Callable, NamedTuple

import torch
from torch import Tensor

from .base import DivergenceError, Module
from .filtering import ParticleFilter
from .model_based_api import FilteringModel
from .resampling import MultinomialResampler, SystematicResampler


class IF2Result(NamedTuple):
    """Last valid parameter swarm, its means, and perturbed-model likelihoods.

    A failed pass stops the whole batch and preserves the preceding swarm.
    ``completed_iterations`` counts complete accepted passes, not individual rows.
    These likelihoods must not be used as endpoint scores for the static model.
    """

    parameters: Tensor
    parameter_swarm: Tensor
    parameter_history: Tensor
    perturbed_log_likelihoods: Tensor
    completed_iterations: int
    success: bool
    failure_reason: str | None


class _LastState(Module):
    def forward(self, state, weight, likelihood, **data):
        self.state, self.weight = state, weight
        return likelihood


class IteratedFiltering(Module):
    """Fit a bootstrap model by perturbing parameters carried by its particles.

    ``model_factory(parameters)`` must return a ``FilteringModel`` accepting
    parameters of shape (batch, particles, coordinates). Model states must have
    shape (batch, particles, state_dimension), with the same dtype and device.
    The original prior.sample, dynamic.sample and observation.fitness/score
    interfaces are used; no model callback implements parameter perturbations.
    Parameter constraints belong in the factory, which must be side-effect free
    apart from the random draws made by its samplers.

    The initial scale is applied before each pass. The process scale is applied
    before the prior at observation zero and before each later transition.
    Initial-value-only coordinates therefore use zero process scale. A prior
    requiring separate parameters before and after a pre-observation transition
    is outside this interface; adapt that model explicitly rather than assume
    that the two parameter values are interchangeable.

    Pass m, observation t uses cooling_rate ** ((m + t/T) / cooling_interval)
    when within_pass_cooling is True; otherwise t/T is omitted. Cooling starts
    at m=0. A full terminal parameter swarm is retained between passes.

    All stochastic model callbacks must use the generator controlled by the
    caller if exact replay is required. A failed batch is not retried here.
    """

    def __init__(
        self,
        model_factory: Callable[[Tensor], FilteringModel],
        *,
        perturbation_scale,
        initial_scale=0.0,
        generator: torch.Generator = torch.default_generator,
        cooling_rate: float = 0.9,
        cooling_interval: float = 1.0,
        within_pass_cooling: bool = False,
        multinomial: bool = False,
    ):
        super().__init__()
        if not isfinite(cooling_rate) or not 0 < cooling_rate <= 1:
            raise ValueError("cooling_rate must be finite and in (0, 1]")
        if not isfinite(cooling_interval) or cooling_interval <= 0:
            raise ValueError("cooling_interval must be finite and positive")
        self.model_factory = model_factory
        self.perturbation_scale, self.initial_scale = perturbation_scale, initial_scale
        self.generator = generator
        self.cooling_rate, self.cooling_interval = cooling_rate, cooling_interval
        self.within_pass_cooling, self.multinomial = within_pass_cooling, multinomial

    @torch.no_grad()
    def forward(
        self,
        initial_parameters: Tensor,
        observation: Tensor,
        n_particles: int,
        n_iterations: int,
        *,
        control: Tensor | None = None,
        time: Tensor | None = None,
        series_metadata: Tensor | None = None,
    ) -> IF2Result:
        """Run complete PF passes; return failure if a pass is incomplete/nonfinite."""
        for name, value, minimum in [
            ("n_particles", n_particles, 1),
            ("n_iterations", n_iterations, 0),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        p = initial_parameters.detach().clone()
        if p.ndim != 2 or min(p.shape) < 1 or not p.is_floating_point():
            raise ValueError(
                "initial_parameters must be a nonempty floating (B, P) tensor"
            )
        if not torch.isfinite(p).all():
            raise ValueError("initial_parameters must be finite")
        if (
            observation.ndim < 3
            or observation.shape[0] < 1
            or observation.shape[1] != p.shape[0]
        ):
            raise ValueError("observation must have shape (T, B, ...) with T > 0")
        if (
            observation.device != p.device
            or torch.device(self.generator.device) != p.device
        ):
            raise ValueError(
                "parameters, observations and generator must share a device"
            )

        def scales(value):
            result = torch.as_tensor(value, dtype=p.dtype, device=p.device)
            try:
                result = torch.broadcast_to(result, p.shape)[:, None, :]
            except RuntimeError as exc:
                raise ValueError("scales must broadcast to (B, P)") from exc
            if not torch.isfinite(result).all() or (result < 0).any():
                raise ValueError("scales must be finite and nonnegative")
            return result

        process_scale, initial_scale = (
            scales(self.perturbation_scale),
            scales(self.initial_scale),
        )
        batch, coordinates = p.shape
        length = observation.shape[0]
        swarm = p[:, None, :].expand(batch, n_particles, coordinates).clone()
        history, likelihoods, failure = [p], [], None
        resampler_type = (
            MultinomialResampler if self.multinomial else SystematicResampler
        )

        def model(parameters):
            ssm = self.model_factory(parameters)
            if (
                not isinstance(ssm, FilteringModel)
                or ssm.has_proposal
                or ssm.has_initial_proposal
            ):
                raise ValueError("model_factory must return a bootstrap FilteringModel")
            return ssm

        # Reject unsupported models even when zero iterations are requested.
        model(swarm)

        def perturb(parameters, scale, cooling):
            noise = torch.randn(
                parameters.shape,
                device=p.device,
                dtype=p.dtype,
                generator=self.generator,
            )
            return parameters + noise * scale * cooling

        def pack(state, parameters):
            if (
                state.ndim != 3
                or state.shape[:2] != parameters.shape[:2]
                or state.shape[-1] < 1
            ):
                raise ValueError("model states must have shape (B, J, state_dimension)")
            if state.dtype != parameters.dtype or state.device != parameters.device:
                raise ValueError(
                    "model states and parameters must have the same dtype and device"
                )
            return torch.cat((state, parameters), dim=-1)

        for iteration in range(n_iterations):

            def cooling(t):
                within = t / length if self.within_pass_cooling else 0.0
                return self.cooling_rate ** (
                    (iteration + within) / self.cooling_interval
                )

            initial = perturb(swarm, initial_scale, cooling(0))

            class Prior(Module):
                def sample(self, batch_size, n_particles, **data):
                    parameters = perturb(initial, process_scale, cooling(0))
                    state = model(parameters).prior_model.sample(
                        batch_size=batch_size, n_particles=n_particles, **data
                    )
                    return pack(state, parameters)

            class Dynamic(Module):
                def sample(self, prev_state, t, **data):
                    parameters = perturb(
                        prev_state[..., -coordinates:], process_scale, cooling(t)
                    )
                    state = model(parameters).dynamic_model.sample(
                        prev_state=prev_state[..., :-coordinates], t=t, **data
                    )
                    return pack(state, parameters)

            class Observation(Module):
                def fitness(self, state, observation, **data):
                    ssm = model(state[..., -coordinates:])
                    fitness = ParticleFilter._get_fitness_func(ssm)
                    return fitness(
                        state=state[..., :-coordinates], observation=observation, **data
                    )

            ssm = FilteringModel(
                prior_model=Prior(),
                dynamic_model=Dynamic(),
                observation_model=Observation(),
            )
            pf = ParticleFilter(resampler=resampler_type(self.generator), SSM=ssm)
            last = _LastState()
            try:
                factors = pf(
                    n_particles=n_particles,
                    time_extent=length - 1,
                    aggregation_function=last,
                    observation=observation,
                    control=control,
                    time=time,
                    series_metadata=series_metadata,
                )
                if factors.shape[0] != length:
                    raise DivergenceError("incomplete IF2 trajectory")
                valid_weights = (
                    not torch.isnan(last.weight).any()
                    and not torch.isposinf(last.weight).any()
                    and torch.isfinite(last.weight).any(dim=1).all()
                )
                if not valid_weights or not all(
                    torch.isfinite(x).all() for x in (factors, last.state)
                ):
                    raise DivergenceError("nonfinite IF2 state, weights or likelihood")
                terminal, _ = resampler_type(self.generator)(last.state, last.weight)
                proposal = terminal[..., -coordinates:]
                if not torch.isfinite(proposal).all():
                    raise DivergenceError("nonfinite terminal parameter swarm")
            except DivergenceError as exc:
                failure = str(exc)
                break
            swarm = proposal.detach().clone()
            history.append(swarm.mean(dim=1))
            likelihoods.append(factors.sum(dim=0).reshape(batch))

        ll = torch.stack(likelihoods) if likelihoods else p.new_empty((0, batch))
        return IF2Result(
            history[-1],
            swarm,
            torch.stack(history),
            ll,
            len(likelihoods),
            failure is None,
            failure,
        )
