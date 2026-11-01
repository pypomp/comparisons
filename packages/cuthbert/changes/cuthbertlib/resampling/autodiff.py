"""Implements decorators for automatic differentiation of resampling schemes.

Current supported is the stop_gradient resampling scheme, which provides the
classical Fisher estimates for the score function via automatic differentiation.
This can be wrapped around a resampling scheme such as multinomial or systematic
resampling.

See [Scibior and Wood (2021)](https://arxiv.org/abs/2106.10314) for more details.
"""

import math
from functools import wraps

import jax
import jax.numpy as jnp

from cuthbertlib.resampling.protocols import Resampling
from cuthbertlib.resampling.utils import apply_resampling_indices
from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike


def discounted_stop_gradient_decorator(func: Resampling, alpha: float) -> Resampling:
    """Discount inherited log-weight derivatives without changing the forward PF.

    This implements the DMOP weight recursion for a reparameterizable bootstrap
    particle filter that resamples at EVERY observation. ``func`` must return
    equally weighted particles (e.g. systematic or multinomial resampling).
    Do not compose this with adaptive/no-resampling wrappers: skipping calls
    discounts by resampling events rather than observation times. Guided
    proposals and discrete transition scores require separate treatment.

    Args:
        func: An unconditional, equally weighted resampling function.
        alpha: Static discount in [0, 1]. One recovers the existing stop-gradient
            correction; zero removes inherited weight derivatives, while keeping
            pathwise derivatives through the particle states.

    Returns:
        A resampler with unchanged forward particles, ancestry and weights.

    Raises:
        ValueError: If alpha is nonfinite or outside [0, 1].
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")

    # Keep the undiscounted baseline on Cuthbert's existing ADPF implementation.
    if alpha == 1:
        return stop_gradient_decorator(func)

    @wraps(func)
    def wrapped(key, logits, positions, n):
        idx, base_logits, particles = func(
            key, jax.lax.stop_gradient(logits), positions, n
        )
        selected = apply_resampling_indices(logits, idx)
        correction = alpha * (selected - jax.lax.stop_gradient(selected))
        return idx, base_logits + correction, particles

    return jax.jit(wrapped, static_argnames=("n",))


def stop_gradient_decorator(func: Resampling) -> Resampling:
    """Wrap a Resampling function to use stop gradient resampling.

    Args:
        func: A resampling function with signature
              (key, logits, positions, n) -> (indices, logits_out, positions_out).

    Returns:
        A Resampling function implementing stop gradient resampling.
    """
    # Build a descriptive docstring that includes the wrapped function doc
    wrapped_doc = func.__doc__ or ""
    doc = f"""
    Stop gradient resampling decorator.

    This wrapper will call the provided resampling function, and then apply 
    the stop gradient trick of [Scibior and Wood (2021)](https://arxiv.org/abs/2106.10314). 
    Resulting estimates of the score function (i.e., the gradient of the 
    log-likelihood with respect to model parameters) are unbiased, 
    corresponding to the classical Fisher estimate.

    Wrapped resampler documentation:
    {wrapped_doc}
    """

    @wraps(func)
    def _wrapped(
        key: Array, logits: ArrayLike, positions: ArrayTreeLike, n: int
    ) -> tuple[Array, Array, ArrayTree]:
        idx_base, logits_base, positions_base = func(
            key, jax.lax.stop_gradient(logits), positions, n
        )

        logits = jnp.asarray(
            logits_base
            + apply_resampling_indices(logits, idx_base)
            - jax.lax.stop_gradient(apply_resampling_indices(logits, idx_base))
        )
        return idx_base, logits, positions_base

    # Attach the composed docstring and return a jitted version
    _wrapped.__doc__ = doc
    return jax.jit(_wrapped, static_argnames=("n",))
