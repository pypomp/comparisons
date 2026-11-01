"""Added stop-gradient weight correction through the existing PFJAX callbacks.

This is separate from PFJAX's original score/RB estimators and OT resampling.
"""

import math
import numbers

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .particle_filter import particle_filter
from .particle_resamplers import resample_multinomial


def particle_filter_mop(
    model, key, y_meas, theta, n_particles, *, alpha=1.0, history=False
):
    """Bootstrap PF with an added differentiable likelihood-weight recursion.

    Uses the existing particle_filter, multinomial resampling and ordinary
    model pf_init/pf_step. The model must declare _bootstrap=True, use
    reparameterized state sampling, and have no pf_aux. General proposals,
    score/Fisher accumulators and OT are outside this interface.

    Forward values are those of the bootstrap PF. alpha=0 gives the original
    pathwise derivative through selected states; alpha=1 adds the weight
    correction; 0<alpha<1 discounts it at each resampling. State derivatives
    are not stopped. This function and its alpha=1 correction are additions,
    not an upstream PFJAX method. No exact/unbiased finite-particle gradient
    of log likelihood is asserted. The numerical derivative of repeated
    forward evaluations cannot validate a stop-gradient estimator.

    For selected log weights l, c=alpha*(l-stop_gradient(l)). Before the next
    measurement, c-logmeanexp(c) is added to its log weight. The denominator
    removes the derivative counted by PFJAX's sum of log-normalizers.
    Original state and resampling-history output shapes are preserved.
    """
    if not getattr(model, "_bootstrap", False) or callable(
        getattr(model, "pf_aux", None)
    ):
        raise ValueError(
            "particle_filter_mop requires a bootstrap model without pf_aux"
        )
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, numbers.Real)
        or not math.isfinite(alpha)
        or not 0 <= alpha <= 1
    ):
        raise ValueError("alpha must be a fixed finite number in [0, 1]")

    class CorrectedModel:
        def pf_init(self, key, y_init, theta):
            x, w = model.pf_init(key, y_init, theta)
            return {"state": x, "correction": jnp.zeros_like(w)}, w

        def pf_step(self, key, x_prev, y_curr, theta):
            x, w = model.pf_step(key, x_prev["state"], y_curr, theta)
            return {"state": x, "correction": jnp.zeros_like(w)}, w + x_prev[
                "correction"
            ]

    def resample(key, x_particles_prev, logw):
        out = resample_multinomial(
            key, x_particles_prev["state"], jax.lax.stop_gradient(logw)
        )
        selected = logw[out["ancestors"]]
        correction = alpha * (selected - jax.lax.stop_gradient(selected))
        correction -= logsumexp(correction) - jnp.log(logw.size)
        return {
            "x_particles": {"state": out["x_particles"], "correction": correction},
            "ancestors": out["ancestors"],
        }

    result = particle_filter(
        CorrectedModel(),
        key,
        y_meas,
        theta,
        n_particles,
        resampler=resample,
        history=history,
    )
    result["x_particles"] = result["x_particles"]["state"]
    return result
