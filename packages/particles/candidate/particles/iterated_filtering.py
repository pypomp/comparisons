"""Iterated filtering with the existing Bootstrap and SMC implementations.

This module adds IF2; it does not add an automatic differentiation method.
Model parameters are perturbed in user-chosen estimation coordinates. The
factory maps those coordinates to a normal StateSpaceModel (including any
log/logit transforms). No transition density is required.
"""

from collections import namedtuple
from numbers import Integral

import numpy as np

from particles import FeynmanKac, SMC
from particles import resampling as rs
from particles.state_space_models import Bootstrap


IF2Step = namedtuple("IF2Step", "theta_particles theta_mean loglik smc")
IF2Result = namedtuple("IF2Result", "theta_particles means loglik")


def _inputs(data, theta_particles, scales):
    theta = np.array(theta_particles, dtype=float, copy=True)
    if theta.ndim != 2 or min(theta.shape) == 0 or not np.isfinite(theta).all():
        raise ValueError("theta_particles must be a finite, nonempty (N, P) array")
    if len(data) == 0:
        raise ValueError("data must contain at least one observation")
    scale = np.asarray(scales, dtype=float)
    if scale.shape not in [(theta.shape[1],), (len(data), theta.shape[1])]:
        raise ValueError("scales must have shape (P,) or (T, P)")
    if not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError("scales must be finite and nonnegative")
    return theta, np.broadcast_to(scale, (len(data), theta.shape[1])).copy()


def _perturb(theta, scale):
    # Do not consume random numbers for coordinates whose scale is zero.
    result = theta.copy()
    active = scale > 0
    if np.any(active):
        result[:, active] += np.random.normal(size=(len(theta), active.sum())) * scale[active]
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite perturbed parameter")
    return result


def _pack(theta, x):
    x = np.asarray(x)
    if x.ndim < 1 or len(x) != len(theta) or x.dtype.kind not in "fiu":
        raise ValueError("model states must be numeric arrays with first axis N")
    if not np.isfinite(x).all():
        raise FloatingPointError("nonfinite model state")
    out = np.empty(len(theta), dtype=[("theta", float, (theta.shape[1],)),
                                     ("x", x.dtype, x.shape[1:])])
    out["theta"], out["x"] = theta, x
    return out


class _PerturbedBootstrap(FeynmanKac):
    """Augment particles through FK callbacks, without replacing the SMC loop."""

    def __init__(self, model_factory, data, theta, scales, vectorized):
        super().__init__(T=len(data))
        self.model_factory, self.data = model_factory, data
        self.theta, self.scales, self.vectorized = theta, scales, vectorized

    def _apply(self, method, theta, t=None, xp=None, x=None):
        def call(th, previous, current, n):
            fk = Bootstrap(ssm=self.model_factory(th), data=self.data)
            if method == "M0":
                return np.asarray(fk.M0(n))
            if method == "M":
                return np.asarray(fk.M(t, previous))
            return np.broadcast_to(fk.logG(t, previous, current), (n,)).copy()

        if self.vectorized:
            return call(theta, xp, x, len(theta))
        # Scalar factory support lets existing models be used without rewriting
        # their distributions for particle-specific parameter arrays.
        parts = []
        for i, th in enumerate(theta):
            previous = None if xp is None else xp[i:i + 1]
            current = None if x is None else x[i:i + 1]
            parts.append(call(th.copy(), previous, current, 1))
        return np.concatenate(parts, axis=0)

    def M0(self, N):
        theta = _perturb(self.theta, self.scales[0])
        return _pack(theta, self._apply("M0", theta))

    def M(self, t, xp):
        theta = _perturb(xp["theta"], self.scales[t])
        return _pack(theta, self._apply("M", theta, t=t, xp=xp["x"]))

    def logG(self, t, xp, x):
        previous = None if xp is None else xp["x"]
        lw = self._apply("logG", x["theta"], t=t, xp=previous, x=x["x"])
        # An impossible particle (-inf) is allowed, an impossible whole cloud
        # or undefined measurement is not silently converted to usable weights.
        if np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any():
            raise FloatingPointError("invalid or all -inf observation log weights")
        return lw

    def time_to_resample(self, smc):
        # IF2 here always selects ancestors between observations, including
        # exactly uniform weights. ESSrmin=1 alone does not ensure this.
        return True


def if2_step(model_factory, data, theta_particles, scales, *,
             vectorized=False, resampling="multinomial"):
    """Run one perturbed bootstrap filter and return an equal-weight cloud.

    Parameters
    ----------
    model_factory : callable
        Takes a (P,) estimation-coordinate vector and returns an ordinary
        StateSpaceModel with PX0, PX, PY. With vectorized=True it instead takes
        (N,P); its distributions must return N particle-specific values. The
        factory and distribution constructors must not sample or mutate inputs.
    data : list-like
        Nonempty observations. PX0 samples the state of data[0], as in Bootstrap.
    theta_particles : array, shape (N,P)
        Complete equal-weight parameter cloud, not just its mean.
    scales : array, shape (P,) or (T,P)
        Gaussian random-walk standard deviations. Row 0 is applied before PX0;
        row t before PX(t,...). Zero means no perturbation. For initial-value
        parameters set rows 1: to zero, and row 0 to their initial scale.
    vectorized : bool
        False uses an unchanged scalar-parameter model per particle (potentially
        slow). True requires a factory supporting particle-specific parameters.
    resampling : str
        Multinomial, systematic, stratified or residual; used at every observation
        transition and once after the last observation. No SQMC, guided PF or
        APF here. SSP uses a separate Numba RNG and is excluded from this API's
        NumPy-state continuation contract.

    Returns
    -------
    IF2Step
        theta_particles is the complete resampled terminal cloud; theta_mean
        is its arithmetic mean in estimation coordinates. loglik is from the
        *perturbed* model, not a likelihood evaluation of theta_mean. smc is the
        original SMC object, including weighted particles and normalizers.

    Notes
    -----
    Uses NumPy's global RNG, as the original package does. Set np.random.seed
    outside the function for reproducibility; preserve RNG state as well as the
    cloud for continuation. Do not share this RNG across concurrent threads.
    Invalid parameters, states or weights raise; failed runs are not retried.
    """
    theta, scale = _inputs(data, theta_particles, scales)
    if resampling not in ("multinomial", "systematic", "stratified", "residual"):
        raise ValueError("unsupported native resampling scheme")
    fk = _PerturbedBootstrap(model_factory, data, theta, scale, vectorized)
    pf = SMC(fk=fk, N=len(theta), resampling=resampling, collect="off")
    pf.run()  # original time loop, selection, normalization and logLt accumulation
    if not np.isfinite(pf.logLt) or not np.isfinite(pf.W).all():
        raise FloatingPointError("nonfinite filter result")
    indices = rs.resampling(resampling, pf.W, M=len(theta))
    terminal = pf.X["theta"][indices].copy()
    return IF2Step(terminal, terminal.mean(axis=0), float(pf.logLt), pf)


def iterated_filtering(model_factory, data, theta_particles, scales, niter, *,
                       cooling=0.9, start_iteration=0, vectorized=False,
                       resampling="multinomial"):
    """Repeat if2_step, carrying the whole cloud between passes.

    Pass m uses scales * cooling**m, with m starting at start_iteration.
    Cooling must lie in (0,1]. States are freshly initialized each pass; parameter
    particles continue from the previous final resampling. No centering around a
    point estimate, best-iterate selection, optimizer or final scoring is added.
    means includes the initial mean followed by each resampled terminal mean.
    niter=0 returns an unchanged copy and an empty loglik array.
    """
    for name, value in [("niter", niter), ("start_iteration", start_iteration)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if not np.isscalar(cooling) or not np.isfinite(cooling) or not 0 < cooling <= 1:
        raise ValueError("cooling must lie in (0,1]")
    theta, scale = _inputs(data, theta_particles, scales)
    means, loglik = [theta.mean(axis=0)], []
    for m in range(start_iteration, start_iteration + niter):
        step = if2_step(model_factory, data, theta, scale * cooling**m,
                        vectorized=vectorized, resampling=resampling)
        theta = step.theta_particles
        means.append(step.theta_mean)
        loglik.append(step.loglik)
    return IF2Result(theta, np.asarray(means), np.asarray(loglik))
