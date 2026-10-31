"""Frozen 600-month Dacca model in Cuthbert's bootstrap callback interface.

Equations, clipping, permanent violation counter, initial normalization and
additive observation floor match PyPOMP's Gaussian-noise Dacca benchmark.
The observation expression avoids overflow in a branch whose contribution
already rounds to zero beside the original 1e-18 floor.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = dict(np.load(ROOT / "input/frozen.npz"))
NAMES = DATA["paramnames"].tolist()
STATES = DATA["statenames"].tolist()
INDEX = {name: i for i, name in enumerate(NAMES)}
FIXED = ["rho", "c", "alpha", "delta", "Y_0"]
IVPS = ["S_0", "I_0", "R1_0", "R2_0", "R3_0"]
FREE = [name for name in NAMES if name not in FIXED]
FIDX = [INDEX[name] for name in FREE]
IVP_FREE = [FREE.index(name) for name in IVPS]
POSITIVE = ["gamma", "epsilon", "m", "sigma", "tau"]
LTOL = float(np.log(1e-18))


def to_est(theta):
    """Keep finite coordinates only; fixed zero/one parameters remain exact."""
    theta = np.asarray(theta, dtype=np.float64)
    values = theta[FIDX].copy()
    for name in POSITIVE:
        values[FREE.index(name)] = np.log(theta[INDEX[name]])
    values[FREE.index("beta_trend")] *= 100
    initial = theta[[INDEX[name] for name in IVPS]]
    values[IVP_FREE] = np.log(initial / initial.sum())
    return values


def from_est(parameters):
    """Original exp/identity/softmax transformations, including the IVP gauge."""
    parameters = jnp.asarray(parameters)
    values = {name: parameters[i] for i, name in enumerate(FREE)}
    for name in POSITIVE:
        values[name] = jnp.exp(values[name])
    values["beta_trend"] /= 100
    # Match the original transformation's arithmetic. No artificial bounds.
    proportions = jnp.exp(parameters[jnp.asarray(IVP_FREE)])
    proportions = proportions / jnp.sum(proportions)
    values.update(dict(zip(IVPS, proportions)))
    values.update(rho=jnp.asarray(0., parameters.dtype), c=jnp.asarray(1., parameters.dtype),
                  alpha=jnp.asarray(1., parameters.dtype), delta=jnp.asarray(.02, parameters.dtype),
                  Y_0=jnp.asarray(0., parameters.dtype))
    return jnp.stack([values[name] for name in NAMES])


def initial_state(theta):
    proportions = theta[jnp.asarray([INDEX[n] for n in ["S_0", "I_0", "Y_0", "R1_0", "R2_0", "R3_0"]])]
    population = jnp.asarray(DATA["covars"][0, 2], theta.dtype)
    p = population * proportions / jnp.sum(proportions)
    return jnp.stack([p[0], p[1], p[2], jnp.asarray(0., theta.dtype), p[3], p[4], p[5], jnp.asarray(0., theta.dtype)])


def euler(theta, state, covariates, dt, noise):
    """One original Euler step driven by an externally supplied N(0,1) word."""
    S, I, Y, deaths, q0, q1, q2, count = state
    get = lambda n: theta[INDEX[n]]
    trend, dpopdt, pop = covariates[:3]
    seasons = covariates[3:9]
    beta = jnp.exp(get("beta_trend") * trend + jnp.dot(seasons, theta[6:12]))
    omega = jnp.exp(jnp.dot(seasons, theta[22:28]))
    neps = 3 * get("epsilon")
    pass0, pass1, pass2, pass3 = get("gamma")*I, q0*neps, q1*neps, q2*neps
    dw = noise * jnp.sqrt(dt)
    effective_I = (I / pop) ** get("alpha")
    infections = (omega + (beta + get("sigma")*dw/dt)*effective_I)*S
    delta = get("delta")
    disease = get("m")*I
    next_S = S + (dpopdt + delta*pop - infections - delta*S + pass3 + get("rho")*Y)*dt
    next_I = I + (get("c")*infections - disease - delta*I - pass0)*dt
    next_Y = Y + ((1-get("c"))*infections - delta*Y - get("rho")*Y)*dt
    next_q0 = q0 + (pass0-pass1-q0*delta)*dt
    next_q1 = q1 + (pass1-pass2-q1*delta)*dt
    next_q2 = q2 + (pass2-pass3-q2*delta)*dt
    next_deaths = deaths + disease*dt
    unbounded = jnp.stack([next_S, next_I, next_Y, next_deaths, next_q0, next_q1, next_q2])
    next_count = count + ((next_S < 0) | (next_I < 0) | (next_Y < 0) | (next_deaths < 0) | (next_q0 < 0) | (next_q1 < 0) | (next_q2 < 0))
    return jnp.r_[jnp.clip(unbounded, 0), next_count]


def observation(theta, state, y):
    deaths, count = state[3], state[7]
    v = theta[INDEX["tau"]] * deaths
    valid = jnp.isfinite(v) & (count <= 0)
    # Safe inactive inputs prevent 0*NaN in vectorized reverse-mode branches.
    mu = jnp.where(valid, deaths, jnp.asarray(1., deaths.dtype))
    scale = jnp.where(valid, v, jnp.asarray(1., deaths.dtype)) + 1e-18
    z = jnp.clip((y-mu)/scale, -1e10, 1e10)
    normal = -.5*jnp.log(2*jnp.pi) - jnp.log(scale) - .5*z*z
    value = jnp.logaddexp(normal, jnp.asarray(LTOL, deaths.dtype))
    return jnp.where(valid, value, jnp.asarray(LTOL, deaths.dtype))


def inputs(T=600):
    assert 0 < T <= 600
    return dict(y=jnp.asarray(DATA["ys"][:T], dtype=jnp.float32),
                cov=jnp.asarray(DATA["covars"][:20*T].reshape(T,20,9)),
                dt=jnp.asarray(DATA["dt"][:20*T].reshape(T,20)))


def factory(parameters):
    theta = from_est(parameters)

    def init_sample(key):
        return initial_state(theta)

    def propagate_sample(key, state, model_inputs):
        state = state.at[3].set(0.)  # The sole resetting accumulator is Mn.
        def one(carry, item):
            x, key = carry
            key, noise_key = jax.random.split(key)
            cv, dt = item
            noise = jax.random.normal(noise_key, (1,), dtype=parameters.dtype)[0]
            return (euler(theta,x,cv,dt,noise), key), None
        return jax.lax.scan(one, (state,key), (model_inputs["cov"],model_inputs["dt"]))[0][0]

    def log_potential(previous, state, model_inputs):
        return observation(theta,state,model_inputs["y"])

    return init_sample, propagate_sample, log_potential


STARTS = np.stack([to_est(theta) for theta in DATA["thetas"]]).astype(np.float32)
PUBLISHED = to_est(DATA["theta_published"][0]).astype(np.float32)
PROCESS_SCALE = np.asarray([0. if n in IVPS else .02 for n in FREE], dtype=np.float32)
INITIAL_SCALE = np.asarray([.16 if n in IVPS else .02 for n in FREE], dtype=np.float32)
