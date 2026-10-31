"""Dacca for native PFJAX, plus explicitly local IF2/DMOP extensions.

Native PFJAX particle_filter executes PF and OT filtering. DMOP uses its
resampler hook and a zero-valued correction carried in the particle state.
The normalization correction is essential: raw selected log weights alone
would double-count derivatives in PFJAX's logsumexp accumulator.
"""
from functools import partial
import jax
import jax.numpy as jnp
import optax
import pfjax
from pfjax.particle_resamplers import resample_multinomial, resample_ot
import dacca_model as m


class Dacca(pfjax.BaseModel):
    def __init__(self):
        super().__init__(bootstrap=True)

    def pf_init(self, key, y_init, theta):
        init, prop, pot = m.factory(theta)
        x0 = init(key)
        x = prop(key, x0, y_init)
        return x, pot(x0, x, y_init)

    def pf_step(self, key, x_prev, y_curr, theta):
        _, prop, pot = m.factory(theta)
        x = jax.checkpoint(prop)(key, x_prev, y_curr)
        return x, pot(x_prev, x, y_curr)

    def meas_lpdf(self, y_curr, x_curr, theta):
        return m.observation(m.from_est(theta), x_curr, y_curr['y'])


class CorrectedDacca(Dacca):
    def pf_init(self, key, y_init, theta):
        x, w = super().pf_init(key, y_init, theta)
        return {'x': x, 'correction': jnp.asarray(0., w.dtype)}, w

    def pf_step(self, key, x_prev, y_curr, theta):
        x, w = super().pf_step(key, x_prev['x'], y_curr, theta)
        return {'x': x, 'correction': jnp.asarray(0., w.dtype)}, w + x_prev['correction']


def corrected_resampler(alpha):
    def resample(key, x_particles_prev, logw):
        out = resample_multinomial(key, x_particles_prev['x'], jax.lax.stop_gradient(logw))
        selected = logw[out['ancestors']]
        correction = alpha * (selected - jax.lax.stop_gradient(selected))
        correction -= jax.scipy.special.logsumexp(correction) - jnp.log(logw.size)
        return {'x_particles': {'x': out['x_particles'], 'correction': correction},
                'ancestors': out['ancestors']}
    return resample


def objective(kind, J, T, alpha=.97):
    data = m.inputs(T)
    model = CorrectedDacca() if kind == 'dmop' else Dacca()
    if kind == 'dmop':
        resampler = corrected_resampler(alpha)
    elif kind == 'ot':
        # Native defaults, including native scaling and Sinkhorn tolerances.
        resampler = resample_ot
    elif kind == 'ot_unrolled':
        # Explicit diagnostic, not native defaults: differentiate 100 Sinkhorn
        # iterations directly when the default implicit linear solve fails.
        resampler = partial(resample_ot, sinkhorn_kwargs={'implicit_diff':None,
                            'min_iterations':100,'max_iterations':100})
    elif kind == 'native':
        resampler = resample_multinomial
    else:
        raise ValueError(kind)
    def loglik(p, key):
        return pfjax.particle_filter(model, key, data, p, J, resampler=resampler)['loglik']
    return loglik


def adam_fit(kind, J, T, steps, alpha=.97, learning_rate=.001):
    """Short execution check, not the formal paper-setting comparison."""
    obj = objective(kind, J, T, alpha)
    adam = optax.adam(learning_rate)
    def fit(p, key):
        def step(carry, i):
            p, state, key = carry
            key, sk = jax.random.split(key)
            value, g = jax.value_and_grad(lambda q: -obj(q, sk)/T)(p)
            updates, state = adam.update(g, state, p)
            p = optax.apply_updates(p, updates)
            return (p, state, key), (p, value, jnp.linalg.norm(g))
        (_, _, _), trace = jax.lax.scan(step, (p, adam.init(p), key), jnp.arange(steps))
        return trace
    return jax.jit(fit)


def if2(J, T, steps, cooling=.5):
    """Local IF2 extension using native PFJAX multinomial resampling.

    Perturbed parameter swarms persist between iterations, IVPs are perturbed
    only at initialization, and the terminal resampling gives equal weights.
    This extension is not an upstream PFJAX estimation facility.
    """
    data = m.inputs(T)
    init_scale = jnp.asarray([.16 if n in m.IVPS else 0. for n in m.FREE])
    step_scale = jnp.asarray([0. if n in m.IVPS else .02 for n in m.FREE])
    model = Dacca()
    def fit(p, key):
        swarm = jnp.broadcast_to(p, (J, len(m.FREE)))
        def iteration(carry, i):
            swarm, key = carry
            key, ik = jax.random.split(key)
            swarm = swarm + cooling**(i/50) * init_scale * jax.random.normal(ik, swarm.shape)
            states = jax.vmap(lambda p: m.initial_state(m.from_est(p)))(swarm)
            def month(carry, item):
                swarm, states, key = carry
                n, inp = item
                key, nk, pk, rk = jax.random.split(key, 4)
                swarm = swarm + cooling**((i+n/T)/50) * step_scale * jax.random.normal(nk, swarm.shape)
                states, logw = jax.vmap(model.pf_step, (0,0,None,0))(jax.random.split(pk,J),states,inp,swarm)
                out = resample_multinomial(rk, {'state': states, 'parameter': swarm}, logw)
                res = out['x_particles']
                return (res['parameter'],res['state'],key),jax.scipy.special.logsumexp(logw)-jnp.log(J)
            (swarm,_,key), ll = jax.lax.scan(month,(swarm,states,key),(jnp.arange(T),data))
            return (swarm,key),(jnp.mean(swarm,axis=0),jnp.sum(ll))
        _, trace=jax.lax.scan(iteration,(swarm,key),jnp.arange(steps))
        return trace
    return jax.jit(fit)
