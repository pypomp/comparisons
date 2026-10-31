"""Official Dynamax SGD using a local sample-only full-Dacca PF objective.

This is an estimator adapter, NOT a native Dynamax particle filter or an
implementation of monthly transition densities / EM sufficient statistics.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dynamax.ssm import SSM
from dynamax.parameters import ParameterProperties
import dacca_model as m


def systematic(key,logw):
    J=len(logw)
    weights=jax.nn.softmax(jax.lax.stop_gradient(logw))
    uniforms=(jax.random.uniform(key,(),dtype=logw.dtype)+jnp.arange(J))/J
    return jnp.clip(jnp.searchsorted(jnp.cumsum(weights),uniforms,method='sort'),0,J-1)


def local_pf(p,key,y,J=16,alpha=1.):
    """600-month model for full input y; same key order as validated Cuthbert.

    In every month: resample, perform all 20 Euler steps, observe. No monthly
    Gaussian approximation or transition-density calculation is introduced.
    """
    T=len(y);data=m.inputs(T);data['y']=y
    init,propagate,observation=m.factory(p)
    ik,fk=jax.random.split(key)
    prepare_keys=jax.random.split(fk,T)
    step_keys=jnp.concatenate((ik[None],prepare_keys[:-1]))
    states=jax.vmap(init)(jax.random.split(ik,J))
    def step(carry,item):
        x,w,ll=carry;inp,key=item
        keys=jax.random.split(key,J+1)
        indices=systematic(keys[0],w)
        correction=alpha*(w[indices]-jax.lax.stop_gradient(w[indices]))
        ancestors=x[indices]
        x=jax.vmap(propagate,(0,0,None))(keys[1:],ancestors,inp)
        w=jax.vmap(observation,(0,0,None))(ancestors,x,inp)+correction
        ll=ll+(jax.scipy.special.logsumexp(w)-jax.scipy.special.logsumexp(correction))
        return (x,w,ll),None
    return jax.lax.scan(jax.checkpoint(step),(states,jnp.zeros(J,p.dtype),jnp.array(0.,p.dtype)),(data,step_keys))[0][2]


class DaccaSGD(SSM):
    """Reuse SSM.fit_sgd; unsupported density and EM methods stay unavailable."""
    def __init__(self,J=16,alpha=1.):self.J,self.alpha=J,alpha
    @property
    def emission_shape(self):return (1,)
    @property
    def inputs_shape(self):return (2,)
    def initial_distribution(self,*args,**kwargs):
        raise NotImplementedError('Estimator-only adapter: use Dacca initial_state simulator')
    def transition_distribution(self,*args,**kwargs):
        raise NotImplementedError('Dacca monthly transition is sample-only; density is unavailable')
    def emission_distribution(self,*args,**kwargs):
        raise NotImplementedError('Estimator-only adapter uses the original observation potential, including its floor')
    def marginal_log_prob(self,params,emissions,inputs=None):
        if inputs is None:raise ValueError('Provide the explicit independent PF key as auxiliary inputs')
        key=jax.random.wrap_key_data(inputs[0].astype(jnp.uint32))
        return local_pf(params,key,emissions[:,0],self.J,self.alpha)


def key_bank(key,updates,T=600):
    """One epoch only: each independent random key is consumed exactly once."""
    keys=jax.random.split(key,updates)
    words=jax.random.key_data(keys)
    inputs=jnp.broadcast_to(words[:,None,:],(updates,T,2))
    emissions=jnp.broadcast_to(m.inputs(T)['y'].reshape(1,T,1),(updates,T,1))
    return keys,emissions,inputs


def paper_optimizer(M,beta1=.9):
    # Same coordinate rates, warmup and cosine decay as frozen paper settings.
    import pypomp as pp
    values={}
    for n in m.FREE:
        rate=.05 if n in m.IVPS or n in ['gamma','beta_trend','sigma','tau'] else .1
        values[n]=np.r_[np.linspace(rate*.1,rate,min(10,M)),np.full(max(0,M-10),rate)]
    eta=pp.LearningRate(values).cosine_decay(final_factor=.05,M=M).to_array(m.FREE,M)
    return optax.chain(optax.scale_by_adam(b1=beta1,b2=.999,eps=1e-8),
                       optax.scale_by_schedule(lambda i:-eta[i]))


def explicit_sgd(model,p,keys,optimizer,T=600):
    state=optimizer.init(p);y=m.inputs(T)['y'];trace=[p];losses=[]
    step=jax.jit(jax.value_and_grad(lambda p,k:-local_pf(p,k,y,model.J,model.alpha)/T))
    for key in keys:
        loss,gradient=step(p,key)
        updates,state=optimizer.update(gradient,state,p)
        p=optax.apply_updates(p,updates)
        trace.append(p);losses.append(loss)
    return p,jnp.stack(losses),jnp.stack(trace)


def local_if2(p,key,J=16,T=600,M=1,cooling=.5):
    """Local IF2 extension with all 23 paper coordinates and no density."""
    data=m.inputs(T)
    initial_sd=jnp.asarray([.16 if n in m.IVPS else 0. for n in m.FREE])
    process_sd=jnp.asarray([0. if n in m.IVPS else .02 for n in m.FREE])
    swarm=jnp.broadcast_to(p,(J,len(p)))
    def iteration(swarm,item):
        i,key=item;key,pk=jax.random.split(key)
        swarm=swarm+cooling**(i/50)*initial_sd*jax.random.normal(pk,swarm.shape)
        x=jax.vmap(lambda p:m.initial_state(m.from_est(p)))(swarm)
        def step(carry,item):
            swarm,x,key=carry;n,inp=item
            key,qk,xk,rk=jax.random.split(key,4)
            swarm=swarm+cooling**((i+n/T)/50)*process_sd*jax.random.normal(qk,swarm.shape)
            x=jax.vmap(lambda p,k,x:m.factory(p)[1](k,x,inp))(swarm,jax.random.split(xk,J),x)
            w=jax.vmap(lambda p,x:m.observation(m.from_est(p),x,inp['y']))(swarm,x)
            idx=systematic(rk,w)
            return (swarm[idx],x[idx],key),jax.scipy.special.logsumexp(w)-jnp.log(J)
        (swarm,_,_),ll=jax.lax.scan(step,(swarm,x,key),(jnp.arange(T),data))
        return swarm,(jnp.mean(swarm,0),ll.sum())
    swarm,(history,ll)=jax.lax.scan(iteration,swarm,(jnp.arange(M),jax.random.split(key,M)))
    return history[-1],history,ll
