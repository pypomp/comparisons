"""Independent IF2 recursion with explicitly matched dtype/JIT boundaries."""
from pathlib import Path
import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import pfjax
import bridge
import dacca_model as m
from check_dacca import reference_if2

OUT=Path(os.environ.get('PFJAX_VALIDATION_OUTPUT',
    'outputs/competitor_estimation/implementation_validation_20261030/pfjax'))
OUT.mkdir(parents=True,exist_ok=True)
model=bridge.Dacca();data=m.inputs(600);J=8
dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
initial=jnp.broadcast_to(jnp.asarray(m.PUBLISHED,dtype=dtype),(J,23))
rows=[]

def reference(key,data,q,iteration):
    ik,k,fk=jax.random.split(key,3)
    # Match public API dtype before arithmetic, while keeping this recursion
    # independent of particle_filter, iterated_filter and their callbacks.
    scale=jnp.asarray(m.PROCESS_SCALE,q.dtype)
    start=jnp.asarray(m.INITIAL_SCALE,q.dtype)
    q=q+0.5**(iteration/50.)*start*jax.random.normal(ik,q.shape,dtype=q.dtype)
    keys=jax.random.split(k,J+1)
    x,w=jax.vmap(model.pf_init,(0,None,0))(keys[1:],jax.tree.map(lambda a:a[0],data),q)
    def step(carry,item):
        k,x,q,w,ll=carry;t,observation=item
        k,rk=jax.random.split(k)
        ancestors=jax.random.choice(rk,jnp.arange(J),(J,),p=jax.nn.softmax(w))
        q,x=q[ancestors],x[ancestors]
        keys=jax.random.split(k,J+1)
        sk=jax.vmap(lambda k:jax.random.split(k))(keys[1:])
        factor=0.5**((iteration+t/600)/50.)
        noise=jax.vmap(lambda k:jax.random.normal(k,(23,),dtype=q.dtype))(sk[:,0])
        q=q+factor*scale*noise
        x,w=jax.vmap(model.pf_step,(0,0,None,0))(sk[:,1],x,observation,q)
        return (keys[0],x,q,w,ll+jax.scipy.special.logsumexp(w)),None
    (_,x,q,w,ll),_=jax.lax.scan(step,(keys[0],x,q,w,jax.scipy.special.logsumexp(w)),
                                (jnp.arange(1,600),jax.tree.map(lambda a:a[1:],data)))
    ancestors=jax.random.choice(fk,jnp.arange(J),(J,),p=jax.nn.softmax(w))
    return q[ancestors],ll-600*jnp.log(J)

for seed in [20261030,20261031,20261101]:
    candidate=reference_cloud=old_cloud=initial
    for i in range(2):
        k=jax.random.fold_in(jax.random.PRNGKey(seed),i)
        a=jax.jit(lambda q:pfjax.iterated_filter_step(model,k,data,q,m.PROCESS_SCALE,
            initial_scale=m.INITIAL_SCALE,iteration=i,cooling_rate=.5,
            cooling_interval=50.,within_pass_cooling=True))(candidate)
        b,ll=jax.jit(lambda q:reference(k,data,q,i))(reference_cloud)
        old,old_ll=reference_if2(model,k,data,old_cloud,i)
        row=dict(seed=seed,pass_index=i,new_reference_parameters_max=float(jnp.max(jnp.abs(a['parameter_particles']-b))),
            new_reference_loglik_abs=float(jnp.abs(a['loglik']-ll)),
            original_reference_parameters_max=float(jnp.max(jnp.abs(a['parameter_particles']-old))),
            original_reference_loglik_abs=float(jnp.abs(a['loglik']-old_ll)))
        rows.append(row);candidate=a['parameter_particles'];reference_cloud=b;old_cloud=old
result=dict(dtype=str(dtype),rows=rows,compiled_independent_reference_exact=all(
    r['new_reference_parameters_max']==r['new_reference_loglik_abs']==0 for r in rows))
name='diagnostic_float64.json' if jax.config.x64_enabled else 'diagnostic_float32.json'
(OUT/name).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert result['compiled_independent_reference_exact'], 'Matched reference must agree exactly; retain failure report.'
