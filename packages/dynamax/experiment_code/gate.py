"""One bounded full-size scalar API gate before the formal array starts."""
from pathlib import Path
import json,time,os,hashlib
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dynamax.parameters import ParameterProperties
import bridge as b
import dacca_model as m
import engine as e
import worker as w
ROOT=Path(__file__).resolve().parent
class Done(Exception):pass


class ReferenceOptimizer:
    """Plain original Optax chain, stops after the requested real updates."""
    def __init__(self,q,M,steps):self.q=q;self.base=b.paper_optimizer(M);self.state=self.base.init(q);self.n=0;self.steps=steps
    def init(self,q):np.testing.assert_array_equal(q,self.q);return self.state
    def update(self,g,state,params=None):
        updates,state=self.base.update(g,state);self.q=jax.block_until_ready(optax.apply_updates(self.q,updates));self.state=state;self.n+=1
        if self.n==self.steps:raise Done()
        return updates,state


def main():
    begin=time.perf_counter();device=jax.devices()[0];assert device.platform=='gpu' and 'A40' in device.device_kind
    w.verify_model(600) # Loads the original package's CSV files and checks all data.
    root=ROOT/'gate_results';root.mkdir(exist_ok=True)
    q=jnp.asarray(np.load(ROOT/'input/paper_starts_estimation_free.npy')[0]);trial=jnp.array([0]);seed=22631
    checks={}
    for alpha,M in [(1.,425),(.97,400)]:
        train=e.keys(seed,trial,jnp.arange(M))[:,0];words=jax.random.key_data(train)
        ys=jnp.broadcast_to(m.inputs(600)['y'][None,:,None],(M,600,1));ins=jnp.broadcast_to(words[:,None],(M,600,2))
        ref=ReferenceOptimizer(q,M,2)
        try:b.DaccaSGD(5000,alpha).fit_sgd(q,ParameterProperties(True),ys,ins,optimizer=ref,batch_size=1,num_epochs=1,shuffle=False)
        except Done:pass
        got,meta,active,failed=e.fit_gradient(q[None],root/('alpha'+str(alpha))/'checkpoint.npz',trial,seed,0,M,alpha,5000,600,M,
            'gate:'+w.sha(ROOT/'source_manifest.json')+':'+str(alpha),stop_after=2)
        np.testing.assert_allclose(got[0],ref.q,rtol=2e-5,atol=2e-5)
        saved,a=e.load(root/('alpha'+str(alpha))/'checkpoint.npz','gate:'+w.sha(ROOT/'source_manifest.json')+':'+str(alpha))
        differences=[]
        for i,v in enumerate(jax.tree.leaves(ref.state)):
            np.testing.assert_allclose(a[f'state_{i}'],v,rtol=2e-5,atol=2e-5)
            differences.append(float(np.max(np.abs(np.asarray(a[f'state_{i}'])-np.asarray(v)))))
        assert ref.n==saved['iterations']==2 and bool(active[0]) and not bool(failed[0])
        checks[str(alpha)]=dict(passed=True,parameter_max_difference=float(jnp.max(jnp.abs(got[0]-ref.q))),optimizer_state_max_differences=differences,
            reference_updates=2,checkpointed_updates=saved['iterations'],actual_call='original scalar DaccaSGD -> unchanged SSM.fit_sgd -> original paper Optax chain',
            first_update_including_compile=saved['first_updates_including_internal_compile'],fit_seconds_including_compile=saved['fit_seconds'])
    published=jnp.asarray(m.PUBLISHED);key=jax.random.key(22641);values={}
    for alpha in [1.,.97]:
        value,g=jax.jit(jax.value_and_grad(lambda p:b.local_pf(p,key,m.inputs(600)['y'],5000,alpha)))(published)
        assert np.isfinite(float(value)) and np.isfinite(np.asarray(g)).all();values[str(alpha)]=float(value)
    np.testing.assert_allclose(values['1.0'],values['0.97'],rtol=0,atol=.02)
    # Check that leading batching does not change generated normal words. This
    # narrows the earlier batch discrepancy; it does not certify PF trajectories.
    ks=jax.random.split(jax.random.key(22651),4)
    def particle_words(k):
        def step(key,_):
            key,noise=jax.random.split(key);return key,jax.random.normal(noise,(1,))[0]
        return jax.lax.scan(step,k,None,length=20)[1]
    random_fn=jax.jit(lambda k:jax.vmap(particle_words)(jax.random.split(k,5001)[1:]))
    serial=jnp.stack([random_fn(k) for k in ks]);batched=jax.jit(jax.vmap(random_fn))(ks)
    random_equal=bool(np.array_equal(np.asarray(serial),np.asarray(batched)))
    random_max_difference=float(jnp.max(jnp.abs(serial-batched)))
    record=dict(passed=True,T=600,J=5000,checks=checks,alpha_forward_values=values,alpha_forward_difference=values['1.0']-values['0.97'],
        GPU_random_words_batch_equal=random_equal,GPU_random_words_max_difference=random_max_difference,
        batch_PF_wrapper_approved=False,elapsed_seconds=time.perf_counter()-begin,
        slurm_job=os.environ.get('SLURM_JOB_ID'),source_manifest_sha256=w.sha(ROOT/'source_manifest.json'))
    e.write(root/'PASSED.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        e.write(ROOT/'gate_results/FAILED.json',dict(error=repr(exc),slurm_job=os.environ.get('SLURM_JOB_ID')));raise
