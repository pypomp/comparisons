"""Small real Dacca mechanism tests; no GPU performance claim."""
from pathlib import Path
import tempfile,json
import jax
import jax.numpy as jnp
import numpy as np
from dynamax.parameters import ParameterProperties
import bridge as b
import dacca_model as m
import engine as e
ROOT=Path(__file__).resolve().parent


def main():
    T,J,B=12,16,1;q0=jnp.asarray(np.load(ROOT/'input/paper_starts_estimation_free.npy')[:B]);ids=jnp.array([3]);seed,warm=731,739
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
        tmp=Path(tmp);refs=[]
        train=e.keys(seed,ids,jnp.arange(3))
        for i in range(B):
            ys=jnp.broadcast_to(m.inputs(T)['y'][None,:,None],(3,T,1));ins=jnp.broadcast_to(jax.random.key_data(train[:,i])[:,None],(3,T,2))
            q,_=b.DaccaSGD(J,1.).fit_sgd(q0[i],ParameterProperties(True),ys,ins,optimizer=b.paper_optimizer(3),batch_size=1,num_epochs=1,shuffle=False)
            refs.append(q)
        full,meta,active,failed=e.fit_gradient(q0,tmp/'full/checkpoint.npz',ids,seed,warm,3,1.,J,T,4,'full',stop_after=3)
        _,am,_,_=e.fit_gradient(q0,tmp/'resume/checkpoint.npz',ids,seed,warm,3,1.,J,T,4,'resume',stop_after=1)
        resumed,rm,_,_=e.fit_gradient(q0,tmp/'resume/checkpoint.npz',ids,seed,warm,3,1.,J,T,4,'resume',stop_after=3)
        np.testing.assert_allclose(full,jnp.stack(refs),rtol=3e-5,atol=3e-5)
        np.testing.assert_allclose(full,resumed,rtol=3e-5,atol=3e-5)
        _,aa=e.load(tmp/'full/checkpoint.npz','full');_,bb=e.load(tmp/'resume/checkpoint.npz','resume')
        for k in ['state_0','state_1','state_2','state_3']:np.testing.assert_allclose(aa[k],bb[k],rtol=3e-5,atol=3e-5)
        assert int(aa['state_0'])==3 and np.asarray(active).all() and not np.asarray(failed).any()
        # IF2 chunk/restart preserves complete swarms and accepted-pass cooling.
        af,afm,_,_=e.fit_if2(q0,tmp/'if2full/checkpoint.npz',ids,seed,4,.5,J,T,2,'if2full')
        ar,arm,_,_=e.fit_if2(q0,tmp/'if2resume/checkpoint.npz',ids,seed,4,.5,J,T,2,'if2resume',stop_after=2)
        ar,arm,_,_=e.fit_if2(q0,tmp/'if2resume/checkpoint.npz',ids,seed,4,.5,J,T,2,'if2resume')
        np.testing.assert_array_equal(af,ar)
        _,aa=e.load(tmp/'if2full/checkpoint.npz','if2full');_,bb=e.load(tmp/'if2resume/checkpoint.npz','if2resume')
        np.testing.assert_array_equal(aa['swarm'],bb['swarm'])
        assert afm['iterations']==arm['iterations']==4
        # A completed stage is loaded without any new fit iterations.
        again,againmeta,_,_=e.fit_if2(q0,tmp/'if2full/checkpoint.npz',ids,seed,4,.5,J,T,2,'if2full')
        assert againmeta['fit_seconds']==afm['fit_seconds'];np.testing.assert_array_equal(again,af)
        e.save(tmp/'unclean.npz',dict(identity='unclean',in_flight=True),q=q0)
        blocked=False
        try:e.load(tmp/'unclean.npz','unclean')
        except RuntimeError:blocked=True
        assert blocked
        result=dict(passed=True,T=T,J=J,B=B,official_fit_sgd_reference=True,native_parameter_max_difference=float(jnp.max(jnp.abs(full-jnp.stack(refs)))),
            resume_parameter_max_difference=float(jnp.max(jnp.abs(full-resumed))),Adam_count=3,scalar_API_production_path=True,
            IF2_full_swarm_resume_exact=True,complete_checkpoint_does_not_refit=True,unsafe_inflight_resume_blocked=True,
            scope='Small mechanism check, not final Dacca optimization or GPU speed')
        e.write(ROOT/'results/local_validation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
