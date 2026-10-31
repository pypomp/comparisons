"""One bounded A40 run: new public API, original Dacca and native PF timing."""
from pathlib import Path
import os, sys, json, time, hashlib, traceback, importlib.metadata
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'source/candidate'), str(ROOT/'source')]
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import model as m
import keyed
import public_api as api

OUT = ROOT/'validation_results'
OUT.mkdir(exist_ok=True)
def write(name, value):
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sync(value):
    return [x.numpy() for x in tf.nest.flatten(value)]
def measure(fn, arguments, repeats=3, warmups=1):
    t=time.perf_counter();fn.get_concrete_function(*arguments(0));build=time.perf_counter()-t
    warm=[];times=[];values=[]
    for i in range(warmups+repeats):
        t=time.perf_counter();v=sync(fn(*arguments(i)));dt=time.perf_counter()-t
        if i<warmups:warm.append(dt)
        else:times.append(dt);values.append(v)
    return {'graph_build_seconds':build,'warmup_seconds':warm,'timed_seconds':times,
            'median_synchronized_seconds':float(np.median(times))},values
def stats(x):
    x=np.asarray(x,dtype=np.float64);w=np.exp(x-x.max())
    return {'R':len(x),'logmeanlik':float(x.max()+np.log(w.mean())),
            'MCSE':float(w.std(ddof=1)/np.sqrt(len(x))/w.mean()),
            'likelihood_ESS':float(w.sum()**2/(w*w).sum())}

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for n,h in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    gpus=tf.config.list_physical_devices('GPU');assert gpus,'GPU required; CPU fallback prohibited'
    for g in gpus:tf.config.experimental.set_memory_growth(g,True)
    details=[tf.config.experimental.get_device_details(g) for g in gpus]
    assert all('A40' in g.get('device_name','') for g in details),details
    expected=json.loads((ROOT/'source/runtime_versions.json').read_text())
    versions={k:importlib.metadata.version(k) for k in expected};assert versions==expected
    environment={'GPU':details,'versions':versions,'candidate_path':tfp.__file__,
       'graph_options':tf.config.optimizer.get_experimental_options(),'XLA':False,
       'note':'native TF graph; no automatic GPU-to-CPU or XLA fallback'}
    write('environment.json',environment)
    results={}
    with tf.device('/GPU:0'):
        probe=tf.matmul(tf.ones([4,4]),tf.ones([4,4]));assert 'GPU' in probe.device;sync(probe)
        model=keyed.Dacca(T=600)
        published=m.to_est(m.DATA['theta_published']).reshape(1,23)
        e=tf.constant(published);members=tf.constant([[0,0]],tf.int32)
        args=lambda i:(e,tf.constant([24881001,i]),members)
        # Published point, 600 real observations plus strictly zero dummy t0.
        pristine=api.pf_kernel(model,5000,pristine=True)
        candidate=api.pf_kernel(model,5000,alpha=1.)
        for label,fn in [('original_native',pristine),('candidate_alpha1',candidate)]:
            timing,values=measure(fn,args,7,2)
            raw=[float(v[0].sum()) for v in values]
            assert all(np.isfinite(raw))
            assert all(v[0].shape==(601,1) and (v[0][0]==0).all() for v in values)
            timing.update(raw_loglik=raw,T=600,J=5000,batch=1,output_device=fn(*args(50)).device)
            assert 'GPU' in timing['output_device']
            results[label]=timing;write('partial.json',results)
        a=sync(pristine(*args(60)))[0];b=sync(candidate(*args(60)))[0]
        np.testing.assert_array_equal(a,b)
        results['native_forward_exact']=True
        # Full-data independent target gate, 36 distinct members.
        control=api.pf_kernel(model,5000)
        raw=[];seconds=[]
        for j in range(3):
            p=tf.constant(np.repeat(published,12,axis=0));ids=tf.constant([[11,j*12+i] for i in range(12)])
            t=time.perf_counter();v=control(p,tf.constant([24881002,0]),ids).numpy()
            raw.extend(v.sum(0).tolist());seconds.append(time.perf_counter()-t)
        observed=stats(raw);ref=json.loads((ROOT/'source/input/reference_published_pf.json').read_text())['published']
        delta=observed['logmeanlik']-ref['logmeanlik'];se=float(np.hypot(observed['MCSE'],ref['MCSE']))
        passed=bool(abs(delta)+1.96*se<=5 and observed['MCSE']<=1 and observed['likelihood_ESS']>=10)
        results['original_target_control']={'passed':passed,'raw_loglik':raw,'summary':observed,'reference':ref,
           'difference':delta,'combined_MCSE':se,'seconds':seconds}
        write('partial.json',results);assert passed,'Original Dacca PF statistical control failed'
        # At one point, each public gradient uses identical PF noise.
        old=api.pf_kernel(model,5000,pristine=True,differentiate=True)
        old_inc,old_g=sync(old(*args(70)))
        results['gradients']={}
        for alpha in [1.,.97]:
            fn=api.pf_kernel(model,5000,alpha=alpha,differentiate=True)
            t=time.perf_counter();inc,g=sync(fn(*args(70)));dt=time.perf_counter()-t
            np.testing.assert_array_equal(inc,old_inc);assert np.isfinite(g).all()
            if alpha==1.:np.testing.assert_array_equal(g,old_g)
            results['gradients'][str(alpha)]={'same_forward':True,'finite':True,
              'max_gradient_difference_from_pristine_alpha1':float(np.max(np.abs(g-old_g))),
              'graph_and_first_execution_seconds':dt}
        write('partial.json',results)
        # Public IF2 is measured B1 and B12; formal error isolation uses B1.
        results['throughput']={}
        for B in [1,12]:
            ps=tf.constant(np.repeat(published,B,axis=0));ids=tf.constant([[20,i] for i in range(B)])
            cloud=tf.broadcast_to(ps[None],[5000,B,23]);fn=api.if2_kernel(model,5000)
            arg=lambda i:(cloud,tf.constant(0.),tf.constant([24881003,i]),ids)
            timing,vals=measure(fn,arg,2,1)
            assert all(np.isfinite(x).all() for v in vals for x in v)
            timing.update(batch=B,per_endpoint_seconds=timing['median_synchronized_seconds']/B,
                          formal_batch_use='B1 for exception isolation' if B==1 else 'throughput diagnostic only')
            results['throughput'][f'IF2_B{B}']=timing;write('partial.json',results)
            for alpha in [1.,.97]:
                fn=api.gradient_kernel(model,5000,alpha)
                arg=lambda i:(ps,tf.constant([24881004,i]),ids)
                timing,vals=measure(fn,arg,2,1)
                assert all(np.isfinite(x).all() for v in vals for x in v)
                timing.update(batch=B,per_endpoint_seconds=timing['median_synchronized_seconds']/B)
                results['throughput'][f'gradient_{alpha}_B{B}']=timing;write('partial.json',results)
        # Actual one-update public optimizer regression, unchanged frozen rates.
        rates=tf.constant(np.load(ROOT/'source/input/learning_rates_400.npy')[0])
        ps=tf.constant(np.repeat(published,12,axis=0));ids=tf.constant([[30,i] for i in range(12)])
        direct=api.Adam(model,ps,5000,1.);branch=api.Adam(model,ps,5000,1.)
        initial=direct.snapshot();other=branch.snapshot()
        assert all(np.array_equal(initial[k],other[k]) for k in initial)
        one=sync(direct.step(tf.constant([24881005,0]),ids,rates))
        two=sync(branch.step(tf.constant([24881005,0]),ids,rates))
        for a,b in zip(one,two):np.testing.assert_array_equal(a,b)
        assert one[-1].all()
        # Deliberate test-only gradient NaN checks row masks and zero moments.
        def poisoned(p,key,member):
            grad=tf.ones_like(p)
            grad=tf.tensor_scatter_nd_update(grad,[[3,0]],[tf.constant(float('nan'))])
            return tf.zeros([12]),grad
        test=api.Adam(model,ps,5000,1.,kernel=poisoned)
        before=test.snapshot();v=sync(test.step(tf.constant([24881006,0]),ids,rates));after=test.snapshot()
        assert v[-1].tolist()==[i!=3 for i in range(12)]
        np.testing.assert_array_equal(after['parameters'][3],before['parameters'][3])
        for k in before:
            if k.startswith('opt_') and np.ndim(before[k])==1:
                assert after[k][3]==before[k][3]
        results['Adam_regression']={'same_initial_parameters_and_zero_state':True,'same_alpha1_update_exact':True,
            'one_nonfinite_gradient_row_preserved_other_11':True,'rates_first':rates.numpy().tolist(),
            'beta1':.9,'beta2':.999,'epsilon_hat':1e-8,'clipping':None}
        results['peak_gpu_memory']=tf.config.experimental.get_memory_info('GPU:0')
        results['passed']=True;write('COMPLETE.json',results)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write('failure.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise
