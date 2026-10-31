"""Fixed public-interface estimation comparison. No retries or best-loss selection."""
from pathlib import Path
import argparse, hashlib, json, time, sys, traceback, importlib.metadata
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/candidate'),str(ROOT/'source')]
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import model as m, keyed, public_api as api

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f:np.savez(f,**state)
    path.with_suffix('.tmp').replace(path)
def sync(value):return [x.numpy() for x in tf.nest.flatten(value)]
def seed(domain,iteration):
    return tf.random.experimental.stateless_fold_in(tf.constant([24881000,domain],tf.int32),iteration)
def check_frozen():
    doc=json.loads((ROOT/'manifest.json').read_text())
    for n,h in doc['sha256'].items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'protocol.json').read_text())['ready']
    return sha(ROOT/'manifest.json')
def setup(out):
    expected=json.loads((ROOT/'source/runtime_versions.json').read_text())
    versions={k:importlib.metadata.version(k) for k in expected};assert versions==expected
    gpus=tf.config.list_physical_devices('GPU');assert gpus,'GPU required'
    for g in gpus:tf.config.experimental.set_memory_growth(g,True)
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    details=[tf.config.experimental.get_device_details(g) for g in gpus]
    assert all('A40' in g.get('device_name','') for g in details)
    write(out/'environment.json',{'GPU':details,'versions':versions,'tensorflow':tf.__version__,'tfp':tfp.__version__,
          'candidate_path':tfp.__file__,'XLA':False,'CPU_fallback':False,
          'graph_options':tf.config.optimizer.get_experimental_options()})

def gradient_regression(model,out):
    """Pre-fit GPU check against the historical wrapper at one published point.

    The old wrapper is used only as a reference in this validation; all fits
    call the new public API directly. The result is not optimization evidence.
    """
    e=tf.constant(m.to_est(m.DATA['theta_published']).reshape(1,23))
    members=tf.constant([[90,0]],tf.int32);key=seed(4000,0);rows=[]
    for alpha,kind in [(1.,'native'),(.97,'discounted')]:
        original=keyed.gradient(model,5000,kind,alpha,False)
        candidate=api.gradient_kernel(model,5000,alpha)
        t=time.perf_counter();a=original(e,key,members);old_values=sync(a);old_s=time.perf_counter()-t
        t=time.perf_counter();b=candidate(e,key,members);new_values=sync(b);new_s=time.perf_counter()-t
        assert all('GPU' in x.device for x in a+b)
        for old,new in zip(old_values,new_values):
            assert np.isfinite(new).all()
            np.testing.assert_allclose(old,new,rtol=2e-5,atol=1e-5)
        rows.append({'alpha':alpha,'loss_and_gradient_within_tolerance':True,'rtol':2e-5,'atol':1e-5,
                     'gradient_bitwise_equal':bool(np.array_equal(old_values[1],new_values[1])),
                     'max_gradient_absolute_difference':float(np.max(np.abs(old_values[1]-new_values[1]))),
                     'reference_compile_first_seconds':old_s,'public_compile_first_seconds':new_s})
    write(out/'gradient_regression.json',{'passed':True,'T':600,'J':5000,'batch':1,
         'actual_GPU':True,'old_wrapper_reference_only':True,'rows':rows})

def prefix(model,initial,cases,out):
    fn=api.if2_kernel(model,5000);allrows=[];parameters=[];active=[]
    batch_start=time.perf_counter()
    for c,p in zip(cases,initial):
        where=out/f"case{c['case_id']:02d}";where.mkdir()
        ids=tf.constant([[c['start_id'],c['replicate']]],tf.int32)
        cloud=tf.broadcast_to(tf.constant(p)[None,None],[5000,1,23])
        timings=[];failure=None;accepted=0
        args=(cloud,tf.constant(0.),seed(1000+c['case_id'],0),ids)
        t=time.perf_counter();fn.get_concrete_function(*args);graph=time.perf_counter()-t
        # Separate disposable warmup; never shares a training or scoring stream.
        t=time.perf_counter()
        try:sync(fn(cloud,tf.constant(0.),seed(2000+c['case_id'],0),ids));warm=time.perf_counter()-t
        except Exception as e:
            warm=time.perf_counter()-t;failure={'stage':'warmup','exception':repr(e),'traceback':traceback.format_exc()}
        save(where/'initial_cloud.npz',{'cloud':cloud.numpy()})
        if failure is None:
            for i in range(175):
                write(where/'inflight.json',{'iteration':i})
                t=time.perf_counter()
                try:
                    new,ll=fn(cloud,tf.constant(float(i)),seed(10,i),ids)
                    new_np,ll_np=sync((new,ll));seconds=time.perf_counter()-t
                    finite=np.isfinite(new_np).all() and np.isfinite(ll_np).all()
                    natural=m.natural(tf.reduce_mean(new,0)).numpy()
                    finite=finite and np.isfinite(natural).all()
                    if not finite:raise FloatingPointError('Nonfinite public IF2 cloud/likelihood/natural point')
                    cloud=new;accepted+=1
                except Exception as e:
                    seconds=time.perf_counter()-t
                    failure={'stage':'fit','iteration':i,'exception':repr(e),'traceback':traceback.format_exc()}
                timings.append(seconds)
                append(where/'calls.jsonl',{'iteration':i,'synchronized_seconds':seconds,'accepted':failure is None})
                save(where/'checkpoint.npz',{'cloud':cloud.numpy(),'accepted':np.array(accepted)})
                (where/'inflight.json').unlink()
                write(out/'status.json',{'case_id':c['case_id'],'iteration':i+1,'target':175,'phase':'IF2'})
                if failure is not None:break
        e=tf.reduce_mean(cloud,0).numpy()[0];parameters.append(e);active.append(failure is None)
        row={**c,'accepted_iterations':accepted,'fit_failed':failure is not None,'failure':failure,
             'fit_seconds':float(sum(timings)),'call_seconds':timings,'graph_build_seconds':graph,'warmup_seconds':warm}
        write(where/'COMPLETE.json',row);allrows.append(row)
    save(out/'prefix.npz',{'parameters':np.asarray(parameters),'active':np.asarray(active)})
    # No PF call here: construct and save the canonical zero Adam state once.
    canonical=api.Adam(model,tf.constant(np.asarray(parameters)),5000,1.,active=np.asarray(active)).snapshot()
    save(out/'expected_Adam_initial.npz',canonical)
    write(out/'expected_Adam_initial_hashes.json',{k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in canonical.items()})
    doc={'rows':allrows,'physical_fit_seconds':sum(x['fit_seconds'] for x in allrows),
         'physical_wall_seconds':time.perf_counter()-batch_start,'prefix_sha256':sha(out/'prefix.npz'),
         'IF2_calls_per_case_target':175,'batch_size':1,'all_cases_retained':True}
    write(out/'COMPLETE.json',doc)

def adam(model,initial,cases,method,out):
    pref=None;active=np.ones(12,dtype=bool)
    if method in ['B','C']:
        root=ROOT/'results/prefix';pref=json.loads((root/'COMPLETE.json').read_text())
        assert sha(root/'prefix.npz')==pref['prefix_sha256']
        dat=np.load(root/'prefix.npz');initial=dat['parameters'];active=dat['active']
    alpha=.97 if method=='C' else 1.
    e=tf.constant(initial);ids=tf.constant([[x['start_id'],x['replicate']] for x in cases],tf.int32)
    engine=api.Adam(model,e,5000,alpha,active=active)
    initial_state=engine.snapshot();save(out/'initial_Adam.npz',initial_state)
    # Hash array values, not ZIP timestamps. B/C must start from identical state.
    state_hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in initial_state.items()}
    if pref is not None:
        expected=json.loads((ROOT/'results/prefix/expected_Adam_initial_hashes.json').read_text())
        assert state_hashes==expected,'B/C parameters and complete zero Adam state must match the common prefix record'
    write(out/'initial_state_hashes.json',state_hashes)
    rates=np.load(ROOT/'source/input/learning_rates_400.npy');assert rates.shape==(400,23)
    t=time.perf_counter();engine.step.get_concrete_function(seed(100,0),ids,tf.constant(rates[0]));graph=time.perf_counter()-t
    t=time.perf_counter();execution_failure=None
    try:sync(engine.step(seed(3000,0),ids,tf.constant(rates[0])))
    except Exception as exc:
        execution_failure={'stage':'warmup','exception':repr(exc),'traceback':traceback.format_exc()}
    warm=time.perf_counter()-t;engine.restore(initial_state)
    assert all(np.array_equal(engine.snapshot()[k],v) for k,v in initial_state.items())
    calls=[];failed_iteration=[None if x else 'IF2_prefix' for x in active];stage_start=time.perf_counter()
    if execution_failure is not None:
        engine.active.assign(tf.zeros_like(engine.active))
        failed_iteration=['batch_warmup_execution_exception' if x else 'IF2_prefix' for x in active]
        write(out/'execution_failure.json',execution_failure)
    domain=20 if method=='A' else 30  # B/C identical gradient noise by explicit design.
    for i in range(400):
        if not engine.active.numpy().any():break
        write(out/'inflight.json',{'iteration':i,'domain':domain})
        old_state=engine.snapshot();old_active=engine.active.numpy().copy();t=time.perf_counter()
        try:value=sync(engine.step(seed(domain,i),ids,tf.constant(rates[i])))
        except Exception as exc:
            seconds=time.perf_counter()-t;calls.append(seconds)
            engine.restore(old_state);engine.active.assign(tf.zeros_like(engine.active))
            execution_failure={'stage':'fit','iteration':i,'exception':repr(exc),'traceback':traceback.format_exc(),
                 'scope':'unrecoverable whole-call runtime failure, not attributed to one numerical row; all last valid states retained; no replay'}
            for row in np.flatnonzero(old_active):failed_iteration[row]='batch_execution_exception_at_'+str(i)
            write(out/'execution_failure.json',execution_failure)
            append(out/'calls.jsonl',{'iteration':i,'synchronized_seconds':seconds,'execution_failed':True,'active':engine.active.numpy().tolist()})
            save(out/'checkpoint.npz',engine.snapshot());(out/'inflight.json').unlink();break
        seconds=time.perf_counter()-t
        params,loss,g,new_active=value
        for row in np.flatnonzero(old_active & ~new_active):failed_iteration[row]=i
        if (old_active & ~new_active).any():save(out/f'failure_{i:04d}.npz',{'parameters':params,'loss':loss,'gradient':g,'active':new_active})
        calls.append(seconds)
        append(out/'calls.jsonl',{'iteration':i,'synchronized_seconds':seconds,'active':new_active.tolist()})
        save(out/'checkpoint.npz',engine.snapshot())
        (out/'inflight.json').unlink()
        if (i+1)%10==0:write(out/'status.json',{'method':method,'iteration':i+1,'target':400,'active':int(new_active.sum()),'fit_seconds':sum(calls)})
    final=engine.parameters();natural=m.natural(tf.constant(final)).numpy()
    rows=[]
    for i,c in enumerate(cases):
        rows.append({**c,'method':method,'parameters_natural':natural[i].tolist(),'parameter_names':m.NAMES,
           'fit_failed':not bool(engine.active.numpy()[i]),'iterations_completed':int(engine.accepted.numpy()[i]),
           'IF2_iterations_completed':0 if pref is None else pref['rows'][i]['accepted_iterations'],
           'failure_iteration':failed_iteration[i],'gradient_discount':alpha,
           'whole_call_execution_failure':execution_failure is not None,
           'train_root':24881000,'gradient_domain':domain,'member_key':[c['start_id'],c['replicate']],
           'endpoint_rule':'last valid iterate; no selection by training loss',
           'parameters_estimation':final[i].tolist()})
    write(out/'endpoints.json',rows)
    write(out/'FITS_COMPLETE.json',{'method':method,'count':len(rows),'fit_failed':sum(x['fit_failed'] for x in rows),
         'gradient_fit_synchronized_seconds':sum(calls),'gradient_stage_wall_seconds':time.perf_counter()-stage_start,
         'graph_build_seconds':graph,'first_execution_warmup_seconds':warm,
         'prefix_fit_seconds_charged_to_method':0 if pref is None else pref['physical_fit_seconds'],
         'prefix_wall_seconds_charged_to_method':0 if pref is None else pref['physical_wall_seconds'],
         'prefix_graph_seconds_charged_to_method':0 if pref is None else sum(x['graph_build_seconds'] for x in pref['rows']),
         'prefix_warmup_seconds_charged_to_method':0 if pref is None else sum(x['warmup_seconds'] for x in pref['rows']),
         'timing_scope':'synchronized fit calls are subsets of stage wall; graph/warmup separate; do not add nested times',
         'prefix_physically_reused':method in ['B','C'],'call_seconds':calls,
         'initial_state_array_hashes':state_hashes,'independent_original_PyPOMP_scoring_required':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['prefix','A','B','C'],required=True);a=ap.parse_args()
    out=ROOT/'results'/a.method;out.mkdir(parents=True,exist_ok=True)
    if (out/'STARTED.json').exists():raise RuntimeError('This method already started; no retry or silent replay')
    manifest=check_frozen();write(out/'STARTED.json',{'time':time.time(),'manifest':manifest})
    try:
        setup(out)
        cases=[{'case_id':2*s+r,'start_id':s,'replicate':r} for s in range(6) for r in range(2)]
        initial=np.load(ROOT/'source/input/paper_starts_estimation_free.npy')[[c['start_id'] for c in cases]]
        with tf.device('/GPU:0'):
            probe=tf.matmul(tf.ones([4,4]),tf.ones([4,4]));assert 'GPU' in probe.device
            model=keyed.Dacca(T=600)
            if a.method=='prefix':prefix(model,initial,cases,out)
            else:
                gradient_regression(model,out)
                adam(model,initial,cases,a.method,out)
    except BaseException as e:
        write(out/'failure.json',{'exception':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
