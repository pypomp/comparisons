"""One method and batch of independent fixed-iteration Dacca fits."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,time
import jax
import jax.numpy as jnp
import numpy as np
import pypomp as pp
import engine as e
import dacca_model as m
ROOT=Path(__file__).resolve().parent
METHODS=['IF2','IFAD_097','SGD_alpha1']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(raw):
    a=np.asarray(raw,dtype=float)
    if not np.isfinite(a).all():return dict(valid=False,logmeanlik=None,MCSE=None,likelihood_ESS=None)
    w=np.exp(a-a.max())
    return dict(valid=True,logmeanlik=float(a.max()+np.log(w.mean())),MCSE=float(w.std(ddof=1)/np.sqrt(len(w))/w.mean()),likelihood_ESS=float(w.sum()**2/(w*w).sum()))


def verify_model(T=600):
    st=pp.models.dhaka(dt=None,nstep=20).to_struct()
    np.testing.assert_array_equal(np.asarray(st.ys).reshape(-1),np.asarray(m.DATA['ys']).reshape(-1))
    np.testing.assert_allclose(st.covars_extended,m.DATA['covars'],rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(st.dt_array_extended,m.DATA['dt'],rtol=1e-6,atol=1e-8)
    return st._replace(ys=st.ys[:T],times=st.times[:T],covars_extended=st.covars_extended[:20*T+1],dt_array_extended=st.dt_array_extended[:20*T],nstep_array=st.nstep_array[:T])


def score(q,path,config,trial,method,identity,cap,struct):
    ev=config['evaluation'];R=ev['batch_R'];J=ev['J'];path=Path(path)
    if path.exists():
        record=read(path);assert record['identity']==identity and record['parameters_hash']==e.ah(q)
        if record['complete']:return record
    else:record=dict(identity=identity,parameters_hash=e.ah(q),raw=[],complete=False,compile_seconds=0.,evaluation_seconds=0.,extended=None,
        backend='original pinned PyPOMP',J=J,trial_id=trial,method=method)
    theta=m.from_est(q)[jnp.asarray([m.NAMES.index(n) for n in struct.param_names])]
    def make_keys(offset):return e.keys(config['randomness']['evaluation']+METHODS.index(method),jnp.array([trial]),jnp.arange(offset,offset+R))
    fn=jax.jit(lambda p,ks:pp.functional.pfilter(struct,jnp.broadcast_to(p,(R,len(p))),J=J,keys=ks)['logLik'].reshape(R))
    tick=time.perf_counter();compiled=fn.lower(theta,make_keys(0)).compile();record['memory']=e.memory_gate(compiled,cap)
    record['compile_seconds']+=time.perf_counter()-tick
    target=ev['maximum_R'] if record['extended'] else ev['initial_R']
    while len(record['raw'])<target and not e.STOP:
        tick=time.perf_counter();values=np.asarray(jax.block_until_ready(compiled(theta,make_keys(len(record['raw'])))))
        record['evaluation_seconds']+=time.perf_counter()-tick
        record['raw'].extend([float(v) if np.isfinite(v) else None for v in values])
        if len(record['raw'])==ev['initial_R'] and record['extended'] is None:
            first=stats(record['raw']);record['initial_precision']=first
            record['extended']=not first['valid'] or first['likelihood_ESS']<ev['minimum_ESS'] or first['MCSE']>ev['maximum_MCSE']
            target=ev['maximum_R'] if record['extended'] else ev['initial_R']
        record.update(stats(record['raw']));record['complete']=len(record['raw'])==target
        record['R']=len(record['raw']);record['precision_passed']=record['valid'] and record['likelihood_ESS']>=ev['minimum_ESS'] and record['MCSE']<=ev['maximum_MCSE']
        e.write(path,record)
    return record


def run(config_path,method,batch_index,local=False,output=None):
    config=read(config_path)
    if not local:
        assert config['status']=='APPROVED',config['status']
        assert config['model']==dict(T=600,Euler_steps=20,J=5000,free_coordinates=23)
        assert config['start_indices']==list(range(6)) and config['repetitions']==2
    spec=config['methods'][method];T=config['model']['T'];J=config['model']['J']
    pairs=[(i,r) for i in config['start_indices'] for r in range(config['repetitions'])]
    B=config['batch_size'];pairs=pairs[batch_index*B:(batch_index+1)*B];assert pairs
    trials=jnp.asarray([i*100+r for i,r in pairs]);starts=np.load(ROOT/'input/paper_starts_estimation_free.npy')
    q0=jnp.asarray(np.stack([starts[i] for i,_ in pairs]));identity=sha(config_path)+':'+sha(ROOT/'source_manifest.json')+':'+method+':'+str(batch_index)
    root=Path(output) if output else ROOT/'runs'/method/f'batch{batch_index:02d}';root.mkdir(parents=True,exist_ok=True)
    e.signals();begin=time.perf_counter();device=jax.devices()[0];cap=None
    if not local:
        assert device.platform=='gpu' and 'A40' in device.device_kind,jax.devices()
        dm=device.memory_stats();cap=min(60*1024**3,int(.85*(dm['bytes_limit']-dm.get('bytes_in_use',0))))
    struct=verify_model(T)
    environment=dict(identity=identity,method=method,pairs=pairs,trials=np.asarray(trials).tolist(),initial_parameters_hash=e.ah(q0),
        devices=[str(d) for d in jax.devices()],jax=jax.__version__,pypomp_file=pp.__file__,cap_bytes=cap,
        slurm_job=os.environ.get('SLURM_JOB_ID'),started_at=datetime.now(timezone.utc).isoformat())
    e.write(root/'environment.json',environment);q=q0;prefix=None;active=jnp.ones(len(q),bool);failed=jnp.zeros(len(q),jnp.int32)
    if spec['IF2_passes']:
        seed=config['randomness']['IF2'] if method=='IF2' else config['randomness']['IFAD_prefix']
        q,prefix,active,failed=e.fit_if2(q0,root/'if2/checkpoint.npz',trials,seed,spec['IF2_passes'],spec['cooling'],J,T,config['IF2_chunk'],identity+':if2',cap)
        if not prefix['complete']:e.write(root/'status.json',dict(phase='paused_IF2',identity=identity));return root
        np.savez(root/'IF2_parameters.npz',parameters=np.asarray(q),initial_parameters=np.asarray(q0),trials=np.asarray(trials))
    gradient=None
    if spec['gradient_updates']:
        q,gradient,active,failed=e.fit_gradient(q,root/'gradient/checkpoint.npz',trials,config['randomness']['gradient'],config['randomness']['warmup'],
            spec['gradient_updates'],spec['alpha'],J,T,config['gradient_chunk'],identity+':gradient',initial_active=active,initial_failed=failed)
        if not gradient['complete']:e.write(root/'status.json',dict(phase='paused_gradient',identity=identity));return root
    fit_seconds=sum(s['fit_seconds'] for s in [prefix,gradient] if s is not None)
    prepare_seconds=sum(s['prepare_seconds'] for s in [prefix,gradient] if s is not None)
    endpoint=dict(identity=identity,method=method,pairs=pairs,parameters_hash=e.ah(q),initial_parameters_hash=e.ah(q0),
        active=np.asarray(active).tolist(),failed_at=np.asarray(failed).tolist(),IF2=prefix,gradient=gradient,
        batch_fit_seconds=fit_seconds,batch_prepare_seconds=prepare_seconds,
        attribution='Native Dynamax SSM.fit_sgd optimizer API plus local PF/IF2/IFAD extensions; no native Dynamax PF or Dacca EM')
    e.save(root/'endpoints.npz',endpoint,parameters=q,initial_parameters=q0,trials=trials,active=active,failed_at=failed)
    e.write(root/'endpoints.json',endpoint)
    scores=[]
    for row,((i,rep),trial) in enumerate(zip(pairs,np.asarray(trials))):
        if e.STOP:break
        scores.append(score(q[row],root/f'score_start{i:02d}_rep{rep:02d}.json',config,int(trial),method,identity,cap,struct))
    complete=len(scores)==len(q) and all(v['complete'] for v in scores)
    status=dict(phase='complete' if complete else 'paused_scoring',complete=complete,identity=identity,method=method,
        points=len(q),failed_fits=int(np.sum(~np.asarray(active))),precision_failures=sum(not s['precision_passed'] for s in scores if s['complete']),
        batch_fit_seconds=fit_seconds,batch_prepare_seconds=prepare_seconds,invocation_wall_seconds=time.perf_counter()-begin,
        total_score_seconds=sum(s['evaluation_seconds'] for s in scores),total_score_compile_seconds=sum(s['compile_seconds'] for s in scores),
        slurm_job=os.environ.get('SLURM_JOB_ID'),finished_at=datetime.now(timezone.utc).isoformat())
    e.write(root/'status.json',status)
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='terminal_manifest.json'}
    e.write(root/'terminal_manifest.json',dict(identity=identity,files=files));return root


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--method',choices=METHODS,required=True);ap.add_argument('--batch-index',type=int,required=True);args=ap.parse_args()
    run(args.config,args.method,args.batch_index)
