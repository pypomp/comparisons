"""Verify raw task outputs; partial and low-precision results stay visible."""
from pathlib import Path
import hashlib,json,math,statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
hash_z=lambda z:hashlib.sha256(np.asarray(z,dtype='<f8').tobytes()).hexdigest()
config=read(ROOT/'protocol.json');config_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
starts=np.load(ROOT/'input/paper_starts_estimation_free.npy')
rows=[]
for task,(start,rep) in enumerate((s,r) for s in config['starts'] for r in range(config['repeats'])):
    base=ROOT/'remote/results/tasks'/str(task);folder=base/f'start_{start}_rep_{rep}'
    row=dict(task_id=task,start=start,replicate=rep,phase='not_collected',endpoint=False,score_finished=False,precision_ok=False)
    if (base/'status.json').exists():row['phase']=read(base/'status.json')['phase']
    if (folder/'endpoint.json').exists():
        e=read(folder/'endpoint.json');assert e['config_sha256']==config_hash
        np.testing.assert_array_equal(e['initial_parameters'],starts[start])
        calls=[read(p) for p in sorted((folder/'calls').glob('call_*.json'))]
        assert len(calls)==e['objective_calls']
        eligible=[c for c in calls if c['eligible'] and c['error'] is None]
        for c in calls:
            assert c['eligible']==(c['finish_fit_seconds']<=config['budget_seconds'])
            seed=int(np.random.SeedSequence([config['seed'],start,rep,c['call']]).generate_state(1)[0])
            assert seed==c['requested_seed']
            if c['PF'] is not None:
                assert c['PF']['seed']==seed%(2**31-1)
                assert c['PF']['observations_weighted']==600 and len(c['PF']['incremental_loglik'])==600
        if eligible:
            best=max(eligible,key=lambda c:c['loglik'])
            assert e['best_call']==best['call'] and e['parameter_sha256']==hash_z(best['parameters'])==hash_z(e['parameters'])
        else:assert e['fit_failed'] and e['parameters'] is None
        row.update(endpoint=True,fit_failed=e['fit_failed'],parameter_sha256=e['parameter_sha256'],
            fitting_seconds=e['charged_fit_seconds'],PF_calls=e['PF_calls'],failed_calls=e['failed_calls'],late_calls=e['late_calls'])
    if (folder/'independent_score.json').exists():
        s=read(folder/'independent_score.json');assert s['config_sha256']==config_hash
        assert s['endpoint_sha256']==e['parameter_sha256']
        row.update(score_finished=s['finished'],precision_ok=s['precision_ok'])
        if not s.get('fit_failed'):
            batches=[read(p) for p in sorted((folder/'score_batches').glob('batch_*.json'))]
            raw=[v for batch in batches for v in batch['raw_loglik']]
            assert raw==s['raw_loglik'] and len(raw)==s['repetitions'] and len(raw)<=288
            for batch in batches:
                assert batch['endpoint_sha256']==e['parameter_sha256'] and batch['config_sha256']==config_hash
                expected=int(np.random.SeedSequence([config['score']['seed'],start,rep,batch['batch']]).generate_state(1)[0])
                assert batch['seed']==expected and len(batch['raw_loglik'])==36
                assert (batch['J'],batch['T'])==(5000,600)
            if raw and all(math.isfinite(v) for v in raw):
                m=max(raw);w=[math.exp(v-m) for v in raw];mean=statistics.mean(w)
                actual=dict(logmeanlik=m+math.log(mean),mcse=statistics.stdev(w)/(math.sqrt(len(w))*mean),likelihood_ess=sum(w)**2/sum(v*v for v in w))
                for k,v in actual.items():assert math.isclose(v,s['summary'][k],abs_tol=1e-9,rel_tol=1e-10)
                assert s['precision_ok']==(actual['mcse']<=1 and actual['likelihood_ess']>=10)
                row.update(actual)
            row.update(score_repetitions=len(raw),score_seconds=s['score_wall_seconds'])
    rows.append(row)
summary=dict(method=config['method'],native_optimizer=False,device='CPU',cpus_per_task=1,scope='12 endpoints; CPU optimizer costs are not a same-hardware ranking against GPU IF2/IFAD',
    expected=12,endpoints=sum(r['endpoint'] for r in rows),scores_finished=sum(r['score_finished'] for r in rows),precision_ok=sum(r['precision_ok'] for r in rows),rows=rows)
out=ROOT.parents[3]/'outputs/competitor_estimation/pypfilt/formal_v1';out.mkdir(parents=True,exist_ok=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
