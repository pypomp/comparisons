"""Audit collected terminal tasks without running a filter or optimizer."""
from pathlib import Path
import argparse,datetime,hashlib,json,numpy as np
R=Path(__file__).resolve().parent;F=R/'formal_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def psha(v):return hashlib.sha256(np.asarray(v,dtype='<f8').tobytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def audit(tasks, task_directories=None):
 jobs=read(F/'jobs.json');cfg=read(F/'protocol.json');starts=np.load(R/'input/paper_starts_estimation_free.npy')
 protocol=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 rows=[];allseeds=set()
 for t in tasks:
  p=(task_directories or {}).get(t,F/'gl/results'/f'task_{t:02d}');manifest=read(p/'terminal_manifest.json')
  assert manifest['source_manifest_sha256']==jobs['source_manifest_sha256']
  for name,h in manifest['sha256'].items():assert sha(p/name)==h,(t,name)
  e=read(p/'endpoint.json');ini=read(p/'initial.json');nm=read(p/'NM_initial.json');rt=read(p/'runtime.json')
  assert e['task_id']==t and e['protocol_sha256']==protocol
  assert (e['start_id']*2+e['replicate'])*2+int(e['method']=='B')==t
  assert np.array_equal(ini['parameters'],starts[e['start_id']])
  assert psha(ini['parameters'])==ini['parameter_sha256']==e['initial_estimation_sha256']
  assert psha(e['parameters_estimation'])==e['estimation_sha256']
  assert psha(e['parameters_natural'])==e['natural_sha256']
  th=dict(zip(e['estimation_parameter_names'],e['parameters_estimation']))
  for n in ['gamma','epsilon','m','sigma','tau']:th[n]=np.exp(th[n])
  th['beta_trend']/=100
  ivps=['S_0','I_0','R1_0','R2_0','R3_0'];iv=np.exp([th[n] for n in ivps]);iv/=iv.sum()
  th.update(dict(zip(ivps,iv)));th.update(rho=0.,c=1.,alpha=1.,delta=.02,Y_0=0.)
  assert np.allclose([th[n] for n in e['parameter_names']],e['parameters_natural'],rtol=1e-14,atol=1e-14)
  assert read(p/'score_input.json')==[e] and e['scoring_used_for_selection'] is False
  calls=[read(x) for x in sorted((p/'NM_calls').glob('call_*.json'))]
  assert len(calls)==e['PF_calls']
  for i,c in enumerate(calls):
   assert c['call']==i and c['observations']==600
   assert c['parameter_sha256']==psha(c['parameters'])
   assert c['eligible']==(c['finish_seconds']<=1800)
   assert abs(c['seconds']-(c['finish_seconds']-c['start_seconds']))<1e-9
   assert c['seed']==24981000+100000*t+50000+i and c['seed'] not in allseeds
   allseeds.add(c['seed'])
  eligible=[c for c in calls if c['eligible']]
  best=max(eligible,key=lambda c:c['loglik']) if eligible else None
  assert e['accepted_PF_calls']==len(eligible) and e['late_PF_calls']==len(calls)-len(eligible)
  if best:
   assert e['best_training_call']==best['call'] and e['best_training_loglik']==best['loglik']
   assert e['parameters_estimation']==best['parameters']
  passes=read(p/'IF2_passes.json') if (p/'IF2_passes.json').exists() else []
  assert len(passes)==e['attempted_IF2_passes']
  if e['method']=='B':
   initial=np.load(p/'IF2_initial.npz');assert np.array_equal(initial['cloud'],np.tile(ini['parameters'],(5000,1)))
   accepted=[]
   for i,c in enumerate(passes):
    assert c['iteration']==i and c['accepted']==(c['finish_seconds']<=900)
    assert c['actual_SMC_class']=='particles.core.SMC'
    sp=p/c['swarm_file'];assert sha(sp)==c['swarm_sha256'];z=np.load(sp)
    assert z['cloud'].shape==(5000,23) and np.isfinite(z['cloud']).all()
    assert z['rng_keys'].shape==(624,) and str(z['rng_name'])=='MT19937'
    assert np.allclose(z['cloud'].mean(axis=0),c['mean'],rtol=1e-12,atol=1e-12)
    assert psha(c['mean'])==c['mean_sha256']
    if c['accepted']:accepted.append(c)
   assert len(accepted)==e['accepted_IF2_passes'] and len(passes)-len(accepted)==e['late_IF2_passes']
   assert nm['parameters']==(accepted[-1]['mean'] if accepted else ini['parameters'])
   assert read(p/'IF2_endpoint.json')['parameters']==nm['parameters']
  else:assert nm['parameters']==ini['parameters'] and not passes
  assert psha(nm['parameters'])==nm['parameter_sha256']==e['NM_initial_sha256']
  assert e['charged_fit_seconds']>=max(c['finish_seconds'] for c in calls)
  assert rt['process_seconds']>=e['charged_fit_seconds']
  assert abs(e['budget_overshoot_seconds']-max(0,e['charged_fit_seconds']-1800))<1e-9
  rows.append(dict(task=t,method=e['method'],start=e['start_id'],replicate=e['replicate'],fit_failed=e['fit_failed'],files_hashed=len(manifest['sha256']),terminal_manifest_sha256=sha(p/'terminal_manifest.json'),parameter_sha256=e['natural_sha256'],accepted_IF2_passes=e['accepted_IF2_passes'],late_IF2_passes=e['late_IF2_passes'],PF_calls=e['PF_calls'],late_PF_calls=e['late_PF_calls'],charged_fit_seconds=e['charged_fit_seconds'],process_seconds=rt['process_seconds'],independent_score_available=False))
 return dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,partial=True,tasks=tasks,expected_total=24,scored_endpoints=0,fit_failures=sum(x['fit_failed'] for x in rows),training_seed_collisions=0,all_terminal_hashes_verified=True,initials_and_selection_and_cost_checked=True,rows=rows,not_optimization_effect_evidence=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('tasks',nargs='+',type=int);a=ap.parse_args();d=audit(a.tasks)
 dest=F/'partial_audit'/('tasks'+'_'.join(map(str,a.tasks))+'_audit.json');dest.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='rows'},indent=2))
