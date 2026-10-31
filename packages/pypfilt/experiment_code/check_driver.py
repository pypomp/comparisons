"""Small tests of budget exclusion, interrupted replay, and evaluation recovery."""
from pathlib import Path
import copy,json,shutil,time
import fit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'checks';OUT.mkdir(exist_ok=True)
base=json.loads((ROOT/'protocol_draft.json').read_text())
base.update(T=24,J=64,starts=[0],repeats=1,budget_seconds=.025)

def fresh(name):
    p=OUT/name
    if p.exists():shutil.rmtree(p)
    return p

calls=[]
def late_pf(theta,T,J,seed):
    calls.append(seed);time.sleep(.005 if len(calls)==1 else .04)
    return dict(loglik=1. if len(calls)==1 else 999.,seed=seed)
p=fresh('late');r=fit.fit_one(base,0,0,p,late_pf)
assert r['best_training_loglik']==1. and r['late_calls']==1 and r['PF_calls']==2
assert r['charged_fit_seconds']>base['budget_seconds']
assert fit.fit_one(base,0,0,p,lambda *_:(_ for _ in ()).throw(AssertionError('repeat'))) == r

config=copy.deepcopy(base);config['budget_seconds']=.15
count=0
def interrupted_pf(theta,T,J,seed):
    global count
    count+=1;time.sleep(.004)
    if count==4:raise fit.Interrupted('controlled test')
    return dict(loglik=float(count),seed=seed)
p=fresh('resume')
try:fit.fit_one(config,0,0,p,interrupted_pf)
except fit.Interrupted:pass
else:raise AssertionError('no interruption')
state=fit.read(p/'fit_state.json');assert state['phase']=='interrupted'
before=list((p/'calls').glob('call_*.json'));assert len(before)==3
hashes={q.name:q.read_bytes() for q in before}
resumed=[]
def resume_pf(theta,T,J,seed):
    resumed.append(seed);time.sleep(.008)
    return dict(loglik=float(len(resumed)+4),seed=seed)
r=fit.fit_one(config,0,0,p,resume_pf)
assert r['interrupted_PF_attempts']==1 and len(resumed)==r['PF_calls']-3
assert all((p/'calls'/name).read_bytes()==value for name,value in hashes.items())
eligible=[fit.read(q) for q in sorted((p/'calls').glob('call_*.json')) if fit.read(q)['eligible']]
assert r['best_training_loglik']==max(row['loglik'] for row in eligible)
assert all(row['finish_fit_seconds']<=config['budget_seconds'] for row in eligible)

# Low ESS must retain all 288 raw repetitions, and recovery must not rescore
# the first two finished batches or change the fitted endpoint.
config['score'].update(J=64,batch_repetitions=36,stages=list(range(36,289,36)))
endpoint=fit.read(p/'endpoint.json');endpoint['config_sha256']=fit.sha(config)
fit.write(p/'endpoint.json',endpoint)
scored=[]
def low_ess_ref(theta,T,J,R,seed):
    scored.append(seed)
    return dict(raw_loglik=[0.]+[-100.]*(R-1),compile_seconds=0.,seconds=.001)
first=fit.score_one(config,0,0,p,lambda _:len(scored)<2,low_ess_ref)
assert not first['finished'] and first['repetitions']==72
second=fit.score_one(config,0,0,p,reference=low_ess_ref)
assert second['finished'] and not second['precision_ok'] and second['repetitions']==288
assert len(scored)==8 and len(set(scored))==8 and second['endpoint_unchanged']

q=fresh('precise_score');q.mkdir();fit.write(q/'endpoint.json',endpoint)
precise=[]
def precise_ref(theta,T,J,R,seed):
    precise.append(seed)
    return dict(raw_loglik=[0.]*R,compile_seconds=0.,seconds=.001)
score=fit.score_one(config,0,0,q,reference=precise_ref)
assert score['finished'] and score['precision_ok'] and score['repetitions']==36 and len(precise)==1
fit.write(OUT/'verification.json',dict(late_candidate_excluded=True,late_cost_counted=True,
    interrupted_objectives_replayed_without_PF=True,interrupted_cost_retained=True,
    completed_endpoint_not_refit=True,score_resume_no_duplicate=True,
    low_precision_288_retained=True,precise_score_stops_at_36=True))
print((OUT/'verification.json').read_text())
