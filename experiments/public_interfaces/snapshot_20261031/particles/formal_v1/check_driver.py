"""Deadline accounting checks and actual small new-IF2 + native-PF NM smoke."""
from pathlib import Path
import sys,json,time
import numpy as np
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import fit

class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t

def main():
    out=ROOT/'local_tests';out.mkdir(exist_ok=True)
    cfg=dict(status='draft',T=2,J=4,budget_seconds=25.,IF2_prefix_seconds=10.,
        IF2_cooling50=.5,train_root=24981000,methods=['A','B'],starts=list(range(6)),repeats=2)
    clock=Clock();seen=[]
    def pf(z,J,seed,T):
        clock.t+=6;seen.append((np.asarray(z).copy(),seed))
        return SimpleNamespace(logLt=-100.+len(seen),t=T)
    clouds=[]
    def step(factory,data,cloud,scales,vectorized,resampling):
        assert vectorized is True and resampling=='multinomial'
        clouds.append(cloud.copy());clock.t+=7
        new=cloud+.1
        return SimpleNamespace(theta_particles=new,theta_mean=new.mean(0),loglik=-50.,smc=SimpleNamespace())
    b=fit.fit(cfg,1,out/'B_deadline',pf=pf,public_if2=step,clock=clock)
    assert b['charged_fit_seconds']==26 and b['accepted_IF2_passes']==1 and b['late_IF2_passes']==1
    assert b['accepted_PF_calls']==1 and b['late_PF_calls']==1 and b['best_training_call']==0
    np.testing.assert_array_equal(clouds[1],clouds[0]+.1)
    initial=np.asarray(json.loads((out/'B_deadline/NM_initial.json').read_text())['parameters'])
    np.testing.assert_array_equal(initial,clouds[1].mean(0))
    np.testing.assert_array_equal(b['parameters_estimation'],seen[0][0])
    aclock=Clock();aseq=[]
    def apf(z,J,seed,T):
        aclock.t+=6;aseq.append(seed);return SimpleNamespace(logLt=-100.+len(aseq),t=T)
    a=fit.fit(cfg,0,out/'A_deadline',pf=apf,clock=aclock)
    assert a['charged_fit_seconds']==30 and a['accepted_PF_calls']==4 and a['late_PF_calls']==1
    assert a['best_training_call']==3 and not a['public_IF2_called']
    assert len(set(aseq+[s for _,s in seen]))==len(aseq)+len(seen)
    assert a['initial_estimation_sha256']==b['initial_estimation_sha256']
    before=clock.t
    again=fit.fit(cfg,1,out/'B_deadline',pf=pf,public_if2=step,clock=clock)
    assert clock.t==before and again==b
    fclock=Clock()
    def failure(*a,**kw):fclock.t+=3;raise FloatingPointError('injected failure')
    failed=fit.fit(cfg,3,out/'B_failure',pf=pf,public_if2=failure,clock=fclock)
    assert failed['fit_failed'] and failed['failed_stage']=='IF2' and failed['IF2_seconds']==3
    assert failed['PF_calls']==0 and failed['charged_fit_seconds']==3
    real=dict(cfg,T=24,J=64,budget_seconds=2.,IF2_prefix_seconds=1.)
    actual=[fit.fit(real,i,out/('actual_'+str(i))) for i in range(2)]
    assert all(not x['fit_failed'] and x['PF_calls']>0 for x in actual)
    assert actual[1]['accepted_IF2_passes']>0 and actual[1]['public_IF2_called']
    assert actual[0]['initial_estimation_sha256']==actual[1]['initial_estimation_sha256']
    record=dict(passed=True,late_calls_rejected=True,late_cost_retained=True,full_swarm_carried=True,
        IF2_late_cloud_not_used_for_NM=True,common_initial_exact=True,training_seeds_distinct=True,
        completed_fit_not_repeated=True,failure_not_retried=True,failed_cost_retained=True,
        actual_T24_J64_new_interface_and_native_PF_smoke=actual)
    fit.write(out/'checks.json',record);print(json.dumps(fit.safe(record),indent=2))
if __name__=='__main__':main()
