"""Synthetic cost accounting controls, no GPU or numerical model execution."""
from pathlib import Path
import copy,json,math,tempfile
from estimate_budget import estimate
P=Path(__file__).resolve().parent
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def check():
    with tempfile.TemporaryDirectory(prefix='maze15-cost-controls-') as temp:
        p=Path(temp);arms=['native_dpf','native_stop','discount097']
        write(p/'formal15/protocol.json',dict(fits=15,arms=arms))
        write(p/'manifest.json',dict(files={}))
        write(p/'bridge_audit.json',dict(passed=True))
        receipt=dict(state='COMPLETED',exit_code='0:0',ElapsedRaw=500)
        write(p/'bridge_collection/collection.json',receipt)
        write(p/'bridge_collection/raw/data/prepared.json',dict(elapsed_seconds=50))
        fixtures={}
        for arm,w in zip(arms,[.5,.6,.7]):
            rows=[dict(name=f'cost{i}',complete_step_seconds=w+.01*i,serialization_seconds=.01) for i in range(3)]
            rows += [dict(name='step2_network',complete_step_seconds=.1,serialization_seconds=.01),
                     dict(name='original_cold',complete_step_seconds=10+w,serialization_seconds=.01),
                     dict(name='original_repeat',complete_step_seconds=w,serialization_seconds=.01)]
            evaluations=[dict(batch=b,seconds=t,variant=v) for v in ['eager','network'] for b,t in [(4,.05),(16,.1),(60,.19),(64,.2)]]
            root=p/'bridge_collection/raw/bridge_results'/arm
            result=dict(passed=True,CPU_only=False,elapsed_seconds=20,rows=rows,evaluations=evaluations)
            write(root/'result.json',result);write(root/'setup.json',dict(seconds=.25));fixtures[arm]=(root,result)
        budget=estimate(p)
        assert budget['fit_count']==15 and sum(x['fits'] for x in budget['methods'].values())==15
        assert budget['bridge_actual_GPU_seconds']==500
        assert budget['requested_total_including_bridge_seconds']==500+sum(5*x['requested_seconds'] for x in budget['methods'].values())
        assert budget['training_requested_GPU_hours']==budget['training_requested_GPU_seconds']/3600
        for arm,w in zip(arms,[.5,.6,.7]):
            m=budget['methods'][arm]
            assert math.isclose(m['train_center_seconds'],1400*(w+.01)+100*.1)
            assert math.isclose(m['evaluation_seconds'],630*.2+100*.1+3*.19)
            assert m['requested_seconds']%60==0 and m['requested_seconds']>=m['per_fit_request_before_rounding_seconds']
        # A serialized budget must compare exactly with a recomputation when
        # preparing release; integer dict keys must not break this invariant.
        assert json.loads(json.dumps(budget))==estimate(p)
        denied=[]
        write(p/'bridge_audit.json',dict(passed=False))
        try:estimate(p)
        except AssertionError:denied.append('failed_audit')
        else:raise AssertionError('failed audit accepted')
        write(p/'bridge_audit.json',dict(passed=True))
        write(p/'bridge_collection/collection.json',dict(receipt,state='RUNNING'))
        try:estimate(p)
        except AssertionError:denied.append('running_allocation')
        else:raise AssertionError('running allocation accepted')
        write(p/'bridge_collection/collection.json',receipt)
        root,result=fixtures['native_dpf'];modified=copy.deepcopy(result)
        for row in modified['rows']:row['complete_step_seconds']=100
        write(root/'result.json',modified)
        try:estimate(p)
        except AssertionError:denied.append('over_safety_ceiling')
        else:raise AssertionError('excessive limit accepted')
        return dict(passed=True,synthetic_only=True,actual_GPU_forecast_available=False,
                    cases_checked=['15fits','full_and_tail_training_counts','shape_specific_validation_test_counts',
                                   'bridge_cost_once','GPU_seconds_to_hours','minute_rounding','JSON_recompute_equality'],
                    refused=denied,model_calls=0,GPU_calls=0)
if __name__=='__main__':
    out=P/'budget_controls.json';assert not out.exists()
    with out.open('x') as f:json.dump(check(),f,indent=2);f.write('\n')
    print('Synthetic15-fit cost accounting passed; real budget still requires GPU records')
