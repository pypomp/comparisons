"""Non-numerical controls for the selected fifteen and per-method job limits."""
from pathlib import Path
import copy,json
from guard import validate
P=Path(__file__).resolve().parent

def check():
    protocol=json.loads((P/'protocol.json').read_text())
    budget=dict(passed=True,fit_count=15,requested_total_including_bridge_seconds=18000,
                methods={arm:dict(requested_minutes=m) for arm,m in zip(protocol['arms'],[18,20,22])})
    approval=dict(passed=True,fit_count=15,approved_indices=sorted(t['array_index'] for t in protocol['selected_tasks']),planned_total_cap=18000)
    for i,t in enumerate(protocol['selected_tasks']):
        actual=validate(t['array_index'],'maze15-'+t['arm']+'-20261102',budget['methods'][t['arm']]['requested_minutes'],protocol,budget,approval)
        assert actual['seed']==[0,10,20,30,40][i//3] and actual['arm']==protocol['arms'][i%3]
    denied=[]
    for index in sorted(set(range(36))-set(approval['approved_indices'])):
        try:validate(index,'maze15-native_dpf-20261102',18,protocol,budget,approval)
        except (AssertionError,StopIteration,KeyError):denied.append(index)
        else:raise AssertionError('Unselected legacy task allowed')
    controls=[]
    def reject(name,pr,bu,ap,job='maze15-native_dpf-20261102',minutes=18):
        try:validate(0,job,minutes,pr,bu,ap)
        except (AssertionError,StopIteration,KeyError):controls.append(name)
        else:raise AssertionError('Tampering accepted: '+name)
    for key,value in [('fits',36),('cases',12),('training_repetitions',2)]:
        pr=copy.deepcopy(protocol);pr[key]=value;reject(key,pr,budget,approval)
    pr=copy.deepcopy(protocol);pr['selected_tasks'][0]['seed']=10;reject('wrong_seed',pr,budget,approval)
    pr=copy.deepcopy(protocol);pr['selected_tasks'][0]['legacy_case']=1;reject('wrong_case',pr,budget,approval)
    pr=copy.deepcopy(protocol);pr['selected_tasks'][1]=pr['selected_tasks'][0];reject('duplicate_task',pr,budget,approval)
    reject('wrong_method_name',protocol,budget,approval,job='maze15-discount097-20261102')
    reject('wrong_limit',protocol,budget,approval,minutes=45)
    ap=copy.deepcopy(approval);ap['fit_count']=36;reject('old_release_count',protocol,budget,ap)
    ap=copy.deepcopy(approval);ap['planned_total_cap']+=60;reject('wrong_budget',protocol,budget,ap)
    return dict(passed=True,selected_tasks_validated=15,unselected_legacy_tasks_rejected=denied,
                tamper_controls_rejected=controls,model_calls=0,GPU_calls=0,
                numeric_source_or_old_frozen_protocol_modified=False,
                runtime_source_hash_and_actual_scontrol_checks_not_exercised_by_synthetic_control=True)

if __name__=='__main__':
    output=P/'selection_controls.json';assert not output.exists()
    with output.open('x') as f:json.dump(check(),f,indent=2);f.write('\n')
    print('15 selections and21 excluded legacy tasks checked;10 tamper controls rejected')
