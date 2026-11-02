"""Reject any task outside the actual approved fifteen and measured time limits."""
from pathlib import Path
import hashlib,json,os,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate(index,job_name,minutes,protocol,budget,approval):
    assert protocol['fits']==15 and protocol['cases']==5 and protocol['training_repetitions']==1
    selected=protocol['selected_tasks'];assert len(selected)==15
    assert len({r['array_index'] for r in selected})==15
    assert sorted({r['legacy_case'] for r in selected})==[0,2,4,6,8]
    task=next(x for x in selected if x['array_index']==index)
    assert task['legacy_case']==index//3 and task['arm']==protocol['arms'][index%3]
    assert task['seed']==10*(task['legacy_case']//2) and task['repetition']==0
    assert job_name=='maze15-'+task['arm']+'-20261102'
    assert minutes==budget['methods'][task['arm']]['requested_minutes']
    assert budget['fit_count']==15 and budget['passed'] and approval['fit_count']==15 and approval['passed']
    assert approval['approved_indices']==sorted(x['array_index'] for x in selected)
    assert approval['planned_total_cap']==budget['requested_total_including_bridge_seconds']
    return task
def main():
    manifest=read(P/'manifest.json')
    for n,h in manifest['files'].items():assert sha(P/n)==h,n
    approval=read(ROOT/'release_approval.json')
    assert approval['overlay_manifest_sha256']==sha(P/'manifest.json')
    assert approval['manifest_sha256']==sha(ROOT/'manifest.json')
    assert approval['audit_sha256']==sha(ROOT/'bridge_audit.json')
    assert approval['budget_sha256']==sha(P/'budget.json')
    # SLURM_TIMELIMIT is not uniformly exported. Inspect this actual allocation
    # using its explicit task id, never infer resources from a template file.
    query=subprocess.run(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],capture_output=True,text=True)
    assert query.returncode==0,query.stderr
    fields=dict(x.split('=',1) for x in query.stdout.split() if '=' in x)
    pieces=fields['TimeLimit'].split(':');assert len(pieces)==3 and '-' not in pieces[0]
    seconds=int(pieces[0])*3600+int(pieces[1])*60+int(pieces[2]);assert seconds%60==0
    task=validate(int(os.environ['SLURM_ARRAY_TASK_ID']),os.environ['SLURM_JOB_NAME'],seconds//60,
                  read(P/'protocol.json'),read(P/'budget.json'),approval)
    destination=ROOT/'guard_records';destination.mkdir(exist_ok=True)
    with (destination/(task['arm']+'_'+str(task['legacy_case']).zfill(2)+'.json')).open('x') as f:
        json.dump(dict(passed=True,task=task,job=os.environ['SLURM_JOB_ID'],array_job=os.environ['SLURM_ARRAY_JOB_ID'],
                       actual_TimeLimit_seconds=seconds,overlay_manifest_sha256=sha(P/'manifest.json'),
                       release_sha256=sha(ROOT/'release_approval.json')),f,indent=2)
    print('Approved one of fifteen tasks:',task)
if __name__=='__main__':main()
