"""After actual result review, freeze the fifteen-task selection and budget."""
from pathlib import Path
import ast,datetime,hashlib,json,tarfile
from estimate_budget import estimate
from check_selection import check
P=Path(__file__).resolve().parent;ROOT=P.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not (P/'manifest.json').exists() and not (ROOT/'release_approval.json').exists()
    assert read(P/'authorization.json')['fits']==15
    audit=read(ROOT/'bridge_audit.json');assert audit['passed']
    budget=read(P/'budget.json');assert budget==estimate(ROOT)
    assert read(P/'selection_controls.json')==check()
    assert read(P/'budget_controls.json')['passed']
    for name,h in read(ROOT/'manifest.json')['files'].items():assert sha(ROOT/name)==h
    assert audit['manifest_sha256']==sha(ROOT/'manifest.json')
    names=['protocol.json','authorization.json','README.md','estimate_budget.py','budget.json',
           'guard.py','train.sbatch','check_selection.py','selection_controls.json','check_budget.py','budget_controls.json']
    for n in names:
        if n.endswith('.py'):ast.parse((P/n).read_text())
    manifest=dict(files={n:sha(P/n) for n in names},parent_numerical_manifest_sha256=sha(ROOT/'manifest.json'))
    with (P/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    approval=dict(passed=True,fit_count=15,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       authorization_sha256=sha(P/'authorization.json'),manifest_sha256=sha(ROOT/'manifest.json'),
       overlay_manifest_sha256=sha(P/'manifest.json'),budget_sha256=sha(P/'budget.json'),
       audit_sha256=sha(ROOT/'bridge_audit.json'),data_sha256=audit['data_sha256'],
       planned_total_cap=budget['requested_total_including_bridge_seconds'],
       approved_indices=sorted(t['array_index'] for t in read(P/'protocol.json')['selected_tasks']),
       rationale='User explicitly authorized15fits after validation, with measured runtime budget. Original36-task release remains prohibited; numerical source and validation thresholds unchanged.')
    with (ROOT/'release_approval.json').open('x') as f:json.dump(approval,f,indent=2);f.write('\n')
    with tarfile.open(P/'release.tar.gz','x:gz') as t:
        for n in names+['manifest.json']:t.add(P/n,arcname='formal15/'+n)
        for n in ['bridge_audit.json','release_approval.json']:t.add(ROOT/n,arcname=n)
    print(json.dumps(approval,indent=2))
if __name__=='__main__':main()
