"""One new sparse array per method: five paired seeds, actual measured limits."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
P=Path(__file__).resolve().parent;ROOT=P.parent
SSH=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def call(command,label,data=None):
    r=subprocess.run(SSH+[command],input=data,capture_output=True)
    (P/(label+'.stdout')).write_bytes(r.stdout);(P/(label+'.stderr')).write_bytes(r.stderr)
    assert r.returncode==0,r.stderr.decode(errors='replace')
    return r.stdout.decode()
def main():
    assert not (P/'submission_intent.json').exists() and not (P/'jobs.json').exists()
    assert set(read(ROOT/'jobs.json'))=={'bridge'} and not (ROOT/'formal_submission_intent.json').exists()
    protocol=read(P/'protocol.json');budget=read(P/'budget.json');release=read(ROOT/'release_approval.json')
    assert protocol['fits']==budget['fit_count']==release['fit_count']==15
    assert release['audit_sha256']==sha(ROOT/'bridge_audit.json') and read(ROOT/'bridge_audit.json')['passed']
    assert release['budget_sha256']==sha(P/'budget.json') and release['overlay_manifest_sha256']==sha(P/'manifest.json')
    for n,h in read(P/'manifest.json')['files'].items():assert sha(P/n)==h
    for n,h in read(ROOT/'manifest.json')['files'].items():assert sha(ROOT/n)==h
    remote=read(ROOT/'jobs.json')['bridge']['remote']
    intent=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),fits=15,
                overlay_manifest_sha256=sha(P/'manifest.json'),release_sha256=sha(ROOT/'release_approval.json'),
                budget_sha256=sha(P/'budget.json'),reserved_total=budget['requested_total_including_bridge_seconds'],remote=remote)
    write(P/'submission_intent.json',intent)
    # No frozen parent file is uploaded or overwritten. Review files are new.
    call(f'test ! -e {remote}/formal15 && test ! -e {remote}/release_approval.json && test ! -e {remote}/bridge_audit.json && tar -xzkf - -C {remote}',
         'deployment',(P/'release.tar.gz').read_bytes())
    code="""from pathlib import Path
import hashlib,json
p=Path('.')
def sha(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  while b:=f.read(8388608):h.update(b)
 return h.hexdigest()
for manifest,base in [('manifest.json',p),('formal15/manifest.json',p/'formal15')]:
 for n,h in json.loads((p/manifest).read_text())['files'].items():assert sha(base/n)==h,n
a=json.loads((p/'release_approval.json').read_text());assert a['fit_count']==15
assert sha(p/'data/prepared.pt')==a['data_sha256']
assert sha(p/'bridge_audit.json')==a['audit_sha256']
assert sha(p/'formal15/manifest.json')==a['overlay_manifest_sha256']
assert not list((p/'claims').glob('native_*.json')) and not list((p/'claims').glob('discount*.json'))
print('Frozen numerical source, overlay, original prepared data and actual review verified; no prior fit claim')
"""
    call(f'cd {remote} && python3 -c {shlex.quote(code)}','remote_source_data_check')
    jobs={}
    for arm in protocol['arms']:
        selected=[t for t in protocol['selected_tasks'] if t['arm']==arm];assert len(selected)==5
        indices=sorted(t['array_index'] for t in selected);minutes=budget['methods'][arm]['requested_minutes']
        assert 0<minutes<=45
        name='maze15-'+arm+'-20261102'
        previous=call(f'sacct -X -n -P --name={name} -S 2026-11-02 --format=JobID,State',arm+'_prior_name')
        assert not previous.strip(),previous
        entry=dict(arm=arm,indices=indices,task_mapping=selected,seconds_per_task=minutes*60,
                   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_name=name,remote=remote)
        write(P/(arm+'_submission_intent.json'),entry)
        command=f'cd {remote} && sbatch --parsable --job-name={name} --array='+','.join(map(str,indices))+f'%1 --time={minutes} formal15/train.sbatch'
        response=call(command,arm+'_submit');job=response.strip().split(';')[0];assert job.isdigit()
        entry['array_job_id']=job;write(P/(arm+'_receipt.json'),entry);jobs[arm]=entry
        temp=P/'jobs.next.json';write(temp,jobs);temp.replace(P/'jobs.json')
    assert sum(len(x['indices']) for x in jobs.values())==15
    print(json.dumps(jobs,indent=2))
if __name__=='__main__':main()
