"""Submit the prepared Cuthbert comparison once, after actual GPU result review."""
from pathlib import Path
import ast,datetime,hashlib,json,shlex,subprocess,tarfile

ROOT=Path(__file__).resolve().parent
r=ROOT/'cuthbert_formal_v3';gate=ROOT/'cuthbert_gate_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (r/'jobs.json').exists(),'Already submitted; inspect jobs.json'
approval=json.loads((r/'release_approval.json').read_text())
collection=json.loads((ROOT/'cuthbert_gate_v3_collected/collection.json').read_text())
assert collection['all_transferred_hashes_verified'] and collection['terminal_manifest_available']
assert approval['approved'] and approval['upstream_failure_comparison_reviewed']
assert approval['formal_driver_sha256']==sha(r/'run.py')
assert approval['gate_source_manifest_sha256']==sha(gate/'source_manifest.json')
raw=ROOT/'cuthbert_gate_v3_collected/raw/results'
assert approval['gate_complete_sha256']==sha(raw/'COMPLETE.json')
complete=json.loads((raw/'COMPLETE.json').read_text());assert complete['core_checks_passed']
assert collection['scheduler_row'][1]=='COMPLETED' and collection['scheduler_row'][2]=='0:0'
for n in ('Dacca_systematic_results.json','Dacca_multinomial_results.json','paired_ADPF_J32_results.json','paired_ADPF_J5000_results.json','public_cost_results.json'):
    assert json.loads((raw/n).read_text())['passed'],n
cost=json.loads((raw/'public_cost_results.json').read_text())['measurements']
predicted=35*cost['public_IF2_5_passes']['median_timed_seconds']+1200*max(cost[f'public_gradient_alpha_{v}']['median_timed_seconds'] for v in (1.0,.97))
forecast=1.5*predicted+3*sum(v['compile_seconds'] for v in cost.values())+300
assert forecast<3120,forecast
assert sha(r/'score.py')=='8aec1ba02982a1ad9bff227b300806eaafa972a1e9fd00b71bef2c5f0ebdc508'
for p in r.glob('*.py'):ast.parse(p.read_text())
files={str(p.relative_to(r)):sha(p) for p in sorted(r.rglob('*')) if p.is_file() and p.name not in ('source_manifest.json','deployment.tar.gz') and '__pycache__' not in p.parts}
manifest=dict(files=files,gate_source_manifest_sha256=sha(gate/'source_manifest.json'),revision='Cuthbert public interface v3',formal_release_approved=True)
(r/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'deployment.tar.gz','w:gz') as tf:
    for n in [*files,'source_manifest.json']:tf.add(r/n,arcname=n)
remote='/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/cuthbert_formal_v3'
ssh=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
with (r/'deployment.tar.gz').open('rb') as f:
    subprocess.run(ssh+[f'test ! -e {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}/logs && tar -xzf - -C {shlex.quote(remote)}'],stdin=f,check=True,timeout=120)
gate_job=json.loads((gate/'jobs.json').read_text())['job_id']
verify="from pathlib import Path;import hashlib,json;r=Path('.');m=json.loads((r/'source_manifest.json').read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m['files'].items());print('SOURCE_VERIFIED',len(m['files']))"
response=subprocess.run(ssh+[f'cd {shlex.quote(remote)} && python3 -c {shlex.quote(verify)} && sbatch --parsable --dependency=afterok:{gate_job} formal.sbatch cuthbert'],capture_output=True,text=True,check=True,timeout=45)
job=response.stdout.strip().splitlines()[-1].split(';')[0];assert job.isdecimal()
record=dict(job_id=job,remote=remote,gate_job=gate_job,submitted_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_sha256=sha(r/'source_manifest.json'),array='0-11%3',time_limit_per_case_seconds=3360,formal_allocation_gpu_hours_upper=11.2,new_validation_and_formal_gpu_hours_upper=700/60,forecast_seconds=forecast,submission_stdout=response.stdout)
(r/'jobs.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
