"""Freeze and submit this new diagnostic revision once, using existing SSH."""
from pathlib import Path
import ast,datetime,hashlib,json,shlex,subprocess,tarfile

root=Path(__file__).resolve().parent
remote='/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/cuthbert_gpu_diagnosis_v1'
ssh=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
assert not (root/'jobs.json').exists(),'Already submitted; read jobs.json'
for name in ('execute.py','formal_smoke.py','run_recorded.py','measure.py'):
    ast.parse((root/name).read_text())
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('source_manifest.json','deployment.tar.gz','deploy.py') and '__pycache__' not in p.parts}
manifest={'files':files,'previous_gate_source_manifest_sha256':'a7230485e25ad8623e27b2af24f32e0e9f01e3657dcb1229809766af1f00d9a2','algorithm_source_changes':[]}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(root/'deployment.tar.gz','w:gz') as tf:
    for n in [*files,'source_manifest.json']:tf.add(root/n,arcname=n)
cmd=f'test ! -e {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)}/logs && tar -xzf - -C {shlex.quote(remote)}'
with (root/'deployment.tar.gz').open('rb') as f:
    subprocess.run(ssh+[cmd],stdin=f,check=True,timeout=120)
verify="import hashlib,json;from pathlib import Path;p=Path('.');m=json.loads((p/'source_manifest.json').read_text());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m['files'].items());print('SOURCE_VERIFIED',len(m['files']))"
cmd=f'cd {shlex.quote(remote)} && python3 -c {shlex.quote(verify)} && sbatch --parsable diagnose.sbatch'
r=subprocess.run(ssh+[cmd],capture_output=True,text=True,check=True,timeout=45)
lines=r.stdout.strip().splitlines();job=lines[-1].split(';')[0];assert job.isdecimal(),r.stdout
doc={'job_id':job,'remote':remote,'submitted_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU':'A40','time_limit_seconds':1680,'source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),'prior_job':65097220,'native_timing_repeated':False,'formal_fit_calls':0,'submission_stdout':r.stdout}
(root/'jobs.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(doc,indent=2))
