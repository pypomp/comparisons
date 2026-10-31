from pathlib import Path
import datetime,hashlib,json,subprocess
root=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/pydpf/formal_v1');path=root/'jobs.json'
assert not path.exists(),'Submission record exists; do not resubmit'
config=json.loads((root/'protocol.json').read_text());assert config['status']=='frozen' and sum(x['allocation_seconds'] for x in config['allocation_by_phase'].values())==108000
assert json.loads((root/'results/preflight_reference_cpu.json').read_text())['passed']
manifest=json.loads((root/'source_manifest.json').read_text())
for name,value in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
state=dict(state='submitting',recorded_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_gate_job=65097403,formal_allocation_seconds=108000,source_manifest_sha256=hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),remote_root=str(root),jobs={},submission_attempts=[])
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
save()
for phase in ['warm','A','B','C']:
 seconds=config['allocation_by_phase'][phase]['allocation_seconds'];dependency=65097403 if phase in ['warm','A'] else state['jobs']['warm']['job_id'];duration=f'{seconds//3600:02d}:{seconds//60%60:02d}:00'
 command=['sbatch','--parsable',f'--job-name=pydpf-api-{phase}',f'--time={duration}',f'--dependency=afterok:{dependency}',f'--export=ALL,PYDPF_PHASE={phase}','formal.sbatch']
 result=subprocess.run(command,cwd=root,text=True,capture_output=True);state['submission_attempts'].append(dict(phase=phase,command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr));save()
 if result.returncode:state['state']='partial_submission_failed_no_automatic_retry';save();raise RuntimeError(result.stderr)
 job=int(result.stdout.strip().split(';')[0]);state['jobs'][phase]=dict(job_id=job,depends_afterok=dependency,allocation_seconds=seconds,GPUs=1,GPU_model='A40');save()
state['state']='submitted_waiting_GPU_gate';save();print(json.dumps(state,indent=2))
