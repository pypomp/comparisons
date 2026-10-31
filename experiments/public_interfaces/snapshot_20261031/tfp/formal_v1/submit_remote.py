"""Unique four-job submission. A partial or uncertain attempt is never replayed."""
from pathlib import Path
import json,subprocess,time,hashlib,os
root=Path(__file__).resolve().parent;os.chdir(root)
manifest=json.loads((root/'manifest.json').read_text())
for n,h in manifest['sha256'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
assert not (root/'jobs.json').exists(),'Existing submission record; do not duplicate or replay'
(root/'SUBMISSION_LOCK').mkdir()
doc={'gate_job_id':65096288,'remote_root':str(root),'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
     'submitted':False,'requested_total_GPU_hours':20,'methods':{},'started_unix':time.time()}
def save():
    temp=root/'jobs.tmp';temp.write_text(json.dumps(doc,indent=2)+'\n');temp.replace(root/'jobs.json')
save();(root/'logs').mkdir(exist_ok=True)
for method,hours in [('prefix',8),('A',4),('B',4),('C',4)]:
    ids=[65096288]+([doc['methods']['prefix']['job_id']] if method in ['B','C'] else [])
    command=['sbatch','--parsable','--job-name=tfp-public-'+method,
             '--time='+str(hours)+':00:00','--dependency=afterok:'+':'.join(map(str,ids)),'formal.sbatch',method]
    result=subprocess.run(command,check=True,text=True,capture_output=True)
    (root/('submission_'+method+'.txt')).write_text(result.stdout+result.stderr)
    job=int(result.stdout.strip().split(';')[0]);doc['methods'][method]={'job_id':job,'time_limit_hours':hours,'dependency_afterok':ids,'command':command,'submitted_unix':time.time()};save()
doc['submitted']=True;save();print(json.dumps(doc,indent=2))
