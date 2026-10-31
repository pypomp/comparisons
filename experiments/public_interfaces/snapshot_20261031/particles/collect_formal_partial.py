"""Collect only explicitly requested, terminal CPU tasks. Never fit or score."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,tarfile,datetime
ROOT=Path(__file__).resolve().parent
FORMAL=ROOT/'formal_v1'
SSH=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('tasks',type=int,nargs='+');a=ap.parse_args()
    tasks=sorted(set(a.tasks));assert tasks and all(0<=i<24 for i in tasks)
    jobs=json.loads((FORMAL/'jobs.json').read_text());job=jobs['fit_job'];remote=jobs['remote']
    ids=','.join(f'{job}_{i}' for i in tasks)
    raw=subprocess.run(SSH+[f'sacct -X -n -P -j {ids} --format=JobID,State,ExitCode,ElapsedRaw,NodeList,AllocCPUS'],check=True,capture_output=True,text=True).stdout
    records={line.split('|')[0]:line.split('|') for line in raw.splitlines() if line.strip()}
    for i in tasks:
        row=records[f'{job}_{i}'];assert row[1]=='COMPLETED' and row[2]=='0:0',(i,row)
    names='_'.join(map(str,tasks));dest=FORMAL/'partial_audit';dest.mkdir(exist_ok=True)
    (dest/f'tasks{names}_sacct.txt').write_text(raw)
    # Read immutable source contents only; runtime/result directories are excluded.
    code='''from pathlib import Path
import json,hashlib
r=Path(%r);p=r/'manifest.json';h=hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads(p.read_text())['sha256']
for n,v in m.items():assert hashlib.sha256((r.parent/n).read_bytes()).hexdigest()==v,n
print(json.dumps(dict(passed=True,source_files=len(m),manifest_sha256=h)))
'''%remote
    command='python3 -c '+shlex.quote(code)
    src=json.loads(subprocess.run(SSH+[command],check=True,capture_output=True,text=True).stdout)
    assert src['manifest_sha256']==jobs['source_manifest_sha256']
    (dest/f'tasks{names}_remote_source.json').write_text(json.dumps(src,indent=2)+'\n')
    local=FORMAL/'gl';local.mkdir(exist_ok=True)
    new=[i for i in tasks if not (local/'results'/f'task_{i:02d}'/'terminal_manifest.json').exists()]
    if new:
        target=dest/f'tasks{names}_raw.tar.gz'
        paths=[f'results/task_{i:02d}' for i in new]
        cmd='cd '+shlex.quote(remote)+' && tar -czf - '+ ' '.join(map(shlex.quote,paths))
        with target.open('wb') as f:subprocess.run(SSH+[cmd],stdout=f,check=True)
        with tarfile.open(target) as t:
            assert all(not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in t.getmembers())
            t.extractall(local)
    receipt=dict(collected_tasks=tasks,newly_collected_tasks=new,source=src,scheduler={str(i):records[f'{job}_{i}'] for i in tasks},
        collected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scoring_run=False,fitting_run=False)
    (dest/f'tasks{names}_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
