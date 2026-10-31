"""Collect a stopped Cuthbert check once and verify its source and all results."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,tarfile

ROOT=Path(__file__).resolve().parent
ssh=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('revision',choices=['cuthbert_gpu_diagnosis_v1','cuthbert_gate_v3']);a=ap.parse_args()
    source=ROOT/a.revision;job=json.loads((source/'jobs.json').read_text());out=ROOT/(a.revision+'_collected')
    if (out/'collection.json').exists():
        print('Already collected; read',out/'collection.json');return
    remote_code='''
from pathlib import Path
import datetime,hashlib,io,json,subprocess,sys,tarfile
root=Path(REMOTE);job=JOB
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=subprocess.run(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'],capture_output=True,text=True,check=True)
row=next(x.split('|') for x in r.stdout.splitlines() if x.split('|')[0]==job)
assert row[1].split()[0] in ('COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED'),row
assert not subprocess.check_output(['squeue','-h','-j',job]).strip(),'Job remains in queue'
m=json.loads((root/'source_manifest.json').read_text())
for n,h in m['files'].items():assert sha(root/n)==h,n
paths=[p for name in ('results','logs') for p in sorted((root/name).rglob('*')) if p.is_file()]
paths += [root/n for n in ('source_manifest.json','protocol.json','source_review.json') if (root/n).exists()]
before={str(p.relative_to(root)):sha(p) for p in paths}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tf:
 for p in paths:tf.add(p,arcname=str(p.relative_to(root)))
 record=json.dumps(dict(job_id=job,scheduler_row=row,scheduler_stdout=r.stdout,remote=str(root),source_files_verified=len(m['files']),source_manifest_sha256=sha(root/'source_manifest.json'),files=before,collected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2).encode()
 entry=tarfile.TarInfo('collection_record.json');entry.size=len(record);tf.addfile(entry,io.BytesIO(record))
after={str(p.relative_to(root)):sha(p) for p in paths}
assert before==after,'Stopped result files changed during collection'
sys.stdout.buffer.write(buf.getvalue())
'''.replace('REMOTE',repr(job['remote'])).replace('JOB',repr(job['job_id']))
    result=subprocess.run(ssh+['python3 -'],input=remote_code.encode(),capture_output=True,timeout=90)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    out.mkdir(exist_ok=True);archive=out/'remote_results.tar.gz';archive.write_bytes(result.stdout)
    with tarfile.open(archive) as tf:
        for member in tf:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts,member.name
            p=out/'raw'/member.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(member).read())
    raw=out/'raw';record=json.loads((raw/'collection_record.json').read_text())
    assert record['source_manifest_sha256']==job['source_manifest_sha256']
    assert sha((raw/'source_manifest.json').read_bytes())==job['source_manifest_sha256']
    for n,h in record['files'].items():assert sha((raw/n).read_bytes())==h,n
    terminal=raw/'results/terminal_manifest.json';count=0
    if terminal.exists():
        m=json.loads(terminal.read_text());assert m['source_manifest_sha256']==job['source_manifest_sha256']
        for n,h in m['files'].items():assert sha((raw/n).read_bytes())==h,n;count+=1
    record.update(archive_sha256=sha(archive.read_bytes()),all_transferred_hashes_verified=True,terminal_manifest_available=terminal.exists(),terminal_files_verified=count)
    (out/'collection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),job=job['job_id'],scheduler=record['scheduler_row'],files_verified=len(record['files']),terminal_files_verified=count)))

if __name__=='__main__':main()
