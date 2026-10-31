"""One collection of formal results; never submit or start a daemon."""
from pathlib import Path
import io,json,shlex,subprocess,tarfile
root=Path(__file__).resolve().parent
jobs=json.loads((root/'jobs.json').read_text())
command='cd '+shlex.quote(jobs['remote_root'])+' && tar --exclude="*/jax_cache" --exclude="*/tmp" --exclude="*/runtime" --exclude="*.tmp" -czf - results logs manifest.json protocol.json'
r=subprocess.run(['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes',
    '-o','ConnectTimeout=15','greatlakes',command],capture_output=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
destination=root/'remote';destination.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
    for item in t.getmembers():
        p=Path(item.name)
        if p.is_absolute() or '..' in p.parts or item.issym() or item.islnk():raise RuntimeError('Unsafe archive member '+item.name)
    t.extractall(destination,filter='data')
print(json.dumps({'collected':True,'array_job_id':jobs['array_job_id'],'destination':str(destination)}))
