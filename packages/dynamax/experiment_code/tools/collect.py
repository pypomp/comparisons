"""One-shot collection only; does not submit or poll persistently."""
from pathlib import Path
import json,subprocess,tempfile,tarfile,hashlib
R=Path(__file__).resolve().parent.parent
j=json.loads((R/'jobs.json').read_text())
with tempfile.NamedTemporaryFile(dir=R,suffix='.tar.gz',delete=False) as f:
    archive=Path(f.name)
    cmd=f"cd '{j['remote_root']}'\nset -- logs results\nfor name in gate_results runs; do if test -d \"$name\"; then set -- \"$@\" \"$name\"; fi; done\ntar -czf - \"$@\""
    p=subprocess.run(['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes',cmd],stdout=f,stderr=subprocess.PIPE,timeout=120)
try:
    if p.returncode not in [0,1]:raise RuntimeError(p.stderr.decode())
    dst=R/'gl';dst.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        ms=t.getmembers()
        assert all(not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
        t.extractall(dst,members=ms)
finally:archive.unlink(missing_ok=True)
verified=0
for manifest in (dst/'runs').glob('*/batch*/terminal_manifest.json') if (dst/'runs').exists() else []:
    data=json.loads(manifest.read_text())
    for name,value in data['files'].items():assert hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest()==value,name
    verified+=1
print(json.dumps({'collected':True,'terminal_batches_verified':verified,'tar_exit':p.returncode,'tar_stderr':p.stderr.decode()}))
