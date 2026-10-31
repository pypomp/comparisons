"""One-shot recovery of stopped jobs; never submit, fit, or score."""
from pathlib import Path
import argparse, datetime, hashlib, json, shlex, subprocess, tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = '/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030'
SSH = ['ssh', '-S', '/tmp/greatlakes-codex-login2.sock', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'greatlakes']
SPECS = {
    'jax': ('jax_gate_v2', 'source_manifest.json', 'files', '.', '65097220', ['results', 'logs']),
    'tfp': ('tfp', 'manifest.json', 'sha256', '.', '65096288', ['validation_results', 'logs']),
    'pydpf': ('pydpf/repair_v2', 'source_manifest.json', 'files', '.', '65097403', ['results', 'logs']),
    'particles': ('particles/formal_v1', 'manifest.json', 'sha256', '..', '65097887,65097888',
                  ['results/task_%02d' % i for i in range(6, 24)] + ['scoring', 'endpoints.json', 'logs']),
}

def ssh_python(code):
    return subprocess.run(SSH + ['python3 -'], input=code, text=True, capture_output=True, check=True, timeout=300).stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('package', choices=SPECS)
    a = ap.parse_args()
    rel, manifest, key, source_base, jobs, selected = SPECS[a.package]
    dest = ROOT / 'recovery_20261031' / a.package
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / 'receipt.json'
    if receipt.exists():
        print('Already collected: ' + str(receipt)); return
    command = 'sacct -X -n -P -j ' + jobs + ' --format=JobID,State,ExitCode,ElapsedRaw,NodeList,AllocCPUS,AllocTRES,Start,End,Reason'
    sacct = subprocess.run(SSH + [command], capture_output=True, text=True, check=True, timeout=60).stdout
    (dest / 'sacct.txt').write_text(sacct)
    rows = [x.split('|') for x in sacct.splitlines() if x.strip()]
    expected = ({'65097887_' + str(i) for i in range(24)} | {'65097888'}) if a.package == 'particles' else {jobs}
    found = {x[0]: x for x in rows}
    assert expected <= found.keys(), (expected - found.keys())
    assert all(found[k][1].split()[0] in {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY'} for k in expected)
    selected = selected + [manifest]
    code = '''from pathlib import Path
import hashlib,json
r=Path(%r); names=%r
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/%r).read_text())[%r]; base=r/%r
mismatches=[n for n,h in m.items() if not (base/n).is_file() or sha(base/n)!=h]
files={}
missing=[]
for n in names:
 p=r/n
 if not p.exists(): missing.append(n); continue
 for f in ([p] if p.is_file() else p.rglob('*')):
  if f.is_file():
   assert not f.is_symlink(),str(f)
   files[str(f.relative_to(r))]={'sha256':sha(f),'bytes':f.stat().st_size}
print(json.dumps({'source_manifest_sha256':sha(r/%r),'source_file_count':len(m),'source_mismatches':mismatches,'files':files,'missing_optional_outputs':missing}))
''' % (REMOTE + '/' + rel, selected, manifest, key, source_base, manifest)
    before = json.loads(ssh_python(code))
    local_manifest = ROOT / rel / manifest
    assert hashlib.sha256(local_manifest.read_bytes()).hexdigest() == before['source_manifest_sha256']
    (dest / 'remote_before.json').write_text(json.dumps(before, indent=2) + '\n')
    names = [n for n in selected if n not in before['missing_optional_outputs']]
    archive = dest / 'raw.tar.gz'
    with archive.open('wb') as f:
        subprocess.run(SSH + ['tar -czf - -C ' + shlex.quote(REMOTE + '/' + rel) + ' ' + ' '.join(map(shlex.quote, names))], stdout=f, check=True, timeout=900)
    after = json.loads(ssh_python(code))
    (dest / 'remote_after.json').write_text(json.dumps(after, indent=2) + '\n')
    assert before == after, 'Stopped output changed during collection'
    out = dest / 'raw'
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            assert not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
            assert item.isfile() or item.isdir(), item.name
        tar.extractall(out, filter='data')
    for n, rec in before['files'].items():
        assert hashlib.sha256((out / n).read_bytes()).hexdigest() == rec['sha256'], n
    doc = dict(collected_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               package=a.package, job_ids=sorted(expected), source_manifest_sha256=before['source_manifest_sha256'],
               source_files=before['source_file_count'], source_mismatches=before['source_mismatches'],
               transferred_files=len(before['files']), transferred_bytes=sum(x['bytes'] for x in before['files'].values()),
               transfer_before_after_and_local_hashes_match=True,
               transfer_manifest_created_at_collection=True,
               fitting_run=False, scoring_run=False,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               missing_optional_outputs=before['missing_optional_outputs'])
    receipt.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps(doc, indent=2))

if __name__ == '__main__': main()
