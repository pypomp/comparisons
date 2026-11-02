"""Transfer one stable stopped authorized task, never another running task.

This is an external collector, outside both frozen manifests. No model calls.
An existing attempt needs manual inspection; this script never retries it.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import tarfile

P = Path(__file__).resolve().parent
SSH = ['ssh', '-S', '/tmp/greatlakes-codex-login2.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', 'greatlakes']
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED', 'DEADLINE'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8388608):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def task_mapping(jobs, arm, index):
    job = jobs[arm]
    task = next(x for x in job['task_mapping'] if x['array_index'] == index)
    assert index in job['indices'] and len(job['indices']) == 5
    assert task['arm'] == arm and task['legacy_case'] in [0, 2, 4, 6, 8]
    assert task['seed'] == 10 * (task['legacy_case'] // 2)
    assert task['repetition'] == 0
    assert job['array_job_id'].isdigit()
    return job, task, f"{arm}_{task['legacy_case']:02d}"


def main(a):
    job, task, key = task_mapping(read(P/'jobs.json'), a.arm, a.index)
    destination = P/'collected'/key
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination/'collection.json').exists()
    assert not (destination/'collection_attempt.json').exists()
    task_id = f"{job['array_job_id']}_{a.index}"

    def call(command):
        result = subprocess.run(SSH+[command], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return result.stdout

    scheduler = """import json,subprocess
t=TASK
a=subprocess.run(['sacct','-n','-P','-j',t,'--format=JobID%40,JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,TimelimitRaw,AllocTRES%100'],capture_output=True,text=True)
q=subprocess.run(['squeue','-h','-j',t,'-o','%i|%T|%M|%R'],capture_output=True,text=True)
assert a.returncode==q.returncode==0,(a.stderr,q.stderr)
print(json.dumps(dict(sacct=a.stdout,squeue=q.stdout)))
""".replace('TASK', repr(task_id))
    scheduling = json.loads(call('python3 -c '+shlex.quote(scheduler)))
    rows = [x.split('|') for x in scheduling['sacct'].splitlines()
            if x.split('|')[0] == task_id]
    assert len(rows) == 1 and not scheduling['squeue'].strip(), scheduling
    fields = rows[0]
    assert fields[2].split()[0] in TERMINAL, fields
    assert int(fields[6])*60 == job['seconds_per_task'], fields
    write(destination/'collection_attempt.json', dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        task=task, task_id=task_id, scheduler=scheduling))

    # Explicit immutable sources and this task's files only. No root tar,
    # bridge_results, other results, other claims, or shared tensor data cache.
    selection = """from pathlib import Path
import hashlib,json
p=Path('.');key=KEY;array=ARRAY;index=INDEX
names=set(json.loads((p/'manifest.json').read_text())['files'])
names|={'formal15/'+n for n in json.loads((p/'formal15/manifest.json').read_text())['files']}
names|={'manifest.json','formal15/manifest.json','release_approval.json','bridge_audit.json','data/prepared.json'}
optional=[key+'_terminal.json',key+'.log',key+'_launch_exception.txt','claims/'+key+'.json','guard_records/'+key+'.json',f'train15-{array}_{index}.log']
missing=[n for n in optional if not (p/n).exists()]
names|={n for n in optional if (p/n).exists()}
names|={str(x) for x in (p/'results'/key).rglob('*') if x.is_file()}
hashes={}
for name in sorted(names):
 f=p/name
 assert f.is_file() and not f.is_symlink(),name
 h=hashlib.sha256()
 with f.open('rb') as stream:
  while b:=stream.read(8388608):h.update(b)
 hashes[name]=h.hexdigest()
print(json.dumps(dict(files=hashes,missing_optional=missing)))
""".replace('KEY', repr(key)).replace('ARRAY', repr(job['array_job_id'])).replace('INDEX', repr(a.index))
    remote = job['remote']
    snapshot = json.loads(call('cd '+shlex.quote(remote)+' && python3 -c '+shlex.quote(selection)))
    hashes = snapshot['files']
    for name in hashes:
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert not name.startswith('results/') or name.startswith('results/'+key+'/')
        assert name != 'data/prepared.pt' and not name.startswith('bridge_results/')
    write(destination/'transfer_hashes.json', hashes)
    write(destination/'selection.json', snapshot)
    command = 'tar -czf - -C '+shlex.quote(remote)+' -- '+' '.join(shlex.quote(n) for n in hashes)
    with (destination/'collected.tar.gz').open('xb') as stream:
        result = subprocess.run(SSH+[command], stdout=stream, stderr=subprocess.PIPE)
    (destination/'transfer.stderr').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr
    raw = destination/'raw'
    raw.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(destination/'collected.tar.gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in hashes and member.name not in seen
            target = raw/member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(archive.extractfile(member).read())
            assert sha(target) == hashes[member.name], member.name
            seen.add(member.name)
    assert seen == set(hashes)
    receipt = dict(task=task, task_id=task_id, job=fields[1], state=fields[2],
                   exit_code=fields[3], ElapsedRaw=int(fields[4]), node=fields[5],
                   time_limit_minutes=int(fields[6]), AllocTRES=fields[7],
                   scheduler=scheduling, transferred_files=len(hashes),
                   transfer_SHA_verified=True, source_manifest_sha256=sha(raw/'manifest.json'),
                   overlay_manifest_sha256=sha(raw/'formal15/manifest.json'),
                   scientific_audit_complete=False, missing_optional=snapshot['missing_optional'],
                   other_task_results_collected=False, shared_cache_collected=False)
    write(destination/'collection.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['native_dpf', 'native_stop', 'discount097'], required=True)
    parser.add_argument('--index', required=True, type=int)
    main(parser.parse_args())
