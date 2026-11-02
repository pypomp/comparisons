"""Exclusive bridge or one formal array task, with immutable source checks."""
import argparse,datetime,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
from verify_source import verify,ROOT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def child(args,log,arm='native_dpf'):
    env=os.environ.copy();base=ROOT/'source'/('candidate' if arm=='discount097' else 'upstream')
    env['PYTHONPATH']=str(base)+':'+env['RUNTIME_DEPS']
    with log.open('xb') as f:
        return subprocess.run([sys.executable,*args],env=env,stdout=f,stderr=subprocess.STDOUT).returncode

def main(a):
    verify();cfg=json.loads((ROOT/'protocol.json').read_text())
    job=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
    if a.stage=='bridge':key='bridge'
    else:
        index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert 0<=index<36
        case=index//3;arm=cfg['arms'][index%3];key=f'{arm}_{case:02d}'
        release=json.loads((ROOT/'release_approval.json').read_text())
        assert release['passed'] and release['manifest_sha256']==sha(ROOT/'manifest.json')
        assert sha(ROOT/'bridge_audit.json')==release['audit_sha256']
        assert release['planned_total_cap']<=cfg['resources']['scope_cap_GPU_seconds']
        assert release['data_sha256']==json.loads((ROOT/'data/prepared.json').read_text())['cache_sha256']
    claims=ROOT/'claims';claims.mkdir(exist_ok=True)
    write(claims/(key+'.json'),dict(job=job,node=node,stage=a.stage,key=key,
          array_job=os.getenv('SLURM_ARRAY_JOB_ID'),array_task=os.getenv('SLURM_ARRAY_TASK_ID'),
          UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    rc=1;error=None
    try:
        if a.stage=='bridge':
            (ROOT/'data').mkdir(exist_ok=False);(ROOT/'bridge_results').mkdir(exist_ok=False)
            rc=child(['prepare_data.py','--csv',cfg['data']['csv'],'--output','data/prepared.pt'],ROOT/'prepare_data.log')
            if rc:raise RuntimeError(f'data preparation exited {rc}')
            for arm in cfg['arms']:
                rc=child(['bridge.py','--arm',arm,'--data','data/prepared.pt','--device','cuda:0',
                          '--output',str(Path('bridge_results')/arm)],ROOT/(arm+'_bridge.log'),arm)
                if rc:raise RuntimeError(f'{arm} bridge exited {rc}')
        else:
            (ROOT/'results').mkdir(exist_ok=True)
            rc=child(['train.py','--arm',arm,'--case',str(case),'--data','data/prepared.pt',
                      '--output',str(Path('results')/key)],ROOT/(key+'.log'),arm)
            if rc:raise RuntimeError(f'{key} training/evaluation exited {rc}')
        verify();rc=0
    except BaseException:
        error=traceback.format_exc();(ROOT/(key+'_launch_exception.txt')).write_text(error)
        rc=rc if rc else 1
    finally:
        paths=list((ROOT/('bridge_results' if a.stage=='bridge' else 'results/'+key)).rglob('*'))
        if a.stage=='bridge' and (ROOT/'data/prepared.json').exists():paths.append(ROOT/'data/prepared.json')
        paths.append(claims/(key+'.json'))
        write(ROOT/(key+'_terminal.json'),dict(stage=a.stage,key=key,job=job,node=node,returncode=rc,
              manifest_sha256=sha(ROOT/'manifest.json'),files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()},
              excluded_regenerable_cache='data/prepared.pt',exception=error))
    return rc

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['bridge','train'],required=True)
    sys.exit(main(p.parse_args()))
