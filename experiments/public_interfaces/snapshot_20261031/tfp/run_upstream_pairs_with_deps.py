"""Run every upstream pytest test file in isolated, original/candidate pairs.

Each file has a fixed 180s cap. Timeouts, collection errors, aborts and skips
remain visible; none count as passes. This bounds runaway probabilistic tests
and lets all other files execute after one upstream dependency failure.
"""
from pathlib import Path
import os, sys, json, time, subprocess, hashlib, shutil, concurrent.futures
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT.parent/'upstream_repositories/tfp'
PYTHON=ROOT.parents[1]/'competitor-estimation/tfp/local-venv/bin/python'
OUT=ROOT/'upstream_tests_with_deps';OUT.mkdir(exist_ok=True)
TIMEOUT=180

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def prepare():
    inventories={}
    for kind in ['original','candidate']:
        target=OUT/kind
        if not target.exists():shutil.copytree(UPSTREAM,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if kind=='candidate':
            cand=ROOT/'source/candidate';base=ROOT/'source/upstream'
            for f in cand.rglob('*.py'):
                rel=f.relative_to(cand)
                if 'tensorflow_probability'!=rel.parts[0]:continue
                old=base/rel
                if not old.exists() or f.read_bytes()!=old.read_bytes():
                    dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        inventories[kind]={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in target.rglob('*.py')}
    files=sorted(set(UPSTREAM.rglob('*_test.py'))|set(UPSTREAM.rglob('test_*.py')))
    excluded=[str(p.relative_to(UPSTREAM)) for p in files if p.relative_to(UPSTREAM).parts[0]=='discussion']
    files=[str(p.relative_to(UPSTREAM)) for p in files if p.relative_to(UPSTREAM).parts[0]!='discussion']
    files.sort(key=lambda f:(0 if '/experimental/mcmc/' in f else 1,f))
    write(OUT/'inventory.json',{'official_source_test_named_files':len(list(UPSTREAM.rglob('*test*.py'))),
        'pytest_files':files,'excluded_by_official_conftest':excluded,
        'non_test_utility_files_are_not_pytest_files':True,'per_file_timeout_seconds':TIMEOUT,
        'parallel_file_pairs':2,'scope':'all official pytest test files except official discussion ignore; spinoffs and substrates attempted too',
        'sha256':inventories})
    return files

def run_one(kind,rel):
    name=rel.replace('/','__').removesuffix('.py');path=OUT/'records'/name/kind
    record=path.with_suffix('.json')
    if record.exists():return json.loads(record.read_text())
    path.parent.mkdir(parents=True,exist_ok=True)
    repo=OUT/kind
    env=dict(os.environ,PYTHONPATH=str(repo)+os.pathsep+str(ROOT/'test-deps')+os.pathsep+str(ROOT/'test-science-deps'),
             TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',CUDA_VISIBLE_DEVICES='')
    xml=path.with_suffix('.xml')
    cmd=[str(PYTHON),'-m','pytest',rel,'-q','--tb=short','--continue-on-collection-errors',
         '--junitxml='+str(xml)]
    t=time.perf_counter();result={'kind':kind,'file':rel,'command':cmd,'cwd':str(repo),'timeout_seconds':TIMEOUT}
    with path.with_suffix('.log').open('w') as log:
        try:
            p=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=TIMEOUT)
            result.update(exit_code=p.returncode,state='finished')
        except subprocess.TimeoutExpired:
            result.update(exit_code=124,state='timeout')
    result['elapsed_seconds']=time.perf_counter()-t
    if xml.exists():
        root=ET.parse(xml).getroot();suites=root.findall('testsuite') if root.tag!='testsuite' else [root]
        result['pytest_counts']={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        result['failed_tests']=[{'name':t.attrib.get('name'),'class':t.attrib.get('classname'),
             'message':(t.find('failure') if t.find('failure') is not None else t.find('error')).attrib.get('message','')}
             for t in root.iter('testcase') if t.find('failure') is not None or t.find('error') is not None]
    write(record,result);return result
def pair(rel):
    rows=[run_one(k,rel) for k in ['original','candidate']]
    write(OUT/'pairs'/(rel.replace('/','__')+'.json'),{'file':rel,'results':rows,
        'same_exit_status':rows[0]['exit_code']==rows[1]['exit_code'],
        'candidate_only_failure':rows[0]['exit_code']==0 and rows[1]['exit_code']!=0})
    return rows
def main():
    lock=OUT/'RUNNING.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Existing upstream runner is active; no duplicate')
    write(lock,{'pid':os.getpid(),'started_unix':time.time()})
    files=prepare();allrows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(pair,rel):rel for rel in files}
        for future in concurrent.futures.as_completed(pending):
            rows=future.result();allrows.extend(rows)
            write(OUT/'status.json',{'completed_pairs':len(allrows)//2,'expected_pairs':len(files),
                'original':sum(r['kind']=='original' and r['exit_code']==0 for r in allrows),
                'candidate':sum(r['kind']=='candidate' and r['exit_code']==0 for r in allrows),
                'last':pending[future],'finished':False})
    write(OUT/'COMPLETE.json',{'all_files_attempted':True,'all_tests_passed':all(r['exit_code']==0 for r in allrows),
          'records':allrows,'file_count':len(files),'paired_scope':True,'timeouts_not_passes':True})
    lock.unlink()

if __name__=='__main__':main()
