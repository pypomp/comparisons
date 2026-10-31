"""Bounded paired GPU diagnosis; no formal optimization or tolerance changes."""
from pathlib import Path
import datetime,hashlib,json,os,platform,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
protocol=json.loads((ROOT/'protocol.json').read_text())
old=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006')
deps=[ROOT/'test_deps',old/'cuthbert_dacca_20261027/source',old/'cuthbert_dacca_20261027/deps',old/'competitor_estimation_20261028/pfjax/deps']
results={}
started=time.monotonic()
def run(label,command,source='candidate',x64=False,timeout=180):
    env=os.environ.copy()
    paths=[ROOT/'cuthbert'/source,ROOT/'cuthbert/validation',*deps]
    env.update(PYTHONPATH=':'.join(map(str,paths)),JAX_PLATFORMS='cuda',JAX_ENABLE_X64=str(x64),PYTHONDONTWRITEBYTECODE='1')
    cwd=ROOT/'cuthbert'/source
    cmd=[sys.executable,str(ROOT/'run_recorded.py'),'--record',str(OUT/f'{label}.json'),'--cwd',str(cwd),'--timeout',str(timeout),'--',sys.executable,*map(str,command)]
    result=subprocess.run(cmd,env=env)
    results[label]=result.returncode
    (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')

for source in ('upstream','candidate'):
    run(source+'_original_failures',['-m','pytest','-vv','--tb=long','-rA',*protocol['focus_tests'],f'--junitxml={OUT/source}_original_failures.xml','-o','cache_dir=/tmp/cuthbert-diagnosis-'+source],source,x64=True,timeout=180)
for source in ('upstream','candidate'):
    run(source+'_PF_regression',['-m','pytest','-vv','--tb=long','-rA','tests/cuthbert/smc/test_particle_filters.py',f'--junitxml={OUT/source}_PF_regression.xml','-o','cache_dir=/tmp/cuthbert-diagnosis-pf-'+source],source,x64=True,timeout=240)
run('candidate_public_tests',['-m','pytest','-vv','--tb=long','-rA','tests/cuthbert/smc/test_parameter_estimation.py',f'--junitxml={OUT}/candidate_public_tests.xml','-o','cache_dir=/tmp/cuthbert-diagnosis-public'],x64=True,timeout=240)
run('Dacca_integration',[ROOT/'cuthbert/validation/check_dacca_api.py','--output',OUT/'Dacca_integration_results.json'],timeout=180)
run('public_cost',[ROOT/'measure.py','cuthbert','public','--output',OUT/'public_cost_results.json'],timeout=120)
run('ABC_driver_smoke',[ROOT/'formal_smoke.py','--package','cuthbert','--case','0','--output',OUT/'driver_smoke','--smoke'],timeout=120)
record={'diagnosis_complete':True,'exit_codes':results,'all_commands_passed':all(x==0 for x in results.values()),'node':platform.node(),'job':os.environ.get('SLURM_JOB_ID'),'elapsed_seconds':time.monotonic()-started,'formal_fit_calls':0,'native_timing_repeated':False,'tolerance_changes':False,'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='terminal_manifest.json'}
(OUT/'terminal_manifest.json').write_text(json.dumps({'files':files,'source_manifest_sha256':sha(ROOT/'source_manifest.json')},indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(0 if record['all_commands_passed'] else 1)
