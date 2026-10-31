"""GPU checks for the revised public API; formal fitting requires local review."""
from pathlib import Path
import datetime,hashlib,json,os,platform,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
protocol=json.loads((ROOT/'protocol.json').read_text())
old=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006')
deps=[ROOT/'test_deps',old/'cuthbert_dacca_20261027/source',old/'cuthbert_dacca_20261027/deps',old/'competitor_estimation_20261028/pfjax/deps']
results={};started=time.monotonic()

def run(label,command,x64=False,timeout=180):
    env=os.environ.copy()
    paths=[ROOT/'cuthbert/candidate',ROOT/'cuthbert/validation',*deps]
    env.update(PYTHONPATH=':'.join(map(str,paths)),JAX_PLATFORMS='cuda',JAX_ENABLE_X64=str(x64),PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,str(ROOT/'run_recorded.py'),'--record',str(OUT/f'{label}.json'),'--cwd',str(ROOT/'cuthbert/candidate'),'--timeout',str(timeout),'--',sys.executable,*map(str,command)]
    results[label]=subprocess.run(cmd,env=env).returncode
    (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')

run('prior_upstream_failures',['-m','pytest','-vv','--tb=long','-rA',*protocol['focus_tests'],f'--junitxml={OUT}/prior_upstream_failures.xml','-o','cache_dir=/tmp/cuthbert-v3-focus'],True,180)
run('PF_regression',['-m','pytest','-vv','--tb=long','-rA','tests/cuthbert/smc/test_particle_filters.py',f'--junitxml={OUT}/PF_regression.xml','-o','cache_dir=/tmp/cuthbert-v3-pf'],True,240)
run('public_tests',['-m','pytest','-vv','--tb=long','-rA','tests/cuthbert/smc/test_parameter_estimation.py',f'--junitxml={OUT}/public_tests.xml','-o','cache_dir=/tmp/cuthbert-v3-public'],True,240)
for method in ('systematic','multinomial'):
    run('Dacca_'+method,[ROOT/'cuthbert/validation/check_dacca_api.py','--resampling',method,'--output',OUT/f'Dacca_{method}_results.json'],timeout=120)
for particles in (32,5000):
    run(f'paired_ADPF_J{particles}',[ROOT/'cuthbert/validation/check_native_adpf_pairs.py','--particles',particles,'--starts',ROOT/'input/paper_starts_estimation_free.npy','--output',OUT/f'paired_ADPF_J{particles}_results.json'],timeout=120 if particles==32 else 180)
run('public_cost',[ROOT/'measure.py','cuthbert','public','--output',OUT/'public_cost_results.json'],timeout=180)
run('ABC_driver_smoke',[ROOT/'formal_smoke.py','--package','cuthbert','--case','0','--output',OUT/'driver_smoke','--smoke'],timeout=120)
core={k:v for k,v in results.items() if k!='prior_upstream_failures'}
record=dict(checks_completed=True,exit_codes=results,core_checks_passed=all(v==0 for v in core.values()),
    formal_release_approved=False,manual_upstream_failure_review_required=True,
    full_GPU_upstream_suite_passed=False,node=platform.node(),job=os.environ.get('SLURM_JOB_ID'),
    elapsed_seconds=time.monotonic()-started,formal_fit_calls=0,native_timing_repeated=False,
    tolerance_changes=False,captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(OUT/'COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='terminal_manifest.json'}
(OUT/'terminal_manifest.json').write_text(json.dumps(dict(files=files,source_manifest_sha256=sha(ROOT/'source_manifest.json')),indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(0 if record['core_checks_passed'] else 1)
