"""One bounded GPU preflight. Preserve every phase exit status, including no tests."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True);started=time.monotonic();records=[]
steps=[('upstream_collection',[sys.executable,'-m','pytest','upstream','--collect-only','-q'],180,{5}),('candidate_cuda_tests',[sys.executable,'-m','pytest','gpu_tests','-q','--junitxml=results/gpu_tests.xml'],600,{0}),('driver_cuda_checks',[sys.executable,'driver_validation/check_driver.py'],300,{0}),('dacca_cuda_correctness',[sys.executable,'check_dacca_cuda.py','--output','results/dacca_cuda_correctness.json'],900,{0}),('native_PF_timing_and_public_rates',[sys.executable,'gpu_benchmark.py'],1800,{0})]
for name,command,limit,ok in steps:
    remaining=3200-(time.monotonic()-started)
    if remaining<60:records.append(dict(name=name,state='not_run_budget_limit',passed=False));break
    before=time.perf_counter()
    with (OUT/f'{name}.log').open('w') as log:
        try:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(limit,remaining-30));code=result.returncode
        except subprocess.TimeoutExpired:code=124
    records.append(dict(name=name,command=command,exit_code=code,expected_exit_codes=sorted(ok),passed=code in ok,seconds=time.perf_counter()-before,note='No tests exist in pinned upstream; pytest exit5 is absence, not a passed suite' if name=='upstream_collection' else 'GPU-only; no CPU fallback'))
    (OUT/'preflight_progress.json').write_text(json.dumps(records,indent=2)+'\n')
passed=len(records)==len(steps) and all(r['passed'] for r in records)
(OUT/'preflight_complete.json').write_text(json.dumps(dict(passed=passed,phases=records,elapsed_seconds=time.monotonic()-started,finished_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID')),indent=2)+'\n')
raise SystemExit(0 if passed else 1)
