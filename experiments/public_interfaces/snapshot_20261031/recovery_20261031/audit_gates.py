"""Audit recorded GPU checks; do not rerun failed checks or change tolerances."""
from pathlib import Path
import datetime,hashlib,json,statistics
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for pkg in ['jax','tfp','pydpf']:
 rec=read(R/pkg/'receipt.json')
 assert not rec['source_mismatches'] and rec['transfer_before_after_and_local_hashes_match']
 for n,d in read(R/pkg/'remote_before.json')['files'].items():assert sha(R/pkg/'raw'/n)==d['sha256']
j=R/'jax/raw';m=read(j/'results/terminal_manifest.json')
assert m['source_manifest_sha256']==sha(j/'source_manifest.json')
for n,h in m['files'].items():assert sha(j/n)==h,n
timing={}
for pkg in ['cuthbert','pfjax']:
 d=read(j/'results'/f'{pkg}_native_timing.json');v=d['measurements']['native_pf']
 assert d['J']==5000 and d['T']==600 and d['B']==1 and d['device_kind']=='NVIDIA A40'
 values=[x['seconds'] for x in v['calls'] if x['kind']=='timed'];assert len(values)==7
 assert statistics.median(values)==v['median_timed_seconds'] and all(x['finite'] for x in v['calls'])
 timing[pkg]=dict(native_PF_median_seconds=statistics.median(values),compile_seconds=v['compile_seconds'],
                  batch=1,GPU='A40',node='gl1524',measured_calls=7,not_an_optimization_result=True)
tfp=read(R/'tfp/raw/validation_results/partial.json')
for k in ['original_native','candidate_alpha1']:
 d=tfp[k];assert len(d['timed_seconds'])==7 and statistics.median(d['timed_seconds'])==d['median_synchronized_seconds']
assert tfp['original_native']['raw_loglik']==tfp['candidate_alpha1']['raw_loglik']
timing['tfp']=dict(native_PF_median_seconds=tfp['original_native']['median_synchronized_seconds'],
                   candidate_forward_median_seconds=tfp['candidate_alpha1']['median_synchronized_seconds'],
                   batch=1,GPU='A40',node='gl1511',measured_calls=7,not_an_optimization_result=True)
py=read(R/'pydpf/raw/results/preflight_complete.json')
assert not py['passed'] and read(R/'pydpf/raw/results/dacca_cuda_correctness.json')['passed']
assert read(R/'pydpf/raw/results/benchmark_complete.json')['passed']
assert '1 failed, 33 passed' in (R/'pydpf/raw/results/candidate_cuda_tests.log').read_text()
costs={}
for pkg in ['jax','tfp','pydpf']:
 row=(R/pkg/'sacct.txt').read_text().strip().split('|')
 assert row[1:3]==['FAILED','1:0'];costs[pkg]=int(row[3])
cancelled=[x.split('|') for x in (R/'cancelled_formal_sacct.txt').read_text().splitlines() if x.strip()]
assert len(cancelled)==10 and all(x[1]=='CANCELLED' and int(x[3])==0 and x[5]=='None' for x in cancelled)
failures={
 'cuthbert':dict(GPU_Dacca_check_passed=True,Dacca_J=32,full_GPU_suite_passed=False,
                 suite_timeout_seconds=1200,failures_visible_before_timeout=3,
                 failure_test_names_available=False,public_cost_measurement_run=False,
                 next_check='Identify each failing test and run unchanged upstream and candidate tests under identical GPU settings; retain timeout as incomplete coverage.'),
 'pfjax':dict(configured_GPU_suite_passed=72,GPU_Dacca_check_passed=False,
              failed_comparison='public gradient versus historical implementation',
              violating_coordinate=1,absolute_difference=0.0006084442138671875,rtol=2e-5,atol=2e-4,
              public_cost_measurement_run=False,
              next_check='Compare repeated original controls and per-step traces under the same compilation and input treatment; keep existing tolerances.'),
 'pydpf':dict(GPU_extension_tests_passed=33,GPU_extension_tests_failed=1,GPU_Dacca_check_passed=True,Dacca_J=8,
              driver_GPU_check_passed=True,failed_comparison='float32 multinomial original versus default/alpha1 exact equality',
              maximum_difference=2.384185791015625e-7,rtol=0,atol=0,
              next_check='Repeat original-versus-original GPU control and isolate which compared tensor differs; diagnose CUDA determinism before changing implementation.'),
 'tfp':dict(GPU_Dacca_gradient_check_passed=False,native_forward_same_seed_exact=True,
            failed_comparison='alpha1 gradient versus original exact equality',
            mismatched_coordinates=22,total_coordinates=23,maximum_difference=1.4156103e-7,rtol=0,atol=0,
            next_check='Repeat original controls and compare identical graph construction and reduction paths; keep exact-equality failure visible.')}
doc=dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         artifact_integrity_passed=True,all_GPU_correctness_gates_passed=False,
         gate_actual_GPU_seconds=costs,prior_failed_deployment_GPU_seconds={'jax':73,'pydpf':234},
         formal_GPU_jobs_cancelled_before_start=True,formal_GPU_fit_endpoints=0,
         cancellation_reason_in_sacct='None; dependency declarations and cancellation times retained, actor not established',
         failures=failures,native_PF_timing=timing,
         tolerance_changes=False,new_submissions=False,
         transfer_manifest_note='JAX has original terminal manifest, also checked. PyDPF and TFP transfer hashes were created at collection; not represented as runner terminal manifests.')
(R/'GPU_audit.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(doc,indent=2))
