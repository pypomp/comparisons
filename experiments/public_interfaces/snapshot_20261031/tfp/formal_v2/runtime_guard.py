"""No fitting unless the completed A40 gate supports every frozen allocation."""
from pathlib import Path
import argparse, json, hashlib, math, time

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['prefix','A','B','C']);a=p.parse_args()
out=ROOT/'results'/a.method;out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ROOT/'protocol.json').read_text());gate=ROOT.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(gate/'manifest.json')==protocol['validation_manifest_sha256']
assert sha(ROOT/'source/public_api.py')==json.loads((gate/'manifest.json').read_text())['sha256']['source/public_api.py']
record={'method':a.method,'checked_unix':time.time(),'fitting_started':False,
        'gate_job_id':65096288,'gate_manifest_sha256':sha(gate/'manifest.json')}
try:
    complete=gate/'validation_results/COMPLETE.json';data=json.loads(complete.read_text())
    assert data['passed'] and data['native_forward_exact'] and data['original_target_control']['passed']
    assert data['Adam_regression']['same_alpha1_update_exact']
    assert data['Adam_regression']['one_nonfinite_gradient_row_preserved_other_11']
    for alpha in ['1.0','0.97']:
        assert data['gradients'][alpha]['same_forward'] and data['gradients'][alpha]['finite']
    env=json.loads((gate/'validation_results/environment.json').read_text())
    assert env['XLA'] is False and all('A40' in x['device_name'] for x in env['GPU'])
    assert env['versions']==json.loads((ROOT/'source/runtime_versions.json').read_text())
    def timing(name):
        d=data['throughput'][name];v=d['timed_seconds']
        assert len(v)>=2 and all(math.isfinite(x) and x>0 for x in v)
        return d,max(v)
    prefix,if2=timing('IF2_B1');_,native=timing('gradient_1.0_B12');_,discount=timing('gradient_0.97_B12')
    estimates={'prefix':2100*if2*1.10+12*(prefix['graph_build_seconds']+max(prefix['warmup_seconds']))+600,
               'A':400*native*1.20+2100,'B':400*native*1.20+2100,'C':400*discount*1.20+2100}
    limits={'prefix':28800,'A':14400,'B':14400,'C':14400}
    record.update(gate_COMPLETE_sha256=sha(complete),measured_call_seconds_max={'IF2_B1':if2,'native_gradient_B12':native,'discounted_gradient_B12':discount},
                  upper_planning_seconds=estimates,allocation_limit_seconds=limits,
                  allocation_total_GPU_hours=sum(limits.values())/3600,
                  estimate_is_measured_rate_based_not_actual_fit_time=True,
                  all_methods_within_limit=all(estimates[k]<=limits[k] for k in limits))
    assert record['all_methods_within_limit'],'Measured speed does not fit the frozen allocation plan; no fitting allowed'
    if a.method in ['B','C']:
        prefix_root=ROOT/'results/prefix';done=json.loads((prefix_root/'COMPLETE.json').read_text())
        assert sha(prefix_root/'prefix.npz')==done['prefix_sha256']
        terminal=json.loads((prefix_root/'terminal_manifest.json').read_text())
        assert terminal['source_manifest_sha256']==sha(ROOT/'manifest.json')
        assert terminal['sha256']['prefix.npz']==done['prefix_sha256']
        record.update(shared_prefix_sha256=done['prefix_sha256'],prefix_terminal_manifest_sha256=sha(prefix_root/'terminal_manifest.json'))
    record['passed']=True
except BaseException as e:
    record.update(passed=False,error=repr(e))
    (out/'GUARD_REJECTED.json').write_text(json.dumps(record,indent=2)+'\n')
    raise
(out/'runtime_guard.json').write_text(json.dumps(record,indent=2)+'\n')
