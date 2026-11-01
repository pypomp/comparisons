"""Validate measured CUDA correctness and fixed resource caps before any fitting."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def verify(config,gate):
 expected=config['GPU_gate'];manifest=load(gate/'source_manifest.json');assert sha(gate/'source_manifest.json')==expected['source_manifest_sha256']
 for name,digest in manifest['files'].items():assert sha(gate/name)==digest,name
 results=gate/'results';complete=load(results/'preflight_complete.json');assert complete['job']==str(expected['job']) and complete['passed'] and all(r['passed'] for r in complete['phases'])
 xml=ET.parse(results/'gpu_tests.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
 assert sum(int(s.attrib.get('tests',0)) for s in suites)==34
 assert sum(int(s.attrib.get(k,0)) for s in suites for k in ['failures','errors','skipped'])==0
 assert load(results/'dacca_cuda_correctness.json')['passed']
 regression=load(gate/'driver_validation/driver_checks.json');assert regression['passed'] and regression['device']=='cuda:0'
 benchmark=load(results/'benchmark_complete.json');assert benchmark['passed'] and benchmark['same_seed_alpha_forward_and_final_rng_exact']
 rates=benchmark['public_rates'];warm=next(r for r in rates if r['operation']=='public_IF2' and r['B']==1);assert warm['passes']==3 and warm['complete_swarm']
 gradients={float(r['alpha']):r['warm_median_seconds'] for r in rates if r['operation']=='public_gradient' and r['B']==12};assert set(gradients)=={1.,.97}
 cost={'warm':len(config['rows'])*config['warm_IF2_iterations']*warm['seconds_per_pass'],'A':config['gradient_iterations']*gradients[1.],'B':config['gradient_iterations']*gradients[1.],'C':config['gradient_iterations']*gradients[.97]}
 allocations=config['allocation_by_phase'];assert sum(a['allocation_seconds'] for a in allocations.values())<=config['formal_GPU_hour_cap']*3600
 checks={}
 for phase,seconds in cost.items():
  entry=allocations[phase];required=seconds*config['budget_margin_factor']+entry['preparation_reserve_seconds']+entry['score_reserve_seconds'];checks[phase]=dict(measured_fit_projection_seconds=seconds,required_with_margin_and_reserves_seconds=required,allocation_seconds=entry['allocation_seconds'],passed=required<=entry['allocation_seconds'])
 assert all(c['passed'] for c in checks.values()),json.dumps({'budget_gate_failed':checks})
 resolved=dict(config);resolved['resource_estimate']={'IF2_B1_seconds_per_pass':warm['seconds_per_pass'],'gradient_B12_seconds':max(gradients.values())};resolved['GPU_gate_evidence_sha256']={str(p.relative_to(gate)):sha(p) for p in [results/'preflight_complete.json',results/'gpu_tests.xml',results/'dacca_cuda_correctness.json',gate/'driver_validation/driver_checks.json',results/'benchmark_complete.json']}
 return resolved,dict(passed=True,preflight_job=expected['job'],budget_checks=checks,measured_fit_aggregate_seconds=sum(cost.values()),total_allocation_seconds=sum(a['allocation_seconds'] for a in allocations.values()),no_fitting_performed=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resolved,evidence=verify(load(a.protocol),a.gate);a.output.mkdir(parents=True,exist_ok=True)
 for name,value in [('resolved_protocol.json',resolved),('GPU_gate.json',evidence)]:
  path=a.output/name;text=json.dumps(value,indent=2,allow_nan=False)+'\n'
  if path.exists():assert path.read_text()==text
  else:path.write_text(text)
 print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
