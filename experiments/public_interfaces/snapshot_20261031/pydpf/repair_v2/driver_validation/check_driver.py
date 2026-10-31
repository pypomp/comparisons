"""Small CUDA driver regression only; no formal fitting claim."""
from pathlib import Path
import json,shutil,time
import torch
import driver as d
assert torch.cuda.is_available(), 'CUDA required, no fallback'
ROOT=Path(__file__).resolve().parent;config=d.read(ROOT/'protocol_smoke.json');out=ROOT/'driver_checks'
assert not out.exists(),'Do not overwrite existing driver evidence'
out.mkdir();d.warm(config,out,'cuda:0',True)
# Match uninterrupted and resumed runs with full optimizer state.
d.grad(config,'B',out,'cuda:0',True,stop_after=1)
partial=torch.load(out/'B/checkpoint.pt',weights_only=True);assert partial['iteration']==1 and partial['phase']=='pending'
d.grad(config,'B',out,'cuda:0',True)
full=ROOT/'driver_checks_uninterrupted';full.mkdir();shutil.copytree(out/'warm',full/'warm');d.grad(config,'B',full,'cuda:0',True)
a=torch.load(out/'B/checkpoint.pt',weights_only=True);b=torch.load(full/'B/checkpoint.pt',weights_only=True)
assert torch.equal(a['parameters'],b['parameters']) and torch.equal(a['generator_state'],b['generator_state'])
for i,s in a['optimizer']['state'].items():
 for k,v in s.items():assert torch.equal(v,b['optimizer']['state'][i][k])
d.grad(config,'C',out,'cuda:0',True);c=torch.load(out/'C/checkpoint.pt',weights_only=True)
assert a['initial_parameter_sha256']==c['initial_parameter_sha256'] and a['initial_Adam_state_sha256']==c['initial_Adam_state_sha256']
for i in range(3):
 x=d.read(out/'B/events'/f'step_{i:05d}.json');y=d.read(out/'C/events'/f'step_{i:05d}.json');assert x['seed']==y['seed'] and x['generator_after_sha256']==y['generator_after_sha256']
# Forced row-local nonfinite gradient only in a small test; no production retry.
fault=ROOT/'driver_checks_forced_failure';fault.mkdir();shutil.copytree(out/'warm',fault/'warm');orig=torch.autograd.grad;calls=[0]
def faulty(*args,**kwargs):
 vals=list(orig(*args,**kwargs))
 if calls[0]==1:
  vals=[g.clone() for g in vals]
  for g in vals:g[1]=float('nan')
 calls[0]+=1;return tuple(vals)
torch.autograd.grad=faulty
try:d.grad(config,'B',fault,'cuda:0',True)
finally:torch.autograd.grad=orig
f=torch.load(fault/'B/checkpoint.pt',weights_only=True);assert f['rows'][1]['fit_failed'] and f['rows'][1]['accepted_steps']==1 and torch.equal(f['parameters'][0],b['parameters'][0])
saved=torch.load(fault/'B/failures/row_01_last_valid.pt',weights_only=True);assert torch.equal(f['parameters'][1],saved['parameters'][1])
for i in range(3):assert d.read(fault/'B/events'/f'step_{i:05d}.json')['generator_after_sha256']==d.read(full/'B/events'/f'step_{i:05d}.json')['generator_after_sha256']
for values in saved['optimizer']['state'].values():assert values['step'].item()==1
report=dict(passed=True,device='cuda:0',formal=False,T=12,J=32,rows=2,gradient_steps=3,full_optimizer_resume_exact=True,BC_initial_parameters_and_empty_Adam_hash_identical=True,BC_per_iteration_rng_identical=True,forced_failure_preserves_other_row_and_rng=True,failed_row_last_valid_parameters_and_optimizer_retained=True,swarm_reset=False)
d.atomic(ROOT/'driver_checks.json',report);print(json.dumps(report,indent=2))
