"""Small CPU/CUDA driver regression only; no formal fitting claim."""
from pathlib import Path
import argparse,json,shutil,time
import torch
import driver as d
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output-root',required=True);a=p.parse_args();device=a.device
if device.startswith('cuda'):assert torch.cuda.is_available(), 'CUDA required, no fallback'
config=d.read(ROOT/'protocol_smoke.json');ROOT=Path(a.output_root);ROOT.mkdir(parents=True,exist_ok=True);out=ROOT/'driver_checks'
assert not out.exists(),'Do not overwrite existing driver evidence'
out.mkdir();d.warm(config,out,device,True)
# Match uninterrupted and resumed runs with full optimizer state.
d.grad(config,'B',out,device,True,stop_after=1)
partial=torch.load(out/'B/checkpoint.pt',weights_only=True);assert partial['iteration']==1 and partial['phase']=='pending'
d.grad(config,'B',out,device,True)
full=ROOT/'driver_checks_uninterrupted';full.mkdir();shutil.copytree(out/'warm',full/'warm');d.grad(config,'B',full,device,True)
a=torch.load(out/'B/checkpoint.pt',weights_only=True);b=torch.load(full/'B/checkpoint.pt',weights_only=True)
assert torch.equal(a['parameters'],b['parameters']) and torch.equal(a['generator_state'],b['generator_state'])
for i,s in a['optimizer']['state'].items():
 for k,v in s.items():assert torch.equal(v,b['optimizer']['state'][i][k])
d.grad(config,'C',out,device,True);c=torch.load(out/'C/checkpoint.pt',weights_only=True)
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
try:d.grad(config,'B',fault,device,True)
finally:torch.autograd.grad=orig
f=torch.load(fault/'B/checkpoint.pt',weights_only=True);assert f['rows'][1]['fit_failed'] and f['rows'][1]['accepted_steps']==1 and torch.equal(f['parameters'][0],b['parameters'][0])
saved=torch.load(fault/'B/failures/row_01_last_valid.pt',weights_only=True);assert torch.equal(f['parameters'][1],saved['parameters'][1])
for i in range(3):assert d.read(fault/'B/events'/f'step_{i:05d}.json')['generator_after_sha256']==d.read(full/'B/events'/f'step_{i:05d}.json')['generator_after_sha256']
for values in saved['optimizer']['state'].values():assert values['step'].item()==1
# A finite gradient can overflow Adam's second moment without overflowing params.
overflow=ROOT/'driver_checks_finite_gradient_overflow';overflow.mkdir();shutil.copytree(out/'warm',overflow/'warm');calls=[0]
def huge_finite(*args,**kwargs):
 vals=list(orig(*args,**kwargs))
 if calls[0]==1:
  vals=[g.clone() for g in vals]
  for g in vals:g[1]=1e30
  assert all(torch.isfinite(g).all() for g in vals)
 calls[0]+=1;return tuple(vals)
torch.autograd.grad=huge_finite
try:d.grad(config,'B',overflow,device,True)
finally:torch.autograd.grad=orig
h=torch.load(overflow/'B/checkpoint.pt',weights_only=True);last=torch.load(overflow/'B/failures/row_01_last_valid.pt',weights_only=True)
assert h['rows'][1]['fit_failed'] and h['rows'][1]['accepted_steps']==1 and h['rows'][1]['failure']=='nonfinite Adam optimizer state'
assert torch.equal(h['parameters'][0],b['parameters'][0]) and torch.equal(h['parameters'][1],last['parameters'][1])
for i,state in h['optimizer']['state'].items():
 for name in ['exp_avg','exp_avg_sq']:
  assert torch.isfinite(state[name]).all() and state[name][1]==last['optimizer']['state'][i][name][1]
event=d.read(overflow/'B/events/step_00001.json');assert not event['optimizer_state_finite'][1] and event['natural_parameters_finite'][1]
# Finite estimation coordinates whose natural mapping overflows also fail.
natural=ROOT/'driver_checks_natural_overflow';natural.mkdir();shutil.copytree(out/'warm',natural/'warm');original_step=torch.optim.Adam.step;calls=[0]
def overflowing_step(self,*args,**kwargs):
 result=original_step(self,*args,**kwargs)
 if calls[0]==1:
  with torch.no_grad():self.param_groups[0]['params'][0][1]=1000.
 calls[0]+=1;return result
torch.optim.Adam.step=overflowing_step
try:d.grad(config,'B',natural,device,True)
finally:torch.optim.Adam.step=original_step
n=torch.load(natural/'B/checkpoint.pt',weights_only=True);nlast=torch.load(natural/'B/failures/row_01_last_valid.pt',weights_only=True)
assert n['rows'][1]['fit_failed'] and n['rows'][1]['accepted_steps']==1 and n['rows'][1]['failure']=='nonfinite natural parameter update'
assert torch.equal(n['parameters'][0],b['parameters'][0]) and torch.equal(n['parameters'][1],nlast['parameters'][1])
report=dict(passed=True,device=device,formal=False,finite_gradient_Adam_overflow_detected_and_rolled_back=True,natural_parameter_overflow_detected_and_rolled_back=True,T=12,J=32,rows=2,gradient_steps=3,full_optimizer_resume_exact=True,BC_initial_parameters_and_empty_Adam_hash_identical=True,BC_per_iteration_rng_identical=True,forced_failure_preserves_other_row_and_rng=True,failed_row_last_valid_parameters_and_optimizer_retained=True,swarm_reset=False)
d.atomic(ROOT/'driver_checks.json',report);print(json.dumps(report,indent=2))
