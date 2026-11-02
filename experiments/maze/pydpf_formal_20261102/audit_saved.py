"""Saved tensors only. No model, particle-filter or optimizer imports/calls."""
import argparse, hashlib, json, math
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes()
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def finite(a):
    if torch.is_tensor(a):return bool(torch.isfinite(a).all())
    if isinstance(a,dict):return all(finite(x) for x in a.values())
    if isinstance(a,(tuple,list)):return all(finite(x) for x in a)
    return True
def compare(a,b):
    exact={k:equal(a[k],b[k]) for k in ['loss','outputs','components','model_after_forward','rng','after_step']}
    ga,gb=a['gradient'],b['gradient'];assert ga.keys()==gb.keys()
    assert all((ga[k] is None)==(gb[k] is None) for k in ga)
    va=torch.cat([x.flatten().double() for x in ga.values() if x is not None])
    vb=torch.cat([x.flatten().double() for x in gb.values() if x is not None])
    f=finite(va) and finite(vb)
    rel=float((va-vb).norm()/va.norm().clamp_min(1e-30)) if f else None
    mx=float((va-vb).abs().max()) if f else None
    return dict(exact=exact,gradient_elements=va.numel(),gradient_bitwise=equal(va,vb),
                finite=f,relative_L2=rel,max_abs=mx,
                passed=all(exact.values()) and f and rel<=2e-6 and mx<=2e-6)

def bridge(root):
    result=read(root/'result.json');assert result['passed']
    for row in result['rows']:assert sha(root/(row['name']+'.pt'))==row['sha256']
    checks=[]
    for k in range(3):
        pair=compare(load(root/f'step{k}_eager.pt'),load(root/f'step{k}_network.pt'))
        assert pair==result['pairs'][k] and pair['passed'];checks.append(pair)
    selfcheck=compare(load(root/'original_cold.pt'),load(root/'original_repeat.pt'))
    restorecheck=compare(load(root/'original_cold.pt'),load(root/'restored_initial.pt'))
    assert selfcheck==result['self_repeat'] and restorecheck==result['restore_check']
    assert selfcheck['passed'] and restorecheck['passed']
    for b in result.get('evaluation_sizes',read(root/'setup.json')['shapes']):
        a,bv=load(root/f'eval_{b}_eager.pt'),load(root/f'eval_{b}_network.pt')
        assert equal(a,bv) and finite(a)
    # Initials are full states; successful update 1 is the common input for 2.
    for k in range(3):
        original=load(root/f'step{k}_eager.pt')
        steps=[float(v['step']) for v in original['after_step']['optimizer']['state'].values() if 'step' in v]
        assert steps and all(v==k+1 for v in steps)
    return dict(passed=True,arm=result['arm'],CPU_only=result['CPU_only'],checks=checks,
                self_repeat=selfcheck,restore=restorecheck,result_sha256=sha(root/'result.json'),
                all_evaluations_exact=True)

def training(root,cpu=False):
    identity=read(root/'identity.json');assert identity['CPU_fixture']==cpu
    epochs=[json.loads(x) for x in (root/'epochs.jsonl').read_text().splitlines()]
    calls=[json.loads(x) for x in (root/'training.jsonl').read_text().splitlines()]
    fit=read(root/'fit_status.json');done=read(root/'COMPLETE.json')
    assert fit['completed'] and done['fit_complete'] and fit['accepted']==fit['expected_updates']==len(calls)
    assert len(epochs)==identity['epochs']
    val=[]
    for i,e in enumerate(epochs):
        assert e['epoch']==i and sha(root/f'epoch_{i:03d}.pt')==e['checkpoint_sha256']
        raw=load(root/f'validation_{i:03d}.pt');assert finite(raw)
        sq=((raw['pose']-raw['truth'])*1000).square().sum(-1)
        assert equal(sq,raw['squared_error'])
        mse=math.fsum(sq.double().tolist())/len(sq)
        assert mse==e['validation_mse'];val.append(mse)
    selected=min(range(len(val)),key=lambda k:(val[k],k))
    score=read(root/'test_result.json');assert score['selected_epoch']==selected==fit['best_epoch']
    assert score['selected_sha256']==sha(root/f'epoch_{selected:03d}.pt')
    mse=[];testids=None
    for rep in range(3):
        raw=load(root/f'test_{rep}.pt');assert finite(raw)
        sq=((raw['pose']-raw['truth'])*1000).square().sum(-1)
        assert equal(sq,raw['squared_error'])
        mse.append(math.fsum(sq.double().tolist())/len(sq))
        if testids is None:testids=raw['ids']
        else:assert equal(testids,raw['ids'])
    assert mse==score['repeat_mse'] and math.sqrt(math.fsum(mse)/3)==score['pooled_rmse']
    for epoch in range(len(epochs)):
        order=load(root/f'order_{epoch:03d}.pt')['ids']
        valid=load(root/f'validation_{epoch:03d}.pt')['ids']
        assert not set(order.tolist())&set(valid.tolist())
        assert not set(order.tolist())&set(testids.tolist())
        assert not set(valid.tolist())&set(testids.tolist())
    return dict(passed=True,CPU_only=cpu,identity=identity,updates=len(calls),
                validation_selected_epoch=selected,recomputed_test_RMSE=score['pooled_rmse'])

def audit_gpu(root,receipt,source):
    rec=read(receipt);assert rec['state']=='COMPLETED' and rec['exit_code']=='0:0' and not rec['squeue'].strip()
    manifest=read(source/'manifest.json')['files']
    for name,h in manifest.items():assert sha(source/name)==h and sha(root/name)==h
    hashes=read(Path(receipt).parent/'transfer_hashes.json')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual==hashes
    terminal=read(root/'bridge_terminal.json')
    assert terminal['returncode']==0 and terminal['stage']=='bridge'
    for name,h in terminal['files'].items():assert sha(root/name)==h
    data=read(root/'data/prepared.json')
    assert data['normalization']['training_only'] and not data['sample_only']
    split=data['split'];assert [len(split[k]) for k in ['train','validation','test']]==[900,400,700]
    assert len(set(sum(split.values(),[])))==2000
    generator=torch.Generator().manual_seed(20261101)
    assert torch.randperm(2000,generator=generator).tolist()==split['train']+split['validation']+split['test']
    checks={arm:bridge(root/'bridge_results'/arm) for arm in ['native_dpf','native_stop','discount097']}
    costs={}
    for arm in checks:
        assert not checks[arm]['CPU_only']
        p=root/'bridge_results'/arm;r=read(p/'result.json');env=read(p/'environment.json')
        assert env['job']==rec['job'] and env['node']==rec['node'] and 'A40' in env['gpu']
        assert env['strict_determinism'] and not env['tf32_matmul'] and env['tf32_cudnn'] and not env['cudnn_benchmark']
        assert read(p/'data.json')['sha256']==data['cache_sha256']
        big=max(x['complete_step_seconds'] for x in r['rows'] if x['name'].startswith('cost'))
        tail=next(x['complete_step_seconds'] for x in r['rows'] if x['name']=='step2_network')
        assert sorted(r['evaluation_sizes'])==[4,16,60,64]
        # Each original evaluation warms this same shape before its paired
        # graph-adapted (native inference fallback) evaluation. Cold time is
        # retained separately and the forecast also reserves 600 seconds.
        ev=max(x['seconds'] for x in r['evaluations'] if x['variant']=='network')
        forecast=1.35*(1400*big+100*tail+733*ev)+600
        costs[arm]=dict(warm_B64_step_seconds=big,tail_B4_step_seconds=tail,
                        evaluation_batch_seconds=ev,forecast_seconds=forecast,cap_seconds=2700,passed=forecast<=2700)
        assert costs[arm]['passed'],costs[arm]
    # Same initial parameters across arm packages. Different derivative methods
    # are NOT required to have identical gradients or optimizer updates.
    initial=[load(root/'bridge_results'/a/'initial.pt')['model'] for a in checks]
    assert all(equal(initial[0],x) for x in initial[1:])
    return dict(passed=True,checks=checks,costs=costs,data_sha256=data['cache_sha256'],
                manifest_sha256=sha(source/'manifest.json'),collection_sha256=sha(receipt),
                actual_bridge_GPU_seconds=rec['ElapsedRaw'],planned_total_cap=rec['ElapsedRaw']+36*2700,
                formal_fits=0,scope_cap=108000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--training',action='store_true');p.add_argument('--cpu',action='store_true')
    p.add_argument('--receipt');p.add_argument('--source');a=p.parse_args()
    assert not Path(a.output).exists()
    r=(audit_gpu(Path(a.root),a.receipt,Path(a.source)) if a.receipt else
       training(Path(a.root),a.cpu) if a.training else bridge(Path(a.root)))
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'passed':r['passed'],'output':a.output}))
