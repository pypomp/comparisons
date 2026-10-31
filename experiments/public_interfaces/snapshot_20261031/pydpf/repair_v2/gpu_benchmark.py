"""GPU-only native PF timing and public API rate checks; no fit selection."""
from pathlib import Path
import gc,hashlib,json,os,statistics,time,traceback
import numpy as np
import torch
import pydpf
import api_adapter as api
import dacca_model as model_code
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,torch.Tensor):return clean(x.detach().cpu().tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def write(n,d):
    p=OUT/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n');t.replace(p)
def sync():torch.cuda.synchronize()
def hash_tensor(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def main():
    assert torch.cuda.is_available() and 'A40' in torch.cuda.get_device_name(0),'CUDA A40 required; no CPU fallback'
    torch.set_num_threads(1);original=api.original_package()
    write('runtime.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),memory=torch.cuda.get_device_properties(0).total_memory,job=os.environ.get('SLURM_JOB_ID'),host=os.environ.get('SLURMD_NODENAME'),dtype='float32',threads=torch.get_num_threads(),candidate_path=pydpf.__file__,upstream_path=original.__file__))
    model=model_code.Dacca('cuda:0',600,True);published=torch.tensor(model_code.to_est(model_code.DATA['theta_published']),device='cuda:0');timings=[]
    timings=json.loads((ROOT/'native_pf_timing_prior.json').read_text())['records']
    write('native_pf_timing_reused.json',dict(source_job=65096176,source_sha256=hashlib.sha256((ROOT/'native_pf_timing_prior.json').read_bytes()).hexdigest(),rerun=False))
    rates=[]
    for B in [1,12]:
      params=published.repeat(B,1)
      if B==1:
        sync();begin=time.perf_counter();warm=api.warm_if2(model,params,5000,3,api.seed(44781000,0));sync();seconds=time.perf_counter()-begin
        assert warm.success and warm.completed_iterations==3 and tuple(warm.parameter_swarm.shape)==(1,5000,23)
        rates.append(dict(operation='public_IF2',B=1,passes=3,seconds=seconds,seconds_per_pass=seconds/3,complete_swarm=True))
      for alpha in [1.,.97]:
        samples=[]
        for rep in range(3):
            p=params.detach().clone().requires_grad_();sync();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            value=-api.factors(model,p,5000,api.seed(44781000,1,B,rep),alpha=alpha).sum(0).reshape(B)/600
            grad=torch.autograd.grad(value.sum(),p)[0];sync();elapsed=time.perf_counter()-begin
            assert torch.isfinite(value).all() and torch.isfinite(grad).all()
            samples.append(dict(repetition=rep,warmup=rep==0,seconds=elapsed,peak_memory=torch.cuda.max_memory_allocated(),forward_hash=hash_tensor(value),gradient_norm=float(torch.linalg.vector_norm(grad)),rng_hash=hash_tensor(model.generator.get_state())))
            del p,value,grad
        rates.append(dict(operation='public_gradient',B=B,alpha=alpha,records=samples,warm_median_seconds=statistics.median(x['seconds'] for x in samples[1:])))
        write('public_api_rates.json',rates);gc.collect();torch.cuda.empty_cache()
    for B in [1,12]:
        one=next(r for r in rates if r['operation']=='public_gradient' and r['B']==B and r['alpha']==1)
        disc=next(r for r in rates if r['operation']=='public_gradient' and r['B']==B and r['alpha']==.97)
        assert all(a['forward_hash']==b['forward_hash'] and a['rng_hash']==b['rng_hash'] for a,b in zip(one['records'],disc['records']))
    write('benchmark_complete.json',dict(passed=True,public_rates=rates,native_pf_medians={n:statistics.median(r['seconds'] for r in timings if r['package']==n and not r['warmup']) for n in ['original','candidate']},same_seed_alpha_forward_and_final_rng_exact=True))
if __name__=='__main__':
    try:main()
    except Exception as e:write('benchmark_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
