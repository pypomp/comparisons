"""Three consecutive original AdamW steps, paired with FCNN execution only.

The GPU run checks full B64 and the actual B4 tail. The CPU run substitutes
B2/J8 but keeps the complete T100 sequence, all model parameters and optimizer.
"""
import argparse, json, time, traceback
from pathlib import Path
import torch, pydpf
from common import ROOT, ref, write, sha, rng, set_rng, state, restore, step, seed_call, ordinary, environment
from training_graph import install

def equal(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes()
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def compare(a,b):
    fields=['loss','outputs','components','model_after_forward','rng','after_step']
    exact={k:equal(a[k],b[k]) for k in fields}
    ga,gb=a['gradient'],b['gradient']; assert ga.keys()==gb.keys()
    assert all((ga[k] is None)==(gb[k] is None) for k in ga)
    va=torch.cat([x.flatten().double() for x in ga.values() if x is not None])
    vb=torch.cat([x.flatten().double() for x in gb.values() if x is not None])
    finite=bool(torch.isfinite(va).all() and torch.isfinite(vb).all())
    rel=float((va-vb).norm()/va.norm().clamp_min(1e-30)) if finite else None
    mx=float((va-vb).abs().max()) if finite else None
    return dict(exact=exact, gradient_elements=va.numel(), gradient_bitwise=equal(va,vb),
                finite=finite,relative_L2=rel,max_abs=mx,
                passed=all(exact.values()) and finite and rel<=2e-6 and mx<=2e-6)

def main(a):
    out=Path(a.output);out.mkdir(exist_ok=False)
    device=torch.device(a.device);gpu=device.type=='cuda';torch.set_num_threads(1)
    data=torch.load(a.data,map_location='cpu',weights_only=False,mmap=True)
    write(out/'data.json',dict(path=a.data,sha256=sha(a.data),normalization=data['normalization']))
    rows=[];pairs=[];evaluations=[];begin=time.perf_counter()
    sizes=(a.batch,4);ids=data['split']['train']
    evaluation_sizes=sorted(set([a.batch,4]+([16,60] if gpu else [])))
    batches={b:tuple(x.contiguous() for x in ref.batch(data,ids[:b],device,99)) for b in evaluation_sizes}
    with pydpf.utils.set_deterministic_mode(True,False):
        write(out/'environment.json',environment(device))
        e,eo,es,eg=ref.build(a.arm,0,device);e.update()
        initial=state(e,eo,es,eg)
        g,go,gs,gg=ref.build(a.arm,0,device);g.update()
        assert equal(initial['model'],ref.cpu(g.SSM.state_dict()))
        torch.save(initial,out/'initial.pt')
        # Construct both graph shapes before any optimization. They must read
        # the same parameter storage after each original AdamW update.
        saved=state(g,go,gs,gg);routers=install(g,sizes,a.particles,gpu)
        assert equal(saved,state(g,go,gs,gg))
        setup=sum(r.setup_seconds for v in routers.values() for r in v.runners.values())
        write(out/'setup.json',dict(seconds=setup,shapes=list(sizes),networks=len(routers)))
        def run(variant,label,start,b,seed):
            model,opt,sched,gen=(e,eo,es,eg) if variant=='eager' else (g,go,gs,gg)
            start_hash=ref.digest(start)
            restore(model,opt,sched,gen,start)
            ref.sync(device);t=time.perf_counter()
            rec=step(model,opt,sched,gen,batches[b],a.particles,seed,out/(label+'_rejected.pt'))
            assert ref.digest(start)==start_hash,'Saved starting checkpoint was mutated'
            ref.sync(device);elapsed=time.perf_counter()-t
            serial=time.perf_counter();torch.save(rec,out/(label+'.pt'));serial=time.perf_counter()-serial
            row=dict(name=label,variant=variant,batch=b,kernel_seconds=rec['kernel_seconds'],
                     complete_step_seconds=elapsed,serialization_seconds=serial,sha256=sha(out/(label+'.pt')))
            rows.append(row);print(json.dumps(row),flush=True)
            return rec
        # Original self repeat of exactly the same full optimizer step.
        cold=run('eager','original_cold',initial,a.batch,310000000)
        self_again=run('eager','original_repeat',initial,a.batch,310000000)
        self_check=compare(cold,self_again);write(out/'self_repeat.json',self_check)
        assert self_check['passed'],self_check
        current=initial
        # The third step uses the actual training tail B4. Each new step starts
        # at the previous ORIGINAL accepted optimizer state, never a tuned point.
        for k,b in enumerate([a.batch,a.batch,4]):
            results={}
            for v in (['eager','network'] if k%2==0 else ['network','eager']):
                results[v]=run(v,f'step{k}_{v}',current,b,310000000+k)
            pair=compare(results['eager'],results['network']);pairs.append(pair)
            write(out/f'pair{k}.json',pair);assert pair['passed'],pair
            current=results['eager']['after_step']
        # Three warm complete B64 steps from a fixed updated checkpoint, with
        # optimizer/gradient serialization costs separated from the main timer.
        for k in range(3):
            run('network',f'cost{k}',current,a.batch,310000100+k)
        # Common ordinary-PF evaluation delegates to native FCNN in no_grad.
        for b in evaluation_sizes:
            eval_raw={}
            for v,model,opt,sched,gen in [('eager',e,eo,es,eg),('network',g,go,gs,gg)]:
                restore(model,opt,sched,gen,current);model.eval();pf=ordinary(model,gen);pf.eval();pf.update()
                seed_call(gen,410000000+b);ref.sync(device);t=time.perf_counter()
                with torch.inference_mode(): pose,truth=ref.prediction(pf,batches[b],a.particles)
                ref.sync(device);elapsed=time.perf_counter()-t
                raw=dict(pose=ref.cpu(pose),truth=ref.cpu(truth),model=ref.cpu(model.SSM.state_dict()),rng=rng(gen))
                torch.save(raw,out/f'eval_{b}_{v}.pt');eval_raw[v]=raw
                evaluations.append(dict(batch=b,variant=v,seconds=elapsed))
            assert equal(eval_raw['eager'],eval_raw['network']) and ref.finite(eval_raw)
        # Loading a saved checkpoint preserves graph parameter addresses.
        restored=run('network','restored_initial',initial,a.batch,310000000)
        restore_check=compare(cold,restored);assert restore_check['passed'],restore_check
        write(out/'result.json',dict(passed=True,arm=a.arm,CPU_only=not gpu,steps=3,
              pairs=pairs,self_repeat=self_check,restore_check=restore_check,rows=rows,
              evaluations=evaluations,evaluation_sizes=evaluation_sizes,elapsed_seconds=time.perf_counter()-begin,
              original_model_filter_loss=True,formal_fits=0,heldout_test_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=64);p.add_argument('--particles',type=int,default=100)
    a=p.parse_args()
    try: main(a)
    except BaseException:
        out=Path(a.output)
        if out.exists(): (out/'exception.txt').write_text(traceback.format_exc())
        raise
