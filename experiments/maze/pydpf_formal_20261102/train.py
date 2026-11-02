"""Frozen supervised maze training. One case/arm, no retries or test selection."""
import argparse, json, math, os, time, traceback
from pathlib import Path
import torch, pydpf
from common import ROOT, ref, write, append, sha, rng, state, restore, step, seed_call, ordinary, environment
from training_graph import install

def evaluate(pf,gen,data,ids,particles,device,seed,save,batch_size=64):
    poses=[];truths=[];keys=[];timings=[]
    pf.eval();pf.update()
    with torch.inference_mode():
        for bi,chunk in enumerate(ids.split(batch_size)):
            key=seed+bi;seed_call(gen,key);ref.sync(device);t=time.perf_counter()
            pose,truth=ref.prediction(pf,ref.batch(data,chunk,device,99),particles)
            ref.sync(device);timings.append(time.perf_counter()-t)
            poses.append(ref.cpu(pose));truths.append(ref.cpu(truth));keys.append(key)
    pose=torch.cat(poses);truth=torch.cat(truths)
    squared=((pose-truth)*1000).square().sum(-1)
    raw=dict(ids=ids,pose=pose,truth=truth,squared_error=squared,keys=keys,rng_after=rng(gen),seconds=timings)
    torch.save(raw,save)  # save before a nonfinite check
    assert ref.finite(raw),'Nonfinite ordinary-PF prediction; raw saved'
    return math.fsum(squared.double().tolist())/len(ids)

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'protocol.json').read_text());device=torch.device(a.device)
    assert a.arm in cfg['arms'] and 0<=a.case<12
    torch.set_num_threads(1);start=time.perf_counter()
    data=torch.load(a.data,map_location='cpu',weights_only=False,mmap=True)
    if a.cpu_fixture:
        assert device.type=='cpu' and not os.getenv('SLURM_JOB_ID')
        meta={'cache_sha256':sha(a.data)}
        epochs,particles,batch_size=2,8,4
    else:
        meta=json.loads(Path(a.data).with_suffix('.json').read_text())
        assert sha(a.data)==meta['cache_sha256'] and data['normalization']['training_only']
        assert [len(data['split'][k]) for k in ['train','validation','test']]==[900,400,700]
        epochs,particles,batch_size=cfg['epochs'],cfg['particles'],cfg['batch_size']
        assert (epochs,particles,batch_size)==(100,100,64)
    train_count=len(data['split']['train'])
    expected_updates=epochs*math.ceil(train_count/batch_size)
    initial_seed=cfg['initialization_seeds'][a.case//2]
    with pydpf.utils.set_deterministic_mode(True,False):
        write(out/'environment.json',environment(device))
        write(out/'identity.json',dict(arm=a.arm,case=a.case,initialization_seed=initial_seed,
              repetition=a.case%2,data_sha256=meta['cache_sha256'],protocol_sha256=sha(ROOT/'protocol.json'),
              CPU_fixture=a.cpu_fixture,epochs=epochs,particles=particles,batch_size=batch_size))
        dpf,opt,sched,gen=ref.build(a.arm,initial_seed,device);dpf.update()
        initial=state(dpf,opt,sched,gen);torch.save(initial,out/'initial.pt')
        batches=tuple(sorted({batch_size,train_count%batch_size}-{0}))
        routers=install(dpf,batches,particles,device.type=='cuda')
        assert ref.digest(initial)==ref.digest(state(dpf,opt,sched,gen))
        write(out/'setup.json',dict(seconds=sum(r.setup_seconds for v in routers.values() for r in v.runners.values())))
        pf=ordinary(dpf,gen)
        train_ids=data['split']['train'];val_ids=data['split']['validation']
        best=math.inf;best_epoch=None;accepted=0;completed=False;failure=None
        ordergen=torch.Generator().manual_seed(210000000+a.case)
        try:
            for epoch in range(epochs):
                epoch_begin=time.perf_counter()
                order=train_ids[torch.randperm(train_count,generator=ordergen)]
                torch.save(dict(ids=order,rng_after=ordergen.get_state()),out/f'order_{epoch:03d}.pt')
                for bi,ids in enumerate(order.split(batch_size)):
                    seed=310000000+a.case*100000+epoch*100+bi
                    t=time.perf_counter()
                    rec=step(dpf,opt,sched,gen,ref.batch(data,ids,device,99),particles,seed,
                             out/f'rejected_{epoch:03d}_{bi:02d}.pt')
                    accepted+=1
                    torch.save(rec['rng'],out/f'rng_{epoch:03d}_{bi:02d}.pt')
                    row=dict(epoch=epoch,batch=bi,n=len(ids),seed=seed,accepted=accepted,
                             loss=float(rec['loss']),components=rec['components'].tolist(),
                             gradient_sha256=rec['gradient_sha256'],before_sha256=rec['before_sha256'],
                             after_sha256=rec['after_sha256'],rng_sha256=ref.digest(rec['rng']),
                             kernel_seconds=rec['kernel_seconds'],step_seconds=time.perf_counter()-t)
                    append(out/'training.jsonl',row)
                    del rec
                sched.step();dpf.update();dpf.eval()
                metric=evaluate(pf,gen,data,val_ids,particles,device,
                        410000000+a.case*100000+epoch*100,out/f'validation_{epoch:03d}.pt',batch_size)
                checkpoint=state(dpf,opt,sched,gen)
                checkpoint.update(epoch=epoch,accepted=accepted,validation_mse=metric,
                                  order_rng=ordergen.get_state())
                torch.save(checkpoint,out/f'epoch_{epoch:03d}.pt')
                if metric<best:best=metric;best_epoch=epoch
                row=dict(epoch=epoch,accepted=accepted,validation_mse=metric,best_epoch=best_epoch,
                         best_validation_mse=best,seconds=time.perf_counter()-epoch_begin,
                         checkpoint_sha256=sha(out/f'epoch_{epoch:03d}.pt'))
                append(out/'epochs.jsonl',row);print(json.dumps(row),flush=True)
            completed=True
        except Exception:
            failure=traceback.format_exc();(out/'fit_exception.txt').write_text(failure)
            torch.save(dict(state=state(dpf,opt,sched,gen),accepted=accepted),out/'failure_last.pt')
        torch.save(dict(state=state(dpf,opt,sched,gen),accepted=accepted),out/'final.pt')
        write(out/'fit_status.json',dict(completed=completed,accepted=accepted,best_epoch=best_epoch,
              best_validation_mse=best if math.isfinite(best) else None,failed=failure is not None,
              elapsed_seconds=time.perf_counter()-start,expected_updates=expected_updates))
        # Even an incomplete fit retains and tests its prior validation-selected
        # checkpoint, explicitly flagged. No checkpoint means no manufactured RMSE.
        scores=[]
        if best_epoch is not None:
            chosen=out/f'epoch_{best_epoch:03d}.pt';checkpoint=torch.load(chosen,map_location='cpu',weights_only=False)
            restore(dpf,opt,sched,gen,checkpoint);pf.eval();pf.update()
            for rep in range(3):
                scores.append(evaluate(pf,gen,data,data['split']['test'],particles,device,
                              510000000+a.case*10000+rep*100,out/f'test_{rep}.pt',batch_size))
            write(out/'test_result.json',dict(selected_epoch=best_epoch,selected_sha256=sha(chosen),
                  repeat_mse=scores,pooled_rmse=math.sqrt(math.fsum(scores)/3),
                  test_trajectories=len(data['split']['test']),ordinary_PF=True,test_selection=False,fit_complete=completed))
        write(out/'COMPLETE.json',dict(arm=a.arm,case=a.case,fit_complete=completed,
              test_available=bool(scores),elapsed_seconds=time.perf_counter()-start,
              initial_model_sha256=ref.digest(initial['model'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--case',required=True,type=int)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--cpu-fixture',action='store_true')
    a=p.parse_args()
    try:main(a)
    except BaseException:
        out=Path(a.output)
        if out.exists(): (out/'execution_exception.txt').write_text(traceback.format_exc())
        raise
