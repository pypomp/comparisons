"""Estimate 15-fit costs from the single completed GPU bridge, without PF calls."""
from pathlib import Path
import argparse,hashlib,json,math,statistics

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def estimate(parent):
    audit=read(parent/'bridge_audit.json');assert audit['passed']
    root=parent/'bridge_collection/raw';receipt=read(parent/'bridge_collection/collection.json')
    assert receipt['state']=='COMPLETED' and receipt['exit_code']=='0:0'
    protocol=read(parent/'formal15/protocol.json');assert protocol['fits']==15
    arms=protocol['arms'];results={a:read(root/'bridge_results'/a/'result.json') for a in arms}
    data_seconds=read(root/'data/prepared.json')['elapsed_seconds']
    # This residual is observed allocation time not covered by the data-loader
    # timer or three arm timers. It includes imports/launch/I/O; no causal split
    # is claimed. Charge the entire nonnegative residual to EACH fit forecast.
    shared_residual=max(0.,receipt['ElapsedRaw']-data_seconds-sum(x['elapsed_seconds'] for x in results.values()))
    estimates={}
    for arm,result in results.items():
        assert result['passed'] and not result['CPU_only']
        rows={x['name']:x for x in result['rows']}
        warm=[rows[f'cost{i}']['complete_step_seconds'] for i in range(3)]
        tail=rows['step2_network']['complete_step_seconds']
        ev={x['batch']:x['seconds'] for x in result['evaluations'] if x['variant']=='network'}
        assert set(ev)=={4,16,60,64}
        # 100*(14 full+one tail) train steps, 100*(6 full+one B16) val,
        # 3*(10 full+one B60) independent test batches.
        train_center=1400*statistics.median(warm)+100*tail
        train_upper=1400*max(warm)+100*tail
        evaluation=630*ev[64]+100*ev[16]+3*ev[60]
        setup=read(root/'bridge_results'/arm/'setup.json')['seconds']
        cold_excess=max(0.,rows['original_cold']['complete_step_seconds']-rows['original_repeat']['complete_step_seconds'])
        body_residual=max(0.,result['elapsed_seconds']-sum(x['complete_step_seconds']+x['serialization_seconds'] for x in rows.values())-sum(x['seconds'] for x in result['evaluations'])-setup)
        # A bridge raw record contains gradient, outputs and post-step AdamW;
        # it is larger than an epoch checkpoint. Use its maximum write time as
        # a conservative proxy for each of 100 epoch checkpoints, final+initial.
        serialized=max(x['serialization_seconds'] for x in rows.values())*102
        startup=shared_residual+body_residual+cold_excess+setup
        recurring_center=train_center+evaluation+serialized
        recurring_upper=train_upper+evaluation+serialized
        forecast=recurring_center+startup
        safety=1.35*recurring_upper+startup
        requested=60*math.ceil(safety/60)
        assert requested>0 and requested<=2700,{'arm':arm,'requested':requested,'components':locals().keys()}
        estimates[arm]=dict(warm_complete_step_seconds=warm,tail_step_seconds=tail,
            evaluation_seconds_by_batch={str(k):v for k,v in ev.items()},train_center_seconds=train_center,
            train_upper_seconds=train_upper,evaluation_seconds=evaluation,
            checkpoint_write_allowance_seconds=serialized,cold_step_excess_seconds=cold_excess,
            graph_setup_seconds=setup,arm_unattributed_body_seconds=body_residual,
            shared_unattributed_allocation_seconds_charged_per_fit=shared_residual,
            per_fit_forecast_seconds=forecast,per_fit_request_before_rounding_seconds=safety,
            requested_seconds=requested,requested_minutes=requested//60,
            fits=5,estimated_GPU_seconds=5*forecast,reserved_GPU_seconds=5*requested)
    projected=sum(x['estimated_GPU_seconds'] for x in estimates.values())
    reserved=sum(x['reserved_GPU_seconds'] for x in estimates.values())
    return dict(passed=True,fit_count=15,method_count=3,paired_seeds=5,
       GPU='A40, same type as bridge validation',methods=estimates,
       bridge_actual_GPU_seconds=receipt['ElapsedRaw'],
       training_forecast_GPU_seconds=projected,training_requested_GPU_seconds=reserved,
       training_forecast_GPU_hours=projected/3600,training_requested_GPU_hours=reserved/3600,
       forecast_total_including_bridge_seconds=projected+receipt['ElapsedRaw'],
       requested_total_including_bridge_seconds=reserved+receipt['ElapsedRaw'],
       no_queue_time_in_GPU_hours=True,headroom_fraction_on_recurring_upper=0.35,
       startup_allocation_is_conservative_unattributed_not_causally_measured=True,
       inference_shape_samples_per_method=1,
       limitations=['No formal100-epoch timing yet.','Input batch transfer, per-step RNG/log writes and long-run variability are covered by headroom, not separately measured.','Whole bridge unattributed residual charged per fit; not an isolated process-startup measurement.'],
       source_manifest_sha256=sha(parent/'manifest.json'),bridge_audit_sha256=sha(parent/'bridge_audit.json'),
       collection_sha256=sha(parent/'bridge_collection/collection.json'),protocol_15_sha256=sha(parent/'formal15/protocol.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',default=str(Path(__file__).resolve().parent.parent));p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);assert not out.exists()
    result=estimate(Path(a.parent))
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
