"""Gate and independently recompute all 36 final endpoint records from finished audits.
Does not fetch remote artifacts, fit or run PF; writes only this delivery's outputs.
"""
from pathlib import Path
import csv,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'work/competitor-estimation/tfp';OUT=ROOT/'outputs/competitor_estimation/tfp/formal_v2_final'
METHODS={'IF2':64995117,'IFAD':64995118,'ADPF':64995119}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(raw):
 x=np.asarray(raw,np.float64);assert len(x)>=2 and np.isfinite(x).all();w=np.exp(x-x.max());return {'R':len(x),'logmeanlik':float(x.max()+np.log(w.mean())),'MCSE':float(w.std(ddof=1)/np.sqrt(len(x))/w.mean()),'likelihood_ESS':float(w.sum()**2/np.dot(w,w))}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 missing=[str(BASE/f'formal_{m}_audit_{j}/audit.json') for m,j in METHODS.items() if not (BASE/f'formal_{m}_audit_{j}/audit.json').exists()]
 if missing:
  (OUT/'delivery_manifest.json').write_text(json.dumps({'package':'TFP','expected_endpoints':36,'numeric_audit_passed':False,'visual_review_passed':False,'root_review_passed':False,'release_ready':False,'missing_method_audits':missing},indent=2)+'\n');print('Not ready: missing completed method audits');return
 rows=[];summaries={};times={};audit_paths={};hash_evidence={};eval_keys=[]
 import jax
 import jax.numpy as jnp
 source=BASE/'formal_v2';manifest=sha(source/'manifest.json');frozen=read(source/'manifest.json')
 for n,d in frozen['sha256'].items():assert sha(source/n)==d,n
 for method,job in METHODS.items():
  folder=BASE/f'formal_{method}_audit_{job}';a=read(folder/'audit.json');audit_paths[method]=str(folder/'audit.json');retr=read(folder/'raw/retrieval_manifest.json');raw=folder/'raw/results'/method
  assert a['passed'] and a['state']=='COMPLETED' and a['exit_code']=='0:0' and a['scored_endpoints']==12
  assert a['source_manifest_sha256']==retr['source_manifest_sha256']==manifest and a['frozen_source_files_verified']==28
  for n,d in retr['files'].items():assert sha(folder/'raw'/n)==d,n
  sched=retr['scheduler']['stdout'].strip().split('|');assert int(sched[0])==job and sched[2:4]==['COMPLETED','0:0']
  assert a['node']==sched[7] and a['configuration']['GPU']==[{'compute_capability':[8,6],'device_name':'NVIDIA A40'}]
  assert a['configuration']['TF']=='2.20.0' and a['configuration']['TFP']=='0.25.0'
  comp=read(raw/'COMPLETE.json');assert comp['endpoint_count']==12 and comp['independent_scoring_finished'] and comp['manifest']==manifest
  originals=read(raw/'scores.json');assert len(originals)==12 and len(a['endpoints'])==12
  params=np.load(raw/'final_natural.npy');assert params.shape==(12,28)
  for case,(o,e,p) in enumerate(zip(originals,a['endpoints'],params)):
   assert o['case_id']==e['case_id']==case and o['method']==method and o['start']==e['start'] and o['replicate']==e['replicate']
   assert o['parameter_sha256']==e['parameter_sha256']==hashlib.sha256(p.tobytes()).hexdigest()
   rawll=o.get('raw_loglik',[]);computed=stat(rawll) if len(rawll)>1 else {'R':len(rawll),'logmeanlik':None,'MCSE':None,'likelihood_ESS':None}
   for k,v in computed.items():
    if v is not None:np.testing.assert_allclose(v,e[k],rtol=1e-11,atol=1e-9);np.testing.assert_allclose(v,o['summary'][k],rtol=1e-11,atol=1e-9)
   assert o['fitting_failed']==e['fitting_failed'] and o['failed']==e['evaluation_failed'] and o['precision_passed']==e['precision_passed']
   expected=int(np.random.SeedSequence([287041,list(METHODS).index(method),o['start'],o['replicate'],991]).generate_state(1)[0]%(2**31-1));assert o['seed']==e['seed']==expected
   keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.key(expected),i))(jnp.arange(len(rawll),dtype=jnp.uint32));eval_keys.extend(tuple(map(int,k)) for k in np.asarray(jax.random.key_data(keys)))
   floor=bool(len(rawll)>0 and np.all(np.abs(np.asarray(rawll)-600*np.log(1e-18))<.05))
   rows.append({'package':'TFP','method':method,'case_id':case,'start':o['start'],'replicate':o['replicate'],**computed,'fit_failed':bool(o['fitting_failed']),'evaluation_failed':bool(o['failed']),'score_floor_saturated':floor,'precision_passed':bool(o['precision_passed']),'accepted_final_stage_updates':o['accepted_final_stage_updates'],'parameter_sha256':o['parameter_sha256'],'evaluation_seed':expected})
  subset=[x for x in rows if x['method']==method];good=[x for x in subset if not x['fit_failed'] and not x['evaluation_failed'] and x['logmeanlik'] is not None and np.isfinite(x['logmeanlik'])]
  vals=np.array([x['logmeanlik'] for x in good]);summaries[method]={'expected_endpoints':12,'observed_endpoints':len(subset),'finite_success_count':len(good),'median_successful_finite_logmeanlik':float(np.median(vals)) if len(vals) else None,'minimum_successful_finite_logmeanlik':float(vals.min()) if len(vals) else None,'maximum_successful_finite_logmeanlik':float(vals.max()) if len(vals) else None,'fit_failures':sum(x['fit_failed'] for x in subset),'evaluation_failures':sum(x['evaluation_failed'] for x in subset),'imprecise_or_failed_scores':sum(not x['precision_passed'] for x in subset),'score_floor_saturated':sum(x['score_floor_saturated'] for x in subset),'raw_PF_draws':sum(x['R'] for x in subset),'MCSE_range':[min(x['MCSE'] for x in subset if x['MCSE'] is not None),max(x['MCSE'] for x in subset if x['MCSE'] is not None)],'likelihood_ESS_range':[min(x['likelihood_ESS'] for x in subset if x['likelihood_ESS'] is not None),max(x['likelihood_ESS'] for x in subset if x['likelihood_ESS'] is not None)]}
  t=a['timing'];fit=t['IF2_1950_synchronized_calls_seconds'] if method=='IF2' else t['total_fitting_calls_seconds'];prefix=fit if method=='IF2' else t['IF2_prefix_175_synchronized_calls_seconds'] if method=='IFAD' else 0.;grad=fit-prefix
  times[method]={'job_id':job,'node':a['node'],'GPU':'NVIDIA A40','IF2_synchronized_calls_seconds':prefix,'gradient_synchronized_calls_seconds':grad,'fitting_synchronized_calls_seconds':fit,'startup_PF_control_seconds_first_includes_compile':t['ordinary_native_PF_control_seconds_first_includes_trace_compile'],'XLA_probe_process_seconds':t['XLA_probe_process_seconds'],'TF_fit_process_seconds':t['TF_fitting_process_seconds_including_import_control_prepare_IO'],'evaluation_explicit_compile_seconds':t['evaluation_explicit_compile_seconds'],'evaluation_PF_execution_seconds':t['evaluation_PF_execution_seconds'],'evaluation_process_wall_seconds':t['evaluation_process_wall_seconds'],'Slurm_allocation_seconds':int(sched[4]),'Slurm_allocation_GPU_hours':int(sched[4])/3600,'all_other_allocation_seconds':int(sched[4])-fit,'IF2_XLA':method in ['IF2','IFAD'],'gradient_XLA':False if method!='IF2' else None,'full_method_audit_timing':t}
  assert times[method]['all_other_allocation_seconds']>=0
  hash_evidence[method]={'audit_sha256':sha(folder/'audit.json'),'retrieval_manifest_sha256':sha(folder/'raw/retrieval_manifest.json'),'raw_archive_sha256':sha(folder/f'{method}_completed_raw.tar.gz'),'collected_files':len(retr['files']),'producer_terminal_manifest_present':retr['producer_terminal_manifest_present']}
 assert len(rows)==36 and len({(x['method'],x['start'],x['replicate']) for x in rows})==36 and len(eval_keys)==len(set(eval_keys))
 summary={'package':'TFP','experiment':'Original Dacca, fixed iterations, three methods','expected_endpoints':36,'observed_endpoints':36,'rows':rows,'summary_by_method':summaries,'timing_by_method':times,'timing_scope':'Each synchronized call updates the whole batch of12 filters at T600/J5000. IF2=1950 passes, IFAD=175IF2+400Adam, native ADPF=425Adam from raw starts. Different fixed iteration counts, not equal wall time. Slurm occupation includes probe, graph preparation, imports, control PF, fitting, IO, final PF scoring and process overhead. Nested timing scopes must not be added twice. No prefix is physically shared between the IF2 arm and the IFAD arm.','total_allocation_seconds':sum(t['Slurm_allocation_seconds'] for t in times.values()),'total_fitting_calls_seconds':sum(t['fitting_synchronized_calls_seconds'] for t in times.values()),'total_raw_PF_draws':len(eval_keys),'unique_actual_evaluation_keys':len(set(eval_keys)),'source_manifest_sha256':manifest,'audit_paths':audit_paths,'method_evidence':hash_evidence,'source_and_model':{'T':600,'Euler_steps':20,'process_noise':'Gaussian','J':5000,'batch_size':12,'starts':6,'replicates':2,'free_coordinates':23,'TF':'2.20.0','TFP':'0.25.0','evaluation':'Independent original PyPOMP17f8798, J5000, R36 then72/144/288 if ESS<10 or MCSE>1; all failures and imprecise scores retained','resampling':'Native TFP systematic explicitly every step, not upstream adaptive ESS.5 default'},'native_and_extensions':{'ADPF':'Native TFP particle_filter, unbiased_gradients=True, native Keras Adam, local per-filter RNG layout; raw starts, not paper IFAD1','IF2':'Local parameter perturbation and cooling driver using native TFP sequential_monte_carlo/systematic resampling','IFAD':'Same local IF2 prefix, local alpha.97 gradient-discount extension around native TFP, native Keras Adam','XLA':'IF2 kernels selected XLA. Native and discounted gradients failed optional XLA compilation and used ordinary TensorFlow GPU graphs; errors retained. native=false is XLA selection only.'},'limitations':['All three methods run on A40 GPUs, but IF2/IFAD use gl1517 and ADPF gl1504; same physical GPU across all three was not achieved.','Only6 starts x2 repeats; a fixed-iteration experiment does not establish equal-cost superiority.','No original producer terminal manifests or retained full-run initial Adam checkpoints; retrieval SHA and frozen-source/regression evidence are provided.','No whole-experiment speed ratio against other packages is established by this package-only figure.','score_floor_saturated flags raw PF repetitions within0.05 of600*log(1e-18), the original observation floor; it is retained separately from fit failure.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (OUT/'endpoints.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 gate={'package':'TFP','expected_endpoints':36,'observed_endpoints':36,'numeric_audit_passed':True,'visual_review_passed':False,'root_review_passed':False,'release_ready':False,'summary_sha256':sha(OUT/'summary.json'),'endpoint_csv_sha256':sha(OUT/'endpoints.csv'),'method_evidence':hash_evidence,'pending':['Visual inspection of final figures','Independent parent review of whole-package evidence'],'generated_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'delivery_manifest.json').write_text(json.dumps(gate,indent=2)+'\n')
 print(json.dumps({'numeric_audit_passed':True,'endpoint_count':36,'summary_by_method':summaries,'total_allocation_seconds':summary['total_allocation_seconds']},indent=2))
if __name__=='__main__':main()
