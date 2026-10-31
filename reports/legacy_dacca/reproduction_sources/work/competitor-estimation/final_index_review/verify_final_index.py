"""Independently verify the actual968 final index after the parent creates it.
No fake inputs, fitting, scoring, source mutations or status writes.
"""
from pathlib import Path
import collections,csv,hashlib,json,math,statistics,tarfile
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'outputs/competitor_estimation';DEST=OUT/'final_comparison';HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def close(a,b):
 if a is None or b is None:assert a is b,(a,b)
 else:assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-8),(a,b)
def main():
 assert (DEST/'index_manifest.json').exists(),'No actual final index: verification not run'
 manifest=read(DEST/'index_manifest.json');assert manifest['release_ready']
 for name,h in manifest['files'].items():assert sha(DEST/name)==h,name
 result=read(DEST/'results.json');rows=result['rows'];assert result['all_requested_packages_delivered'] and len(rows)==968
 scope=collections.Counter(r['experiment_scope'] for r in rows);assert scope=={'paper100 common starts':800,'six common starts x two repeats':168},scope
 assert len({(r['experiment_scope'],r['package'],r['method'],r['start'],r['replicate']) for r in rows})==968
 assert set(result['reproduction_archives'])=={'Paper two-package comparison','TFP','PyDPF','Dynamax','PFJAX','particles','pypfilt'}
 for name,a in result['reproduction_archives'].items():
  p=Path(a['path']);assert p.is_file() and p.stat().st_size==a['bytes'] and sha(p)==a['sha256'];assert a==manifest['source_delivery_archives'][name]
 for package,source in result['provenance'].items():assert sha(Path(source['summary']))==source['summary_sha256']
 original=[]
 def add(package,method,start,rep,r,ll,fit,precision,floor,failed=False,scope='six common starts x two repeats'):
  original.append({'package':package,'method':method,'start':start,'replicate':rep,'experiment_scope':scope,'logmeanlik':r[ll] if finite(r[ll]) else None,'MCSE':(r.get('MCSE',r.get('mcse')) if finite(r.get('MCSE',r.get('mcse'))) else None),'likelihood_ESS':(r.get('likelihood_ESS',r.get('likelihood_ess')) if finite(r.get('likelihood_ESS',r.get('likelihood_ess'))) else None),'R':r.get('R',r.get('repetitions',r.get('score_repetitions'))),'fit_failed':bool(r[fit]),'evaluation_failed':bool(failed),'precision_passed':bool(r[precision]),'score_floor_saturated':bool(floor)})
 for package,relative in [('TFP','tfp/formal_v2_final'),('PyDPF','pydpf/formal_v1_final'),('Dynamax','dynamax/dacca_formal_final')]:
  folder=OUT/relative;d=read(folder/'summary.json');m=read(folder/'delivery_manifest.json');assert all(m.get(k) for k in ['numeric_audit_passed','visual_review_passed','root_review_passed','release_ready'])
  for n,h in m['sha256'].items():assert sha(folder/n)==h,n
  for r in d['rows']:
   explicit=[r[k] for k in ['evaluation_failed','score_failure','score_failed'] if k in r]
   if explicit:assert len(set(explicit))==1;failed=explicit[0]
   elif package=='PyDPF':failed=not r['finished'] or not finite(r['logmeanlik'])
   elif package=='Dynamax':assert len(r['score_raw'])==r['R'] and all(finite(v) for v in r['score_raw']);failed=False
   else:raise AssertionError('Missing explicit score failure evidence')
   add(package,r['method'],r['start'],r['replicate'],r,'logmeanlik','fit_failed','precision_passed',r['score_floor_saturated'],failed)
  cost=result['cost_by_experiment'][package];close(cost['formal_allocation_seconds'],d['total_allocation_seconds'])
  for method,t in cost['by_method'].items():
   if package=='TFP':
    raw=d['timing_by_method'][method];close(t['fitting_seconds'],raw['fitting_synchronized_calls_seconds']);close(t['score_execution_seconds'],raw['evaluation_PF_execution_seconds']);close(t['score_compile_seconds'],raw['evaluation_explicit_compile_seconds']);close(t['score_process_seconds'],raw['evaluation_process_wall_seconds']);close(t['allocation_seconds'],raw['Slurm_allocation_seconds'])
   elif package=='PyDPF':
    raw=d['timing'][method];close(t['fitting_seconds'],raw['batch_fitting_seconds']);close(t['score_execution_seconds'],raw['score_execution_seconds']);close(t['score_compile_seconds'],raw['score_compilation_seconds']);close(t['score_process_seconds'],raw['score_process_wall_seconds']);close(t['allocation_seconds'],raw['allocation_seconds'])
   else:
    rr=[r for r in d['rows'] if r['method']==method];close(t['fitting_seconds'],sum(r['fit_seconds'] for r in rr));close(t['score_execution_seconds'],sum(r['score_seconds'] for r in rr));close(t['score_compile_seconds'],sum(r['score_compile_seconds'] for r in rr));assert t['allocation_seconds'] is None and t['score_process_seconds'] is None
    close(cost['prior_package_and_gate_allocation_seconds'],d['prior_package_and_gate_allocation_seconds'])
   assert t['fit_compile_seconds'] is None
 for package in ['particles','pypfilt']:
  d=read(OUT/package/'formal_v1/results.json');cost=result['cost_by_experiment'][package];assert cost['resource']=='CPU core seconds';close(cost['formal_allocation_seconds'],d['totals']['Slurm_allocated_CPU_seconds'])
  t=cost['by_method']['PF + external Nelder-Mead'];close(t['fitting_seconds'],d['totals']['charged_fitting_seconds']);close(t['score_process_seconds'],d['totals']['scoring_wall_seconds']);close(t['score_execution_seconds'],sum(r['score_execution_seconds'] for r in d['rows']));close(t['score_compile_seconds'],sum(r['score_compile_seconds'] for r in d['rows']))
  for r in d['rows']:add(package,'PF + external Nelder-Mead',r['start'],r['replicate'],r,'logmeanlik','fit_failed','precision_ok',bool(r['score_raw']) and all(v==-24867.998046875 for v in r['score_raw']),not r['score_finished'])
 pf=read(OUT/'pfjax/evaluation_v2/audit.json');cost=result['cost_by_experiment']['PFJAX'];close(cost['formal_allocation_seconds'],pf['original_job_allocation_seconds']);close(cost['repair_rescoring_allocation_seconds'],pf['repair_allocation_seconds']);close(cost['total_formal_and_repair_seconds'],pf['total_allocation_seconds']);assert cost['total_formal_and_repair_seconds']==2792
 for r in read(OUT/'pfjax/evaluation_v2/endpoints.json'):add('PFJAX',r['method'],r['start'],r['repeat'],r,'logmeanlik','failed','precision_passed',False,r.get('evaluation_failed',r.get('score_failure',False)))
 paper=read(ROOT/'outputs/cuthbert_dacca_paper/audited_v2/audit.json');cost=result['cost_by_experiment']['Paper two-package comparison'];close(cost['formal_allocation_seconds'],sum(x['allocation_seconds'] for x in paper['physical_costs']));assert cost['physical_costs']==paper['physical_costs']
 for r in paper['scores']:add(r['backend'],r['method'],r['start'],None,r,'log_likelihood','fit_failure','precision_ok',r['score_floor_saturated'],r['score_failure'],'paper100 common starts')
 byid=lambda r:(r['experiment_scope'],r['package'],r['method'],r['start'],r['replicate'])
 saved={byid(r):r for r in rows};assert len(original)==968
 for r in original:
  other=saved[byid(r)]
  for key,value in r.items():
   if key in ['logmeanlik','MCSE','likelihood_ESS']:close(value,other[key])
   else:assert value==other[key],(byid(r),key)
  assert other['score_failure']==other['evaluation_failed']
 raw_by_id={}
 def put(package,method,start,rep,raw,scope='six common starts x two repeats'):
  key=(scope,package,method,start,rep);assert key not in raw_by_id;raw_by_id[key]=raw
 for row in paper['scores']:
  raw=row.get('raw_log_likelihoods',[])
  if not row['fit_finite']:assert not raw
  put(row['backend'],row['method'],row['start'],None,raw,'paper100 common starts')
 for package in ['particles','pypfilt']:
  for row in read(OUT/package/'formal_v1/results.json')['rows']:put(package,'PF + external Nelder-Mead',row['start'],row['replicate'],row['score_raw'])
 for row in read(OUT/'dynamax/dacca_formal_final/summary.json')['rows']:put('Dynamax',row['method'],row['start'],row['replicate'],row['score_raw'])
 for row in read(OUT/'pydpf/formal_v1_final/summary.json')['rows']:
  score=read(Path(row['raw_score_path']));assert score['parameter_sha256']==row['parameter_sha256'];put('PyDPF',row['method'],row['start'],row['replicate'],score['raw_loglik'])
 for method in ['IF2','IFAD','ADPF']:
  with tarfile.open(OUT/'tfp/formal_v2_final/evidence'/method/f'{method}_completed_raw.tar.gz') as t:
   for row in json.load(t.extractfile(f'results/{method}/scores.json')):put('TFP',row['method'],row['start'],row['replicate'],row['raw_loglik'])
 with tarfile.open(OUT/'pfjax/evaluation_v2/reproduction.tar.gz') as t:
  for row in read(OUT/'pfjax/evaluation_v2/endpoints.json'):
   score=json.load(t.extractfile(f"evaluation_v2/gl/results/scores/{row['method']}_s{row['start']}_r{row['repeat']}/score.json"));assert score['parameter_sha']==row['parameter_sha256'];put('PFJAX',row['method'],row['start'],row['repeat'],score['raw'])
 assert set(raw_by_id)==set(saved)
 actual_raw_by_experiment=collections.Counter();skipped=0
 for key,raw in raw_by_id.items():
  row=saved[key];assert row['raw_repetitions']==len(raw)
  if not raw:
   assert row['experiment_scope']=='paper100 common starts' and row['fit_failed'] and row['R']==36 and row['evaluation_skipped_due_to_nonfinite_fit'];skipped+=1
  else:
   assert len(raw)==row['R'] and not row['evaluation_skipped_due_to_nonfinite_fit'] and all(finite(v) for v in raw)
   maximum=max(raw);w=[math.exp(v-maximum) for v in raw];mean=statistics.mean(w)
   close(row['logmeanlik'],maximum+math.log(mean));close(row['MCSE'],statistics.stdev(w)/(math.sqrt(len(w))*mean));close(row['likelihood_ESS'],sum(w)**2/sum(v*v for v in w))
  label='Paper two-package comparison' if row['experiment_scope']=='paper100 common starts' else row['package'];actual_raw_by_experiment[label]+=len(raw)
 assert skipped==7 and sum(actual_raw_by_experiment.values())==result['actual_final_raw_PF_repetitions']==37584
 assert actual_raw_by_experiment['Paper two-package comparison']==28548
 assert sum(row['R'] for row in rows)==37836
 groups=collections.defaultdict(list)
 for r in original:groups[(r['experiment_scope'],r['package'],r['method'])].append(r)
 summaries={(s['scope'],s['package'],s['method']):s for s in result['summaries']};assert set(groups)==set(summaries)
 for key,rr in groups.items():
  s=summaries[key];ff=[r for r in rr if finite(r['logmeanlik'])];fit=[r for r in ff if not r['fit_failed']];ok=[r for r in fit if not r['evaluation_failed']]
  expected={'n':len(rr),'finite_n':len(ff),'complete_fit_finite_n':len(fit),'complete_fit_scored_finite_n':len(ok),'complete_finite_n':len(ok),'fit_failures':sum(r['fit_failed'] for r in rr),'evaluation_failures':sum(r['evaluation_failed'] for r in rr),'low_precision':sum(not r['precision_passed'] for r in ff),'all_precision_not_passed':sum(not r['precision_passed'] for r in rr),'floor_saturated':sum(r['score_floor_saturated'] for r in rr)}
  for k,v in expected.items():assert s[k]==v,(key,k)
  close(s['median_all_finite'],statistics.median(r['logmeanlik'] for r in ff) if ff else None);close(s['median_complete_finite'],statistics.median(r['logmeanlik'] for r in ok) if ok else None)
 for name,values in [('endpoints.csv',rows),('method_summary.csv',result['summaries'])]:
  with (DEST/name).open(newline='') as f:csvrows=list(csv.DictReader(f))
  assert len(csvrows)==len(values)
  for saved_csv,value in zip(csvrows,values):
   for key,v in value.items():assert saved_csv[key]==('' if v is None else str(v)),(name,key)
 assert set(result['algorithm_attribution'])=={'PFJAX','PyDPF','TFP','Dynamax','pypfilt','particles','Paper/Cuthbert','Paper/PyPOMP'}
 for a in result['algorithm_attribution'].values():assert a['native'] and a['evaluated'] and a['limitation'] and Path(a['evidence']['summary']).is_file()
 resources=collections.Counter(c['resource'] for c in result['cost_by_experiment'].values());assert resources=={'A40 GPU seconds':5,'CPU core seconds':2}
 report=(DEST/'report_CN.txt').read_text();assert 'GPU占用（GPU秒）' in report and 'CPU占用（核秒）' in report and '未单独测量' in report
 totals={k:sum(s[k] for s in result['summaries']) for k in ['n','finite_n','complete_fit_scored_finite_n','fit_failures','evaluation_failures','low_precision','all_precision_not_passed','floor_saturated']}
 evidence={'passed':True,'actual_records':968,'all_seven_archives_verified':True,'all_source_rows_preserved':True,'medians_and_denominators_recomputed':True,'measured_costs_and_missing_components_verified':True,'CPU_GPU_resources_separated':True,'counts':totals,'actual_final_raw_PF_repetitions':37584,'actual_final_raw_by_experiment':dict(actual_raw_by_experiment),'all_raw_loglik_MCSE_ESS_independently_recomputed':True,'planned_R_sum_not_actual_draws':37836,'evaluations_skipped_due_to_nonfinite_fit':7,'verified_index_artifact_sha256':{n:sha(DEST/n) for n in ['results.json','endpoints.csv','method_summary.csv','report_CN.txt']},'index_manifest_sha256_before_review_attachment':sha(DEST/'index_manifest.json'),'results_sha256':sha(DEST/'results.json'),'no_refit_or_rescore':True,'synthetic_scientific_inputs_used':False}
 (HERE/'final_968_review.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
