"""Recompute statistics from recorded values. No PF or optimization calls."""
from pathlib import Path
import csv,datetime,hashlib,importlib.util,json,math,statistics
import numpy as np
import jax

R=Path(__file__).resolve().parent
CAMPAIGN=R.parent
P=R/'particles/raw'
OUT=R/'particles'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(raw):
 if not raw or any(v is None or not math.isfinite(v) for v in raw):
  return dict(logmeanlik=None,MCSE=None,likelihood_ESS=None,precision_passed=False)
 m=max(raw);w=[math.exp(v-m) for v in raw];s=math.fsum(w);mean=s/len(w)
 mcse=math.sqrt(math.fsum((v-mean)**2 for v in w)/(len(w)-1)/len(w))/mean
 ess=s*s/math.fsum(v*v for v in w)
 return dict(logmeanlik=m+math.log(mean),MCSE=mcse,likelihood_ESS=ess,precision_passed=mcse<=1 and ess>=10)

spec=importlib.util.spec_from_file_location('fit_audit',CAMPAIGN/'particles/audit_collected_partial.py')
fitmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitmodule)
dirs={i:(CAMPAIGN/'particles/formal_v1/gl/results' if i<6 else P/'results')/f'task_{i:02d}' for i in range(24)}
fit=fitmodule.audit(list(range(24)),task_directories=dirs)
assert fit['passed'] and fit['fit_failures']==0
(OUT/'fit_audit.json').write_text(json.dumps(fit,indent=2)+'\n')
endpoints=read(P/'endpoints.json');protocol=read(P/'scoring/protocol.json');complete=read(P/'scoring/COMPLETE.json')
assert len(endpoints)==24 and complete['endpoint_count']==complete['score_records']==24
assert sha(P/'endpoints.json')==protocol['endpoints_sha256']
assert protocol['J']==5000 and protocol['score_seed']==94981000 and protocol['fit_calls']==complete['fit_calls']==0
assert protocol['stage_totals']==[36,72,144,288] and protocol['methods']==['A','B','C']
assert sha(CAMPAIGN/'particles/formal_v1/score.py')=='8aec1ba02982a1ad9bff227b300806eaafa972a1e9fd00b71bef2c5f0ebdc508'
ref=CAMPAIGN/'particles/source/reference/pypomp'
for n,h in protocol['pp_source_sha256'].items():assert sha(ref/n)==h,n
all_scores=read(P/'scoring/all_scores.json')
assert len(all_scores)==24
rows=[];score_keys=set();chunk_keys=set();leaf_keys=set();raw_count=0
for t,e in enumerate(endpoints):
 assert e==read(dirs[t]/'endpoint.json') and e['task_id']==t
 label=f"{e['method']}_s{e['start_id']}_r{e['replicate']}"
 row=read(P/'scoring'/(label+'.json'))
 assert row['endpoint']==e and row==all_scores[t] and row['fit_failed']==e['fit_failed']
 assert hashlib.sha256(np.asarray(e['parameters_natural'],dtype='<f4').tobytes()).hexdigest()==row['natural_float32_sha256']
 key=jax.random.key(94981000)
 for i in (['A','B','C'].index(e['method']),e['start_id'],e['replicate']):key=jax.random.fold_in(key,i)
 k=tuple(map(int,jax.random.key_data(key)));assert list(k)==row['score_key'] and k not in score_keys;score_keys.add(k)
 for i,recorded in enumerate(row['chunk_keys']):
  child=jax.random.fold_in(key,i);k=tuple(map(int,jax.random.key_data(child)))
  assert list(k)==recorded and k not in chunk_keys;chunk_keys.add(k)
  for leaf in np.asarray(jax.random.key_data(jax.random.split(child,(36,1)))).reshape(-1,2):
   lk=tuple(map(int,leaf));assert lk not in leaf_keys;leaf_keys.add(lk)
 n=len(row['raw']);assert n==row['score_calls']==row['repetitions'] and n in [36,72,144,288]
 assert len(row['chunk_keys'])==len(row['evaluation_seconds'])==n//36
 assert all(v>0 for v in row['evaluation_seconds'])
 assert math.isclose(sum(row['evaluation_seconds']),row['total_evaluation_seconds'],abs_tol=1e-9)
 computed=stats(row['raw'])
 for k in ['logmeanlik','MCSE','likelihood_ESS']:
  assert math.isclose(computed[k],row[k],rel_tol=1e-12,abs_tol=1e-9),(label,k)
 assert computed['precision_passed']==row['precision_passed']
 for stage in [36,72,144,288]:
  if stage<n:assert not stats(row['raw'][:stage])['precision_passed'],(label,stage)
 assert row['precision_passed'] or n==288
 floor=all(x==-24867.998046875 for x in row['raw']);assert floor==row['floor_saturated']
 raw_count+=n
 rows.append(dict(task=t,method=e['method'],start=e['start_id'],replicate=e['replicate'],
                  fit_failed=e['fit_failed'],logmeanlik=computed['logmeanlik'],MCSE=computed['MCSE'],
                  likelihood_ESS=computed['likelihood_ESS'],precision_passed=computed['precision_passed'],
                  repetitions=n,floor_saturated=floor,accepted_IF2_passes=e['accepted_IF2_passes'],
                  charged_fit_seconds=e['charged_fit_seconds'],PF_calls=e['PF_calls'],late_PF_calls=e['late_PF_calls'],
                  late_IF2_passes=e['late_IF2_passes'],evaluation_seconds=row['total_evaluation_seconds']))
assert len(leaf_keys)==raw_count and sum(x['precision_passed'] for x in rows)==complete['score_precision_passed']
sacct={x.split('|')[0]:x.split('|') for x in (OUT/'sacct.txt').read_text().splitlines() if x.strip()}
for t in range(24):
 s=sacct['65097887_'+str(t)];assert s[1:3]==['COMPLETED','0:0'] and int(s[5])==1
 rows[t]['Slurm_CPU_seconds']=int(s[3])
 assert int(s[3])+1>=rows[t]['charged_fit_seconds']
assert sacct['65097888'][1:3]==['COMPLETED','0:0']
summary={}
for m in ['A','B']:
 rs=[x for x in rows if x['method']==m]
 summary[m]=dict(endpoints=len(rs),fit_failures=sum(x['fit_failed'] for x in rs),
                precision_passed=sum(x['precision_passed'] for x in rs),
                median_all_scores_descriptive=statistics.median(x['logmeanlik'] for x in rs),
                median_precision_passed_subset=statistics.median(x['logmeanlik'] for x in rs if x['precision_passed']),
                Slurm_CPU_seconds=sum(x['Slurm_CPU_seconds'] for x in rs),
                charged_fit_seconds=sum(x['charged_fit_seconds'] for x in rs),
                accepted_IF2_passes=[x['accepted_IF2_passes'] for x in rs],
                low_precision_tasks=[x['task'] for x in rs if not x['precision_passed']])
pairs=[dict(start=rows[i]['start'],replicate=rows[i]['replicate'],
            B_minus_A=rows[i+1]['logmeanlik']-rows[i]['logmeanlik'],
            both_scores_precision_passed=rows[i]['precision_passed'] and rows[i+1]['precision_passed']) for i in range(0,24,2)]
paired=dict(n=12,median_B_minus_A_descriptive=statistics.median(x['B_minus_A'] for x in pairs),
            positive=sum(x['B_minus_A']>0 for x in pairs),
            both_precision_passed=sum(x['both_scores_precision_passed'] for x in pairs),
            both_precision_passed_positive=sum(x['both_scores_precision_passed'] and x['B_minus_A']>0 for x in pairs),rows=pairs)
audit=dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
           expected_endpoints=24,unique_endpoints=24,raw_PF_values_recomputed=raw_count,
           score_root=94981000,score_keys_recomputed=True,unique_leaf_keys=len(leaf_keys),
           all_endpoint_and_source_and_terminal_hashes_verified=True,fit_calls_during_audit=0,
           PF_calls_during_audit=0,score_statistics_recomputed_independently=True,
           fit_failures=sum(x['fit_failed'] for x in rows),floor_saturated=sum(x['floor_saturated'] for x in rows),
           precision_passed=sum(x['precision_passed'] for x in rows),
           low_precision_tasks=[x['task'] for x in rows if not x['precision_passed']],
           actual_CPU_allocation_seconds=sum(x['Slurm_CPU_seconds'] for x in rows),
           preflight_CPU_seconds=155,score_GPU_allocation_seconds=int(sacct['65097888'][3]),
           per_method=summary,paired=paired,rows=rows,
           interpretation='A uses native PF with external Nelder-Mead; B adds new public IF2 before the same optimizer. Equal 1800-second budget, different iteration counts. Low precision estimates retained; no universal superiority claim.')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
with (OUT/'endpoints.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in audit.items() if k not in ['rows','paired']},indent=2))
print(json.dumps({k:v for k,v in paired.items() if k!='rows'},indent=2))
