from pathlib import Path
import json,collections,time
R=Path(__file__).resolve().parent;O=R/'upstream_tests_with_deps'
inv=json.loads((O/'inventory.json').read_text());pairs=[]
for f in sorted((O/'pairs').glob('*.json')):pairs.append(json.loads(f.read_text()))
summary={'updated_unix':time.time(),'official_test_named_files':inv['official_source_test_named_files'],'expected_file_pairs':len(inv['pytest_files']),'completed_file_pairs':len(pairs),'all_files_attempted':len(pairs)==len(inv['pytest_files']),'side_summary':{},'candidate_only_failing_files':[],'all_upstream_tests_passed':False,'official_conftest_excludes':inv['excluded_by_official_conftest'],'fixed_per_file_timeout_seconds':180,'first_missing_dependency_attempt_preserved':'upstream_tests/INTERRUPTED_FOR_DEPENDENCIES.json'}
for kind in ['original','candidate']:
 rows=[next(x for x in p['results'] if x['kind']==kind) for p in pairs]
 summary['side_summary'][kind]={'exit_code_counts':dict(collections.Counter(str(x['exit_code']) for x in rows)),'finished_pytest_counts':{k:sum(x.get('pytest_counts',{}).get(k,0) for x in rows) for k in ['tests','errors','failures','skipped']},'timeouts':sum(x['state']=='timeout' for x in rows),'note':'XML counts include only processes that emitted XML; timeout or abort partial progress is not a pass'}
summary['candidate_only_failing_files']=[p['file'] for p in pairs if p['candidate_only_failure']]
additional=[]
for pair in pairs:
 rows={x['kind']:x for x in pair['results']}
 identity=lambda row:{(x.get('class',''),x.get('name','')) for x in row.get('failed_tests',[])}
 for classname,name in sorted(identity(rows['candidate'])-identity(rows['original'])):
  additional.append({'file':pair['file'],'class':classname,'test':name})
summary['candidate_additional_failed_tests_including_common_failing_files']=additional
summary['all_upstream_tests_passed']=summary['all_files_attempted'] and all(x['exit_code']==0 for p in pairs for x in p['results'])
for label in ['official_PF_tests','official_PF_tests_module_binding']:
 p=R/label/'COMPLETE.json'
 if p.exists():summary[label]=json.loads(p.read_text())
(R/'test_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['official_PF_tests','official_PF_tests_module_binding']},indent=2))
