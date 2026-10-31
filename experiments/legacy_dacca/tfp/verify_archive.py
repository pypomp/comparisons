"""Verify the extracted delivery using saved records only (Python + NumPy).
Does not run any particle filter or connect to the cluster.
"""
from pathlib import Path
import hashlib,io,json,tarfile
import numpy as np
R=Path(__file__).resolve().parent
j=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
def stat(x):
 a=np.asarray(x,np.float64);assert len(a)>1 and np.isfinite(a).all();w=np.exp(a-a.max())
 return dict(R=len(a),logmeanlik=float(a.max()+np.log(w.mean())),MCSE=float(w.std(ddof=1)/np.sqrt(len(w))/w.mean()),likelihood_ESS=float(w.sum()**2/np.dot(w,w)))
def main():
 s=j(R/'summary.json');assert len(s['rows'])==s['expected_endpoints']==36;source=j(R/'source/manifest.json')
 for name,digest in source['sha256'].items():assert sha((R/'source'/name).read_bytes())==digest,name
 assert sha((R/'source/manifest.json').read_bytes())==s['source_manifest_sha256'];draws=0
 for name,digest in j(R/'source/reference_source_hashes.json')['sha256'].items():assert sha((R/'reference_source/pypomp'/name).read_bytes())==digest,name
 for name,digest in j(R/'source/source_versions.json')['source_files'].items():assert sha((R/'reference_source/tensorflow_probability/python/experimental/mcmc'/name).read_bytes())==digest,name
 for method,job in [('IF2',64995117),('IFAD',64995118),('ADPF',64995119)]:
  archive=R/'evidence'/method/f'{method}_completed_raw.tar.gz';assert sha(archive.read_bytes())==s['method_evidence'][method]['raw_archive_sha256']
  with tarfile.open(archive) as t:
   data=lambda n:t.extractfile(n).read();read=lambda n:json.loads(data(n))
   ret=read('retrieval_manifest.json');assert sha(data('retrieval_manifest.json'))==s['method_evidence'][method]['retrieval_manifest_sha256']
   for name,digest in ret['files'].items():assert sha(data(name))==digest,name
   sched=ret['scheduler']['stdout'].strip().split('|');assert int(sched[0])==job and sched[2:4]==['COMPLETED','0:0']
   base=f'results/{method}/';rows=read(base+'scores.json');comp=read(base+'COMPLETE.json');assert comp['endpoint_count']==len(rows)==12 and comp['independent_scoring_finished']
   params=np.load(io.BytesIO(data(base+'final_natural.npy')),allow_pickle=False);assert params.shape==(12,28)
   for row,param in zip(rows,params):
    dest=next(x for x in s['rows'] if x['method']==method and x['start']==row['start'] and x['replicate']==row['replicate'])
    assert sha(param.tobytes())==row['parameter_sha256']==dest['parameter_sha256'];raw=row.get('raw_loglik',[]);draws+=len(raw)
    if len(raw)>1:
     for key,val in stat(raw).items():np.testing.assert_allclose(val,dest[key],rtol=1e-11,atol=1e-9)
    assert row['fitting_failed']==dest['fit_failed'] and row['failed']==dest['evaluation_failed'] and row['precision_passed']==dest['precision_passed']
   stages=[('IF2',1950)] if method=='IF2' else [('prefix',175),('Adam',400)] if method=='IFAD' else [('Adam',425)]
   alltime=0.
   for stage,M in stages:
    st=read(base+stage+'/COMPLETE.json');calls=[json.loads(line) for line in data(base+stage+'/calls.jsonl').decode().splitlines()]
    assert st['target']==M and st['done'] and [c['iteration'] for c in calls]==list(range(st['next_iteration']))
    assert st['next_iteration']==M or st.get('all_filters_failed',False)
    accepted=np.array([c['active'] for c in calls],bool).sum(0);assert accepted.tolist()==st['accepted_by_filter']
    np.testing.assert_array_equal([c['seconds'] for c in calls],st['calls_seconds']);alltime+=sum(st['calls_seconds'])
   np.testing.assert_allclose(alltime,s['timing_by_method'][method]['fitting_synchronized_calls_seconds'],rtol=1e-11,atol=1e-8)
 assert draws==s['total_raw_PF_draws'];assert len({(x['method'],x['start'],x['replicate']) for x in s['rows']})==36
 print(json.dumps({'passed':True,'endpoints':36,'raw_PF_draws':draws,'scope':'Saved raw-statistic/source/parameter/scheduler/call-record verification only; no refitting or rescoring'}))
if __name__=='__main__':main()
