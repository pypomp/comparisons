"""Single status read; collect only a completed immutable CPU preflight."""
from pathlib import Path
import subprocess,json,tarfile,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SSH=['ssh','-S','/tmp/greatlakes-codex-login2.sock','-o','BatchMode=yes','-o','ConnectTimeout=15','greatlakes']
def main():
    jobs=json.loads((ROOT/'jobs.json').read_text());job=jobs['preflight_job'];remote=jobs['remote']
    status=subprocess.run(SSH+[f'sacct -X -n -P -j {job} --format=JobID,State,ExitCode,ElapsedRaw,NodeList'],text=True,capture_output=True,check=True).stdout.strip()
    print(status,flush=True);(ROOT/'preflight_sacct.txt').write_text(status+'\n')
    row=next(line.split('|') for line in status.splitlines() if line.split('|')[0]==job)
    if row[1]!='COMPLETED' or row[2]!='0:0':
        print('No collection: job is not successfully completed.');return
    out=ROOT/'gl';out.mkdir(exist_ok=True)
    if not (out/'results/preflight/terminal_manifest.json').exists():
        archive=ROOT/'preflight_results.tar.gz'
        with archive.open('wb') as f:
            subprocess.run(SSH+[f'cd {remote} && tar -czf - results/preflight logs/preflight-{job}.log manifest.json'],stdout=f,check=True)
        with tarfile.open(archive) as t:
            assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
            t.extractall(out)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(out/'manifest.json')==jobs['manifest_sha256']
    source=json.loads((out/'manifest.json').read_text())['sha256']
    for n,h in source.items():assert sha(ROOT/n)==h,n
    p=out/'results/preflight';terminal=json.loads((p/'terminal_manifest.json').read_text())['sha256']
    for n,h in terminal.items():assert sha(out/n)==h,n
    adapter=json.loads((p/'adapter.json').read_text());assert adapter['passed']
    raw=json.loads((p/'PF_raw.json').read_text());assert len(raw)==9
    for i,r in enumerate(raw):
        assert r['iteration']==i and r['warmup']==(i<2)
        for key in ['seed','loglik','X_sha256','W_sha256','rng_sha256','rng_position','observations']:
            assert r['upstream'][key]==r['candidate'][key]
        assert r['upstream']['observations']==600
    med={v:float(np.median([r[v]['seconds'] for r in raw[2:]])) for v in ['upstream','candidate']}
    costs=json.loads((p/'IF2_cost.json').read_text());assert costs['passed'] and costs['J']==5000 and costs['T']==600
    assert len(costs['passes'])==2
    for i,r in enumerate(costs['passes']):
        assert r['smc_class']=='particles.core.SMC'
        with np.load(p/f'IF2_pass{i}.npz') as n:
            assert n['cloud'].shape==(5000,23) and np.isfinite(n['cloud']).all()
            np.testing.assert_array_equal(n['cloud'].mean(0),r['parameter_mean'])
    result=dict(passed=True,job_id=job,state=row[1],exit_code=row[2],allocation_seconds=int(row[3]),node=row[4],
        previous_deployment_failed_seconds=4,source_files_verified=len(source),terminal_files_verified=len(terminal),
        adapter=adapter,PF_warmups=2,PF_timed_calls=7,PF_median_seconds=med,
        upstream_over_candidate_ratio=med['upstream']/med['candidate'],same_seed_native_PF_exact=True,
        IF2_pass_seconds=[r['seconds'] for r in costs['passes']],
        optimizer_benchmark=False,independent_PyPOMP_precision_comparison=False,
        source_manifest_sha256=sha(out/'manifest.json'),terminal_manifest_sha256=sha(p/'terminal_manifest.json'))
    (ROOT/'preflight_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
